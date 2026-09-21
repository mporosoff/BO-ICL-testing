import hashlib
import json

import numpy as np
import pytest

from dataclasses import replace
from boicl.embedding_cache import (
    EmbeddingCache,
    EmbeddingSpec,
    create_embedding_cache,
    sha256_text,
)


SPEC = EmbeddingSpec("mock-embeddings", 2)
ROWS = [{"candidate_id": str(i), "procedure": f"procedure {i}"} for i in range(5)]


def indexed(inputs):
    return {
        "data": [
            {"index": i, "embedding": [float(text[-1]), 1.0]}
            for i, text in enumerate(inputs)
        ]
    }


def rewrite_hash(package, kind):
    manifest = json.loads((package / "manifest.json").read_text())
    path = package / manifest["files"][kind]["name"]
    manifest["files"][kind]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (package / "manifest.json").write_text(json.dumps(manifest))


def test_exact_formatter_preserves_body_and_vector_spaces():
    body = "  50 °C\n\noriginal  whitespace\t"
    llm = EmbeddingSpec.crystal_llm()
    gp = EmbeddingSpec.embedding_gp()
    assert llm.format(body) == "experimental procedure: " + body
    assert gp.format(body) == body
    assert llm.dimensions == 3072 and gp.dimensions == 1536
    assert llm.fingerprint != gp.fingerprint


@pytest.mark.parametrize("bad_index", [0, 2, 4])
def test_partial_provider_responses_preserve_indices_and_resume(tmp_path, bad_index):
    cache = EmbeddingCache(tmp_path / "cache", SPEC)

    def partial(inputs):
        return {
            "data": [
                item
                for item in reversed(indexed(inputs)["data"])
                if item["index"] != bad_index
            ]
        }

    report = cache.prepare(ROWS, partial)
    assert report["missing_ids"] == [str(bad_index)]
    expected = [r for r in ROWS if r["candidate_id"] != str(bad_index)]
    np.testing.assert_equal(
        cache.matrix(expected)[:, 0], [float(r["candidate_id"]) for r in expected]
    )
    reloaded = EmbeddingCache(tmp_path / "cache", SPEC)
    calls = []

    def finish(inputs):
        calls.append(inputs)
        return indexed(inputs)

    assert reloaded.prepare(ROWS, finish)["hit_count"] == 5
    assert calls == [[f"procedure {bad_index}"]]
    np.testing.assert_equal(reloaded.matrix(ROWS)[:, 0], range(5))


def test_duplicate_and_invalid_provider_indices_are_not_assigned(tmp_path):
    cache = EmbeddingCache(tmp_path, SPEC)

    def malformed(inputs):
        return {
            "data": [
                {"index": 0, "embedding": [4, 5]},
                {"index": 0, "embedding": [6, 7]},
                {"index": 99, "embedding": [8, 9]},
                {"index": 2, "embedding": [2, 1]},
            ]
        }

    report = cache.prepare(ROWS[:3], malformed)
    assert report["missing_ids"] == ["0", "1"]
    np.testing.assert_equal(cache.matrix(ROWS[2:3]), [[2, 1]])


def test_cancellation_checkpoints_inflight_batch_and_stops_new_requests(tmp_path):
    stop = [False]
    calls = []

    def provider(inputs):
        calls.append(inputs)
        stop[0] = True
        return indexed(inputs)

    cache = EmbeddingCache(tmp_path, SPEC)
    report = cache.prepare(ROWS, provider, batch_size=2, cancelled=lambda: stop[0])
    assert report["cancelled"] and report["hit_count"] == 2 and len(calls) == 1
    reloaded = EmbeddingCache(tmp_path, SPEC)
    assert reloaded.coverage(ROWS)["missing_ids"] == ["2", "3", "4"]


def test_batch_exception_stops_and_preserves_previous_batch(tmp_path):
    cache = EmbeddingCache(tmp_path, SPEC)
    calls = []

    def provider(inputs):
        calls.append(inputs)
        if len(calls) == 2:
            raise RuntimeError("provider unavailable")
        return indexed(inputs)

    report = cache.prepare(ROWS, provider, batch_size=2)
    assert report["hit_count"] == 2 and len(calls) == 2
    assert EmbeddingCache(tmp_path, SPEC).coverage(ROWS)["hit_count"] == 2


def test_transfer_checksums_identity_and_same_length_wrong_ids(tmp_path):
    source = EmbeddingCache(tmp_path / "source", SPEC)
    source.prepare(ROWS, indexed)
    source.export_package(tmp_path / "export")
    target = EmbeddingCache(tmp_path / "target", SPEC)
    changed = [dict(row, candidate_id="other" + row["candidate_id"]) for row in ROWS]
    report = target.import_package(tmp_path / "export", changed)
    assert report["hit_count"] == 0 and len(report["incompatible_rows"]) == 5
    wrong_model = EmbeddingCache(tmp_path / "wrong", EmbeddingSpec("other-model", 2))
    with pytest.raises(ValueError, match="identity differs"):
        wrong_model.import_package(tmp_path / "export", ROWS)
    path = tmp_path / "export" / "rows.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        target.import_package(tmp_path / "export", ROWS)


def test_transfer_quarantines_nonfinite_malformed_and_duplicate_rows(tmp_path):
    cache = EmbeddingCache(tmp_path / "source", SPEC)
    cache.prepare(ROWS, indexed)
    package = tmp_path / "export"
    cache.export_package(package)
    vectors = np.load(package / "vectors.npy", allow_pickle=False)
    vectors[0, 1] = np.nan
    np.save(package / "vectors.npy", vectors, allow_pickle=False)
    rewrite_hash(package, "vectors")
    rows = [
        json.loads(line) for line in (package / "rows.jsonl").read_text().splitlines()
    ]
    rows[1]["input_sha256"] = "incorrect"
    rows[3] = dict(rows[2])
    (package / "rows.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    rewrite_hash(package, "rows")
    target = EmbeddingCache(tmp_path / "target", SPEC)
    report = target.import_package(package, ROWS)
    assert report["validated_hits"] == ["2", "4"]
    assert len(report["incompatible_rows"]) == 2
    assert len(report["duplicates"]) == 1


def test_object_array_never_deserialized(tmp_path):
    cache = EmbeddingCache(tmp_path / "source", SPEC)
    cache.prepare(ROWS, indexed)
    package = tmp_path / "export"
    cache.export_package(package)
    np.save(
        package / "vectors.npy", np.array([[{"danger": "object"}, 1]] * 5, dtype=object)
    )
    rewrite_hash(package, "vectors")
    with pytest.raises(ValueError, match="allow_pickle=False"):
        EmbeddingCache(tmp_path / "target", SPEC).import_package(package, ROWS)


def test_dynamic_queries_have_separate_namespace_exact_hash_and_resume(tmp_path):
    cache = EmbeddingCache(tmp_path, SPEC)
    cache.prepare(ROWS[:1], indexed)
    calls = []

    def provider(inputs):
        calls.append(inputs)
        return indexed(inputs)

    np.testing.assert_equal(cache.query("procedure 0", provider), [0, 1])
    reloaded = EmbeddingCache(tmp_path, SPEC)
    np.testing.assert_equal(reloaded.query("procedure 0"), [0, 1])
    assert calls == [["procedure 0"]]
    assert reloaded.coverage(ROWS)["hit_count"] == 1
    with pytest.raises(ValueError, match="missing"):
        reloaded.query("procedure 0 ")


@pytest.mark.parametrize("vector", [[1], [1, float("inf")], ["a", "b"]])
def test_malformed_provider_vectors_remain_misses(tmp_path, vector):
    cache = EmbeddingCache(tmp_path, SPEC)
    report = cache.prepare(
        ROWS[:1], lambda _: {"data": [{"index": 0, "embedding": vector}]}
    )
    assert report["missing_ids"] == ["0"]


def test_normalized_legacy_vectors_retain_representation(tmp_path):
    spec = EmbeddingSpec(
        "mock", 2, preprocessing_version="float32-l2-v1", representation="l2-normalized"
    )
    cache = EmbeddingCache(tmp_path, spec)
    with pytest.raises(ValueError, match="raw cache"):
        cache.prepare(ROWS, indexed)
    with pytest.raises(ValueError, match="unit norm"):
        cache._vector([2.0, 0.0])


def test_normalized_legacy_reuse_combines_with_raw_without_reembedding(tmp_path):
    spec = EmbeddingSpec(
        "mock", 2, "experimental procedure: {procedure}", "crystal-prefixed-v1"
    )
    normalized_spec = replace(
        spec, representation="l2-normalized", preprocessing_version="float32-l2-v1"
    )
    legacy = EmbeddingCache(tmp_path / "legacy", normalized_spec)
    record = legacy._records(ROWS[:1])[0]
    legacy._entries[legacy._key(record)] = (
        record,
        np.array([0.6, 0.8], dtype=np.float32),
    )
    legacy.export_package(tmp_path / "transfer")
    combined = create_embedding_cache(tmp_path / "retrieval", spec)
    report = combined.import_package(tmp_path / "transfer", ROWS)
    assert report["normalized_legacy_hits"] == 1
    assert report["imported_representation"] == "l2-normalized"
    calls = []

    def provider(inputs):
        calls.append(inputs)
        return {
            "data": [
                {"index": index, "embedding": [3.0, 4.0]}
                for index, _ in enumerate(inputs)
            ]
        }

    report = combined.prepare(ROWS[:3], provider, batch_size=2)
    assert calls == [
        ["experimental procedure: procedure 1", "experimental procedure: procedure 2"]
    ]
    assert (
        report["hit_count"] == 3
        and report["normalized_legacy_hits"] == 1
        and report["raw_hits"] == 2
    )
    np.testing.assert_allclose(combined.matrix(ROWS[:3]), [[0.6, 0.8]] * 3)
    np.testing.assert_equal(combined.raw.matrix(ROWS[1:3]), [[3, 4]] * 2)
    with pytest.raises(ValueError, match="Raw output is unavailable"):
        combined.matrix(ROWS[:3], normalize=False)
    with pytest.raises(ValueError, match="identity differs"):
        EmbeddingCache(tmp_path / "gp", spec).import_package(
            tmp_path / "transfer", ROWS
        )
    combined.export_package(tmp_path / "collection")
    restored = create_embedding_cache(tmp_path / "restored", spec)
    assert restored.import_package(tmp_path / "collection", ROWS[:3])["hit_count"] == 3
    np.testing.assert_allclose(restored.matrix(ROWS[:3]), combined.matrix(ROWS[:3]))
    np.testing.assert_allclose(restored.query("procedure 4", provider), [0.6, 0.8])
    reloaded = create_embedding_cache(tmp_path / "restored", spec)
    np.testing.assert_allclose(reloaded.query("procedure 4"), [0.6, 0.8])


def test_completed_prepare_compacts_only_verified_obsolete_snapshots(tmp_path):
    cache = EmbeddingCache(tmp_path, SPEC)
    unrelated = tmp_path / "vectors.personal.npy"
    unrelated.write_bytes(b"user-owned file")
    report = cache.prepare(ROWS, indexed, batch_size=1)
    assert report["pruned_checkpoint_files"] == 8
    assert unrelated.read_bytes() == b"user-owned file"
    assert len(list(tmp_path.glob("rows.*.jsonl"))) == 1
    np.testing.assert_equal(EmbeddingCache(tmp_path, SPEC).matrix(ROWS)[:, 0], range(5))


def test_two_cache_instances_refresh_successful_rows_before_generation(tmp_path):
    first = EmbeddingCache(tmp_path, SPEC)
    second = EmbeddingCache(tmp_path, SPEC)
    first.prepare(ROWS[:2], indexed)
    calls = []

    def provider(inputs):
        calls.append(inputs)
        return indexed(inputs)

    second.prepare(ROWS[:3], provider)
    assert calls == [["procedure 2"]]
    np.testing.assert_equal(
        EmbeddingCache(tmp_path, SPEC).matrix(ROWS[:3])[:, 0], range(3)
    )


def test_three_campaigns_share_missing_only_cache_preparation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    spec = replace(SPEC, input_template="experimental procedure: {procedure}")
    caches = [create_embedding_cache(tmp_path, spec) for _ in range(3)]
    start = threading.Barrier(3)
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def provider(inputs):
        calls.append(inputs)
        entered.set()
        assert release.wait(5)
        return indexed(inputs)

    def prepare(cache):
        start.wait(timeout=5)
        return cache.prepare(ROWS, provider)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(prepare, cache) for cache in caches]
        try:
            assert entered.wait(5)
            assert not any(future.done() for future in futures)
        finally:
            release.set()
        reports = [future.result(timeout=10) for future in futures]
    assert len(calls) == 1
    assert sum(report["generated"] for report in reports) == len(ROWS)
    assert all(report["hit_count"] == len(ROWS) for report in reports)
    expected = caches[0].matrix(ROWS)
    for cache in caches[1:]:
        np.testing.assert_array_equal(cache.matrix(ROWS), expected)
    np.testing.assert_array_equal(
        create_embedding_cache(tmp_path, spec).matrix(ROWS), expected
    )
