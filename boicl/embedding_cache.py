"""Exact-input embedding storage and non-executable, checksummed transfer packages.

Provider calls are deliberately injected: callers own credentials, rate limits and
bounded retries. A completed batch is checkpointed even if cancellation follows.
No pickle or provider client is constructed by this module.
"""
from dataclasses import asdict, dataclass, replace
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import uuid

import numpy as np


_CACHE_LOCKS = {}
_CACHE_LOCK_GUARD = threading.Lock()


def _cache_lock(directory):
    key = os.path.normcase(str(Path(directory).resolve()))
    with _CACHE_LOCK_GUARD:
        return _CACHE_LOCKS.setdefault(key, threading.RLock())


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EmbeddingSpec:
    model: str
    dimensions: int
    input_template: str = "{procedure}"
    input_version: str = "procedure-only-v1"
    provider: str = "openai"
    preprocessing_version: str = "raw-v1"
    representation: str = "raw"

    def __post_init__(self):
        if self.dimensions <= 0 or self.representation not in ("raw", "l2-normalized"):
            raise ValueError("Invalid embedding dimensions or representation")
        if self.input_template.count("{procedure}") != 1:
            raise ValueError("Embedding template must contain one {procedure}")

    @classmethod
    def crystal_llm(cls):
        return cls(
            "text-embedding-3-large",
            3072,
            "experimental procedure: {procedure}",
            "crystal-prefixed-v1",
        )

    @classmethod
    def embedding_gp(cls):
        return cls("text-embedding-ada-002", 1536)

    def format(self, procedure):
        if not isinstance(procedure, str) or not procedure.strip():
            raise ValueError("Embedding inputs must be nonempty procedure strings")
        # Deliberately do not strip or normalize the original Unicode/body.
        return self.input_template.replace("{procedure}", procedure)

    @property
    def fingerprint(self):
        return sha256_text(json.dumps(asdict(self), sort_keys=True))


def _atomic_json(path, payload):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_file(directory, name):
    if (
        not isinstance(name, str)
        or Path(name).name != name
        or "/" in name
        or "\\" in name
    ):
        raise ValueError("Package filenames must be simple local filenames")
    path = (directory / name).resolve()
    if path.parent != directory.resolve():
        raise ValueError("Package file escapes its directory")
    return path


class EmbeddingCache:
    """One exact vector-space identity with candidate and dynamic-query namespaces."""

    def __init__(self, directory, spec):
        self.directory = Path(directory)
        self.spec = spec
        self._entries = {}
        self._lock = _cache_lock(directory)
        self.last_report = {}
        self.source_commits = set()
        if (self.directory / "manifest.json").exists():
            self.import_package(self.directory, checkpoint=False)

    def _records(self, records, namespace="candidate"):
        result, seen = [], set()
        for record in records:
            candidate_id = str(record["candidate_id"])
            if candidate_id in seen:
                raise ValueError("Duplicate candidate ID in requested embedding corpus")
            seen.add(candidate_id)
            exact = self.spec.format(record["procedure"])
            result.append(
                {
                    "candidate_id": candidate_id,
                    "input_text": exact,
                    "input_sha256": sha256_text(exact),
                    "namespace": namespace,
                    "source": record.get("source", "campaign input"),
                }
            )
        return result

    @staticmethod
    def _key(row):
        return row.get("namespace", "candidate"), row["input_sha256"]

    def coverage(self, records):
        rows = self._records(records)
        with self._lock:
            hits = [r["candidate_id"] for r in rows if self._key(r) in self._entries]
        hit_ids = set(hits)
        return {
            "requested": len(rows),
            "validated_hits": hits,
            "hit_count": len(hits),
            "missing_ids": [
                r["candidate_id"] for r in rows if r["candidate_id"] not in hit_ids
            ],
            "spec": asdict(self.spec),
        }

    def matrix(self, records, normalize=False):
        rows = self._records(records)
        with self._lock:
            missing = [
                r["candidate_id"] for r in rows if self._key(r) not in self._entries
            ]
            if missing:
                reasons = self.last_report.get("errors", [])
                explanation = (
                    f"; last preparation error: {reasons[0]['reason']}"
                    if reasons and "reason" in reasons[0]
                    else ""
                )
                raise ValueError(
                    f"Missing {len(missing)} validated embeddings: {missing[:5]}{explanation}"
                )
            values = np.array(
                [self._entries[self._key(r)][1] for r in rows], dtype=np.float32
            )
        values = values.reshape(len(rows), self.spec.dimensions)
        if normalize:
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError(
                    "Zero vectors cannot be normalized for cosine retrieval"
                )
            values = values / norms
        return values

    def _vector(self, value):
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or array.shape != (self.spec.dimensions,):
            raise ValueError("Embedding has invalid numeric type or dimension")
        if not np.all(np.isfinite(array)):
            raise ValueError("Embedding contains nonfinite values")
        vector = array.astype(np.float32)
        if not np.all(np.isfinite(vector)):
            raise ValueError("Embedding overflows float32")
        if self.spec.representation == "l2-normalized" and not np.isclose(
            np.linalg.norm(vector), 1, atol=1e-4
        ):
            raise ValueError("Normalized embedding must have unit norm")
        return vector

    def prepare(self, records, embedder, batch_size=64, cancelled=None, progress=None):
        """Generate actual misses. ``embedder(list[str])`` returns indexed API data.

        Responses must carry an explicit ``index`` and ``embedding``. Missing or
        invalid rows remain misses; successful rows never shift to a neighbor.
        The caller may invoke again after failure to retry only remaining inputs.
        """
        if self.spec.representation != "raw":
            raise ValueError(
                "Generate provider output into a raw cache, then derive normalization"
            )
        records = list(records)
        rows = self._records(records)
        with self._lock:
            if (self.directory / "manifest.json").exists():
                self.import_package(self.directory, checkpoint=False)
            report = self._prepare_rows(rows, embedder, batch_size, cancelled, progress)
            report.update(self.coverage(records))
            self.last_report = report
            if not report["missing_ids"] and not report["cancelled"]:
                report["pruned_checkpoint_files"] = self.compact()
            return report

    def _prepare_rows(self, rows, embedder, batch_size, cancelled, progress):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        pending, seen = [], set()
        with self._lock:
            for row in rows:
                key = self._key(row)
                if key not in self._entries and key not in seen:
                    pending.append(row)
                    seen.add(key)
        report = {"generated": 0, "errors": [], "cancelled": False, "batches": 0}
        for start in range(0, len(pending), batch_size):
            if cancelled and cancelled():
                report["cancelled"] = True
                break
            batch = pending[start : start + batch_size]
            try:
                response = embedder([r["input_text"] for r in batch])
                returned_model = (
                    response.get("model")
                    if isinstance(response, dict)
                    else getattr(response, "model", None)
                )
                if returned_model and returned_model != self.spec.model:
                    raise ValueError("Provider returned a different embedding model")
                data = (
                    response.get("data", [])
                    if isinstance(response, dict)
                    else getattr(response, "data", response)
                )
                indexed, duplicates = {}, set()
                for item in data:
                    index = (
                        item.get("index")
                        if isinstance(item, dict)
                        else getattr(item, "index", None)
                    )
                    vector = (
                        item.get("embedding")
                        if isinstance(item, dict)
                        else getattr(item, "embedding", None)
                    )
                    if type(index) is not int or not 0 <= index < len(batch):
                        report["errors"].append(
                            {
                                "batch": report["batches"],
                                "reason": "Invalid provider response index",
                            }
                        )
                        continue
                    if index in indexed:
                        duplicates.add(index)
                    indexed[index] = vector
                with self._lock:
                    for index, row in enumerate(batch):
                        try:
                            if index in duplicates:
                                raise ValueError("Duplicate provider response index")
                            if index not in indexed:
                                raise ValueError("Missing provider response index")
                            vector = self._vector(indexed[index])
                            self._entries[self._key(row)] = (dict(row), vector)
                            report["generated"] += 1
                        except (ValueError, TypeError) as error:
                            report["errors"].append(
                                {
                                    "candidate_id": row["candidate_id"],
                                    "reason": str(error),
                                }
                            )
                    # Save completed work before acknowledging cancellation.
                    self._checkpoint()
            except Exception as error:
                report["errors"].append(
                    {
                        "candidate_ids": [r["candidate_id"] for r in batch],
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                # The injected outbound wrapper has already exhausted its
                # finite retry policy. Stop rather than repeat an auth/error
                # across every remaining batch; a later resume is missing-only.
                report["batches"] += 1
                break
            report["batches"] += 1
            if progress:
                progress(dict(report))
        if cancelled and cancelled():
            report["cancelled"] = True
        return report

    def query(self, procedure, embedder=None, cancelled=None):
        with self._lock:
            if (self.directory / "manifest.json").exists():
                self.import_package(self.directory, checkpoint=False)
            return self._query_locked(procedure, embedder, cancelled)

    def _query_locked(self, procedure, embedder=None, cancelled=None):
        exact = self.spec.format(procedure)
        row = {
            "candidate_id": "query:" + sha256_text(exact),
            "procedure": procedure,
            "source": "dynamic inverse query",
        }
        record = self._records([row], namespace="query")[0]
        key = self._key(record)
        if key not in self._entries:
            if embedder is None:
                raise ValueError("Dynamic query embedding is missing")
            self._prepare_rows([record], embedder, 1, cancelled, None)
        if key not in self._entries:
            raise ValueError("Dynamic query embedding was not completed")
        return self._entries[key][1].copy()

    def _checkpoint(self):
        self.export_package(self.directory, snapshot=True)

    def compact(self):
        """Prune only obsolete snapshots created by this cache, after completion.

        Calls in this application share the directory lock. Do not open the same
        writable cache in a separate application process while preparing it.
        A retained active manifest always references complete immutable files.
        """
        with self._lock:
            manifest_path = self.directory / "manifest.json"
            if not manifest_path.exists():
                return 0
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            keep = {entry["name"] for entry in manifest["files"].values()}
            removed = 0
            for path in self.directory.iterdir():
                if (
                    path.name not in keep
                    and re.fullmatch(
                        r"(?:vectors|rows)\.[a-f0-9]{32}\.(?:npy|jsonl)", path.name
                    )
                    and path.resolve().parent == self.directory.resolve()
                    and path.is_file()
                ):
                    path.unlink()
                    removed += 1
            return removed

    def export_package(self, directory, snapshot=False):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "." + uuid.uuid4().hex if snapshot else ""
        vectors_path = directory / ("vectors" + suffix + ".npy")
        rows_path = directory / ("rows" + suffix + ".jsonl")
        with self._lock:
            entries = list(self._entries.values())
            vectors = np.asarray([v for _, v in entries], dtype=np.float32).reshape(
                -1, self.spec.dimensions
            )
            with vectors_path.open("wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
                for index, (row, _) in enumerate(entries):
                    handle.write(
                        json.dumps(dict(row, row_index=index), ensure_ascii=False)
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            manifest = {
                "schema": "embedding_transfer_v1",
                "cache_version": 1,
                "spec": asdict(self.spec),
                "row_count": len(entries),
                "dtype": "float32",
                # Fresh provider output has no historical source-commit
                # claim. Transfer provenance is preserved when supplied.
                "source_commit": next(iter(self.source_commits))
                if len(self.source_commits) == 1
                else None,
                "source_commits": sorted(self.source_commits),
                "files": {
                    "vectors": {
                        "name": vectors_path.name,
                        "sha256": _file_hash(vectors_path),
                    },
                    "rows": {"name": rows_path.name, "sha256": _file_hash(rows_path)},
                },
            }
            _atomic_json(directory / "manifest.json", manifest)
        return manifest

    def import_package(self, directory, expected_records=None, checkpoint=True):
        with _cache_lock(directory), self._lock:
            return self._import_package_locked(directory, expected_records, checkpoint)

    def _import_package_locked(self, directory, expected_records=None, checkpoint=True):
        """Validate before import; preserve rejected row provenance in a report.

        A checksummed file/spec mismatch rejects the package as a whole. Bad
        individual rows are quarantined while other compatible rows are useful.
        """
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "embedding_transfer_v1"
            or manifest.get("cache_version") != 1
        ):
            raise ValueError("Unsupported embedding transfer schema/version")
        if manifest.get("spec") != asdict(self.spec):
            raise ValueError(
                "Embedding model, dimensions, formatter or preprocessing identity differs"
            )
        provenance_commits = list(manifest.get("source_commits", []))
        if manifest.get("source_commit"):
            provenance_commits.append(manifest["source_commit"])
        if any(not isinstance(commit, str) for commit in provenance_commits):
            raise ValueError("Embedding source commits must be strings")
        paths = {}
        for kind in ("rows", "vectors"):
            item = manifest["files"][kind]
            paths[kind] = _package_file(directory, item["name"])
            if _file_hash(paths[kind]) != item["sha256"]:
                raise ValueError(f"Embedding {kind} file checksum mismatch")
        loaded = np.load(paths["vectors"], allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if loaded.files != ["vectors"]:
                    raise ValueError("NPZ must contain only numeric 'vectors'")
                vectors = loaded["vectors"]
            finally:
                loaded.close()
        else:
            vectors = loaded
        if (
            vectors.dtype.kind not in "fiu"
            or vectors.ndim != 2
            or vectors.shape[1] != self.spec.dimensions
        ):
            raise ValueError("Invalid numeric embedding matrix or dimension")
        if str(vectors.dtype) != manifest.get("dtype"):
            raise ValueError("Embedding dtype differs from declared manifest")
        rows = [
            json.loads(line)
            for line in paths["rows"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) != manifest["row_count"] or len(vectors) != len(rows):
            raise ValueError("Embedding matrix, row mapping and declared counts differ")
        expected = (
            None
            if expected_records is None
            else {r["candidate_id"]: r for r in self._records(expected_records)}
        )
        report = {
            "imported": [],
            "incompatible_rows": [],
            "duplicates": [],
            "unresolved_provenance": [],
        }
        seen_indices, seen_ids, seen_keys = set(), set(), set()
        with self._lock:
            self.source_commits.update(provenance_commits)
            for row in rows:
                try:
                    index = row["row_index"]
                    candidate_id = row["candidate_id"]
                    if type(index) is not int or not 0 <= index < len(vectors):
                        raise ValueError("Invalid row index")
                    if (
                        index in seen_indices
                        or (row.get("namespace", "candidate"), candidate_id) in seen_ids
                        or self._key(row) in seen_keys
                    ):
                        report["duplicates"].append(row)
                        continue
                    seen_indices.add(index)
                    seen_ids.add((row.get("namespace", "candidate"), candidate_id))
                    seen_keys.add(self._key(row))
                    if sha256_text(row["input_text"]) != row["input_sha256"]:
                        raise ValueError("Exact input text hash mismatch")
                    if row.get("namespace", "candidate") not in ("candidate", "query"):
                        raise ValueError("Unknown embedding row namespace")
                    if (
                        expected is not None
                        and row.get("namespace", "candidate") == "candidate"
                    ):
                        if (
                            candidate_id not in expected
                            or expected[candidate_id]["input_text"] != row["input_text"]
                        ):
                            raise ValueError(
                                "Candidate ID or exact formatted input differs from corpus"
                            )
                    vector = self._vector(vectors[index])
                    if not row.get("source"):
                        report["unresolved_provenance"].append(candidate_id)
                    self._entries[self._key(row)] = (dict(row), vector)
                    report["imported"].append(candidate_id)
                except (KeyError, TypeError, ValueError) as error:
                    report["incompatible_rows"].append(
                        {"row": row, "reason": str(error)}
                    )
            if checkpoint:
                self._checkpoint()
                _atomic_json(self.directory / "import_coverage.json", report)
        if expected_records is not None:
            report.update(self.coverage(expected_records))
        self.last_report = report
        return report

    def import_legacy_csv(self, path, checkpoint=False):
        """Read explicit-model CSV vectors, retaining exact stored input keys.

        No implicit model inference, prefix removal, or outcome wrappers are
        introduced. Exact candidate matching remains the coverage/matrix rule.
        """
        report = {"imported": [], "incompatible_rows": []}
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"x", "embedding", "embedding_model"}.issubset(
                reader.fieldnames or []
            ):
                raise ValueError(
                    "Legacy CSV needs exact input text, vectors and explicit embedding_model"
                )
            prefix, suffix = self.spec.input_template.split("{procedure}")
            for index, row in enumerate(reader):
                try:
                    if row["embedding_model"] != self.spec.model:
                        raise ValueError("Embedding model mismatch")
                    exact = row["x"]
                    if (
                        not exact
                        or not exact.startswith(prefix)
                        or (suffix and not exact.endswith(suffix))
                    ):
                        raise ValueError(
                            "Stored exact input does not match required formatter"
                        )
                    digest = sha256_text(exact)
                    if row.get("input_sha256") and row["input_sha256"] != digest:
                        raise ValueError("Exact input hash mismatch")
                    if (
                        row.get("dimensions")
                        and int(row["dimensions"]) != self.spec.dimensions
                    ):
                        raise ValueError("Declared dimensions mismatch")
                    vector = self._vector(ast.literal_eval(row["embedding"]))
                    record = {
                        "candidate_id": row.get("candidate_id") or digest,
                        "input_text": exact,
                        "input_sha256": digest,
                        "namespace": "candidate",
                        "source": "validated legacy CSV; exact stored key and explicit model",
                    }
                    self._entries[self._key(record)] = (record, vector)
                    report["imported"].append(record["candidate_id"])
                except (KeyError, SyntaxError, ValueError, TypeError) as error:
                    report["incompatible_rows"].append(
                        {"row_index": index, "reason": str(error)}
                    )
        if checkpoint:
            self._checkpoint()
        return report


class RetrievalEmbeddingCache:
    """Cosine-retrieval facade retaining raw and legacy-normalized provenance.

    The two on-disk caches never relabel a normalized vector as provider output.
    Candidate/query matrices are derived float32 L2 vectors; generation fills
    only exact-input misses across both stores. GP consumers use raw caches.
    """

    def __init__(self, directory, spec):
        if spec.representation != "raw":
            raise ValueError(
                "Retrieval facade must be configured with the raw provider spec"
            )
        self.directory = Path(directory)
        self.spec = spec
        self.raw = EmbeddingCache(directory, spec)
        normalized_spec = replace(
            spec, representation="l2-normalized", preprocessing_version="float32-l2-v1"
        )
        self.normalized = EmbeddingCache(
            self.directory / "normalized-legacy", normalized_spec
        )
        self.last_report = {}

    def coverage(self, records):
        records = list(records)
        raw_hits = set(self.raw.coverage(records)["validated_hits"])
        normalized_hits = (
            set(self.normalized.coverage(records)["validated_hits"]) - raw_hits
        )
        all_hits = raw_hits | normalized_hits
        hits = [
            str(row["candidate_id"])
            for row in records
            if str(row["candidate_id"]) in all_hits
        ]
        return {
            "requested": len(records),
            "validated_hits": hits,
            "hit_count": len(hits),
            "missing_ids": [
                str(row["candidate_id"])
                for row in records
                if str(row["candidate_id"]) not in all_hits
            ],
            "raw_hits": len(raw_hits),
            "normalized_legacy_hits": len(normalized_hits),
            "output_representation": "derived float32 L2-normalized cosine vectors",
            "spec": asdict(self.spec),
        }

    def prepare(self, records, embedder, batch_size=64, cancelled=None, progress=None):
        records = list(records)
        before = self.coverage(records)
        missing_ids = set(before["missing_ids"])
        missing = [row for row in records if str(row["candidate_id"]) in missing_ids]
        report = self.raw.prepare(
            missing,
            embedder,
            batch_size=batch_size,
            cancelled=cancelled,
            progress=progress,
        )
        report.update(self.coverage(records))
        self.last_report = report
        return report

    def matrix(self, records, normalize=True):
        records = list(records)
        coverage = self.coverage(records)
        if coverage["missing_ids"]:
            errors = self.last_report.get("errors", [])
            raise ValueError(
                f"Missing {len(coverage['missing_ids'])} validated embeddings; preparation errors: {errors[:1]}"
            )
        if not normalize and coverage["normalized_legacy_hits"]:
            raise ValueError(
                "Raw output is unavailable for normalized legacy hits; use cosine normalization"
            )
        raw_hits = set(self.raw.coverage(records)["validated_hits"])
        output = np.empty((len(records), self.spec.dimensions), dtype=np.float32)
        # Batch each representation so a full corpus does not cause thousands
        # of repeated dictionary scans or numeric matrix allocations.
        for cache, use_raw in ((self.raw, True), (self.normalized, False)):
            positions = [
                index
                for index, row in enumerate(records)
                if (str(row["candidate_id"]) in raw_hits) == use_raw
            ]
            if positions:
                output[positions] = cache.matrix(
                    [records[index] for index in positions], normalize=normalize
                )
        return output

    def query(self, procedure, embedder=None, cancelled=None):
        try:
            vector = self.raw.query(procedure)
        except ValueError:
            try:
                vector = self.normalized.query(procedure)
            except ValueError:
                vector = self.raw.query(procedure, embedder, cancelled=cancelled)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Zero query vector cannot be used for cosine retrieval")
        return np.asarray(vector / norm, dtype=np.float32)

    def import_package(self, directory, expected_records=None, checkpoint=True):
        directory = Path(directory)
        if expected_records is not None:
            expected_records = list(expected_records)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") == "retrieval_embedding_collection_v1":
            if manifest.get("spec") != asdict(self.spec):
                raise ValueError("Retrieval collection model/input identity differs")
            reports = {}
            for name, cache in (("raw", self.raw), ("normalized", self.normalized)):
                reports[name] = cache.import_package(
                    directory / name, expected_records, checkpoint=checkpoint
                )
            report = {
                "representations": reports,
                "imported": reports["raw"]["imported"]
                + reports["normalized"]["imported"],
                "incompatible_rows": reports["raw"]["incompatible_rows"]
                + reports["normalized"]["incompatible_rows"],
            }
        else:
            supplied = manifest.get("spec")
            if supplied == asdict(self.raw.spec):
                report = self.raw.import_package(
                    directory, expected_records, checkpoint=checkpoint
                )
                report["imported_representation"] = "raw"
            elif supplied == asdict(self.normalized.spec):
                report = self.normalized.import_package(
                    directory, expected_records, checkpoint=checkpoint
                )
                report["imported_representation"] = "l2-normalized"
            else:
                raise ValueError(
                    "Embedding model, exact formatter, dimensions or raw/normalized preprocessing identity differs"
                )
        if expected_records is not None:
            report.update(self.coverage(expected_records))
        self.last_report = report
        return report

    def export_package(self, directory):
        directory = Path(directory)
        if directory.resolve() == self.directory.resolve():
            raise ValueError(
                "Export a retrieval collection to a separate directory from its active cache"
            )
        directory.mkdir(parents=True, exist_ok=True)
        self.raw.export_package(directory / "raw")
        self.normalized.export_package(directory / "normalized")
        manifest = {
            "schema": "retrieval_embedding_collection_v1",
            "spec": asdict(self.spec),
            "representations": ["raw", "l2-normalized"],
            "output_representation": "derived float32 L2-normalized cosine vectors",
        }
        _atomic_json(directory / "manifest.json", manifest)
        return manifest


def create_embedding_cache(directory, spec):
    """Shared factory: crystal-prefixed retrieval permits normalized reuse."""
    if spec.input_template == "experimental procedure: {procedure}":
        return RetrievalEmbeddingCache(directory, spec)
    return EmbeddingCache(directory, spec)
