import numpy as np
import pandas as pd
import pytest

pytest.importorskip("botorch")
import torch
from boicl.asktellGPR import AskTellGPR
from boicl.embedding_gp import EmbeddingGPEngine, fit_projection
from boicl.pool import Pool


CANDIDATES = [{"candidate_id": str(i), "procedure": f"procedure {i}"} for i in range(7)]
VECTORS = np.array([[i, i * i / 6, np.sin(i)] for i in range(7)], dtype=float)


def test_projection_small_spaces_and_connectivity():
    for count in (1, 2, 3):
        model, coordinates, diagnostics = fit_projection(
            VECTORS[:count], dimensions=32, neighbors=5
        )
        assert coordinates.shape[0] == count
        assert np.isfinite(coordinates).all()
        assert diagnostics["dimensions"] <= max(1, count - 1)
        assert diagnostics["adjustments"]
    separated = np.array([[0, 0], [0, 0.1], [100, 100], [100, 100.1]])
    _, _, diagnostics = fit_projection(separated, dimensions=2, neighbors=1)
    assert diagnostics["neighbors"] > 1
    assert any("Disconnected" in item for item in diagnostics["adjustments"])


def test_projection_reloads_and_invalidates_by_corpus_model_and_settings(tmp_path):
    path = tmp_path / "projection.npz"
    first = EmbeddingGPEngine(CANDIDATES, VECTORS, projection_cache=path)
    second = EmbeddingGPEngine(CANDIDATES, VECTORS, projection_cache=path)
    assert second.projection_loaded
    np.testing.assert_equal(second.coordinates, first.coordinates)
    changed = [dict(c, procedure=c["procedure"] + "changed") for c in CANDIDATES]
    assert not EmbeddingGPEngine(
        changed, VECTORS, projection_cache=path
    ).projection_loaded
    assert not EmbeddingGPEngine(
        CANDIDATES, VECTORS, {"embedding_model": "different"}, path
    ).projection_loaded
    assert not EmbeddingGPEngine(
        CANDIDATES, VECTORS, {"dimensions": 1}, path
    ).projection_loaded


def test_projection_seed_controls_initialization_and_cache_identity(
    tmp_path, monkeypatch
):
    import boicl.embedding_gp as module

    observed = []
    original = module.Isomap.fit_transform

    def inspect_seed(model, vectors):
        observed.append(np.random.random())
        return original(model, vectors)

    monkeypatch.setattr(module.Isomap, "fit_transform", inspect_seed)
    state_before = np.random.get_state()
    path = tmp_path / "projection.npz"
    first = EmbeddingGPEngine(CANDIDATES, VECTORS, {"seed": 0}, path)
    assert observed == [np.random.RandomState(0).random_sample()]
    assert first.projection_diagnostics["seed"] == 0
    assert EmbeddingGPEngine(CANDIDATES, VECTORS, {"seed": 0}, path).projection_loaded
    second = EmbeddingGPEngine(CANDIDATES, VECTORS, {"seed": 7}, path)
    assert observed[-1] == np.random.RandomState(7).random_sample()
    assert not second.projection_loaded
    assert second.projection_fingerprint != first.projection_fingerprint
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    np.testing.assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    with pytest.raises(ValueError, match="seed"):
        fit_projection(VECTORS, seed=-1)


def test_public_posterior_raw_scale_fitted_noise_and_metadata(tmp_path):
    engine = EmbeddingGPEngine(
        CANDIDATES, VECTORS, {"fit_steps": 2}, tmp_path / "projection.npz"
    )
    observations = [
        {"candidate_id": str(i), "moc_wt_pct": y, "esd": 0.5, "GOF": 1.2}
        for i, y in zip((0, 3, 6), (200, 400, 700))
    ]
    engine.fit(observations)
    assert engine.observations[0]["esd"] == 0.5
    ids = [o["candidate_id"] for o in observations]
    means, latent_std = engine.predict(ids)
    _, measurement_std = engine.predict(ids, observation_noise=True)
    np.testing.assert_allclose(means, [200, 400, 700], atol=8)
    assert np.all(measurement_std > latent_std)
    scale = engine.model.outcome_transform.stdvs.item()
    np.testing.assert_allclose(
        measurement_std**2 - latent_std**2,
        engine.model.likelihood.noise.item() * scale**2,
        rtol=1e-5,
    )
    result = engine.suggest([CANDIDATES[1], CANDIDATES[2], CANDIDATES[4]])
    assert {r["candidate_id"] for r in result["records"]} == {"1", "2", "4"}
    assert result["prediction"]["mean"] > 100
    assert "latent" in result["prediction"]["uncertainty_type"]


def test_projection_independent_of_observations_and_single_constant_targets():
    engine = EmbeddingGPEngine(CANDIDATES, VECTORS, {"fit_steps": 1})
    original = engine.coordinates.copy()
    assert engine.suggest(CANDIDATES)["selection_rule"] == "initial_design"
    engine.fit([{"candidate_id": "0", "moc_wt_pct": 83.8}])
    assert all(np.isfinite(x).all() for x in engine.predict(["1"]))
    engine.fit([{"candidate_id": str(i), "moc_wt_pct": 83.8} for i in (0, 3, 6)])
    np.testing.assert_equal(original, engine.coordinates)
    means, std = engine.predict(["1", "2"])
    assert np.isfinite(means).all() and np.isfinite(std).all()
    np.testing.assert_allclose(means, 83.8, atol=1e-5)


def test_legacy_gp_procedure_only_full_corpus_and_no_constructor_requests(tmp_path):
    calls = []

    def embedder(inputs):
        calls.append(inputs)
        return {
            "data": [
                {"index": i, "embedding": VECTORS[int(text[-1])].tolist()}
                for i, text in enumerate(inputs)
            ]
        }

    pool = Pool([c["procedure"] for c in CANDIDATES])
    model = AskTellGPR(
        pool=pool,
        embedding_model="mock",
        embedding_dimensions=3,
        cache_path=tmp_path / "embeddings.csv",
        embedder=embedder,
        n_components=2,
        fit_steps=1,
    )
    assert not calls
    model.tell("procedure 0", 200, train=False)
    model.tell("procedure 3", 400, train=False)
    model.tell("procedure 6", 700)
    assert calls == [[c["procedure"] for c in CANDIDATES]]
    assert model.likelihood is model.regressor.likelihood
    before = model.projection_fingerprint
    pool.choose("procedure 0")
    assert model.predict("procedure 0").mean() == pytest.approx(200, abs=8)
    assert model.projection_fingerprint == before
    assert all("400" not in text and "Q:" not in text for text in calls[0])
    reloaded = AskTellGPR(
        pool=pool,
        embedding_model="mock",
        embedding_dimensions=3,
        cache_path=tmp_path / "embeddings.csv",
        embedder=lambda _: pytest.fail("unexpected API"),
    )
    np.testing.assert_allclose(
        reloaded._query_cache(["procedure 6"]), VECTORS[6:7], rtol=1e-6
    )


def test_legacy_csv_hash_dimension_model_and_malformed_validation(tmp_path):
    path = tmp_path / "cache.csv"
    pd.DataFrame(
        [
            {"x": "good", "embedding": "[1.0, 2.0]", "embedding_model": "mock"},
            {"x": "bad", "embedding": "[1.0]", "embedding_model": "mock"},
            {"x": "other", "embedding": "[1.0, 2.0]", "embedding_model": "other-model"},
            {
                "x": "malformed",
                "embedding": "__import__('os')",
                "embedding_model": "mock",
            },
        ]
    ).to_csv(path, index=False)
    model = AskTellGPR(cache_path=path, embedding_model="mock", embedding_dimensions=2)
    assert model._embeddings_cache["x"].tolist() == ["good"]
    assert len(model.cache_import_errors) == 2


def test_legacy_gp_requires_full_projection_corpus():
    model = AskTellGPR(
        embedding_model="mock",
        embedding_dimensions=2,
        embedder=lambda _: pytest.fail("should fail before embedding"),
    )
    with pytest.raises(ValueError, match="fixed full candidate pool"):
        model.tell("procedure", 2)


def test_embedding_gp_shared_generic_minimize_setting(monkeypatch):
    engine = EmbeddingGPEngine(CANDIDATES, VECTORS, {"maximize": False})
    engine.model = object()
    engine.observations = [{"candidate_id": "0", "value": 0.0}]
    monkeypatch.setattr(
        engine,
        "predict",
        lambda ids, observation_noise=False: (np.array([2.0, -2.0]), np.zeros(2)),
    )
    result = engine.suggest([CANDIDATES[1], CANDIDATES[2]])
    assert result["candidate_id"] == "2" and result["prediction"]["acquisition"] == 2.0


def test_legacy_app_gp_scores_full_pool_and_preserves_measurement_metadata(
    tmp_path, monkeypatch
):
    import boicl
    from boicl.local_app import LocalBOState

    state = LocalBOState(tmp_path)
    state.import_dataset(
        "pool.csv",
        b"procedure,value\nprocedure 0,\nprocedure 1,\nprocedure 2,\nprocedure 3,\n",
    )
    state.config.update(score_limit=1, n_components=32, n_neighbors=5)
    seen = {}

    class FakeGP:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def tell(self, *args, **kwargs):
            pass

        def ask(self, procedures, **kwargs):
            seen["scored"] = list(procedures)
            return [procedures[0]], [1.0], [45.0], [2.0]

    monkeypatch.setattr(boicl, "AskTellGPR", FakeGP)
    result = state._gpr_suggestions(
        state.candidates[1:],
        observations=[
            {
                "candidate_id": "cand-0",
                "procedure": "procedure 0",
                "value": 83.8,
                "esd": 5.67,
                "GOF": 0.548,
                "closure_gap_override": 0.0,
            }
        ],
    )
    assert seen["pool"]._pool == [c["procedure"] for c in state.candidates]
    assert seen["scored"] == [c["procedure"] for c in state.candidates[1:]]
    assert seen["n_components"] == 32 and seen["n_neighbors"] == 5
    assert result[0]["measurement_metadata"][0]["esd"] == 5.67


def test_legacy_app_cache_rejects_invalid_dimensions_and_maps_partial_response(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from boicl.local_app import LocalBOState
    import boicl.local_app as app

    state = LocalBOState(tmp_path)
    state.config.update(optimizer="gpr", embedding_model="text-embedding-ada-002")
    state.import_dataset(
        "pool.csv", b"procedure\nprocedure 0\nprocedure 1\nprocedure 2\n"
    )
    state.cache_dir.mkdir(exist_ok=True)
    state.embedding_cache_path().write_text(
        'x,embedding,embedding_model\n"procedure 0","[1,0]",text-embedding-ada-002\n'
    )
    assert state.embedding_cache_status()["cached_count"] == 0
    calls = []

    def create(**kwargs):
        calls.append(kwargs["input"])
        return {
            "data": [
                {"index": 2, "embedding": [2.0] + [0.0] * 1535},
                {"index": 0, "embedding": [3.0] + [0.0] * 1535},
            ]
        }

    def sdk(**kwargs):
        assert kwargs["max_retries"] == 0
        return SimpleNamespace(
            max_retries=0,
            embeddings=SimpleNamespace(create=create),
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: None)),
        )

    monkeypatch.setattr(app, "OpenAI", sdk)
    with pytest.raises(ValueError, match="1 embedding inputs remain missing"):
        state._cached_embeddings([c["procedure"] for c in state.candidates])
    cache = state._embedding_cache_model()
    np.testing.assert_equal(
        cache.matrix([{"candidate_id": "2", "procedure": "procedure 2"}])[:, 0], [2.0]
    )
    assert state.embedding_cache_status()["cached_count"] == 2
    assert len(calls) == 1
