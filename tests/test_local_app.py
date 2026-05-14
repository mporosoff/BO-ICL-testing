from boicl import AskTellFewShotTopk, Pool
from boicl.local_app import (
    LocalBOState,
    _best_trace,
    _coerce_float,
    _dataset_stats,
    _paper_random_trace,
)


def test_import_dataset_uses_first_column_and_optional_values(tmp_path):
    state = LocalBOState(tmp_path)
    raw = b"procedure,value,uncertainty\nproc a,1.2,0.1\nproc b,,\nproc c,2.5,\n"

    payload = state.import_dataset("dataset.csv", raw)

    assert payload["candidate_count"] == 3
    assert [candidate["procedure"] for candidate in payload["candidates"]] == [
        "proc a",
        "proc b",
        "proc c",
    ]
    assert payload["observations"] == []
    assert payload["label_count"] == 2
    assert "objectives" not in payload["candidates"][0]
    assert state.candidates[0]["objectives"]["value"] == 1.2
    assert state.candidates[0]["uncertainties"]["value"] == 0.1
    assert payload["objective_names"] == ["value"]


def test_import_dataset_can_select_between_multiple_objectives(tmp_path):
    state = LocalBOState(tmp_path)
    raw = b"procedure,yield,selectivity,yield_uncertainty\nproc a,1.2,8.0,0.1\nproc b,2.5,7.0,0.2\n"

    state.import_dataset("dataset.csv", raw)
    payload = state.update_config({"objective_name": "selectivity"})

    assert payload["objective_names"] == ["yield", "selectivity"]
    assert payload["observations"] == []
    assert payload["label_count"] == 2
    assert state.candidates[0]["objectives"]["yield"] == 1.2
    assert payload["dataset_stats"][0]["label"] == "mean"


def test_replicates_keep_candidate_available_until_limit(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure\nproc a\n")
    state.update_config({"replicates_per_candidate": 2})

    assert state.to_json()["available_count"] == 1
    state.add_observation({"candidate_id": "cand-0", "value": 1.0})
    assert state.to_json()["available_count"] == 1
    state.add_observation({"candidate_id": "cand-0", "value": 1.1})
    assert state.to_json()["available_count"] == 0


def test_best_trace_respects_direction():
    observations = [
        {"value": 5.0},
        {"value": 3.0},
        {"value": 7.0},
    ]

    assert [point["best"] for point in _best_trace(observations, "maximize")] == [
        5.0,
        5.0,
        7.0,
    ]
    assert [point["best"] for point in _best_trace(observations, "minimize")] == [
        5.0,
        3.0,
        3.0,
    ]


def test_paper_random_trace_is_monotonic_for_maximization():
    trace = _paper_random_trace([1.0, 2.0, 4.0], "maximize", steps=4)

    assert len(trace) == 4
    assert [point["index"] for point in trace] == [1, 2, 3, 4]
    assert trace[-1]["best"] >= trace[0]["best"]


def test_dataset_stats_include_paper_guides():
    stats = {item["label"]: item["value"] for item in _dataset_stats([1, 2, 3, 4])}

    assert stats["mean"] == 2.5
    assert "75%" in stats
    assert "95%" in stats
    assert "99%" in stats


def test_offline_benchmark_appends_random_config_without_live_observations(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure,value\nproc a,1\nproc b,2\nproc c,3\n")
    state.update_config(
        {
            "acquisition": "random",
            "benchmark_iterations": 2,
            "benchmark_replicates": 3,
            "benchmark_initial_points": 1,
            "benchmark_seed": 7,
        }
    )

    payload = state.run_benchmark({"name": "random smoke"})

    assert payload["observations"] == []
    assert len(payload["benchmark_runs"]) == 1
    run = payload["benchmark_runs"][0]
    assert run["name"] == "random smoke"
    assert len(run["replicate_traces"]) == 3
    assert run["summary"][-1]["count"] == 3


def test_float_coercion_rejects_empty_and_nonfinite():
    assert _coerce_float("1.5") == 1.5
    assert _coerce_float("") is None
    assert _coerce_float("nan") is None


def test_browser_config_tracks_llm_and_inverse_models(tmp_path):
    state = LocalBOState(tmp_path)

    payload = state.update_config(
        {
            "optimizer": "llm",
            "prediction_model": "gpt-4o-mini",
            "inverse_model": "openrouter/mistralai/mistral-7b-instruct:free",
            "llm_samples": 3,
            "inverse_filter": 4,
        }
    )

    assert payload["config"]["optimizer"] == "llm"
    assert payload["config"]["prediction_model"] == "gpt-4o-mini"
    assert payload["config"]["inverse_model"].startswith("openrouter/")
    assert payload["config"]["llm_samples"] == 3
    assert payload["config"]["inverse_filter"] == 4


def test_fewshot_can_use_separate_inverse_model():
    asktell = AskTellFewShotTopk(model="gpt-4o-mini", inverse_model="gpt-4o")

    assert asktell._model == "gpt-4o-mini"
    assert asktell._inverse_model == "gpt-4o"


def test_pool_keeps_embedding_model_setting():
    pool = Pool(["a", "b"], embedding_model="text-embedding-3-small")

    assert pool.embedding_model == "text-embedding-3-small"
