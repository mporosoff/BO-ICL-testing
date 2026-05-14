from io import BytesIO

import numpy as np

from boicl import AskTellFewShotTopk, Pool
from boicl.local_app import (
    DEFAULT_CONFIG,
    DEFAULT_PREDICTION_SYSTEM_MESSAGE,
    LocalBOState,
    _best_trace,
    _clean_api_key_value,
    _coerce_float,
    _dataset_stats,
    _load_env_file,
    _paper_random_trace,
    _write_env_value,
)


def test_import_dataset_uses_first_column_and_optional_values(tmp_path):
    state = LocalBOState(tmp_path)
    raw = b"procedure,value,uncertainty\nproc a,1.2,0.1\nproc b,,\nproc c,2.5,\n"

    payload = state.import_dataset("dataset.csv", raw)

    assert payload["candidate_count"] == 3
    assert payload["config"]["workflow_mode"] == "offline"
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


def test_import_dataset_accepts_npy_table(tmp_path):
    state = LocalBOState(tmp_path)
    out = BytesIO()
    np.save(out, np.array([["proc a", 1.2], ["proc b", 2.5]], dtype=object))

    payload = state.import_dataset("dataset.npy", out.getvalue())

    assert payload["candidate_count"] == 2
    assert payload["label_count"] == 2
    assert payload["objective_names"] == ["objective"]
    assert state.candidates[1]["objectives"]["objective"] == 2.5


def test_defaults_match_paper_style_numeric_settings():
    assert DEFAULT_CONFIG["benchmark_initial_points"] == 1
    assert DEFAULT_CONFIG["batch_size"] == 1
    assert DEFAULT_CONFIG["benchmark_iterations"] == 30
    assert DEFAULT_CONFIG["benchmark_replicates"] == 5
    assert DEFAULT_CONFIG["ucb_lambda"] == 0.1
    assert DEFAULT_CONFIG["prediction_system_message"]


def test_replicates_keep_candidate_available_until_limit(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure\nproc a\n")
    state.update_config({"replicates_per_candidate": 2})

    assert state.to_json()["available_count"] == 1
    state.add_observation({"candidate_id": "cand-0", "value": 1.0})
    assert state.to_json()["available_count"] == 1
    state.add_observation({"candidate_id": "cand-0", "value": 1.1})
    assert state.to_json()["available_count"] == 0


def test_import_procedure_only_dataset_selects_live_mode(tmp_path):
    state = LocalBOState(tmp_path)
    payload = state.import_dataset("dataset.csv", b"procedure\nproc a\nproc b\n")

    assert payload["label_count"] == 0
    assert payload["config"]["workflow_mode"] == "live"


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
    assert payload["progress"]["status"] == "complete"
    assert payload["progress"]["percent"] == 100


def test_campaign_save_load_and_autosave_roundtrip(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure,value\nproc a,1\nproc b,2\n")
    state.add_observation({"candidate_id": "cand-0", "value": 1.0})

    saved = state.save_campaign({"name": "Week one campaign"})
    campaign_id = saved["campaign"]["id"]
    assert campaign_id
    assert (tmp_path / "saved_experiments" / campaign_id / "campaign.json").exists()

    state.add_observation({"candidate_id": "cand-1", "value": 2.0})
    reloaded = LocalBOState(tmp_path)
    payload = reloaded.load_campaign({"id": campaign_id})

    assert payload["campaign"]["name"] == "Week one campaign"
    assert payload["candidate_count"] == 2
    assert len(payload["observations"]) == 2
    assert payload["observations"][-1]["value"] == 2.0


def test_embedding_cache_status_counts_current_dataset_and_model(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure,value\nproc a,1\nproc b,2\n")
    state.cache_dir.mkdir()
    state.embedding_cache_path().write_text(
        "x,embedding,embedding_model\n"
        '"proc a","[1.0, 0.0]",text-embedding-ada-002\n'
        '"other","[0.0, 1.0]",text-embedding-ada-002\n',
        encoding="utf-8",
    )

    status = state.embedding_cache_status()

    assert status["cached_count"] == 1
    assert status["total_count"] == 2
    assert status["missing_count"] == 1


def test_cached_approx_sample_uses_saved_embeddings_without_api(tmp_path):
    state = LocalBOState(tmp_path)
    state.cache_dir.mkdir()
    state.embedding_cache_path().write_text(
        "x,embedding,embedding_model\n"
        '"proc a","[1.0, 0.0]",text-embedding-ada-002\n'
        '"proc b","[0.0, 1.0]",text-embedding-ada-002\n'
        '"target","[0.95, 0.05]",text-embedding-ada-002\n',
        encoding="utf-8",
    )

    assert state._cached_approx_sample(["proc a", "proc b"], "target", 1) == [
        "proc a"
    ]
    assert state.progress_snapshot()["status"] == "complete"


def test_progress_snapshot_tracks_terminal_style_updates(tmp_path):
    state = LocalBOState(tmp_path)

    state.set_progress("Testing progress", 2, 4, "halfway")
    payload = state.to_json()

    assert payload["progress"]["label"] == "Testing progress"
    assert payload["progress"]["percent"] == 50
    assert state.progress_snapshot()["detail"] == "halfway"


def test_float_coercion_rejects_empty_and_nonfinite():
    assert _coerce_float("1.5") == 1.5
    assert _coerce_float("") is None
    assert _coerce_float("nan") is None


def test_api_key_save_sanitizes_prefixed_values_and_load_overrides(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    _write_env_value(env_path, "OPENAI_API_KEY", "OPENAI_API_KEY='sk-local'")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stale")

    _load_env_file(env_path)

    assert _clean_api_key_value("OPENAI_API_KEY=sk-new", "OPENAI_API_KEY") == "sk-new"
    assert env_path.read_text(encoding="utf-8").strip() == "OPENAI_API_KEY=sk-local"
    assert __import__("os").environ["OPENAI_API_KEY"] == "sk-local"


def test_blank_saved_system_message_falls_back_to_default(tmp_path):
    state = LocalBOState(tmp_path)
    state.config["prediction_system_message"] = ""

    assert state.prediction_system_message() == DEFAULT_PREDICTION_SYSTEM_MESSAGE


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
