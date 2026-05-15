import csv
import random
from io import BytesIO, StringIO

import numpy as np

from boicl import AskTellFewShotTopk, Pool
from boicl.local_app import (
    DEFAULT_CONFIG,
    DEFAULT_PREDICTION_SYSTEM_MESSAGE,
    GENERATED_INVERSE_PROMPT_PREFIX,
    GENERATED_PREDICTION_PROMPT_PREFIX,
    INDEX_HTML,
    LocalBOState,
    POOL_BUILDER_HTML,
    RunCancelled,
    _best_trace,
    _clean_api_key_value,
    _coerce_float,
    _dataset_stats,
    _load_env_file,
    _paper_random_trace,
    _retry_delay_seconds,
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
    assert payload["config"]["prediction_system_message"].startswith(
        GENERATED_PREDICTION_PROMPT_PREFIX
    )
    assert payload["config"]["inverse_system_message"].startswith(
        GENERATED_INVERSE_PROMPT_PREFIX
    )
    assert "proc a" in payload["config"]["prediction_system_message"]
    assert "1.2" not in payload["config"]["prediction_system_message"]


def test_pool_builder_import_shape_is_unlabelled_live_pool(tmp_path):
    state = LocalBOState(tmp_path)
    raw = (
        "procedure\n"
        "\"Reduction experiment of WO3/SiO2: ramp to 400 C at 10 C/min, "
        "soak for 4 h.\"\n"
    ).encode("utf-8")

    payload = state.import_dataset(
        "wo3_sio2_reduction_pool.csv", raw, objective_name="alpha phase (%)"
    )

    assert payload["candidate_count"] == 1
    assert payload["label_count"] == 0
    assert payload["config"]["workflow_mode"] == "live"
    assert payload["config"]["objective_name"] == "alpha phase (%)"
    assert payload["objective_names"] == ["alpha phase (%)"]


def test_pool_builder_page_contains_import_controls():
    assert "WO3/SiO2 reduction template" in POOL_BUILDER_HTML
    assert "/api/import-dataset?filename=wo3_sio2_reduction_pool.csv" in POOL_BUILDER_HTML
    assert "Pool size" in POOL_BUILDER_HTML
    assert "Pool cap" not in POOL_BUILDER_HTML
    assert "Alpha phase (%) - Im-3m" in POOL_BUILDER_HTML
    assert "Beta phase (%) - Pm-3n" in POOL_BUILDER_HTML
    assert "Minimum" in POOL_BUILDER_HTML
    assert "Maximum" in POOL_BUILDER_HTML
    assert "Step" in POOL_BUILDER_HTML
    assert "boicl_pool_builder" in POOL_BUILDER_HTML
    assert "boicl_runner" in POOL_BUILDER_HTML
    assert "Open/Focus Runner" in POOL_BUILDER_HTML
    assert "Save the campaign in the runner to keep it for later" in POOL_BUILDER_HTML
    assert "BroadcastChannel" in POOL_BUILDER_HTML


def test_main_app_candidate_labels_surface_variable_values():
    assert "candidateProcedureSummary" in INDEX_HTML
    assert "candidateOptionLabel" in INDEX_HTML
    assert "candidatePreview" in INDEX_HTML
    assert "enter 73.5 for 73.5%, not 0.735" in INDEX_HTML
    assert "openPoolBuilder" in INDEX_HTML
    assert "boicl_pool_builder" in INDEX_HTML
    assert "Pool Builder imported" in INDEX_HTML
    assert "BroadcastChannel" in INDEX_HTML


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
    assert DEFAULT_CONFIG["benchmark_starting_baseline"] == "none"
    assert DEFAULT_CONFIG["ucb_lambda"] == 0.1
    assert DEFAULT_CONFIG["llm_samples"] == 3
    assert DEFAULT_CONFIG["inverse_filter"] == 16
    assert DEFAULT_CONFIG["inverse_random_candidates"] == 0
    assert DEFAULT_CONFIG["inverse_target_multiplier"] == 1.2
    assert DEFAULT_CONFIG["inverse_target_jitter"] == 0.05
    assert DEFAULT_CONFIG["inverse_target_floor_value"] == ""
    assert DEFAULT_CONFIG["greedy_final_iteration"] is False
    assert DEFAULT_CONFIG["api_rate_limit_cooldown_seconds"] == 10.0
    assert DEFAULT_CONFIG["prediction_system_message"]


def test_model_fields_are_real_selectors():
    assert '<select id="embeddingModel"></select>' in INDEX_HTML
    assert '<select id="predictionModel"></select>' in INDEX_HTML
    assert '<select id="inverseModel"></select>' in INDEX_HTML
    assert 'id="modelOptions"' not in INDEX_HTML
    assert 'id="embeddingModelOptions"' not in INDEX_HTML


def test_non_benchmark_progress_clears_stale_partial_run(tmp_path):
    state = LocalBOState(tmp_path)
    partial_run = {"id": "benchmark-1", "summary": [{"index": 1, "mean": 2.0}]}

    state.set_progress(
        "Running benchmark: smoke",
        1,
        10,
        extra={"partial_run": partial_run},
    )
    state.set_progress("Updating suggestions", 0, 1)

    assert "partial_run" not in state.progress_snapshot()


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


def test_best_trace_mean_baseline_can_hide_initial_context_rows():
    observations = [{"value": 1.0}, {"value": 5.0}]

    trace = _best_trace(
        observations,
        "maximize",
        baseline_value=3.0,
        skip_observations=1,
    )

    assert trace == [
        {"index": 1, "value": 3.0, "best": 3.0, "baseline": True},
        {"index": 2, "value": 5.0, "best": 5.0},
    ]


def test_best_trace_mean_baseline_waits_for_first_scored_pool_result():
    observations = [{"value": 1.0}]

    trace = _best_trace(
        observations,
        "maximize",
        baseline_value=3.0,
        skip_observations=1,
    )

    assert trace == [
        {"index": 1, "value": 3.0, "best": 3.0, "baseline": True},
    ]


def test_paper_random_trace_is_monotonic_for_maximization():
    trace = _paper_random_trace([1.0, 2.0, 4.0], "maximize", steps=4)

    assert len(trace) == 4
    assert [point["index"] for point in trace] == [1, 2, 3, 4]
    assert trace[-1]["best"] >= trace[0]["best"]


def test_paper_random_trace_can_start_from_mean_baseline():
    trace = _paper_random_trace(
        [1.0, 3.0, 5.0], "maximize", steps=2, baseline_value=3.0
    )

    assert trace[0] == {"index": 1, "best": 3.0, "baseline": True}
    assert [point["index"] for point in trace] == [1, 2, 3]
    assert trace[1]["best"] >= 3.0


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


def test_offline_benchmark_can_start_plot_from_dataset_mean(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure,value\nproc a,1\nproc b,3\nproc c,5\n")
    state.update_config(
        {
            "acquisition": "random",
            "benchmark_iterations": 1,
            "benchmark_replicates": 1,
            "benchmark_initial_points": 1,
            "benchmark_starting_baseline": "mean",
        }
    )

    payload = state.run_benchmark({"name": "mean baseline"})
    run = payload["benchmark_runs"][0]

    assert run["summary"][0]["index"] == 1
    assert run["summary"][0]["mean"] == 3.0
    assert run["replicate_traces"][0][0]["baseline"] is True
    assert run["replicate_traces"][0][1]["index"] == 2
    assert len(run["summary"]) == 2
    assert all(point["index"] >= 1 for point in run["replicate_traces"][0])
    assert len(run["replicate_observations"][0]) == 2


def test_mean_baseline_progress_stays_on_seed_before_first_bo_result(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "dataset.csv",
        b"procedure,value\nproc a,1\nproc b,2\nproc c,3\nproc d,4\n",
    )
    state.update_config(
        {
            "acquisition": "random",
            "optimizer": "gpr",
            "benchmark_iterations": 2,
            "benchmark_replicates": 1,
            "benchmark_initial_points": 1,
            "benchmark_starting_baseline": "mean",
        }
    )
    captured = {}

    def stop_before_first_bo_result(available, observations, rng, acquisition=None):
        progress = state.progress_snapshot()
        partial = progress["partial_run"]
        captured["current"] = progress["current"]
        captured["total"] = progress["total"]
        captured["detail"] = progress["detail"]
        captured["summary"] = list(partial["summary"])
        raise RunCancelled("stop before first BO result")

    state._benchmark_next_candidate = stop_before_first_bo_result
    payload = state.run_benchmark({"name": "mean progress"})
    partial = payload["benchmark_runs"][0]

    assert captured["current"] == 1
    assert captured["total"] == 3
    assert "experiment 1/3" in captured["detail"]
    assert [point["index"] for point in captured["summary"]] == [1]
    assert captured["summary"][0]["mean"] == 2.5
    assert partial["status"] == "stopped"
    assert [point["index"] for point in partial["summary"]] == [1]


def test_mean_baseline_plot_horizon_ignores_hidden_initial_context(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "dataset.csv",
        b"procedure,value\nproc a,1\nproc b,2\nproc c,3\nproc d,4\n",
    )
    state.update_config(
        {
            "acquisition": "random",
            "optimizer": "gpr",
            "benchmark_iterations": 2,
            "benchmark_replicates": 1,
            "benchmark_initial_points": 2,
            "benchmark_starting_baseline": "mean",
        }
    )

    payload = state.run_benchmark({"name": "mean horizon"})
    run = payload["benchmark_runs"][0]

    assert state.plot_horizon() == 3
    assert [point["index"] for point in run["summary"]] == [1, 2, 3]
    assert [point["index"] for point in payload["random_walk_trace"]] == [1, 2, 3]


def test_llm_benchmark_scores_after_one_initial_point(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "dataset.csv",
        b"procedure,value\nproc a,1\nproc b,2\nproc c,3\nproc d,4\n",
    )
    state.update_config(
        {
            "acquisition": "upper_confidence_bound",
            "optimizer": "llm",
            "benchmark_iterations": 1,
            "benchmark_replicates": 1,
            "benchmark_initial_points": 1,
        }
    )
    calls = []

    def fake_llm_suggestions(available, observations=None, rng=None, k=None, acquisition=None):
        calls.append(
            {
                "available": len(available),
                "observations": len(observations or []),
                "acquisition": acquisition,
            }
        )
        candidate = available[0]
        return [
            {
                "candidate_id": candidate["id"],
                "procedure": candidate["procedure"],
                "acquisition": 1.0,
                "mean": 1.0,
                "source": "llm",
            }
        ]

    state._llm_suggestions = fake_llm_suggestions
    payload = state.run_benchmark({"name": "llm one seed"})

    assert calls == [
        {
            "available": 3,
            "observations": 1,
            "acquisition": "upper_confidence_bound",
        }
    ]
    assert payload["benchmark_runs"][0]["status"] == "complete"


def test_live_llm_suggests_after_one_observation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure\nproc a\nproc b\nproc c\n")
    state.update_config({"acquisition": "upper_confidence_bound", "optimizer": "llm"})
    state.add_observation({"candidate_id": "cand-0", "value": 1.0})
    calls = []

    def fake_llm_suggestions(available, observations=None, rng=None, k=None, acquisition=None):
        calls.append(len(observations or []))
        candidate = available[0]
        return [
            {
                "candidate_id": candidate["id"],
                "procedure": candidate["procedure"],
                "acquisition": 1.0,
                "mean": 1.0,
                "source": "llm",
            }
        ]

    state._llm_suggestions = fake_llm_suggestions
    payload = state.suggest()

    assert calls == [1]
    assert payload["suggestions"][0]["source"] == "llm"


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
    assert state.progress_snapshot()["status"] == "idle"


def test_progress_snapshot_tracks_terminal_style_updates(tmp_path):
    state = LocalBOState(tmp_path)

    state.set_progress("Testing progress", 2, 4, "halfway")
    payload = state.to_json()

    assert payload["progress"]["label"] == "Testing progress"
    assert payload["progress"]["percent"] == 50
    assert state.progress_snapshot()["detail"] == "halfway"


def test_cancel_request_marks_progress_without_state_lock(tmp_path):
    state = LocalBOState(tmp_path)

    state.set_progress("Long task", 1, 10, "working")
    progress = state.request_cancel()

    assert progress["status"] == "cancelling"
    assert state.cancel_event.is_set()
    try:
        state.check_cancelled()
    except RunCancelled as exc:
        assert "stopped" in str(exc)
    else:  # pragma: no cover - explicit assertion path
        raise AssertionError("Expected RunCancelled")


def test_cancelled_benchmark_saves_partial_run_for_export_and_resume(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "dataset.csv",
        b"procedure,value\nproc a,1\nproc b,2\nproc c,3\nproc d,4\n",
    )
    state.update_config(
        {
            "acquisition": "random",
            "optimizer": "gpr",
            "benchmark_iterations": 2,
            "benchmark_replicates": 1,
            "benchmark_initial_points": 2,
        }
    )
    original_next_candidate = state._benchmark_next_candidate

    def stop_before_select(available, observations, rng, acquisition=None):
        state.request_cancel()
        return original_next_candidate(available, observations, rng, acquisition)

    state._benchmark_next_candidate = stop_before_select
    payload = state.run_benchmark({"name": "cancel smoke"})

    assert len(payload["benchmark_runs"]) == 1
    partial = payload["benchmark_runs"][0]
    assert partial["partial"] is True
    assert partial["status"] == "stopped"
    assert len(partial["replicate_observations"][0]) == 2
    assert payload["progress"]["status"] == "cancelled"
    assert "stopped" in payload["last_model_status"].lower()

    rows = list(csv.DictReader(StringIO(state.export_observations_csv())))
    assert len(rows) == 2
    assert {row["source"] for row in rows} == {"offline_benchmark"}
    assert rows[0]["run_status"] == "stopped"

    state._benchmark_next_candidate = original_next_candidate
    resumed = state.run_benchmark({"resume_id": partial["id"]})

    run = resumed["benchmark_runs"][0]
    assert run["partial"] is False
    assert run["status"] == "complete"
    assert len(run["replicate_observations"][0]) == 4
    assert resumed["progress"]["status"] == "complete"


def test_run_and_append_auto_resumes_matching_partial_benchmark(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "dataset.csv",
        b"procedure,value\nproc a,1\nproc b,2\nproc c,3\nproc d,4\n",
    )
    state.update_config(
        {
            "acquisition": "random",
            "optimizer": "gpr",
            "benchmark_iterations": 2,
            "benchmark_replicates": 1,
            "benchmark_initial_points": 1,
        }
    )
    original_next_candidate = state._benchmark_next_candidate

    def fail_once(available, observations, rng, acquisition=None):
        raise RuntimeError("connection lost")

    state._benchmark_next_candidate = fail_once
    payload = state.run_benchmark({"name": "connection smoke"})
    partial = payload["benchmark_runs"][0]

    assert partial["partial"] is True
    assert partial["status"] == "error"
    assert len(partial["replicate_observations"][0]) == 1

    state._benchmark_next_candidate = original_next_candidate
    state.update_config({"api_pause_seconds": 2.0})
    resumed = state.run_benchmark({"name": "connection smoke"})
    run = resumed["benchmark_runs"][0]

    assert len(resumed["benchmark_runs"]) == 1
    assert run["id"] == partial["id"]
    assert run["partial"] is False
    assert run["status"] == "complete"
    assert run["config"]["api_pause_seconds"] == 2.0
    assert any(
        event["message"] == "Resuming partial benchmark 'connection smoke'."
        for event in state.events
    )


def test_run_display_dedupes_live_resume_copy():
    assert "function benchmarkRunsForDisplay()" in INDEX_HTML
    assert "runs[index] = liveRun;" in INDEX_HTML


def test_benchmark_can_use_greedy_final_iteration(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "dataset.csv",
        b"procedure,value\nproc a,1\nproc b,2\nproc c,3\nproc d,4\nproc e,5\n",
    )
    state.update_config(
        {
            "acquisition": "expected_improvement",
            "optimizer": "gpr",
            "benchmark_iterations": 3,
            "benchmark_replicates": 2,
            "benchmark_initial_points": 1,
            "greedy_final_iteration": True,
        }
    )
    seen = []

    def choose_first(available, observations, rng, acquisition=None):
        seen.append(acquisition)
        return available[0]

    state._benchmark_next_candidate = choose_first
    payload = state.run_benchmark({"name": "greedy final smoke"})

    assert seen == [None, None, "greedy", None, None, "greedy"]
    assert payload["benchmark_runs"][0]["config"]["greedy_final_iteration"] is True
    assert payload["progress"]["status"] == "complete"


def test_inverse_target_can_use_supplied_benchmark_observations(tmp_path):
    state = LocalBOState(tmp_path)
    state.update_config({"inverse_target_multiplier": 1.2, "inverse_target_jitter": 0})
    observations = [
        {"procedure": "proc a", "value": 2.0},
        {"procedure": "proc b", "value": 3.0},
    ]

    assert np.isclose(state._inverse_target_display_value(observations), 3.6)


def test_inverse_target_uses_seeded_multiplier_jitter(tmp_path):
    state = LocalBOState(tmp_path)
    state.update_config({"inverse_target_multiplier": 1.2, "inverse_target_jitter": 0.05})
    observations = [{"procedure": "proc b", "value": 3.0}]
    expected_rng = random.Random(11)
    expected = 3.0 * expected_rng.normalvariate(1.2, 0.05)

    assert np.isclose(
        state._inverse_target_display_value(observations, rng=random.Random(11)),
        expected,
    )


def test_inverse_target_floor_prevents_zero_anchored_maximize_target(tmp_path):
    state = LocalBOState(tmp_path)
    state.update_config(
        {
            "inverse_target_multiplier": 1.2,
            "inverse_target_jitter": 0,
            "inverse_target_floor_value": "5",
        }
    )
    observations = [{"procedure": "proc zero", "value": 0.0}]

    assert state._inverse_target_display_value(observations) == 5.0


def test_llm_scored_candidate_count_uses_shortlist(tmp_path):
    state = LocalBOState(tmp_path)
    state.update_config(
        {
            "optimizer": "llm",
            "score_limit": 250,
            "inverse_filter": 16,
            "inverse_random_candidates": 4,
        }
    )

    assert state._llm_scored_candidate_count(500) == 20

    state.update_config({"inverse_filter": 0})

    assert state._llm_scored_candidate_count(500) == 250


def test_api_retry_recovers_from_rate_limit_message(tmp_path):
    state = LocalBOState(tmp_path)
    state.update_config(
        {
            "api_pause_seconds": 0,
            "api_retry_attempts": 3,
            "api_rate_limit_cooldown_seconds": 0,
        }
    )
    attempts = {"count": 0}

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("Error code: 429 - rate limit reached. Please try again in 1ms.")
        return "ok"

    assert state._api_call_with_retries("Retry smoke", flaky_call) == "ok"
    assert attempts["count"] == 2


def test_tpm_rate_limit_uses_cooldown_even_with_short_provider_delay():
    error = RuntimeError(
        "Rate limit reached for gpt-4o on tokens per min (TPM). "
        "Please try again in 666ms."
    )

    delay = _retry_delay_seconds(error, 0, 0.5, 60, rate_limit_cooldown=10)

    assert delay >= 10


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


def test_dataset_prompt_regeneration_preserves_custom_prompt_on_import(tmp_path):
    state = LocalBOState(tmp_path)
    state.config["prediction_system_message"] = "custom prediction prompt"
    state.config["inverse_system_message"] = "custom inverse prompt"

    payload = state.import_dataset("dataset.csv", b"procedure,value\nproc a,1.2\n")

    assert payload["config"]["prediction_system_message"] == "custom prediction prompt"
    assert payload["config"]["inverse_system_message"] == "custom inverse prompt"

    payload = state.regenerate_prompts()

    assert payload["config"]["prediction_system_message"].startswith(
        GENERATED_PREDICTION_PROMPT_PREFIX
    )
    assert "value" in payload["config"]["prediction_system_message"]
    assert payload["config"]["inverse_system_message"].startswith(
        GENERATED_INVERSE_PROMPT_PREFIX
    )
    assert "1.2" not in payload["config"]["inverse_system_message"]


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
