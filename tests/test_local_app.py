import csv
import json
import random
from io import BytesIO, StringIO

import numpy as np

from boicl import AskTellFewShotTopk, Pool
from boicl.llm_model import GaussDist
from boicl.local_app import (
    DEFAULT_CONFIG,
    DEFAULT_PREDICTION_SYSTEM_MESSAGE,
    GENERATED_INVERSE_PROMPT_PREFIX,
    GENERATED_PREDICTION_PROMPT_PREFIX,
    INDEX_HTML,
    LocalBOState,
    OBJECTIVE_BOUNDS_PROMPT_MARKER,
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
        "procedure,alpha phase (%)\n"
        "\"Reduction experiment of WO3/SiO2: ramp to 400 C at 10 C/min, "
        "soak for 4 h.\",\n"
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
    assert "Open Runner Tab" in POOL_BUILDER_HTML
    assert "Reset Builder" in POOL_BUILDER_HTML
    assert "objectivePreviewHeader" in POOL_BUILDER_HTML
    assert "blank until measured" in POOL_BUILDER_HTML
    assert "The objective column is blank until measured." in POOL_BUILDER_HTML
    assert "boicl-focus-runner" in POOL_BUILDER_HTML
    assert "target=\"boicl_runner\"" in POOL_BUILDER_HTML
    assert "Save the campaign in the runner to keep it for later" in POOL_BUILDER_HTML
    assert "BroadcastChannel" in POOL_BUILDER_HTML


def test_main_app_candidate_labels_surface_variable_values():
    assert "candidateProcedureSummary" in INDEX_HTML
    assert "candidateOptionLabel" in INDEX_HTML
    assert "candidatePreview" in INDEX_HTML
    assert "enter 73.5 for 73.5%, not 0.735" in INDEX_HTML
    assert "openPoolBuilder" in INDEX_HTML
    assert "boicl_pool_builder" in INDEX_HTML
    assert "boicl_runner" in INDEX_HTML
    assert "Start Fresh" in INDEX_HTML
    assert "Clear & Re-run" in INDEX_HTML
    assert "Export Archive" in INDEX_HTML
    assert "Import Archive" in INDEX_HTML
    assert "Delete Saved" in INDEX_HTML
    assert "button-grid" in INDEX_HTML
    assert "tooltip-target" in INDEX_HTML
    assert "engineStatus" in INDEX_HTML
    assert "llmPoolScope" in INDEX_HTML
    assert "candidateSearch" in INDEX_HTML
    assert "/api/candidate-search" in INDEX_HTML
    assert "/api/delete-campaign" in INDEX_HTML
    assert "/api/delete-observation" in INDEX_HTML
    assert "Pool Builder imported" in INDEX_HTML
    assert "boicl-focus-runner" in INDEX_HTML
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


def test_import_dataset_tracks_dataset_metadata(tmp_path):
    state = LocalBOState(tmp_path)
    payload = state.import_dataset(
        "Mo-Carburization_Dataset_v1.csv",
        b"procedure,value\nproc a,1\n",
    )

    assert payload["dataset"]["filename"] == "Mo-Carburization_Dataset_v1.csv"
    assert payload["dataset"]["id"].startswith("mo_carburization_dataset_v1_")
    assert state.candidates[0]["dataset_id"] == payload["dataset"]["id"]
    assert payload["candidates"][0]["dataset_id"] == payload["dataset"]["id"]


def test_candidate_search_finds_rows_beyond_public_preview_limit(tmp_path):
    state = LocalBOState(tmp_path)
    rows = ["procedure"] + [f"procedure row {idx}" for idx in range(1, 1001)]
    state.import_dataset("large_pool.csv", "\n".join(rows).encode("utf-8"))

    payload = state.search_candidates("row 900", limit=10)

    assert payload["available_count"] == 1000
    assert payload["candidates"]
    assert payload["candidates"][0]["row"] == 900


def test_import_dataset_accepts_npy_table(tmp_path):
    state = LocalBOState(tmp_path)
    out = BytesIO()
    np.save(out, np.array([["proc a", 1.2], ["proc b", 2.5]], dtype=object))

    payload = state.import_dataset("dataset.npy", out.getvalue())

    assert payload["candidate_count"] == 2
    assert payload["label_count"] == 2
    assert payload["objective_names"] == ["objective"]
    assert state.candidates[1]["objectives"]["objective"] == 2.5


def test_defaults_match_current_numeric_settings():
    assert DEFAULT_CONFIG["benchmark_initial_points"] == 1
    assert DEFAULT_CONFIG["batch_size"] == 1
    assert DEFAULT_CONFIG["benchmark_iterations"] == 30
    assert DEFAULT_CONFIG["benchmark_replicates"] == 5
    assert "benchmark_starting_baseline" not in DEFAULT_CONFIG
    assert DEFAULT_CONFIG["objective_lower_bound"] == ""
    assert DEFAULT_CONFIG["objective_upper_bound"] == ""
    assert DEFAULT_CONFIG["ucb_lambda"] == 0.1
    assert DEFAULT_CONFIG["llm_samples"] == 3
    assert DEFAULT_CONFIG["llm_uncertainty_calibration"] == 1.0
    assert DEFAULT_CONFIG["llm_pool_scope"] == "full"
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
    assert 'id="objectiveLowerBound"' in INDEX_HTML
    assert 'id="objectiveUpperBound"' in INDEX_HTML
    assert 'id="modelOptions"' not in INDEX_HTML
    assert 'id="embeddingModelOptions"' not in INDEX_HTML


def test_initial_random_points_are_capped_at_three(tmp_path):
    state = LocalBOState(tmp_path)

    state.update_config({"benchmark_initial_points": 99})

    assert state.config["benchmark_initial_points"] == 3


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


def test_offline_benchmark_starts_from_real_initial_points(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "dataset.csv",
        b"procedure,value\nproc a,1\nproc b,3\nproc c,5\nproc d,7\n",
    )
    state.update_config(
        {
            "acquisition": "random",
            "benchmark_iterations": 1,
            "benchmark_replicates": 1,
            "benchmark_initial_points": 2,
        }
    )

    payload = state.run_benchmark({"name": "initial points"})
    run = payload["benchmark_runs"][0]

    assert [point["index"] for point in run["summary"]] == [1, 2, 3]
    assert not any(point.get("baseline") for point in run["replicate_traces"][0])
    assert all(point["index"] >= 1 for point in run["replicate_traces"][0])
    assert len(run["replicate_observations"][0]) == 3


def test_benchmark_progress_counts_initial_points_before_first_bo_result(tmp_path):
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
    payload = state.run_benchmark({"name": "initial progress"})
    partial = payload["benchmark_runs"][0]

    assert captured["current"] == 2
    assert captured["total"] == 4
    assert "initialized 2 random initial points" in captured["detail"]
    assert [point["index"] for point in captured["summary"]] == [1, 2]
    assert partial["status"] == "stopped"
    assert [point["index"] for point in partial["summary"]] == [1, 2]


def test_plot_horizon_counts_initial_points_and_bo_iterations(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "dataset.csv",
        b"procedure,value\nproc a,1\nproc b,2\nproc c,3\nproc d,4\nproc e,5\n",
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

    payload = state.run_benchmark({"name": "mean horizon"})
    run = payload["benchmark_runs"][0]

    assert state.plot_horizon() == 4
    assert [point["index"] for point in run["summary"]] == [1, 2, 3, 4]
    assert [point["index"] for point in payload["random_walk_trace"]] == [1, 2, 3, 4]


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
        {"available": 3, "observations": 1, "acquisition": "upper_confidence_bound"}
    ]
    assert payload["benchmark_runs"][0]["status"] == "complete"
    assert len(payload["benchmark_runs"][0]["replicate_observations"][0]) == 2


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
    assert payload["dataset"]["filename"] == "dataset.csv"
    assert payload["dataset"]["id"].startswith("dataset_")
    assert len(payload["observations"]) == 2
    assert payload["observations"][-1]["value"] == 2.0


def test_import_dataset_saves_active_project_and_starts_new_campaign(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("old_dataset.csv", b"procedure,value\nold a,1\nold b,2\n")
    saved = state.save_campaign({"name": "Active campaign"})
    old_id = saved["campaign"]["id"]
    state.update_config({"optimizer": "llm", "benchmark_iterations": 9})
    state.add_observation({"candidate_id": "cand-0", "value": 1.0})

    payload = state.import_dataset("new_pool.csv", b"procedure,value\nnew a,5\n")

    assert payload["campaign"]["saved"] is True
    assert payload["campaign"]["id"] != old_id
    assert payload["dataset"]["filename"] == "new_pool.csv"
    assert payload["config"]["optimizer"] == "llm"
    assert payload["config"]["benchmark_iterations"] == 9
    assert payload["candidate_count"] == 1
    assert payload["observations"] == []
    old_payload = json.loads(
        (tmp_path / "saved_experiments" / old_id / "campaign.json").read_text(
            encoding="utf-8"
        )
    )
    assert old_payload["meta"]["name"] == "Active campaign"
    assert len(old_payload["observations"]) == 1
    new_id = payload["campaign"]["id"]
    assert (tmp_path / "saved_experiments" / new_id / "campaign.json").exists()


def test_delete_observation_removes_live_row_and_autosaves(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("live_pool.csv", b"procedure\nproc a\nproc b\n")
    saved = state.save_campaign({"name": "Live delete"})
    state.add_observation({"candidate_id": "cand-0", "value": 4.0})
    state.add_observation({"candidate_id": "cand-1", "value": 6.0})

    payload = state.delete_observation({"id": "obs-1"})

    assert [obs["id"] for obs in payload["observations"]] == ["obs-2"]
    assert payload["available_count"] == 1
    saved_payload = json.loads(
        (
            tmp_path
            / "saved_experiments"
            / saved["campaign"]["id"]
            / "campaign.json"
        ).read_text(encoding="utf-8")
    )
    assert [obs["id"] for obs in saved_payload["observations"]] == ["obs-2"]
    state.add_observation({"candidate_id": "cand-0", "value": 8.0})
    assert [obs["id"] for obs in state.observations] == ["obs-2", "obs-3"]


def test_import_campaign_archive_restores_and_saves_copy(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("archive_dataset.csv", b"procedure,value\nproc a,1\nproc b,2\n")
    state.save_campaign({"name": "Archive campaign"})
    state.add_observation({"candidate_id": "cand-0", "value": 1.0})
    state.update_config({"acquisition": "random", "benchmark_iterations": 1})
    state.run_benchmark({"name": "archive benchmark"})
    archive = state.export_campaign_archive_json().encode("utf-8")

    restored = LocalBOState(tmp_path)
    payload = restored.import_campaign_archive("archive_backup.json", archive)

    assert payload["campaign"]["saved"] is True
    assert payload["campaign"]["name"] == "Archive campaign"
    assert payload["dataset"]["filename"] == "archive_dataset.csv"
    assert len(payload["observations"]) == 1
    assert len(payload["benchmark_runs"]) == 1
    assert payload["benchmark_runs"][0]["status"] == "complete"
    assert restored.export_campaign_archive_filename().startswith(
        "boicl_archive_campaign_"
    )
    assert (
        tmp_path
        / "saved_experiments"
        / payload["campaign"]["id"]
        / "campaign.json"
    ).exists()


def test_start_fresh_clears_loaded_state_without_deleting_saves(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure,value\nproc a,1\n")
    saved = state.save_campaign({"name": "Keep me"})

    payload = state.start_fresh()

    assert payload["candidate_count"] == 0
    assert payload["observations"] == []
    assert payload["benchmark_runs"] == []
    assert payload["campaign"]["saved"] is False
    assert payload["dataset"]["id"] == ""
    assert (tmp_path / "saved_experiments" / saved["campaign"]["id"] / "campaign.json").exists()


def test_delete_campaign_removes_saved_folder_and_clears_active_save(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure,value\nproc a,1\n")
    saved = state.save_campaign({"name": "Delete me"})
    campaign_id = saved["campaign"]["id"]

    payload = state.delete_campaign({"id": campaign_id})

    assert payload["campaign"]["saved"] is False
    assert payload["campaign"]["name"] == ""
    assert not (tmp_path / "saved_experiments" / campaign_id).exists()
    assert all(campaign["id"] != campaign_id for campaign in payload["campaigns"])
    assert "Deleted saved campaign: Delete me." in payload["last_model_status"]


def test_export_observations_csv_includes_dataset_and_settings_metadata(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "alpha_pool.csv",
        b"procedure,alpha phase (%)\nproc a,1\nproc b,2\nproc c,3\n",
    )
    state.save_campaign({"name": "Alpha campaign"})
    state.add_observation({"candidate_id": "cand-0", "value": 1.0})
    state.update_config({"acquisition": "random", "benchmark_iterations": 1})
    state.run_benchmark({"name": "random metadata"})

    rows = list(csv.DictReader(StringIO(state.export_observations_csv())))

    assert rows
    first = rows[0]
    assert first["campaign_name"] == "Alpha campaign"
    assert first["dataset_filename"] == "alpha_pool.csv"
    assert first["dataset_id"].startswith("alpha_pool_")
    assert first["candidate_id"] == "cand-0"
    assert first["candidate_row"] == "1"
    assert "benchmark_iterations" in first["settings_json"]
    offline = next(row for row in rows if row["source"] == "offline_benchmark")
    assert "benchmark_iterations" in offline["run_settings_json"]
    assert state.export_observations_filename().startswith(
        "boicl_alpha_campaign_alpha_pool_"
    )


def test_live_observation_keeps_model_prediction_for_plot_and_export(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "alpha_pool.csv",
        b"procedure,alpha phase (%)\nproc a,\nproc b,\n",
        objective_name="alpha phase (%)",
    )
    state.suggestions = [
        {
            "candidate_id": "cand-0",
            "procedure": "proc a",
            "acquisition": 1.25,
            "mean": 42.0,
            "std": 3.5,
            "source": "llm",
            "optimizer": "llm",
            "acquisition_function": "upper_confidence_bound",
            "prediction_model": "gpt-4o",
            "inverse_model": "gpt-4o",
            "embedding_model": "text-embedding-ada-002",
            "llm_samples": 3,
            "llm_uncertainty_calibration": 4.33,
            "inverse_filter": 16,
            "inverse_seed": "target-like generated procedure",
        }
    ]

    payload = state.add_observation(
        {"candidate_id": "cand-0", "value": 44.0, "uncertainty": 0.6}
    )

    observation = payload["observations"][0]
    assert observation["prediction"]["mean"] == 42.0
    assert observation["prediction"]["std"] == 3.5
    assert observation["prediction"]["optimizer"] == "llm"
    assert (
        observation["prediction"]["acquisition_function"]
        == "upper_confidence_bound"
    )

    rows = list(csv.DictReader(StringIO(state.export_observations_csv())))
    assert rows[0]["prediction_mean"] == "42.0"
    assert rows[0]["prediction_uncertainty"] == "3.5"
    assert rows[0]["prediction_acquisition"] == "1.25"
    assert rows[0]["prediction_optimizer"] == "llm"
    assert rows[0]["prediction_acquisition_function"] == "upper_confidence_bound"
    assert rows[0]["prediction_model"] == "gpt-4o"
    assert rows[0]["embedding_model"] == "text-embedding-ada-002"
    assert rows[0]["prediction_llm_samples"] == "3"
    assert rows[0]["prediction_llm_uncertainty_calibration"] == "4.33"
    assert rows[0]["prediction_inverse_filter"] == "16"
    assert rows[0]["prediction_inverse_seed"] == "target-like generated procedure"
    assert rows[0]["alpha phase (%)_uncertainty"] == "0.6"
    assert "<th>Model / Acq.</th>" in INDEX_HTML
    assert "<th>Method</th>" in INDEX_HTML


def test_live_plot_collapses_candidate_replicates_but_preserves_raw_rows(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "alpha_pool.csv",
        b"procedure,alpha phase (%)\nproc a,\nproc b,\n",
        objective_name="alpha phase (%)",
    )

    state.add_observation({"candidate_id": "cand-0", "value": 10.0})
    state.add_observation({"candidate_id": "cand-1", "value": 12.0})
    payload = state.add_observation({"candidate_id": "cand-0", "value": 14.0})

    assert len(payload["observations"]) == 3
    points = payload["live_observation_points"]
    assert len(points) == 2
    assert points[0]["value"] == 12.0
    assert round(points[0]["replicate_std"], 6) == round(2**0.5 * 2, 6)
    assert points[0]["replicate_count"] == 2
    assert [point["best"] for point in payload["best_trace"]] == [12.0, 12.0]
    assert payload["best_trace"][0]["best_replicate_count"] == 2

    rows = list(csv.DictReader(StringIO(state.export_observations_csv())))
    assert len([row for row in rows if row["source"] == "live"]) == 3


def test_config_change_clears_stale_model_suggestions(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("alpha_pool.csv", b"procedure\nproc a\nproc b\n")
    state.suggestions = [
        {
            "candidate_id": "cand-0",
            "procedure": "proc a",
            "acquisition": 1.25,
            "mean": 42.0,
            "std": 3.5,
            "source": "llm",
            "optimizer": "llm",
            "acquisition_function": "upper_confidence_bound",
            "prediction_model": "gpt-4o",
        }
    ]

    payload = state.update_config({"acquisition": "greedy"})

    assert payload["suggestions"] == []
    assert "Update suggestions" in payload["last_model_status"]


def test_llm_model_keeps_original_units_when_target_scaling_is_enabled(
    tmp_path, monkeypatch
):
    import boicl as boicl_pkg

    told_values = []
    calibration_values = []

    class FakeAskTellFewShotTopk:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def set_calibration_factor(self, value):
            calibration_values.append(value)

        def tell(self, procedure, value):
            told_values.append((procedure, value))

    monkeypatch.setattr(boicl_pkg, "AskTellFewShotTopk", FakeAskTellFewShotTopk)
    state = LocalBOState(tmp_path)
    state.update_config(
        {
            "optimizer": "llm",
            "objective_scaling": "minmax",
            "inverse_target_floor_value": "5",
            "inverse_target_multiplier": 1.0,
            "inverse_target_jitter": 0.0,
        }
    )
    observations = [
        {"procedure": "zero procedure", "value": 0.0},
        {"procedure": "best procedure", "value": 100.0},
    ]

    _, scaler = state._build_llm_model(observations)

    assert scaler["mode"] == "off"
    assert told_values == [
        ("zero procedure", 0.0),
        ("best procedure", 100.0),
    ]
    assert calibration_values == [1.0]
    assert state._inverse_target_model_value(scaler, [observations[0]]) == 5.0


def test_llm_acquisition_best_uses_original_units_when_scaling_enabled(
    tmp_path, monkeypatch
):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "alpha_pool.csv",
        b"procedure,alpha\nseed zero,0\nseed high,100\ncandidate,50\n",
        objective_name="alpha",
    )
    state.update_config(
        {
            "optimizer": "llm",
            "objective_scaling": "minmax",
            "inverse_filter": 0,
            "batch_size": 1,
        }
    )

    class FakeModel:
        def predict(self, procedures, system_message=""):
            assert system_message
            return [GaussDist(50.0, 0.0) for _ in procedures]

    best_values = []

    def fake_aq(dist, best):
        best_values.append(best)
        return dist.mean() - best

    state._build_llm_model = lambda observations=None: (FakeModel(), {"mode": "off"})
    state._llm_acquisition_callable = lambda acquisition_name: fake_aq

    suggestions = state._llm_suggestions(
        [state.candidates[2]],
        observations=[
            state._observation_from_candidate(state.candidates[0]),
            state._observation_from_candidate(state.candidates[1]),
        ],
        rng=random.Random(0),
        k=1,
    )

    assert suggestions[0]["mean"] == 50.0
    assert best_values == [100.0]


def test_prediction_summary_combines_offline_replicate_predictions(tmp_path):
    state = LocalBOState(tmp_path)
    summary = state._summarize_prediction_points(
        [
            [{"prediction": {"mean": 10.0, "std": 1.0}}],
            [{"prediction": {"mean": 14.0, "std": 3.0}}],
        ]
    )

    assert summary[0]["index"] == 1
    assert summary[0]["mean"] == 12.0
    assert round(summary[0]["std"], 6) == round((8.0 + 5.0) ** 0.5, 6)


def test_plot_intervals_are_clipped_to_objective_display_bounds(tmp_path):
    state = LocalBOState(tmp_path)
    state.update_config({"objective_name": "alpha phase (%)"})

    assert state.to_json()["plot_objective_bounds"] == {"lower": 0.0, "upper": 100.0}

    prediction_summary = state._summarize_prediction_points(
        [[{"prediction": {"mean": 95.0, "std": 30.0}}]]
    )
    assert prediction_summary[0]["mean"] == 95.0
    assert prediction_summary[0]["std"] == 30.0
    assert prediction_summary[0]["lower"] == 65.0
    assert prediction_summary[0]["upper"] == 100.0

    replicate_summary = state._summarize_replicate_traces(
        [[{"index": 1, "best": 80.0}], [{"index": 1, "best": 100.0}]]
    )
    assert replicate_summary[0]["mean"] == 90.0
    assert replicate_summary[0]["lower"] >= 0.0
    assert replicate_summary[0]["upper"] == 100.0


def test_live_random_walk_records_control_points_and_exports_them(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset(
        "alpha_pool.csv",
        b"procedure,alpha phase (%)\nproc a,\nproc b,\nproc c,\n",
        objective_name="alpha phase (%)",
    )

    payload = state.start_live_random_walk({"target_count": 2, "seed": 5})
    assert payload["live_random_walk"]["status"] == "waiting_for_result"
    assert payload["live_random_walk"]["current_candidate"]["id"]

    first_candidate = payload["live_random_walk"]["current_candidate"]["id"]
    payload = state.add_live_random_walk_result({"value": 11.0, "uncertainty": 0.4})

    walk = payload["live_random_walk"]
    assert len(walk["observations"]) == 1
    assert walk["observations"][0]["candidate_id"] == first_candidate
    assert walk["observations"][0]["uncertainty"] == 0.4
    assert payload["live_random_walk_trace"][0]["index"] == 1
    assert payload["live_random_walk_trace"][0]["best"] == 11.0

    rows = list(csv.DictReader(StringIO(state.export_observations_csv())))
    random_row = next(row for row in rows if row["source"] == "live_random_walk")
    assert random_row["run_name"] == "Live random walk"
    assert random_row["experiment_count"] == "1"
    assert random_row["alpha phase (%)"] == "11.0"
    assert random_row["alpha phase (%)_uncertainty"] == "0.4"
    assert '"target_count": 2' in random_row["run_settings_json"]


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


def test_precompute_embeddings_finishes_when_dataset_is_already_cached(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = LocalBOState(tmp_path)
    state.import_dataset("dataset.csv", b"procedure,value\nproc a,\nproc b,\n")
    state.cache_dir.mkdir()
    state.embedding_cache_path().write_text(
        "x,embedding,embedding_model\n"
        '"proc a","[1.0, 0.0]",text-embedding-ada-002\n'
        '"proc b","[0.0, 1.0]",text-embedding-ada-002\n',
        encoding="utf-8",
    )

    payload = state.precompute_embeddings()

    assert payload["progress"]["status"] == "complete"
    assert payload["progress"]["percent"] == 100
    assert payload["progress"]["current"] == 2
    assert payload["progress"]["total"] == 2
    assert "No new embeddings needed" in payload["progress"]["detail"]
    assert "already cached" in payload["last_model_status"]


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

    state.update_config({"score_limit": 10, "llm_pool_scope": "broad"})

    assert state._llm_scored_candidate_count(500) == 10

    state.update_config({"inverse_filter": 0})

    assert state._llm_scored_candidate_count(500) == 10


def test_live_llm_shortlist_uses_full_available_pool_after_one_seed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    state = LocalBOState(tmp_path)
    rows = ["procedure", *[f"proc {index}" for index in range(10)]]
    state.import_dataset("pool.csv", "\n".join(rows).encode("utf-8"))
    state.update_config(
        {
            "optimizer": "llm",
            "batch_size": 1,
            "score_limit": 3,
            "inverse_filter": 2,
            "inverse_random_candidates": 0,
        }
    )
    state.add_observation({"candidate_id": "cand-0", "value": 1})

    class FakeModel:
        def ask(self, *args, **kwargs):
            raise AssertionError("live LLM scoring should bypass the <2-example fallback")

        def predict(self, possible_x, system_message=""):
            assert system_message
            return [
                GaussDist(8.5, 0.7) if procedure == "proc 8" else GaussDist(7.0, 0.2)
                for procedure in possible_x
            ]

    seen = {"retrieval_counts": []}
    targets = iter([10.0, 12.0])

    state._build_llm_model = lambda observations=None: (FakeModel(), {"mode": "off"})
    state._inverse_target_display_value = lambda *args, **kwargs: next(targets)
    state._generate_inverse_text = lambda *args, **kwargs: "same inverse proposal"

    def fake_cached_sample(procedures, query, k, lambda_mult=0.5):
        seen["retrieval_counts"].append(len(procedures))
        return procedures[-k:]

    state._cached_approx_sample = fake_cached_sample

    payload = state.suggest()
    suggestion = payload["suggestions"][0]

    assert seen["retrieval_counts"] == [9]
    assert suggestion["procedure"] == "proc 8"
    assert suggestion["acquisition"] == 8.57
    assert suggestion["mean"] == 8.5
    assert suggestion["std"] == 0.7
    assert len(payload["inverse_designs"]) == 1

    payload = state.suggest()

    assert seen["retrieval_counts"] == [9, 9]
    assert len(payload["inverse_designs"]) == 1
    assert payload["inverse_designs"][0]["target"] == 12.0


def test_live_llm_shortlist_can_prefilter_with_broad_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    state = LocalBOState(tmp_path)
    rows = ["procedure", *[f"proc {index}" for index in range(10)]]
    state.import_dataset("pool.csv", "\n".join(rows).encode("utf-8"))
    state.update_config(
        {
            "optimizer": "llm",
            "batch_size": 1,
            "score_limit": 3,
            "llm_pool_scope": "broad",
            "inverse_filter": 2,
            "inverse_random_candidates": 0,
        }
    )
    state.add_observation({"candidate_id": "cand-0", "value": 1})

    class FakeModel:
        def predict(self, possible_x, system_message=""):
            return [
                GaussDist(3.0, 1.0) if index == 0 else GaussDist(1.0, 0.1)
                for index, _ in enumerate(possible_x)
            ]

    seen = {"retrieval_counts": []}
    state._build_llm_model = lambda observations=None: (FakeModel(), {"mode": "off"})
    state._inverse_target_display_value = lambda *args, **kwargs: 10.0
    state._generate_inverse_text = lambda *args, **kwargs: "target proposal"

    def fake_cached_sample(procedures, query, k, lambda_mult=0.5):
        seen["retrieval_counts"].append(len(procedures))
        return procedures[:k]

    state._cached_approx_sample = fake_cached_sample

    payload = state.suggest()

    assert seen["retrieval_counts"] == [3]
    assert payload["suggestions"][0]["acquisition"] == 3.1
    assert payload["suggestions"][0]["mean"] == 3.0
    assert payload["suggestions"][0]["std"] == 1.0


def test_llm_flat_predictions_use_inverse_retrieval_rank(tmp_path):
    state = LocalBOState(tmp_path)
    rows = ["procedure", *[f"proc {index}" for index in range(5)]]
    state.import_dataset("pool.csv", "\n".join(rows).encode("utf-8"))
    state.update_config(
        {
            "optimizer": "llm",
            "batch_size": 1,
            "inverse_filter": 2,
            "inverse_random_candidates": 0,
            "inverse_target_jitter": 0,
        }
    )
    state.add_observation({"candidate_id": "cand-0", "value": 1})

    class FakeModel:
        def predict(self, possible_x, system_message=""):
            return [GaussDist(0.0, 0.0) for _ in possible_x]

    state._build_llm_model = lambda observations=None: (FakeModel(), {"mode": "off"})
    state._inverse_target_display_value = lambda *args, **kwargs: 5.0
    state._generate_inverse_text = lambda *args, **kwargs: "target-like query"
    state._cached_approx_sample = lambda procedures, query, k, lambda_mult=0.5: procedures[:k]

    suggestions = state._llm_suggestions(
        state.available_candidates(),
        state.active_observations(),
        rng=random.Random(99),
        k=1,
    )

    assert suggestions[0]["procedure"] == "proc 1"
    assert suggestions[0]["source"] == "llm"
    assert suggestions[0]["mean"] == 0.0
    assert "flat" in suggestions[0]["selection_note"].lower()
    assert "inverse-design" in suggestions[0]["selection_note"]
    assert suggestions[0]["inverse_seed"] == "target-like query"


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

    message = state.prediction_system_message()

    assert message.startswith(DEFAULT_PREDICTION_SYSTEM_MESSAGE)
    assert "Prediction task guardrail" in message


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


def test_auto_prompts_refresh_without_adding_bounds_to_llm_context(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("pool.csv", b"procedure,objective\nproc a,\nproc b,\n")

    payload = state.update_config(
        {
            "objective_name": "alpha Mo2C",
            "objective_lower_bound": "0",
            "objective_upper_bound": "100",
        }
    )

    assert "Active objective selected in the tool: alpha Mo2C" in payload["config"][
        "prediction_system_message"
    ]
    assert "bounded from 0 to 100" not in payload["config"]["prediction_system_message"]
    assert "alpha Mo2C" in payload["config"]["inverse_system_message"]
    assert payload["plot_objective_bounds"] == {"lower": 0.0, "upper": 100.0}


def test_custom_prompts_are_not_replaced_and_saved_bounds_are_stripped(tmp_path):
    state = LocalBOState(tmp_path)
    state.import_dataset("pool.csv", b"procedure,objective\nproc a,\nproc b,\n")
    state.update_config(
        {
            "prediction_system_message": (
                "custom prediction\n"
                f"{OBJECTIVE_BOUNDS_PROMPT_MARKER} the active objective is physically "
                "bounded from 0 to 100 in original objective units."
            ),
            "inverse_system_message": (
                "custom inverse\n"
                f"{OBJECTIVE_BOUNDS_PROMPT_MARKER} the active objective is physically "
                "bounded from 0 to 100 in original objective units."
            ),
        }
    )

    payload = state.update_config(
        {
            "objective_name": "alpha Mo2C",
            "objective_lower_bound": "0",
            "objective_upper_bound": "100",
        }
    )

    assert payload["config"]["prediction_system_message"] == "custom prediction"
    assert payload["config"]["inverse_system_message"] == "custom inverse"
    assert state.prediction_system_message().startswith("custom prediction")
    assert "Prediction task guardrail" in state.prediction_system_message()
    assert "not inverse design" in state.prediction_system_message()
    assert "bounded from 0 to 100" not in state.prediction_system_message()
    assert OBJECTIVE_BOUNDS_PROMPT_MARKER not in state.prediction_system_message()
    assert OBJECTIVE_BOUNDS_PROMPT_MARKER not in state.inverse_system_message()


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
