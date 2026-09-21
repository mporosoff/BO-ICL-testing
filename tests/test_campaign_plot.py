from copy import deepcopy
import json

import pytest

from boicl.campaign_plot import comparison_compatibility, measured_points, plot_payload
from boicl.measurement_quality import default_definition


def campaign(cid="main", engine="llm"):
    return {
        "campaign_id": cid,
        "pool_fingerprint": "pool-identity",
        "initialization_fingerprint": "confirmed-three",
        "config": {
            "engine": engine,
            "name": cid,
            "objective": "moc_wt_pct",
            "units": "wt%",
            "direction": "maximize",
            "bounds": [0, 100],
            "new_measurement_budget": 10,
            "repeat_policy": "source_reset_quality_repeats",
            "seed": 616,
        },
        "candidates": [
            {
                "candidate_id": str(index),
                "procedure": f"recipe {index}",
                "hidden_oracle": 999999,
                "objectives": {"moc_wt_pct": 999999},
            }
            for index in range(8)
        ],
        "observations": [
            measurement(str(index), value, seed=True, sigma=sigma)
            for index, value, sigma in [(0, 72.1, 1.5), (1, 83.8, 5.67), (2, 23.4, 1.5)]
        ],
        "suggestions": [],
    }


def measurement(cid, value, seed=False, sigma=None, physical=None):
    return {
        "candidate_id": cid,
        "moc_wt_pct": value,
        "moc_wt_pct_sigma": sigma,
        "observation_id": "observation-" + cid,
        "physical_measurement_id": physical or "physical-" + cid,
        "record_status": "measured",
        "training_included": True,
        "is_seed": seed,
    }


def test_seed_cohort_has_successive_display_positions_without_spending_bo_steps():
    data = campaign()
    original = deepcopy(data)
    payload = plot_payload(data)
    assert data == original
    assert [point["index"] for point in payload["live_observation_points"]] == [1, 2, 3]
    assert [point["axis_label"] for point in payload["live_observation_points"]] == [
        "i1",
        "i2",
        "i3",
    ]
    assert [
        point["optimization_step"] for point in payload["live_observation_points"]
    ] == [0, 0, 0]
    assert [(point["index"], point["best"]) for point in payload["best_trace"]] == [
        (1, 72.1),
        (2, 83.8),
        (3, 83.8),
    ]
    assert all(point["baseline"] for point in payload["best_trace"])
    assert payload["best_trace"][0]["initialization_count"] == 3
    assert payload["initialization_count"] == 3
    assert payload["plot_x_axis"]["initialization_count"] == 3
    assert payload["plot_x_axis"]["initialization_end"] == 3.5
    assert payload["plot_x_axis"]["min"] == 0.5
    assert payload["plot_counts"]["new_completed"] == 0
    data["observations"].append(measurement("3", 0))
    data["observations"].append(measurement("4", 90))
    payload = plot_payload(data)
    assert [point["index"] for point in payload["best_trace"]] == [1, 2, 3, 4, 5]
    assert [point["best"] for point in payload["best_trace"]] == [
        72.1,
        83.8,
        83.8,
        83.8,
        90,
    ]
    assert [point["axis_label"] for point in payload["best_trace"]] == [
        "i1",
        "i2",
        "i3",
        "1",
        "2",
    ]
    assert [point["optimization_step"] for point in payload["best_trace"]] == [
        0,
        0,
        0,
        1,
        2,
    ]
    assert [tick["label"] for tick in payload["plot_x_axis"]["labels"]] == [
        "i1",
        "i2",
        "i3",
        "1",
        "2",
    ]
    assert payload["plot_counts"]["new_completed"] == 2
    assert payload["live_observation_points"][3]["value"] == 0
    assert payload["live_observation_points"][3]["uncertainty"] is None


def test_pending_model_prediction_is_not_measurement_or_incumbent():
    data = campaign()
    data["suggestions"] = [
        {
            "suggestion_id": "proposal-1",
            "candidate_id": "3",
            "status": "pending",
            "prediction": {
                "mean": 99.8,
                "sd": 2,
                "lower95": 94,
                "upper95": 100,
                "uncertainty_type": "latent-function bounded mixture",
            },
        },
        {
            "suggestion_id": "cancelled",
            "candidate_id": "4",
            "status": "cancelled",
            "prediction": {"mean": 10000},
        },
    ]
    payload = plot_payload(data)
    assert len(payload["live_observation_points"]) == 3
    assert payload["best_trace"][-1]["best"] == 83.8
    layer = payload["benchmark_runs"][-1]
    assert layer["summary"] == []
    predicted = layer["prediction_summary"][0]
    assert (
        predicted["index"] == 4 and predicted["mean"] == 99.8 and predicted["std"] == 2
    )
    assert predicted["axis_label"] == "1" and predicted["optimization_step"] == 1
    assert payload["plot_x_axis"]["labels"][-1] == {
        "index": 4,
        "label": "1",
        "initialization": False,
    }
    assert payload["plot_counts"]["new_completed"] == 0
    assert (
        predicted["lower"] == 94
        and predicted["upper"] == 100
        and predicted["measured"] is False
    )


def test_refinement_keeps_original_physical_position_and_prediction():
    data = campaign()
    old = measurement("3", 90, sigma=2)
    old.update(record_status="superseded_refinement", training_included=False)
    data["observations"].extend([old, measurement("4", 91)])
    revised = deepcopy(old)
    revised.update(
        observation_id="refined-3",
        moc_wt_pct=89,
        record_status="measured",
        training_included=True,
    )
    data["observations"].append(revised)
    data["suggestions"] = [
        {
            "status": "measured",
            "observation_id": old["observation_id"],
            "prediction": {"mean": 88, "std": 3},
        }
    ]
    points = measured_points(data)
    assert [
        (point["candidate_id"], point["index"], point["value"]) for point in points[3:]
    ] == [("3", 4, 89), ("4", 5, 91)]
    assert points[3]["prediction"]["mean"] == 88
    repeat = measurement("3", 92, physical="actual-second-synthesis")
    repeat["observation_id"] = "repeated-3"
    data["observations"].append(repeat)
    assert measured_points(data)[-1]["index"] == 6
    assert measured_points(data)[-1]["optimization_step"] == 3


def test_comparison_uses_only_independent_matched_histories():
    main, gp = campaign(), campaign("gp", "gpr_features")
    gp["observations"].append(measurement("3", 95))
    payload = plot_payload(main, comparisons=[gp])
    run = payload["benchmark_runs"][0]
    assert run["summary"][-1]["mean"] == 95
    assert run["summary"][-1]["index"] == 4
    assert run["summary"][-1]["axis_label"] == "1"
    assert run["initialization_count"] == 3
    assert run["summary"][-1]["std"] is None
    assert payload["best_trace"][-1]["best"] == 83.8
    assert len(main["observations"]) == 3
    mismatch = deepcopy(gp)
    mismatch["pool_fingerprint"] = "same-length-different-pool"
    payload = plot_payload(main, comparisons=[mismatch])
    assert not payload["benchmark_runs"]
    assert payload["comparison_diagnostics"][0]["mismatches"] == ["pool_fingerprint"]


def test_plot_does_not_leak_oracle_labels_or_fabricate_sparse_random_baseline():
    data = campaign()
    data["archive"] = [measurement("7", 999999)]
    payload = plot_payload(data)
    assert "999999" not in json.dumps(payload)
    assert payload["dataset_stats"] == [] and payload["random_walk_trace"] == []
    assert payload["live_random_walk"] == {}


def test_random_arm_is_separate_and_can_have_smaller_operator_budget():
    main, control = campaign(), campaign("random")
    control["config"].update(
        selection_policy="random_control",
        comparison_parent_id="main",
        new_measurement_budget=2,
    )
    control["observations"].append(measurement("3", 87))
    payload = plot_payload(main, random_campaign=control)
    assert payload["best_trace"][-1]["best"] == 83.8
    assert payload["live_random_walk_trace"][-1]["best"] == 87
    assert payload["live_random_walk_trace"][-1]["index"] == 4
    assert payload["live_random_walk_trace"][-1]["axis_label"] == "1"
    assert payload["live_random_walk"]["training_shared"] is False
    assert payload["plot_counts"]["new_completed"] == 0
    assert comparison_compatibility(main, control)["measurement_budgets"] == [10, 2]


@pytest.mark.parametrize(
    "method,normalization",
    [("xrd_area_fraction", "all phases"), ("gsas_ii_mass_fraction", "selected phases")],
)
def test_declared_comparisons_still_reject_incompatible_explicit_measurements(
    method, normalization
):
    main, control = campaign(), campaign("control")
    declaration = {
        **default_definition(),
        "quantification_method": "gsas_ii_mass_fraction",
        "normalization": "all phases",
        "historical_policy": "exclude",
        "decision_reason": "Use only the declared reported basis",
    }
    for data in (main, control):
        data["config"]["measurement_definition"] = deepcopy(declaration)
        for row in data["observations"]:
            row["training_included"] = False
    control["config"]["selection_policy"] = "random_control"
    invalid = measurement("3", 90)
    invalid["measurement_quality"] = {
        "quantification_method": method,
        "normalization": normalization,
    }
    control["observations"].append(invalid)
    assert comparison_compatibility(main, control)["mismatches"] == [
        "measurement_definition"
    ]
    with pytest.raises(ValueError, match="Random control"):
        plot_payload(main, random_campaign=control)
    control["observations"].pop()
    control["config"]["measurement_definition"].update(
        quantification_method=method, normalization=normalization
    )
    assert comparison_compatibility(main, control)["mismatches"] == [
        "measurement_definition"
    ]


def test_undeclared_campaigns_do_not_compare_different_reported_explicit_bases():
    main, other = campaign(), campaign("other")
    for data, method in ((main, "gsas_ii_mass_fraction"), (other, "xrd_area_fraction")):
        for row in data["observations"]:
            row["measurement_quality"] = {
                "quantification_method": method,
                "normalization": "all phases",
            }
    assert comparison_compatibility(main, other)["mismatches"] == [
        "measurement_definition"
    ]


def test_generic_minimization_and_unknown_sigma_are_preserved():
    data = campaign()
    data["config"].update(
        objective="cost", units="arbitrary", direction="minimize", bounds=[None, None]
    )
    data["observations"] = [
        dict(measurement("0", None, True), cost=10),
        dict(measurement("3", None), cost=-5),
    ]
    payload = plot_payload(data)
    assert [point["best"] for point in payload["best_trace"]] == [10, -5]
    assert [point["index"] for point in payload["best_trace"]] == [1, 2]
    assert [point["axis_label"] for point in payload["best_trace"]] == ["i1", "1"]
    assert payload["plot_objective_bounds"] == {"lower": None, "upper": None}


def test_empty_initialization_has_no_shaded_region_and_first_new_position_is_one():
    data = campaign()
    data["observations"] = []
    empty = plot_payload(data)
    assert empty["initialization_count"] == 0
    assert empty["plot_x_axis"]["initialization_end"] is None
    assert empty["plot_x_axis"]["labels"] == []
    data["observations"].append(measurement("3", 20))
    payload = plot_payload(data)
    assert payload["best_trace"][0]["index"] == 1
    assert payload["best_trace"][0]["axis_label"] == "1"
    assert payload["best_trace"][0]["optimization_step"] == 1
    assert payload["best_trace"][0]["baseline"] is False


def test_ambiguous_active_refinement_is_rejected():
    data = campaign()
    data["observations"].append(deepcopy(data["observations"][0]))
    with pytest.raises(ValueError, match="Multiple active refinements"):
        measured_points(data)
