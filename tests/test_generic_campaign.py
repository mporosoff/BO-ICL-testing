"""Reusable shared workflow with non-MoC objectives, both directions and controls."""
from copy import deepcopy
import json
import socket

import numpy as np
import pytest
from scipy.stats import norm

from boicl.campaign import CampaignService
from boicl.campaign_config import resolve_config
from boicl.generic_import import (
    FeatureTransform,
    load_generic_package,
    resolve_features,
)
from boicl.structured_gp import (
    PaddedLogitTransform,
    StandardOutcomeTransform,
    StructuredGP,
    bounded_mixture_summary,
    bounded_expected_improvement,
    log_prior,
)


def records():
    return [
        {
            "candidate_id": f"row-{i}",
            "temperature": 10 + 10 * i,
            "time": 2**i,
            "gas": "A" if i % 2 else "B",
            "constant": 7,
            "energy": v,
            "uncertainty": 0.2,
            "gof": 200,
            "closure_gap": -100,
        }
        for i, v in enumerate([-3.0, 1.0, None, None, None, None])
    ]


def feature_spec():
    return [
        {"column": "temperature", "transform": "linear"},
        {"column": "time", "transform": "log2"},
        {"column": "gas", "transform": "categorical"},
        {"column": "constant", "transform": "linear"},
    ]


def overrides():
    return {
        "auto_suggest": False,
        "structured_gp": {
            "burn_in": 15,
            "retained_draws": 40,
            "predict_thin": 4,
            "default_observation_sigma": 0.2,
        },
    }


def generic(service, direction="maximize", bounds=(-5, 5), **extra):
    return service.create_generic(
        records(),
        feature_spec(),
        objective="energy",
        direction=direction,
        bounds=bounds,
        units="eV",
        sigma_column="uncertainty",
        overrides=overrides(),
        **extra,
    )


def suggest(service, cid):
    result = service.start_suggestion(cid, background=False)
    assert result["status"] == "suggested", service.summary(cid)["progress"]
    return service.get(cid)["suggestions"][-1]


def test_mapping_excludes_outcomes_and_quality_and_uses_full_space():
    package = load_generic_package(
        records(),
        feature_spec(),
        objective="energy",
        bounds=[-5, 5],
        sigma_column="uncertainty",
    )
    spec = package["provenance"]["feature_spec"]
    matrix = FeatureTransform(spec).transform(package["candidates"])
    assert matrix.shape == (6, 5)
    np.testing.assert_allclose(matrix[:, 0], np.linspace(0, 1, 6))
    np.testing.assert_allclose(matrix[:, 1], np.linspace(0, 1, 6))
    np.testing.assert_array_equal(matrix[:, -1], 0)
    assert spec[0]["bounds"] == [10, 60]  # not the two initial observations' range
    assert package["observations"][0]["objective_sigma"] == 0.2
    assert (
        package["observations"][0]["gof"] == 200
        and package["observations"][0]["closure_gap"] == -100
    )
    assert all(
        not set(row) & {"energy", "value", "uncertainty", "gof", "closure_gap"}
        for row in package["candidates"]
    )
    changed = records()
    changed[0]["energy"] = 4
    changed[0]["gof"] = 1
    again = load_generic_package(
        changed,
        feature_spec(),
        objective="energy",
        bounds=[-5, 5],
        sigma_column="uncertainty",
    )
    assert (
        again["candidates"] == package["candidates"]
        and again["pool_fingerprint"] == package["pool_fingerprint"]
    )


@pytest.mark.parametrize(
    "spec",
    [
        [{"column": "energy"}],
        [{"column": "gof"}],
        [{"column": "objective_sigma"}],
        [{"column": "temperature", "bounds": [20, 50]}],
        [{"column": "temperature", "transform": "unsupported"}],
        [{"column": "gas", "transform": "categorical", "values": ["A"]}],
    ],
)
def test_invalid_or_leaky_feature_specs_rejected(spec):
    with pytest.raises(ValueError):
        load_generic_package(records(), spec, objective="energy")


def test_custom_uncertainty_mapping_cannot_be_a_feature():
    with pytest.raises(ValueError, match="measurement quality"):
        load_generic_package(
            records(),
            [{"column": "uncertainty"}],
            objective="energy",
            sigma_column="uncertainty",
        )


@pytest.mark.parametrize("direction,best", [("maximize", 1), ("minimize", -3)])
def test_generic_shared_full_lifecycle_both_directions(
    tmp_path, monkeypatch, direction, best
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: pytest.fail("Generic structured GP accessed the network"),
    )
    service = CampaignService(tmp_path)
    cid = generic(service, direction)
    initial = service.summary(cid)
    assert initial["best"] == best and initial["counts"]["new_measurements"] == 0
    assert (
        initial["config"]["objective"] == "energy"
        and initial["config"]["units"] == "eV"
    )
    step = suggest(service, cid)
    assert step["engine_result"]["stage"] == "expected_improvement"
    assert (
        -5
        <= step["prediction"]["lower95"]
        <= step["prediction"]["mean"]
        <= step["prediction"]["upper95"]
        <= 5
    )
    service.reserve(cid, step["suggestion_id"])
    value = -4 if direction == "minimize" else 4
    measured = service.measure(
        cid,
        step["suggestion_id"],
        {"value": value, "objective_sigma": 0},
        refresh=False,
    )
    assert measured["best"] == value and measured["counts"]["new_measurements"] == 1
    assert measured["observations"][-1]["objective_sigma"] == 0
    assert "moc_wt_pct" not in measured["observations"][-1]
    next_step = suggest(service, cid)
    assert next_step["candidate_id"] != step["candidate_id"]
    service.reserve(cid, next_step["suggestion_id"])
    bundle = json.loads(json.dumps(service.export(cid)))
    restored = CampaignService(tmp_path)
    assert (
        restored.export(cid) == bundle
        and restored.summary(cid)["counts"]["pending"] == 1
    )
    assert (
        restored.replay(cid, step["suggestion_id"])["candidate_id"]
        == step["candidate_id"]
    )
    assert restored.list()[0]["data_schema"] == "generic"


def test_generic_gp_uses_actual_dimension_and_explicit_noise_not_moc_metadata(tmp_path):
    service = CampaignService(tmp_path)
    cid = generic(service)
    state = service.get(cid)
    settings = {
        **state["config"]["structured_gp"],
        "objective_field": "value",
        "objective_bounds": [-5, 5],
    }
    gp = StructuredGP(settings).fit(state["candidates"], state["observations"])
    assert gp.dimension == 5 and gp.samples.shape == (40, 6)
    np.testing.assert_array_equal(gp.diagnostics["effective_sigmas"], [0.2, 0.2])
    assert (
        "effective_sigmas_pp" not in gp.diagnostics
        and gp.diagnostics["method_version"] == "structured-gp-generic-v1"
    )
    assert gp.diagnostics["noise_policy"] == "reported_or_fixed"
    assert log_prior(np.r_[np.full(5, np.log(np.sqrt(5))), 0]) == 0
    assert state["config"]["structured_gp"]["logit_delta_pp"] == pytest.approx(0.065)


def test_generic_negative_bounds_padded_transform_and_mixture_moments():
    transform = PaddedLogitTransform(0.065, lower=-5, upper=5).fit([-3, 1])
    y = np.asarray([-5, -3, 0, 1, 5])
    np.testing.assert_allclose(transform.inverse(transform.forward(y)), y, atol=1e-12)
    derivative = (10 + 0.13) / ((y + 5 + 0.065) * (5 + 0.065 - y))
    np.testing.assert_allclose(
        transform.variance(y, 0.2), (0.2 * derivative / transform.scale) ** 2
    )
    summary = bounded_mixture_summary([-100, 100], [0, 0], transform)
    assert summary["mean"][0] == pytest.approx(0)
    assert summary["variance"][0] == pytest.approx(25)
    assert summary["lower95"][0] == pytest.approx(-5) and summary["upper95"][
        0
    ] == pytest.approx(5)


def test_unbounded_generic_objective_has_no_invented_physical_limits(tmp_path):
    service = CampaignService(tmp_path)
    cid = generic(service, bounds=None, direction="minimize")
    step = suggest(service, cid)
    assert (
        step["engine_result"]["diagnostics"]["outcome_transform"]["physical_bounds"]
        is None
    )
    assert (
        step["prediction"]["uncertainty_type"]
        == "latent-function unbounded posterior mixture"
    )
    service.reserve(cid, step["suggestion_id"])
    service.measure(cid, step["suggestion_id"], {"value": -1234}, refresh=False)
    assert service.summary(cid)["best"] == -1234
    transform = StandardOutcomeTransform().fit([10, 14])
    means = np.array([-1.0, 2.0])
    variances = np.array([0.25, 1.0])
    summary = bounded_mixture_summary(means, variances, transform)
    assert summary["mean"][0] == pytest.approx(transform.inverse(0.5))
    assert summary["variance"][0] == pytest.approx(
        (variances.mean() + means.var()) * transform.scale**2
    )
    actual = bounded_expected_improvement(
        means, variances, 0, [-np.inf, np.inf], xi=0, direction="minimize"
    )
    expected = (-means) * norm.cdf(-means / np.sqrt(variances)) + np.sqrt(
        variances
    ) * norm.pdf(means / np.sqrt(variances))
    np.testing.assert_allclose(actual, expected)


def test_random_control_same_initial_state_independent_history_and_reproducible_seed(
    tmp_path,
):
    service = CampaignService(tmp_path)
    cid = generic(service)
    model_step = suggest(service, cid)
    service.reserve(cid, model_step["suggestion_id"])
    service.measure(cid, model_step["suggestion_id"], {"value": 3}, refresh=False)
    control = service.create_control(cid)
    assert service.create_control(cid) == control
    control_state = service.get(control)
    assert control_state["observations"] == service.get(cid)["initial_observations"]
    assert control_state["config"]["comparison_parent_id"] == cid
    step = suggest(service, control)
    assert (
        step["prediction"] is None
        and step["engine_result"]["stage"] == "random_control"
    )
    expected = service.eligible(control_state)[np.random.default_rng(616).integers(4)][
        "candidate_id"
    ]
    assert step["candidate_id"] == expected
    service.reserve(control, step["suggestion_id"])
    service.measure(control, step["suggestion_id"], {"value": -4}, refresh=False)
    assert service.summary(cid)["best"] == 3 and service.summary(control)["best"] == 1
    new = service.create_control(cid, reset=True)
    assert new != control and service.summary(new)["counts"]["new_measurements"] == 0
    assert len(service.list()) == 3


def test_generic_config_updates_revalidate_bounds_feature_mapping_and_objective(
    tmp_path,
):
    service = CampaignService(tmp_path)
    cid = generic(service)
    before = service.export(cid)
    for changes in (
        {"bounds": [0, 2]},
        {"objective": "another_label"},
        {
            "structured_gp": {
                "feature_spec": [
                    {
                        "column": "temperature",
                        "transform": "linear",
                        "bounds": [20, 50],
                        "units": "",
                    }
                ]
            }
        },
    ):
        with pytest.raises(ValueError):
            service.update_config(cid, changes)
        assert service.export(cid) == before
    service.update_config(cid, {"direction": "minimize"})
    assert service.summary(cid)["best"] == -3
    assert service.get(cid)["config"]["llm"]["maximize"] is False


def test_generic_text_only_data_no_chemistry_defaults_and_no_hidden_labels():
    rows = [
        {"text": "Recipe one", "label": -20},
        {"text": "Recipe two", "label": 50},
        {"text": "Recipe three", "label": None},
    ]
    package = load_generic_package(rows, [], objective="label", procedure_column="text")
    assert [r["value"] for r in package["observations"]] == [-20, 50]
    assert all("label" not in c and "value" not in c for c in package["candidates"])
    config = resolve_config(
        "generic_llm",
        {
            "direction": "minimize",
            "bounds": [-100, 200],
            "objective": "loss",
            "units": "J",
        },
    )
    assert (
        config["llm"]["objective_bounds"] == [-100, 200]
        and config["llm"]["maximize"] is False
    )
    assert config["structured_gp"]["noise_policy"] == "reported_or_fixed"
    with pytest.raises(ValueError):
        resolve_config("moc_gp", {"bounds": [-100, 200], "direction": "minimize"})


def test_separate_observation_mapping_preserves_sigma_and_physical_measurement_identity(
    tmp_path,
):
    source = records()
    observed = [
        {
            "candidate_id": "row-0",
            "energy": -2,
            "uncertainty": 0.4,
            "observation_id": "v1",
            "physical_measurement_id": "lab-run-1",
        },
        {
            "candidate_id": "row-1",
            "energy": 1,
            "uncertainty": 0.6,
            "observation_id": "v2",
            "physical_measurement_id": "lab-run-2",
        },
    ]
    package = load_generic_package(
        source,
        feature_spec(),
        objective="energy",
        sigma_column="uncertainty",
        observations=observed,
    )
    assert [r["objective_sigma"] for r in package["observations"]] == [0.4, 0.6]
    assert package["observations"][0]["physical_measurement_id"] == "lab-run-1"
    observed[1]["physical_measurement_id"] = "lab-run-1"
    with pytest.raises(ValueError, match="Multiple active refinements"):
        CampaignService(tmp_path).create_generic(
            source, feature_spec(), objective="energy", observations=observed
        )


def test_generic_identity_preserves_different_protocols_even_with_same_mapped_features():
    rows = [
        {"temperature": 20, "procedure": "Use catalyst A", "energy": None},
        {"temperature": 20, "procedure": "Use catalyst B", "energy": None},
    ]
    package = load_generic_package(
        rows, [{"column": "temperature"}], objective="energy"
    )
    assert len({r["candidate_id"] for r in package["candidates"]}) == 2
    np.testing.assert_array_equal(
        FeatureTransform(package["provenance"]["feature_spec"]).transform(
            package["candidates"]
        ),
        [[0], [0]],
    )


def test_noop_settings_save_preserves_current_suggestion_revision(tmp_path):
    service = CampaignService(tmp_path)
    cid = generic(service)
    step = suggest(service, cid)
    before = service.export(cid)
    service.update_config(
        cid, {"structured_gp": {"retained_draws": 40}, "direction": "maximize"}
    )
    assert service.export(cid) == before
    service.reserve(cid, step["suggestion_id"])
    assert service.summary(cid)["counts"]["pending"] == 1


def test_gap_alias_preserves_zero_and_provenance_across_saved_campaign(tmp_path):
    rows = records()
    for row in rows:
        row.pop("closure_gap")
    rows[0]["gap"] = 0
    rows[1]["gap"] = -2.5
    rows[1]["closure_gap_origin"] = "reported total phase deficit"
    service = CampaignService(tmp_path)
    cid = service.create_generic(rows, feature_spec(), objective="energy")
    state = service.get(cid)
    assert [row["closure_gap"] for row in state["observations"]] == [0, -2.5]
    assert "gap field" in state["observations"][0]["closure_gap_origin"]
    assert (
        state["observations"][1]["closure_gap_origin"] == "reported total phase deficit"
    )
    assert state["provenance"]["quality_aliases"] == {"gap": "closure_gap"}
    assert all(
        "gap" not in row and "closure_gap" not in row for row in state["candidates"]
    )
    assert CampaignService(tmp_path).get(cid)["observations"] == state["observations"]


def test_gap_alias_separate_observations_conflicts_and_missing_values():
    observed = [
        {"candidate_id": "row-0", "energy": -2, "gap": "0", "closure_gap": 0},
        {"candidate_id": "row-1", "energy": 1},
    ]
    package = load_generic_package(
        records(), feature_spec(), objective="energy", observations=observed
    )
    assert package["observations"][0]["closure_gap"] == 0
    assert "closure_gap" not in package["observations"][1]
    assert package["provenance"]["quality_aliases"] == {"gap": "closure_gap"}
    observed[0]["closure_gap"] = 1
    with pytest.raises(ValueError, match="conflicting"):
        load_generic_package(
            records(), feature_spec(), objective="energy", observations=observed
        )


def test_gap_quality_cannot_be_used_as_a_feature_or_procedure():
    rows = records()
    for row in rows:
        row["gap"] = 0
    with pytest.raises(ValueError, match="measurement-quality"):
        load_generic_package(rows, [{"column": "gap"}], objective="energy")
    with pytest.raises(ValueError, match="quality columns"):
        load_generic_package(
            rows, feature_spec(), objective="energy", procedure_column="gap"
        )
