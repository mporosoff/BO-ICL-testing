"""Defaults contract across user entry points; no provider or numerical fitting."""
from copy import deepcopy
import inspect

import pytest

from boicl.campaign_config import resolve_config
from boicl.local_app import _merged_config
from boicl.toolkit_bridge import config_changes, legacy_config


LLM_DEFAULTS = {
    "forward_model": "gpt-4o",
    "inverse_model": "gpt-4o",
    "forward_temperature": 0.7,
    "inverse_temperature": 0.7,
    "forward_max_tokens": 256,
    "inverse_max_tokens": 576,
    "n_samples": 5,
    "min_samples": 2,
    "inverse_n": 1,
    "selector_mode": "nearest",
    "selector_k": 5,
    "fetch_k": 100,
    "shortlist_size": 16,
    "mmr_lambda": 0.5,
    "embedding_model": "text-embedding-3-large",
    "selector_embedding_model": "text-embedding-3-large",
    "acquisition": "expected_improvement",
    "xi": 0,
    "uncertainty_scalar": 1,
    "random_addons": 0,
    "ucb_lambda": 0.5,
    "inverse_multiplier": 1.2,
    "inverse_jitter": 0.05,
    "target_floor": None,
    "target_ceiling": None,
}


@pytest.mark.parametrize(
    "preset", ["moc_llm", "moc_gp", "moc_embedding_gp", "generic_llm", "generic_gp"]
)
def test_shared_defaults_match_corrected_crystal_numerics(preset):
    config = resolve_config(preset)
    assert {key: config["llm"][key] for key in LLM_DEFAULTS} == LLM_DEFAULTS
    assert config["api"] == {
        "maximum_attempts": 8,
        "request_spacing_s": 0.5,
        "base_cooldown_s": 10,
    }
    assert config["embedding_gp"] == {
        "embedding_model": "text-embedding-ada-002",
        "dimensions": 32,
        "neighbors": 5,
        "seed": 616,
    }
    assert config["seed"] == 616 and config["batch_size"] == 1
    assert config["new_measurement_budget"] is None
    assert config["measurements_per_candidate"] == 1
    assert config["independent_campaign_replicates"] == 1
    assert config["auto_suggest"] is True
    assert {
        key: config["structured_gp"][key]
        for key in ("burn_in", "retained_draws", "predict_thin", "proposal_step")
    } == {
        "burn_in": 1000,
        "retained_draws": 4000,
        "predict_thin": 20,
        "proposal_step": 0.3,
    }
    if preset.startswith("generic_"):
        assert config["bounds"] is None and config["units"] == ""
        assert config["llm"]["include_phase_context"] is False
        assert config["llm"]["reference_scale"] == 1
        assert config["structured_gp"]["feature_spec"] == []
        assert config["structured_gp"]["noise_policy"] == "reported_or_fixed"
    else:
        assert config["objective"] == "moc_wt_pct" and config["bounds"] == [0, 100]
        assert config["llm"]["reference_scale"] == 100
        assert config["structured_gp"]["ei_after_unique_measured_designs"] == 10


def test_shared_main_adapter_round_trip_keeps_defaults_and_valid_zeros():
    original = resolve_config(
        "generic_llm",
        {
            "new_measurement_budget": 0,
            "seed": 0,
            "api": {"request_spacing_s": 0, "base_cooldown_s": 0},
            "llm": {
                "uncertainty_scalar": 0,
                "forward_temperature": 0,
                "inverse_temperature": 0,
                "inverse_jitter": 0,
                "target_floor": 0,
                "ucb_lambda": 0,
            },
        },
    )
    before = deepcopy(original)
    visible = legacy_config(original)
    visible["selector_mode"] = original["llm"]["selector_mode"]
    changes = config_changes(visible, original)
    merged = deepcopy(original)
    for key, value in changes.items():
        if isinstance(value, dict):
            merged[key].update(value)
        else:
            merged[key] = value
    result = resolve_config("generic_llm", merged)
    assert result == original == before


def test_fresh_legacy_defaults_and_saved_explicit_settings_remain_distinct():
    fresh = _merged_config()
    assert {
        key: fresh[key]
        for key in (
            "optimizer",
            "acquisition",
            "embedding_model",
            "llm_samples",
            "llm_prediction_temperature",
            "llm_inverse_temperature",
            "selector_k",
            "inverse_design_count",
            "ucb_lambda",
            "n_components",
        )
    } == {
        "optimizer": "llm",
        "acquisition": "expected_improvement",
        "embedding_model": "text-embedding-3-large",
        "llm_samples": 5,
        "llm_prediction_temperature": 0.7,
        "llm_inverse_temperature": 0.7,
        "selector_k": 5,
        "inverse_design_count": 1,
        "ucb_lambda": 0.5,
        "n_components": 32,
    }
    old = {
        "optimizer": "gpr",
        "embedding_model": "text-embedding-ada-002",
        "acquisition": "upper_confidence_bound",
        "llm_samples": 3,
        "llm_prediction_temperature": 0.1,
        "llm_inverse_temperature": 0.05,
        "selector_k": 0,
        "ucb_lambda": 0,
        "inverse_target_jitter": 0,
        "iterations_per_trial": 0,
        "prediction_system_message": "User-written prompt",
        "inverse_system_message": "",
    }
    assert {key: _merged_config(old)[key] for key in old} == old
    assert (
        _merged_config({"optimizer": "gpr"})["embedding_model"]
        == "text-embedding-ada-002"
    )


def test_library_defaults_preserve_explicit_generic_formatters():
    from boicl.asktell import (
        AskTellFewShot,
        AskTellFewShotTopk,
        LabelSimilarityExampleSelector,
    )
    from boicl.pool import Pool
    from boicl.llm_model import get_llm

    model = AskTellFewShotTopk()
    assert model._model == "gpt-4o" and model._inverse_model == "gpt-4o"
    assert model._selector_k == 5 and model._k == 5
    assert model.embedding_model == "text-embedding-3-large"
    assert model.use_logprobs is False and model.llm is None
    assert inspect.signature(get_llm).parameters["temperature"].default == 0.7
    assert inspect.signature(get_llm).parameters["top_p"].default is None
    assert (
        inspect.signature(LabelSimilarityExampleSelector.from_examples)
        .parameters["k"]
        .default
        == 5
    )
    pool = Pool(["recipe"])
    assert pool.embedding_model == "text-embedding-3-large"
    assert pool.format("recipe") == "recipe"
    custom = lambda text: "custom: " + text
    assert Pool(["recipe"], formatter=custom).format("recipe") == "custom: recipe"
    assert AskTellFewShot(x_formatter=custom).format_x("recipe") == "custom: recipe"


def test_cli_creation_uses_shared_resolver_without_numerical_overrides(
    monkeypatch, capsys
):
    import boicl.moc_cli as cli

    calls = []

    class Service:
        def __init__(self, directory):
            pass

        def create(self, preset):
            calls.append(resolve_config(preset))
            return "created"

        def create_pair(self):
            calls.extend([resolve_config("moc_gp"), resolve_config("moc_llm")])
            return {"gp": "gp", "llm": "llm"}

    monkeypatch.setattr(cli, "CampaignService", Service)
    assert cli.main(["init"]) == 0
    assert calls.pop()["engine"] == "gpr_features"
    assert cli.main(["init", "--preset", "moc_llm"]) == 0
    assert calls.pop() == resolve_config("moc_llm")
    assert cli.main(["init", "--pair"]) == 0
    assert [call["engine"] for call in calls] == ["gpr_features", "llm"]
    assert calls[0]["llm"] == calls[1]["llm"]
    capsys.readouterr()
