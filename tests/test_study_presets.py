"""Versioned factory presets are explicit; saved campaign settings remain authoritative."""
from copy import deepcopy
import json
import socket

import pytest

import boicl.campaign_config as configs
from boicl.campaign import CampaignService
from boicl.campaign_config import preset_catalog, preview_preset, resolve_config
from boicl.campaign_plot import plot_payload


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    import openai

    def forbidden(*args, **kwargs):
        pytest.fail("Preset resolution and ledger checks must not call providers")

    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def first(snapshot, eligible, *_):
    return {"candidate_id": eligible[0]["candidate_id"], "score": 0.5}


def measure(service, cid, value=40):
    service.start_suggestion(cid, background=False)
    suggestion = service.get(cid)["suggestions"][-1]["suggestion_id"]
    service.reserve(cid, suggestion)
    service.measure(
        cid,
        suggestion,
        {
            "moc_wt_pct": value,
            "moc_wt_pct_sigma": 1,
            "gof": 1,
            "closure_gap_wt_pct": 0,
            "closure_gap_origin": "Explicit synthetic test measurement",
        },
        refresh=False,
    )


def test_complete_study_defaults_are_separate_from_source_and_generic():
    catalog = {row["preset"]: row for row in preset_catalog()}
    gp = resolve_config("moc_five_gp")
    llm = resolve_config("moc_five_llm")
    for preset, config, suffix in (
        ("moc_five_gp", gp, "six-variable GP"),
        ("moc_five_llm", llm, "BO-ICL LLM"),
    ):
        assert config["name"] == "MoC five-point comparison — " + suffix
        assert config["preset_version"] == catalog[preset]["version"] == "1.0.0"
        assert config["new_measurement_budget"] == 5 and config["batch_size"] == 1
        assert config["auto_suggest"] is False
        assert config["initialization"] == "confirmed_three"
        assert config["bounds"] == [0, 100] and config["seed"] == 616
        assert config["direction"] == "maximize"
        assert config["measurement_definition"]["historical_policy"] == "unresolved"
        assert preview_preset(preset)["config"] == config
    source = resolve_config("moc_gp")
    assert source["structured_gp"]["ei_after_unique_measured_designs"] == 10
    assert source["new_measurement_budget"] is None
    expected_gp = deepcopy(source["structured_gp"])
    expected_gp["ei_after_unique_measured_designs"] = 3
    assert gp["structured_gp"] == expected_gp
    assert gp["structured_gp"]["ei_xi_standardized_logit"] == 0.01
    assert (
        gp["structured_gp"]["burn_in"],
        gp["structured_gp"]["retained_draws"],
        gp["structured_gp"]["predict_thin"],
    ) == (1000, 4000, 20)
    assert llm["llm"] == resolve_config("moc_llm")["llm"]
    assert llm["llm"]["n_samples"] == 5 and llm["llm"]["min_samples"] == 2
    assert llm["llm"]["inverse_n"] == 1
    assert llm["llm"]["forward_system_message"] is None
    assert llm["llm"]["inverse_system_message"] is None
    for preset in ("generic_gp", "generic_llm", "generic_embedding_gp"):
        config = resolve_config(preset)
        assert config["bounds"] is None and config["new_measurement_budget"] is None
        assert config["objective"] == "value" and config["units"] == ""
        assert config["initialization"] == "user_mapped"
        assert config["llm"]["include_phase_context"] is False


def test_saved_overrides_and_factory_version_survive_upgrade_restart_import_checkpoint(
    tmp_path, monkeypatch
):
    service = CampaignService(tmp_path)
    cid = service.create(
        "moc_five_llm",
        overrides={
            "new_measurement_budget": 7,
            "auto_suggest": True,
            "llm": {
                "uncertainty_scalar": 0,
                "manual_inverse_target": 0,
                "forward_system_message": "Custom measured objective",
                "inverse_system_message": "",
            },
        },
    )
    saved = service.export(cid)
    checkpoint = service.save_checkpoint(cid, "Deliberate overrides")
    monkeypatch.setitem(configs.PRESET_VERSIONS, "moc_five_llm", "1.1.0")
    monkeypatch.setitem(
        configs.STUDY_SETTINGS,
        "moc_five_llm",
        {"new_measurement_budget": 9, "auto_suggest": False},
    )
    monkeypatch.setitem(configs.ENGINE_DEFAULTS["llm"], "n_samples", 9)
    assert resolve_config("moc_five_llm")["preset_version"] == "1.1.0"
    assert resolve_config("moc_five_llm")["llm"]["n_samples"] == 9
    assert (
        resolve_config("moc_five_llm", json.loads(json.dumps(saved["config"])))
        == saved["config"]
    )
    resumed = CampaignService(tmp_path)
    assert resumed.export(cid) == saved
    portable = CampaignService(tmp_path / "portable")
    imported = portable.import_bundle(json.loads(json.dumps(saved)))
    assert portable.export(imported) == saved
    fork = resumed.restore_checkpoint(cid, checkpoint["checkpoint_id"])
    assert resumed.get(fork)["config"] == saved["config"]
    assert (
        resumed.get(fork)["provenance"]["preset_at_creation"]
        == saved["provenance"]["preset_at_creation"]
    )
    preview = resumed.preview_preset("moc_five_llm", cid=cid)
    assert (
        preview["version"] == "1.1.0"
        and preview["config"]["new_measurement_budget"] == 9
    )
    assert resumed.export(cid) == saved
    resumed.apply_preset(cid, "moc_five_llm")
    applied = resumed.export(cid)
    assert applied["config"]["preset_version"] == "1.1.0"
    assert applied["config"]["llm"]["forward_system_message"] is None
    assert applied["observations"] == saved["observations"]
    assert (
        applied["provenance"]["preset_at_creation"]
        == saved["provenance"]["preset_at_creation"]
    )


def test_legacy_saved_effective_settings_are_not_claimed_as_current_factory():
    saved = resolve_config(
        "moc_llm", {"llm": {"n_samples": 3}, "new_measurement_budget": 17}
    )
    saved.pop("preset_version")
    saved.pop("preset_provenance")
    migrated = resolve_config("moc_llm", saved)
    assert migrated["preset_version"] == "legacy-unversioned"
    assert migrated["preset_provenance"]["origin"] == "legacy_saved_configuration"
    assert (
        migrated["llm"]["n_samples"] == 3 and migrated["new_measurement_budget"] == 17
    )


def test_five_pair_identical_initialization_independent_histories_and_physical_budget(
    tmp_path,
):
    service = CampaignService(tmp_path, runner=first)
    pair = service.create_pair(study="five_point")
    gp, llm = (service.get(pair[key]) for key in ("gp", "llm"))
    assert len(gp["candidates"]) == len(llm["candidates"]) == 7776
    assert gp["pool_fingerprint"] == llm["pool_fingerprint"]
    assert gp["initialization_fingerprint"] == llm["initialization_fingerprint"]
    assert gp["initial_observations"] == llm["initial_observations"]
    assert [row["moc_wt_pct"] for row in gp["observations"]] == [72.1, 83.8, 23.4]
    assert len(gp["archive"]) == 5 and all(
        not row["training_included"] for row in gp["archive"]
    )
    assert service.jobs == {} and not gp["started"] and not llm["started"]
    for cid in (pair["gp"], pair["llm"]):
        for _ in range(5):
            measure(service, cid)
        assert service.completed(service.get(cid)) == 5
        with pytest.raises(ValueError, match="budget is filled"):
            service.start_suggestion(cid, background=False)
        if cid == pair["gp"]:
            assert service.get(pair["llm"])["observations"] == llm["observations"]
            assert service.get(pair["llm"])["suggestions"] == []
    latest = service.get(pair["gp"])["observations"][-1]
    service.refine(
        pair["gp"],
        latest["observation_id"],
        {"moc_wt_pct": 41},
        "Reanalyzed the same synthesis",
    )
    service.revise_measurement_definition(
        pair["gp"],
        {
            "quantification_method": "gsas_ii_mass_fraction",
            "normalization": "all phases",
        },
        "exclude",
        "Different scientific definition excludes historical reported measurements",
    )
    assert service.completed(service.get(pair["gp"])) == 5
    assert plot_payload(service.get(pair["gp"]))["plot_counts"]["new_completed"] == 5


def test_comparison_reports_current_settings_and_distinct_creation_snapshot(tmp_path):
    service = CampaignService(tmp_path)
    pair = service.create_pair(study="five_point")
    created = deepcopy(pair["manifest"])
    assert created["settings_basis"] == "creation_time_snapshot"
    assert "below 3" in created["method_difference"]
    assert "standardized padded-logit units" in created["method_difference"]
    assert "raw objective units" in created["method_difference"]
    for cid in (pair["gp"], pair["llm"]):
        service.update_config(cid, {"new_measurement_budget": 6})
    service.update_config(
        pair["gp"], {"structured_gp": {"ei_after_unique_measured_designs": 4}}
    )
    current = service.comparison(pair["gp"], pair["llm"])
    assert current["settings_basis"] == "current_resolved_configuration"
    assert (
        "below 4" in current["method_difference"]
        and "below 10" not in current["method_difference"]
    )
    assert current["shared"]["new_measurement_budget"] == 6
    for cid in (pair["gp"], pair["llm"]):
        assert current["methods"][cid]["preset_version"] == "1.0.0"
        assert (
            current["creation_snapshots"][cid]["config"]["new_measurement_budget"] == 5
        )
        exported = service.export(cid)
        assert exported["resolved_method"]["new_measurement_budget"] == 6
        assert exported["provenance"]["matched_pair_at_creation"] == created
    source = service.create_pair()
    assert service.get(source["gp"])["config"]["preset"] == "moc_gp"
    assert "below 10" in source["manifest"]["method_difference"]


def test_explicit_apply_preserves_ledger_decisions_and_pending_reservations(tmp_path):
    service = CampaignService(tmp_path, runner=first)
    cid = service.create("moc_llm", overrides={"auto_suggest": False})
    measure(service, cid)
    service.revise_measurement_definition(
        cid,
        {
            "quantification_method": "gsas_ii_mass_fraction",
            "normalization": "all phases",
        },
        "retain_with_justification",
        "Operator explicitly retains the unknown historical methods",
    )
    service.start_suggestion(cid, background=False)
    suggestion = service.get(cid)["suggestions"][-1]["suggestion_id"]
    service.reserve(cid, suggestion)
    before = service.export(cid)
    preview = service.preview_preset("moc_five_gp", cid=cid)
    assert preview["config"]["structured_gp"]["ei_after_unique_measured_designs"] == 3
    assert service.export(cid) == before
    service.apply_preset(cid, "moc_five_gp")
    after = service.export(cid)
    for key in (
        "observations",
        "initial_observations",
        "archive",
        "candidates",
        "initialization_fingerprint",
        "suggestions",
    ):
        assert after[key] == before[key]
    assert (
        after["config"]["measurement_definition"]
        == before["config"]["measurement_definition"]
    )
    assert after["config"]["new_measurement_budget"] == 5
    assert after["config"]["auto_suggest"] is False
    for preset in ("moc_eight", "generic_gp"):
        with pytest.raises(ValueError, match="initialization|generic preset"):
            service.apply_preset(cid, preset)
        assert service.export(cid) == after
    with pytest.raises(ValueError, match="budget"):
        service.apply_preset(cid, "moc_five_gp", {"new_measurement_budget": 1})
    assert service.export(cid) == after


def test_generic_apply_preserves_mapping_bounds_units_and_custom_prompts(tmp_path):
    service = CampaignService(tmp_path)
    cid = service.create_generic(
        [
            {
                "candidate_id": f"c{i}",
                "temperature": i,
                "gas": "N2" if i % 2 else "H2",
                "loss": i if i < 2 else None,
            }
            for i in range(5)
        ],
        [{"column": "temperature"}, {"column": "gas", "transform": "categorical"}],
        objective="loss",
        units="mg",
        direction="minimize",
        bounds=[-10, 10],
        overrides={
            "llm": {
                "forward_system_message": "My unrelated objective",
                "inverse_system_message": "",
            }
        },
    )
    before = service.export(cid)
    service.apply_preset(cid, "generic_llm")
    after = service.export(cid)
    for key in ("objective", "units", "direction", "bounds"):
        assert after["config"][key] == before["config"][key]
    assert (
        after["config"]["structured_gp"]["feature_spec"]
        == before["config"]["structured_gp"]["feature_spec"]
    )
    assert after["config"]["llm"]["forward_system_message"] == "My unrelated objective"
    assert after["config"]["llm"]["inverse_system_message"] == ""
    assert after["config"]["new_measurement_budget"] is None
    assert after["observations"] == before["observations"]
    with pytest.raises(ValueError, match="MoC preset"):
        service.apply_preset(cid, "moc_five_llm")
    assert service.export(cid) == after
