"""Audit regressions: exact previews, isolated proposal jobs, scientific definitions and Windows persistence."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import threading

import pytest

from boicl.campaign import CampaignService, fingerprint
from boicl.campaign_config import resolve_config
from boicl.campaign_plot import comparison_compatibility
from boicl.moc_import import load_moc_package
from boicl.persistence import atomic_bytes, io_path


def first(snapshot, eligible, *_):
    return {"candidate_id": eligible[0]["candidate_id"], "score": 1}


def make(service, **kwargs):
    return service.create_generic(
        [
            {
                "candidate_id": f"c{i}",
                "procedure": f"Synthetic recipe {i}",
                "x": i,
                "value": [-2, 1, None, None, None][i],
            }
            for i in range(5)
        ],
        [{"column": "x"}],
        preset="generic_llm",
        bounds=[-5, 5],
        overrides={"auto_suggest": False, **kwargs.pop("overrides", {})},
        **kwargs,
    )


def reserve(service, cid):
    assert service.start_suggestion(cid, background=False)["status"] == "suggested"
    sid = service.get(cid)["suggestions"][-1]["suggestion_id"]
    service.reserve(cid, sid)
    return sid


def file_snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in io_path(root).rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("target", [None, 0, -4, 5])
def test_manual_target_generic_roundtrip_and_invalid_bounds(target):
    cfg = resolve_config(
        "generic_llm", {"bounds": [-5, 5], "llm": {"manual_inverse_target": target}}
    )
    assert resolve_config("generic_llm", json.loads(json.dumps(cfg))) == cfg
    for bad in (True, float("nan"), -6, 6):
        with pytest.raises(ValueError, match="target"):
            resolve_config(
                "generic_llm",
                {"bounds": [-5, 5], "llm": {"manual_inverse_target": bad}},
            )
    with pytest.raises(ValueError, match="target bounds"):
        resolve_config(
            "moc_llm", {"llm": {"manual_inverse_target": 0, "target_floor": 1}}
        )


def test_no_call_preview_matches_demo_requests_and_proposals_are_separate(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("No network allowed")
    )
    service = CampaignService(tmp_path)
    cid = make(
        service,
        synthetic_demo=True,
        overrides={"llm": {"manual_inverse_target": 0, "inverse_proposal_count": 3}},
    )
    before, files = service.export(cid), file_snapshot(service.root)
    preview = service.request_preview(cid, role="inverse")
    assert preview["status"] == "exact" and preview["request"]["n"] == 3
    assert preview["target"]["resolved_target"] == 0
    assert service.export(cid) == before and file_snapshot(service.root) == files
    response = service.start_inverse_proposals(cid, background=False)
    assert response["status"] == "proposed"
    after = service.export(cid)
    record = after["inverse_proposals"][-1]
    assert (
        len(record["procedures"])
        == record["requested_count"]
        == record["returned_count"]
        == 3
    )
    assert preview["request"] == record["engine_result"]["request_log"][0]["request"]
    assert {k: v for k, v in after.items() if k != "inverse_proposals"} == {
        k: v for k, v in before.items() if k != "inverse_proposals"
    }
    assert not (service.root / "embeddings").exists()
    assert service.summary(cid)["counts"]["inverse_proposals"] == 1
    forward = service.request_preview(cid, candidate_id="c2")
    service.start_suggestion(cid, background=False)
    suggestion = service.get(cid)["suggestions"][-1]
    logged = next(
        r["request"]
        for r in suggestion["engine_result"]["request_log"]
        if r["role"] == "forward" and r["candidate_id"] == "c2"
    )
    assert logged == forward["request"]
    service.update_config(cid, {"llm": {"inverse_system_message": "Changed now"}})
    recorded = service.request_preview(
        cid, role="inverse", suggestion_id=record["proposal_id"]
    )
    assert (
        recorded["source"] == "recorded" and recorded["request"] == preview["request"]
    )
    assert CampaignService(tmp_path).export(cid) == service.export(cid)


def test_automatic_inverse_preview_draw_matches_next_standalone_request(tmp_path):
    service = CampaignService(tmp_path)
    cid = make(service, synthetic_demo=True)
    for _ in range(2):
        expected = service.request_preview(cid, role="inverse")
        service.start_inverse_proposals(cid, background=False)
        record = service.get(cid)["inverse_proposals"][-1]
        assert expected["target"] == record["target"]
        assert (
            expected["request"] == record["engine_result"]["request_log"][0]["request"]
        )


def test_missing_cache_preview_is_unresolved_and_read_only(tmp_path, monkeypatch):
    from boicl.embedding_cache import EmbeddingCache
    import openai

    monkeypatch.setattr(
        openai, "OpenAI", lambda **k: pytest.fail("Preview created a provider client")
    )
    monkeypatch.setattr(
        EmbeddingCache,
        "prepare",
        lambda *a, **k: pytest.fail("Preview generated embeddings"),
    )
    service = CampaignService(tmp_path)
    cid = make(service)
    before, files = service.export(cid), file_snapshot(service.root)
    result = service.request_preview(cid, candidate_id="c2")
    assert result["status"] == "unresolved" and result["request"] is None
    assert "cached" in result["reason"]
    assert service.export(cid) == before and file_snapshot(service.root) == files
    service.update_config(cid, {"llm": {"selector_mode": "all"}})
    assert service.request_preview(cid, candidate_id="c2")["status"] == "exact"


def test_inverse_cancellation_staleness_and_save_failure_do_not_publish_experiments(
    tmp_path, monkeypatch
):
    from boicl import campaign

    entered, release = threading.Event(), threading.Event()

    def slow(snapshot, count, cancel, progress):
        entered.set()
        assert release.wait(10)
        return {
            "status": "proposed",
            "procedures": ["A synthetic draft"] * count,
            "returned_count": count,
        }

    service = CampaignService(tmp_path, inverse_runner=slow)
    cid = make(service)
    before = service.export(cid)
    service.start_inverse_proposals(cid)
    assert entered.wait(5)
    assert service.start_suggestion(cid)["coalesced"]
    service.update_config(cid, {"name": "Changed during request"})
    release.set()
    service.jobs[cid]["thread"].join(10)
    assert service.get(cid)["inverse_proposals"][-1]["status"] == "cancelled"
    assert service.get(cid)["observations"] == before["observations"]
    assert service.get(cid)["suggestions"] == []
    original = campaign.atomic_json
    calls = 0

    def fail_second_live(path, data):
        nonlocal calls
        if Path(path).name == f"campaign-{cid}.json":
            calls += 1
            if calls == 2:
                raise OSError("simulated proposal result save failure")
        return original(path, data)

    monkeypatch.setattr(campaign, "atomic_json", fail_second_live)
    assert service.start_inverse_proposals(cid, background=False)["status"] == "failed"
    assert service.get(cid)["inverse_proposals"][-1]["status"] == "running"
    monkeypatch.setattr(campaign, "atomic_json", original)
    restarted = CampaignService(tmp_path)
    assert restarted.get(cid)["inverse_proposals"][-1]["status"] == "interrupted"
    assert restarted.jobs == {} and restarted.get(cid)["suggestions"] == []


def test_inverse_import_ownership_type_and_request_count_validation(tmp_path):
    service = CampaignService(tmp_path)
    cid = make(service, synthetic_demo=True)
    service.start_inverse_proposals(cid, background=False)
    original = service.export(cid)
    for change in (
        lambda d: d.update(inverse_proposals=None),
        lambda d: d["inverse_proposals"][0].update(candidate_id="c2"),
        lambda d: d["inverse_proposals"][0].update(proposal_id="../escape"),
        lambda d: d["inverse_proposals"][0].update(procedures=[" "]),
        lambda d: d["inverse_proposals"][0].update(returned_count=0),
        lambda d: d["inverse_proposals"][0]["engine_result"]["request_log"][0][
            "request"
        ].update(n=10),
    ):
        altered = deepcopy(original)
        change(altered)
        with pytest.raises(ValueError):
            service.import_bundle(altered)
    service.update_config(cid, {"engine": "gpr_features"})
    with pytest.raises(ValueError, match="LLM engine"):
        service.start_inverse_proposals(cid)


def explicit(
    method="gsas_ii_mass_fraction", normalization="total refined crystalline mass"
):
    return dict(
        quantification_method=method,
        normalization=normalization,
        source_file="refinement.gpx",
        source_identifier="scan-A",
        refinement_id="revision-2",
        uncertainty_method="reported least-squares esd",
        definition_note="operator-reported method",
    )


def test_unknown_historical_requires_documented_decision_and_incompatible_values_stay_preserved(
    tmp_path,
):
    service = CampaignService(tmp_path, runner=first)
    cid = make(service)
    initial = service.export(cid)
    sid = reserve(service, cid)
    with pytest.raises(ValueError, match="scientific decision"):
        service.measure(cid, sid, {"value": 2, **explicit()}, refresh=False)
    assert service.summary(cid)["counts"]["pending"] == 1
    definition = {
        key: explicit()[key] for key in ("quantification_method", "normalization")
    }
    with pytest.raises(ValueError, match="scientific reason"):
        service.revise_measurement_definition(cid, definition)
    service.revise_measurement_definition(
        cid,
        definition,
        "retain_with_justification",
        "Provisional comparison approved for exploration; historical method still requires validation",
    )
    service.measure(cid, sid, {"value": 2, **explicit()}, refresh=False)
    state = service.export(cid)
    assert state["observations"][:2] == initial["observations"]
    assert state["initialization_fingerprint"] == initial["initialization_fingerprint"]
    assert service.summary(cid)["quality_status"]["scientific_validation_required"]
    assert (
        state["observations"][-1]["measurement_quality"]["source_file"]
        == "refinement.gpx"
    )
    assert (
        state["observations"][-1]["measurement_quality"]["refinement_id"]
        == "revision-2"
    )
    next_sid = reserve(service, cid)
    with pytest.raises(ValueError, match="Incompatible measurement definitions"):
        service.measure(
            cid,
            next_sid,
            {
                "value": 3,
                **explicit("xrd_area_fraction", "sum of integrated pattern areas"),
            },
            refresh=False,
        )
    service.revise_measurement_definition(
        cid,
        {
            "quantification_method": "xrd_area_fraction",
            "normalization": "sum of integrated pattern areas",
        },
        "exclude",
        "Start an area-based cohort; mass and historical definitions are not equivalent",
    )
    excluded = service.export(cid)
    assert all(not r["training_included"] for r in excluded["observations"])
    assert [r["value"] for r in excluded["observations"]] == [-2, 1, 2]
    assert service.summary(cid)["counts"]["new_measurements"] == 1
    service.measure(
        cid,
        next_sid,
        {
            "value": 3,
            **explicit("xrd_area_fraction", "sum of integrated pattern areas"),
        },
        refresh=False,
    )
    assert service.summary(cid)["best"] == 3
    old_id = state["observations"][-1]["observation_id"]
    service.refine(
        cid,
        old_id,
        {
            "value": 4,
            **explicit("xrd_area_fraction", "sum of integrated pattern areas"),
        },
        "Re-quantified the same physical measurement from the integrated pattern",
    )
    assert service.summary(cid)["counts"]["new_measurements"] == 2
    assert service.summary(cid)["best"] == 4
    assert service.get(cid)["initial_observations"] == initial["initial_observations"]
    assert CampaignService(tmp_path).export(cid) == service.export(cid)


def test_definition_comparison_and_independent_control_inclusion(tmp_path):
    service = CampaignService(tmp_path, runner=first)
    a, b = make(service), make(service)
    assert comparison_compatibility(service.get(a), service.get(b))["compatible"]
    mass = {
        "quantification_method": "gsas_ii_mass_fraction",
        "normalization": "all refined phases",
    }
    area = {
        "quantification_method": "xrd_area_fraction",
        "normalization": "integrated pattern area",
    }
    service.revise_measurement_definition(
        a, mass, "exclude", "Different basis; preserve historical data excluded"
    )
    service.revise_measurement_definition(
        b, area, "exclude", "Separate area fraction campaign"
    )
    result = comparison_compatibility(service.get(a), service.get(b))
    assert not result["compatible"] and "measurement_definition" in result["mismatches"]
    control = service.create_control(a)
    assert (
        service.get(control)["initial_observations"]
        == service.get(a)["initial_observations"]
    )
    assert service.active(service.get(control)) == []
    assert comparison_compatibility(service.get(a), service.get(control))["compatible"]


@pytest.mark.parametrize("policy", ["exclude", "retain_with_justification"])
def test_control_definition_is_complete_before_atomic_first_save(
    tmp_path, monkeypatch, policy
):
    from boicl import campaign

    service = CampaignService(tmp_path)
    parent = make(service)
    service.revise_measurement_definition(
        parent,
        {
            "quantification_method": "gsas_ii_mass_fraction",
            "normalization": "all phases",
        },
        policy,
        "Explicit scientific decision for the historical cohort",
    )
    before = service.export(parent)
    files = file_snapshot(service.root)
    write = campaign.atomic_json

    def fail_control_write(path, value):
        if Path(path).name.startswith("campaign-") and value["campaign_id"] != parent:
            assert (
                value["config"]["measurement_definition"]["historical_policy"] == policy
            )
            assert bool(service.active(value)) == (
                policy == "retain_with_justification"
            )
            raise OSError("simulated control save failure")
        return write(path, value)

    monkeypatch.setattr(campaign, "atomic_json", fail_control_write)
    with pytest.raises(OSError, match="control save failure"):
        service.create_control(parent)
    assert list(service.campaigns) == [parent]
    assert file_snapshot(service.root) == files
    assert service.export(parent) == before
    monkeypatch.setattr(campaign, "atomic_json", write)
    control = service.create_control(parent)
    assert service.create_control(parent) == control
    assert len(service.list_checkpoints(control)) == 1
    assert (
        service.get(control)["initial_observations"] == before["initial_observations"]
    )
    assert comparison_compatibility(before, service.get(control))["compatible"]
    restarted = CampaignService(tmp_path)
    assert len(restarted.campaigns) == 2
    assert restarted.export(parent) == before
    assert restarted.export(control) == service.export(control)


def test_confirmed_moc_seeds_load_unknown_without_relabeling_or_rehashing(tmp_path):
    service = CampaignService(tmp_path)
    package = load_moc_package()
    cid = service.create("moc_gp", package)
    data = service.export(cid)
    assert [r["moc_wt_pct"] for r in data["observations"]] == [72.1, 83.8, 23.4]
    assert all(
        r["measurement_quality"]["quantification_method"] == "historical_unspecified"
        for r in data["observations"]
    )
    assert data["initial_observations"] == package["observations"]
    assert data["initialization_fingerprint"] == fingerprint(package["observations"])
    legacy = deepcopy(data)
    for row in legacy["observations"] + legacy["archive"]:
        row.pop("measurement_quality", None)
    legacy.pop("initial_observations")
    legacy.pop("inverse_proposals")
    legacy["config"].pop("measurement_definition")
    legacy["config"]["llm"].pop("manual_inverse_target")
    legacy["config"]["llm"].pop("inverse_proposal_count")
    restored = CampaignService(tmp_path / "legacy").import_bundle(legacy)
    record = CampaignService(tmp_path / "legacy").get(restored)
    assert record["initialization_fingerprint"] == data["initialization_fingerprint"]
    assert all(
        r["measurement_quality"]["quantification_method"] == "historical_unspecified"
        for r in record["observations"]
    )
    invalid = deepcopy(data)
    invalid["initial_observations"][0]["measurement_quality"] = {"schema_version": 999}
    invalid["initialization_fingerprint"] = fingerprint(invalid["initial_observations"])
    with pytest.raises(ValueError, match="measurement-quality schema"):
        service.import_bundle(invalid)
    invalid = deepcopy(data)
    invalid["initial_observations"][0]["measurement_quality"] = {
        "quantification_method": "gsas_ii_mass_fraction"
    }
    invalid["initialization_fingerprint"] = fingerprint(invalid["initial_observations"])
    with pytest.raises(ValueError, match="initialization relabel"):
        service.import_bundle(invalid)
    invalid = deepcopy(data)
    for row in invalid["observations"]:
        row["measurement_quality"] = {"quantification_method": "gsas_ii_mass_fraction"}
    with pytest.raises(ValueError, match="documented refinement"):
        service.import_bundle(invalid)
    for changes in (
        {"moc_wt_pct": 12.3},
        {"moc_wt_pct_sigma": 9.5},
        {"candidate_id": data["observations"][0]["candidate_id"]},
        {"supersedes": data["observations"][1]["observation_id"]},
    ):
        invalid = deepcopy(data)
        invalid["observations"][1].update(changes)
        with pytest.raises(ValueError, match="unchanged|itself"):
            service.import_bundle(invalid)
    initial_id = data["observations"][1]["observation_id"]
    service.refine(
        cid,
        initial_id,
        {"moc_wt_pct": 81.7},
        "Documented reanalysis of the same M12 specimen; preserve its confirmed original",
    )
    refined = service.export(cid)
    assert refined["initial_observations"] == data["initial_observations"]
    assert (
        next(r for r in refined["observations"] if r["observation_id"] == initial_id)[
            "moc_wt_pct"
        ]
        == 83.8
    )
    assert refined["observations"][-1]["moc_wt_pct"] == 81.7
    assert CampaignService(tmp_path).export(cid) == refined


def test_import_requires_documented_same_measurement_acyclic_refinement_lineage(
    tmp_path,
):
    service = CampaignService(tmp_path)
    cid = make(service)
    original_id = service.get(cid)["observations"][0]["observation_id"]
    with pytest.raises(ValueError, match="requires a reason"):
        service.refine(cid, original_id, {"value": -1}, "   ")
    service.refine(cid, original_id, {"value": -1}, "First documented reanalysis")
    first_id = service.get(cid)["observations"][-1]["observation_id"]
    service.refine(cid, first_id, {"value": 0}, "Second documented reanalysis")
    valid = service.export(cid)
    assert CampaignService(tmp_path).export(cid) == valid
    checkpoint = service.save_checkpoint(cid, "Documented refinement chain")
    restored = service.restore_checkpoint(cid, checkpoint["checkpoint_id"])
    assert service.get(restored)["observations"] == valid["observations"]
    final_id = valid["observations"][-1]["observation_id"]
    for changes in (
        {"supersedes": "unknown-record"},
        {"supersedes": final_id},
        {"candidate_id": "c1"},
        {"physical_measurement_id": "different-specimen"},
        {"is_seed": False},
        {"revision_reason": "  "},
    ):
        invalid = deepcopy(valid)
        invalid["observations"][-1].update(changes)
        with pytest.raises(ValueError, match="Refinement"):
            service.import_bundle(invalid)
    invalid = deepcopy(valid)
    invalid["observations"][0].update(
        supersedes=final_id, revision_reason="A cycle is not a valid refinement"
    )
    invalid["observations"][-1].update(
        record_status="superseded_refinement", training_included=False
    )
    with pytest.raises(ValueError, match="cycle"):
        service.import_bundle(invalid)
    invalid = deepcopy(valid)
    branch = deepcopy(invalid["observations"][-1])
    branch.update(
        observation_id="branched-refinement",
        record_status="superseded_refinement",
        training_included=False,
    )
    invalid["observations"].append(branch)
    with pytest.raises(ValueError, match="branch"):
        service.import_bundle(invalid)
    assert service.export(cid) == valid


def test_long_windows_path_and_short_atomic_filename_checkpoint_roundtrip(
    tmp_path, monkeypatch
):
    from boicl import persistence

    deep = tmp_path
    for i in range(6):
        deep = deep / ("scientific-campaign-directory-" + str(i) + "-" + "x" * 24)
    assert len(str(deep)) > 300
    path = io_path(deep) / ("n" * 220 + ".json")
    atomic_bytes(path, b"original")
    assert path.read_bytes() == b"original"
    service = CampaignService(deep, runner=first)
    cid = make(service)
    sid = reserve(service, cid)
    saved = service.save_checkpoint(cid, "Long Windows path")
    original = service.export(cid)
    actual_replace = persistence.os.replace

    def fail_live(src, dst):
        if Path(dst).name == f"campaign-{cid}.json":
            raise OSError("simulated disk replace failure")
        return actual_replace(src, dst)

    points = service.list_checkpoints(cid)
    monkeypatch.setattr(persistence.os, "replace", fail_live)
    with pytest.raises(OSError):
        service.update_config(cid, {"name": "Must roll back"})
    assert service.export(cid) == original and service.list_checkpoints(cid) == points
    assert not list(service.root.rglob(".boicl-*.tmp"))
    monkeypatch.setattr(persistence.os, "replace", actual_replace)
    restarted = CampaignService(deep, runner=first)
    copy = restarted.restore_checkpoint(cid, saved["checkpoint_id"])
    assert restarted.get(copy)["suggestions"][0]["suggestion_id"] == sid
    assert restarted.summary(copy)["counts"]["pending"] == 1
    assert restarted.export(cid) == original
