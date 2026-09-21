"""Campaign lifecycle, pinned imports and untrusted-bundle boundary checks."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import csv
import gzip
import json
import socket
import threading

import pandas as pd
import pytest

from boicl.campaign import (
    CampaignService,
    embedding_spec,
    embedding_cache_directory,
    fingerprint,
)
from boicl.campaign_config import resolve_config, migrate_legacy_settings
from boicl.moc_import import DATA, load_moc_package, validate_candidates, digest


@pytest.fixture(scope="module")
def package():
    return load_moc_package()


def first_runner(snapshot, eligible, cancel, progress):
    return {
        "candidate_id": eligible[0]["candidate_id"],
        "score": 1.0,
        "selection_reason": "Deterministic offline test fixture",
        "prediction": {
            "mean": 60.0,
            "sd": 3.0,
            "uncertainty_type": "synthetic test fixture",
        },
    }


def measurement(value=0, **extra):
    return {
        "moc_wt_pct": value,
        "moc_wt_pct_sigma": 0,
        "gof": 0,
        "closure_gap_wt_pct": 0,
        "closure_gap_origin": "operator explicitly supplied zero override",
        **extra,
    }


def setup_service(tmp_path, package, **overrides):
    service = CampaignService(tmp_path, runner=first_runner)
    cid = service.create("moc_gp", package, {"auto_suggest": False, **overrides})
    return service, cid


def suggest(service, cid):
    response = service.start_suggestion(cid, background=False)
    assert response["status"] == "suggested", service.summary(cid)["progress"]
    return service.get(cid)["suggestions"][-1]["suggestion_id"]


def test_bundled_import_exact_grid_seeds_noise_provenance_and_archive(package):
    assert len(package["candidates"]) == 7776
    assert len({r["candidate_id"] for r in package["candidates"]}) == 7776
    assert [r["moc_wt_pct"] for r in package["observations"]] == [72.1, 83.8, 23.4]
    assert [r["moc_wt_pct_sigma"] for r in package["observations"]] == [1.5, 5.67, 1.5]
    assert [r["gof"] for r in package["observations"]] == [0.517, 0.548, 0.932]
    assert all(
        r["closure_gap_wt_pct"] == 0 and "override" in r["closure_gap_origin"]
        for r in package["observations"]
    )
    assert package["observations"][1]["phase_accounting_residual_wt_pct"] == -0.3
    assert sorted(r["moc_wt_pct"] for r in package["archive"]) == [
        0,
        86.7,
        91.5,
        91.8,
        96.5,
    ]
    assert all(
        r["training_included"] is False and r["closure_gap_wt_pct"] is None
        for r in package["archive"]
    )
    assert package["provenance"]["m12_resolution"]["authoritative"] == 83.8
    assert any(";" in str(r["source_rows"]) for r in package["candidates"])


def test_named_workbook_and_explicit_csv_import_ignore_legacy_pool_labels(
    tmp_path, package, monkeypatch
):
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: pytest.fail("Import must be offline"),
    )
    workbook = tmp_path / "moc.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame([{"procedure": "wrong legacy row", "moc_wt_pct": 99}]).to_excel(
            writer, sheet_name="Pool_import", index=False
        )
        pd.DataFrame(package["candidates"]).to_excel(
            writer, sheet_name="Design_space", index=False
        )
        pd.DataFrame(package["observations"]).to_excel(
            writer, sheet_name="Seed_observations", index=False
        )
    actual = load_moc_package(workbook_bytes=workbook.read_bytes())
    assert actual["pool_fingerprint"] == package["pool_fingerprint"]
    assert [r["moc_wt_pct"] for r in actual["observations"]] == [72.1, 83.8, 23.4]
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    (csv_dir / "import_schema.json").write_bytes(
        (DATA / "import_schema.json").read_bytes()
    )
    for field, name in [
        ("candidates", "MoC_design_space_7776.csv"),
        ("observations", "MoC_seed_observations.csv"),
        ("archive", "MoC_historical_excluded_observations.csv"),
    ]:
        with (csv_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=package[field][0].keys())
            writer.writeheader()
            writer.writerows(package[field])
    csv_package = load_moc_package(csv_dir)
    assert csv_package["pool_fingerprint"] == actual["pool_fingerprint"]
    assert (
        csv_package["provenance"]["m12_resolution"]
        == actual["provenance"]["m12_resolution"]
    )


def test_config_defaults_zero_none_custom_prompts_and_nested_unknown_rejection():
    result = resolve_config(
        "moc_llm",
        {
            "new_measurement_budget": 0,
            "auto_suggest": False,
            "llm": {
                "uncertainty_scalar": 0,
                "xi": 0,
                "inverse_jitter": 0,
                "target_floor": 0,
                "target_ceiling": None,
                "forward_system_message": "Keep this custom forward text.",
                "inverse_system_message": "",
                "selector_mode": "all",
                "selector_k": None,
            },
            "api": {"request_spacing_s": 0, "base_cooldown_s": 0},
        },
    )
    assert resolve_config("moc_llm", json.loads(json.dumps(result))) == result
    assert (
        result["llm"]["uncertainty_scalar"] == 0
        and result["llm"]["inverse_system_message"] == ""
    )
    assert result["llm"]["selector_k"] is None and result["new_measurement_budget"] == 0
    assert resolve_config("moc_gp")["structured_gp"]["retained_draws"] == 4000
    for group in ("llm", "api", "structured_gp", "embedding_gp"):
        with pytest.raises(ValueError, match="Unknown"):
            resolve_config("moc_llm", {group: {"api_key": "secret must never persist"}})
    for group in ("llm", "api"):
        with pytest.raises(ValueError):
            resolve_config("moc_llm", {group: None})
    with pytest.raises(ValueError):
        resolve_config(
            "moc_gp", {"structured_gp": {"metadata_fallbacks": {"api_key": "x"}}}
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"new_measurement_budget": float("nan")},
        {"new_measurement_budget": True},
        {"seed": -1},
        {"workflow_mode": "offline"},
        {"batch_size": 2},
        {"llm": {"n_samples": 3.2}},
        {"llm": {"min_samples": 1}},
        {"llm": {"forward_temperature": 3}},
        {"llm": {"objective_bounds": [-1, 101]}},
        {"llm": {"selector_k": 0}},
        {"llm": {"target_floor": 90, "target_ceiling": 80}},
        {"structured_gp": {"burn_in": -1}},
        {"embedding_gp": {"dimensions": 0}},
    ],
)
def test_invalid_settings_rejected_before_campaign_start(overrides):
    with pytest.raises(ValueError):
        resolve_config("moc_llm", overrides)


def test_legacy_migration_preserves_engine_and_limit_meaning():
    original = {
        "optimizer": "gpr",
        "api_retry_attempts": 8,
        "iterations_per_trial": 0,
        "custom": "retained",
    }
    result = migrate_legacy_settings(original)
    assert result["engine"] == "gpr_embeddings" and result["maximum_attempts"] == 8
    assert result["iteration_limit"] is None and result["legacy_settings"] == original
    assert "observation-count" in result["iteration_limit_meaning"]


def test_initial_counts_source_repeat_policy_and_eight_history(tmp_path, package):
    service, cid = setup_service(tmp_path, package)
    state = service.summary(cid)
    assert state["counts"] == {
        "candidates": 7776,
        "measured": 3,
        "unique_measured": 3,
        "pending": 0,
        "available": 7773,
        "new_measurements": 0,
    }
    assert state["best"] == 83.8 and not state["suggestions"]
    archive_ids = {r["candidate_id"] for r in package["archive"]}
    assert archive_ids <= {
        r["candidate_id"] for r in service.eligible(service.get(cid))
    }
    service.update_config(cid, {"repeat_policy": "exclude_all_historical"})
    assert service.summary(cid)["counts"]["available"] == 7768
    eight = service.create("moc_eight", package, {"auto_suggest": False})
    assert service.summary(eight)["counts"]["measured"] == 8
    assert service.summary(eight)["counts"]["new_measurements"] == 0
    assert service.summary(eight)["best"] == 96.5
    assert len(service.get(eight)["archive"]) == 5


def test_history_lifecycle_zero_measurement_refinement_resume_replay(tmp_path, package):
    service, cid = setup_service(tmp_path, package)
    sid = suggest(service, cid)
    selected = service.get(cid)["suggestions"][-1]["candidate_id"]
    reserved = service.reserve(cid, sid)
    assert (
        reserved["counts"]["pending"] == 1 and reserved["counts"]["available"] == 7772
    )
    saved = service.measure(
        cid, sid, measurement(), request_id="request-1", refresh=False
    )
    assert saved["counts"]["measured"] == 4 and saved["counts"]["new_measurements"] == 1
    actual = saved["observations"][-1]
    assert (
        actual["moc_wt_pct"]
        == actual["moc_wt_pct_sigma"]
        == actual["gof"]
        == actual["closure_gap_wt_pct"]
        == 0
    )
    assert actual["phase_accounting_residual_wt_pct"] is None
    service.measure(cid, sid, measurement(), request_id="request-1", refresh=False)
    assert len(service.get(cid)["observations"]) == 4
    with pytest.raises(ValueError, match="different submission"):
        service.measure(cid, sid, measurement(1), request_id="request-1")
    next_id = suggest(service, cid)
    service.refine(
        cid, actual["observation_id"], measurement(12), "Updated diffraction refinement"
    )
    revised = service.summary(cid)
    assert (
        revised["counts"]["measured"] == 4
        and revised["counts"]["new_measurements"] == 1
    )
    assert revised["observations"][-2]["record_status"] == "superseded_refinement"
    assert (
        revised["observations"][-1]["physical_measurement_id"]
        == actual["physical_measurement_id"]
    )
    assert (
        next(r for r in revised["suggestions"] if r["suggestion_id"] == next_id)[
            "status"
        ]
        == "superseded"
    )
    with pytest.raises(ValueError):
        service.reserve(cid, next_id)
    bundle = service.export(cid)
    restored = CampaignService(tmp_path, runner=first_runner)
    assert restored.export(cid) == bundle
    assert restored.replay(cid, sid)["candidate_id"] == selected
    assert restored.summary(cid)["counts"]["new_measurements"] == 1
    assert (
        "toolkit_code_revision" in bundle
        and bundle["provenance"]["m12_resolution"]["authoritative"] == 83.8
    )


def test_pair_histories_are_independent_after_identical_initialization(
    tmp_path, package
):
    service = CampaignService(tmp_path, runner=first_runner)
    pair = service.create_pair(
        package, {"auto_suggest": False, "new_measurement_budget": 4}
    )
    gp, llm = pair["gp"], pair["llm"]
    assert pair["manifest"]["shared_later_outcomes"] is False
    assert pair["manifest"]["shared"]["new_measurement_budget"] == 4
    assert service.get(gp)["observations"] == service.get(llm)["observations"]
    sid = suggest(service, gp)
    service.reserve(gp, sid)
    service.measure(gp, sid, measurement(20), refresh=False)
    assert service.summary(gp)["counts"]["measured"] == 4
    assert service.summary(llm)["counts"]["measured"] == 3
    assert (
        service.comparison(gp, llm)["initialization_fingerprint"]
        == pair["manifest"]["initialization_fingerprint"]
    )


def test_three_simultaneous_campaign_jobs_remain_independent_after_restart(
    tmp_path, package
):
    release = threading.Event()
    entered = {}
    received = {}

    def runner(snapshot, eligible, cancel, progress):
        cid = snapshot["campaign_id"]
        received.setdefault(cid, []).append(deepcopy(snapshot["observations"]))
        entered[cid].set()
        assert release.wait(30), "Three campaign jobs should enter independently"
        return first_runner(snapshot, eligible, cancel, progress)

    service = CampaignService(tmp_path, runner=runner)
    ids = [
        service.create(
            preset,
            package,
            {"auto_suggest": False, "name": f"Arm {i}", "llm": {"xi": i}},
        )
        for i, preset in enumerate(("moc_gp", "moc_llm", "moc_gp"))
    ]
    entered.update({cid: threading.Event() for cid in ids})
    try:
        for cid in ids:
            service.start_suggestion(cid)
        assert all(event.wait(5) for event in entered.values())
        assert all(service.jobs[cid]["status"] == "running" for cid in ids)
        assert len({service.jobs[cid]["job_id"] for cid in ids}) == 3
        assert all(received[cid] == [package["observations"]] for cid in ids)
    finally:
        release.set()
        for job in service.jobs.values():
            job["thread"].join(20)
    assert all(service.jobs[cid]["status"] == "suggested" for cid in ids)
    suggestions = [service.get(cid)["suggestions"][-1] for cid in ids]
    assert len({row["candidate_id"] for row in suggestions}) == 1
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(
            executor.map(
                lambda pair: service.reserve(*pair),
                [(cid, row["suggestion_id"]) for cid, row in zip(ids, suggestions)],
            )
        )
    for cid, row, value in zip(ids[:2], suggestions[:2], (91, 92)):
        service.measure(cid, row["suggestion_id"], measurement(value), refresh=False)
    service.update_config(ids[0], {"llm": {"xi": 4}})
    assert [service.summary(cid)["counts"]["pending"] for cid in ids] == [0, 0, 1]
    assert [service.get(cid)["config"]["llm"]["xi"] for cid in ids] == [4, 1, 2]
    saved = {cid: service.export(cid) for cid in ids}
    restarted = CampaignService(tmp_path, runner=runner)
    assert restarted.jobs == {}
    assert all(restarted.export(cid) == saved[cid] for cid in ids)
    for cid in ids:
        suggest(restarted, cid)
    assert [r["moc_wt_pct"] for r in received[ids[0]][-1]][-1] == 91
    assert [r["moc_wt_pct"] for r in received[ids[1]][-1]][-1] == 92
    assert received[ids[2]][-1] == package["observations"]
    assert all(received[cid][-1] == saved[cid]["observations"] for cid in ids)


def test_automatic_and_named_checkpoints_restore_exact_saved_points(tmp_path, package):
    service, cid = setup_service(tmp_path, package)
    initial = service.export(cid)
    first_checkpoint = service.list_checkpoints(cid)[0]
    assert first_checkpoint["reason"] == "Campaign created"
    sid = suggest(service, cid)
    service.reserve(cid, sid)
    pending = service.export(cid)
    named = service.save_checkpoint(cid, "Before the first measurement")
    assert named["pending_count"] == 1 and named["new_measurements"] == 0
    assert named["history_revision"] == initial["history_revision"]
    assert named["checkpoint_id"] != first_checkpoint["checkpoint_id"]
    assert service.export(cid) == pending  # manual save is not an experiment mutation
    service.measure(cid, sid, measurement(91), refresh=False)
    measured = service.export(cid)
    measurement_checkpoint = service.list_checkpoints(cid)[0]
    service.refine(
        cid, measured["observations"][-1]["observation_id"], measurement(90), "revision"
    )
    service.update_config(cid, {"llm": {"xi": 0.2}})
    next_sid = suggest(service, cid)
    service.reserve(cid, next_sid)
    service.cancel_reservation(cid, next_sid)
    live = service.export(cid)
    checkpoints = service.list_checkpoints(cid)
    assert len({row["checkpoint_id"] for row in checkpoints}) == len(checkpoints)
    assert {row["reason"] for row in checkpoints} >= {
        "Campaign created",
        "Suggestion started",
        "Suggestion suggested",
        "Experiment reserved",
        "Measurement recorded",
        "Manual checkpoint",
        "Refinement revised",
        "Settings changed",
        "Reservation released",
    }
    # Reads and no-op settings do not manufacture additional saved points.
    service.summary(cid)
    service.view_snapshot(cid)
    service.update_config(cid, {"llm": {"xi": 0.2}})
    assert service.list_checkpoints(cid) == checkpoints
    restarted = CampaignService(tmp_path, runner=first_runner)
    assert restarted.list_checkpoints(cid) == checkpoints
    for checkpoint, expected in (
        (first_checkpoint, initial),
        (named, pending),
        (measurement_checkpoint, measured),
    ):
        fork = restarted.restore_checkpoint(cid, checkpoint["checkpoint_id"])
        actual = restarted.export(fork)
        origin = actual.pop("restored_from_checkpoint")
        assert origin["campaign_id"] == cid
        assert origin["checkpoint_id"] == checkpoint["checkpoint_id"]
        assert fork != cid and fork not in restarted.jobs
        actual["campaign_id"] = cid
        assert actual == expected
        assert restarted.export(cid) == live
        assert len(restarted.list_checkpoints(fork)) == 1
    resumed = restarted.restore_checkpoint(cid, named["checkpoint_id"], "Resumed arm")
    assert restarted.get(resumed)["config"]["name"] == "Resumed arm"
    restarted.measure(resumed, sid, measurement(55), refresh=False)
    assert restarted.get(resumed)["observations"][-1]["moc_wt_pct"] == 55
    assert restarted.export(cid) == live
    assert CampaignService(tmp_path).export(resumed) == restarted.export(resumed)


def test_checkpoint_restoration_validates_integrity_and_campaign_ownership(
    tmp_path, package
):
    service, cid = setup_service(tmp_path, package)
    sid = suggest(service, cid)
    saved = service.save_checkpoint(cid, "Ready to reserve")
    clone = service.restore_checkpoint(cid, saved["checkpoint_id"], "Renamed copy")
    service.reserve(clone, sid)  # display rename preserves a still-current proposal
    with pytest.raises(ValueError, match="Unknown checkpoint"):
        service.restore_checkpoint(clone, saved["checkpoint_id"])
    for bad in ("../outside", "bad", None):
        with pytest.raises(ValueError, match="Invalid checkpoint ID"):
            service.restore_checkpoint(cid, bad)
    with pytest.raises(ValueError, match="name"):
        service.save_checkpoint(cid, " ")
    with pytest.raises(ValueError, match="credential"):
        service.save_checkpoint(cid, "sk-" + "a" * 30)
    directory = tmp_path / "checkpoints" / cid
    path = directory / (saved["checkpoint_id"] + ".json.gz")
    data = json.loads(gzip.decompress(path.read_bytes()))
    data["config"]["name"] = "corrupted"
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    before = service.export(cid)
    with pytest.raises(ValueError, match="checksum"):
        service.restore_checkpoint(cid, saved["checkpoint_id"])
    assert service.export(cid) == before


def test_failed_live_save_removes_unaccepted_checkpoint(tmp_path, package, monkeypatch):
    from boicl import campaign

    service, cid = setup_service(tmp_path, package)
    original = campaign.atomic_json
    before = service.export(cid)
    checkpoints = service.list_checkpoints(cid)

    def fail_live(path, data):
        if path.name == f"campaign-{cid}.json":
            raise OSError("simulated live-write failure")
        return original(path, data)

    monkeypatch.setattr(campaign, "atomic_json", fail_live)
    with pytest.raises(OSError):
        service.update_config(cid, {"name": "Unsaved change"})
    assert service.export(cid) == before
    assert service.list_checkpoints(cid) == checkpoints
    assert len(list((tmp_path / "checkpoints" / cid).glob("*.gz"))) == len(checkpoints)


def test_checkpoint_during_active_job_restarts_without_serializing_worker(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def slow(snapshot, eligible, cancel, progress):
        entered.set()
        assert release.wait(20)
        return first_runner(snapshot, eligible, cancel, progress)

    service = CampaignService(tmp_path, runner=slow)
    cid = service.create_generic(
        [{"x": 0, "value": 1}, {"x": 1}, {"x": 2}],
        [{"column": "x"}],
        overrides={"auto_suggest": False},
    )
    service.start_suggestion(cid)
    try:
        assert entered.wait(5)
        checkpoint = service.save_checkpoint(cid, "While calculating")
        expected = service.export(cid)
        resumed = CampaignService(tmp_path, runner=first_runner)
        assert resumed.jobs == {} and resumed.export(cid) == expected
        fork = resumed.restore_checkpoint(cid, checkpoint["checkpoint_id"])
        assert resumed.get(fork)["rng_state"] == expected["rng_state"]
        assert resumed.summary(fork)["progress"] == {}
        assert suggest(resumed, fork)
        assert service.jobs[cid]["status"] == "running"
        assert service.export(cid) == expected
    finally:
        service.cancel_job(cid)
        release.set()
        service.jobs[cid]["thread"].join(10)
    assert service.jobs[cid]["status"] == "cancelled"


def test_mark_historical_recipe_selection_as_quality_repeat(tmp_path, package):
    wanted = package["archive"][0]["candidate_id"]

    def runner(snapshot, eligible, *_):
        return {
            "candidate_id": next(
                r["candidate_id"] for r in eligible if r["candidate_id"] == wanted
            ),
            "score": 0,
        }

    service, cid = setup_service(tmp_path, package)
    service.runner = runner
    sid = suggest(service, cid)
    assert service.get(cid)["suggestions"][-1]["planned_quality_repeat"] is True
    service.reserve(cid, sid)
    service.measure(cid, sid, measurement(40), refresh=False)
    data = service.get(cid)
    assert len(data["archive"]) == 5 and data["archive"][0]["moc_wt_pct"] == 91.8
    assert data["observations"][-1]["planned_quality_repeat"] is True
    assert service.summary(cid)["counts"]["new_measurements"] == 1


def test_budget_counts_pending_and_completed_cancel_frees_slot(tmp_path, package):
    service, cid = setup_service(tmp_path, package, new_measurement_budget=1)
    sid = suggest(service, cid)
    service.reserve(cid, sid)
    with pytest.raises(ValueError, match="budget"):
        service.start_suggestion(cid, background=False)
    with pytest.raises(ValueError, match="Budget"):
        service.update_config(cid, {"new_measurement_budget": 0})
    service.cancel_reservation(cid, sid)
    next_id = suggest(service, cid)
    service.reserve(cid, next_id)
    service.measure(cid, next_id, measurement(), refresh=False)
    with pytest.raises(ValueError, match="budget"):
        service.start_suggestion(cid, background=False)
    assert service.summary(cid)["counts"]["new_measurements"] == 1


def test_snapshot_reads_and_failed_disk_save_cannot_mutate_campaign(
    tmp_path, package, monkeypatch
):
    from boicl import campaign

    service, cid = setup_service(tmp_path, package)
    old = service.export(cid)
    fetched = service.get(cid)
    fetched["config"]["name"] = "external mutation"
    assert service.get(cid)["config"]["name"] == old["config"]["name"]
    monkeypatch.setattr(
        campaign,
        "atomic_json",
        lambda *args: (_ for _ in ()).throw(OSError("simulated full disk")),
    )
    with pytest.raises(OSError):
        service.update_config(cid, {"name": "must not be published"})
    assert service.export(cid) == old
    with pytest.raises(OSError):
        service.start_suggestion(cid, background=False)
    assert service.export(cid) == old and cid not in service.jobs


def test_nested_settings_preview_preserves_prompts_and_makes_no_calls(
    tmp_path, package
):
    service, cid = setup_service(tmp_path, package)
    service.update_config(
        cid,
        {
            "llm": {
                "forward_system_message": "custom template",
                "inverse_system_message": "",
            }
        },
    )
    before = service.export(cid)
    preview = service.update_config(
        cid, {"llm": {"xi": 0, "uncertainty_scalar": 0}}, apply=False
    )
    assert preview["after"]["llm"]["forward_system_message"] == "custom template"
    assert service.export(cid) == before and service.jobs == {}
    service.update_config(cid, {"llm": {"xi": 0, "uncertainty_scalar": 0}})
    config = service.summary(cid)["config"]
    assert (
        config["llm"]["forward_system_message"] == "custom template"
        and config["llm"]["inverse_system_message"] == ""
    )
    assert config["llm"]["uncertainty_scalar"] == 0
    restored = CampaignService(tmp_path)
    assert restored.get(cid)["config"] == config


def test_concurrent_reservations_cannot_duplicate_candidate(tmp_path, package):
    service, cid = setup_service(tmp_path, package)
    ids = [suggest(service, cid), suggest(service, cid)]
    assert (
        service.get(cid)["suggestions"][0]["candidate_id"]
        == service.get(cid)["suggestions"][1]["candidate_id"]
    )

    def reserve(sid):
        try:
            service.reserve(cid, sid)
            return "saved"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(reserve, ids))
    assert sorted(outcomes) == ["rejected", "saved"]
    assert service.summary(cid)["counts"]["pending"] == 1


def test_coalescing_cancel_and_stale_late_result_never_install_new_suggestion(
    tmp_path, package
):
    entered, release = threading.Event(), threading.Event()

    def slow(snapshot, eligible, cancel, progress):
        entered.set()
        assert release.wait(5)
        return first_runner(snapshot, eligible, cancel, progress)

    service, cid = setup_service(tmp_path, package)
    service.runner = slow
    first = service.start_suggestion(cid)
    assert entered.wait(5)
    second = service.start_suggestion(cid)
    assert second["coalesced"] and second["job_id"] == first["job_id"]
    service.update_config(cid, {"name": "Updated while fitting"})
    service.cancel_job(cid)
    release.set()
    service.jobs[cid]["thread"].join(8)
    state = service.summary(cid)
    assert state["progress"]["status"] == "cancelled"
    assert (
        state["suggestions"][-1]["status"] == "cancelled"
        and state["counts"]["pending"] == 0
    )
    with pytest.raises(ValueError):
        service.reserve(cid, state["suggestions"][-1]["suggestion_id"])
    service.runner = first_runner
    assert suggest(service, cid)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(campaign_id="../outside"),
        lambda d: d["config"]["llm"].update(api_key="secret"),
        lambda d: d["observations"][0].update(training_included="false"),
        lambda d: d["observations"][0].update(moc_wt_pct=float("nan")),
        lambda d: d["observations"][0].update(moc_wt_pct_sigma=-1),
        lambda d: d["observations"].append(deepcopy(d["observations"][0])),
        lambda d: d.update(pool_fingerprint="changed"),
        lambda d: d.update(excluded_ids=["unknown-candidate"]),
        lambda d: d["provenance"].update(authorization="Bearer secret"),
        lambda d: d["candidates"][0].update(procedure="changed original source text"),
        lambda d: d["initial_observations"][1].update(moc_wt_pct=81.7),
        lambda d: d["archive"][0].update(training_included=True),
    ],
)
def test_untrusted_bundle_rejected_before_disk_write(tmp_path, package, mutation):
    service, cid = setup_service(tmp_path, package)
    bad = service.export(cid)
    mutation(bad)
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        service.import_bundle(bad)
    assert set(tmp_path.iterdir()) == before
    assert service.summary(cid)["best"] == 83.8


def test_duplicate_pending_invalid_bundle_and_changed_import_forks(tmp_path, package):
    service, cid = setup_service(tmp_path, package)
    sid = suggest(service, cid)
    service.reserve(cid, sid)
    valid = service.export(cid)
    duplicate = deepcopy(valid["suggestions"][0])
    duplicate["suggestion_id"] = "another"
    bad = deepcopy(valid)
    bad["suggestions"].append(duplicate)
    with pytest.raises(ValueError, match="Duplicate pending"):
        service.import_bundle(bad)
    assert service.import_bundle(valid) == cid
    changed = deepcopy(valid)
    changed["config"]["name"] = "Imported alternative"
    other = service.import_bundle(changed)
    assert other != cid and service.get(other)["imported_from_campaign_id"] == cid
    assert service.summary(other)["counts"]["pending"] == 1


def test_embedding_spec_tracks_model_space_and_templates(tmp_path):
    cfg = resolve_config("moc_llm")
    default = embedding_spec(cfg)
    cfg["llm"]["embedding_model"] = "text-embedding-3-small"
    small = embedding_spec(cfg)
    assert default.dimensions == 3072 and small.dimensions == 1536
    assert (
        default.fingerprint != small.fingerprint
        and small.format("raw") == "experimental procedure: raw"
    )
    assert small.fingerprint in str(embedding_cache_directory(tmp_path, cfg))
    gp = embedding_spec(resolve_config("moc_embedding_gp"))
    assert gp.format("raw") == "raw" and gp.model == "text-embedding-ada-002"
    cfg["llm"]["embedding_model"] = "unsupported-model"
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        embedding_spec(cfg)


def test_real_structured_gp_service_defaults_offline(tmp_path, package, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: pytest.fail("Structured service must make no network calls"),
    )
    service = CampaignService(tmp_path)
    cid = service.create("moc_gp", package, {"auto_suggest": False})
    sid = suggest(service, cid)
    result = service.get(cid)["suggestions"][-1]
    assert result["candidate_id"] == "moc-32c375b3a148b782"
    assert result["engine_result"]["stage"] == "maximin"
    assert result["engine_result"]["diagnostics"]["retained_draws"] == 4000
    assert result["engine_result"]["diagnostics"]["burn_in"] == 1000
    assert (
        0
        <= result["prediction"]["lower95"]
        <= result["prediction"]["mean"]
        <= result["prediction"]["upper95"]
        <= 100
    )
    service.reserve(cid, sid)
    assert service.summary(cid)["counts"]["pending"] == 1


def test_provider_failure_reason_preserved_and_foreign_candidate_not_installed(
    tmp_path, package
):
    service, cid = setup_service(tmp_path, package)
    service.runner = lambda *_: {
        "status": "failed",
        "candidate_id": None,
        "reason": "No candidate has enough valid samples",
    }
    result = service.start_suggestion(cid, background=False)
    assert result["status"] == "failed"
    assert (
        service.summary(cid)["progress"]["detail"]
        == "No candidate has enough valid samples"
    )
    service.runner = lambda *_: {
        "status": "suggested",
        "candidate_id": "foreign-id",
        "score": 99,
    }
    result = service.start_suggestion(cid, background=False)
    assert result["status"] == "failed"
    assert service.get(cid)["suggestions"][-1]["candidate_id"] is None
    assert CampaignService(tmp_path).summary(cid)["counts"]["pending"] == 0


def test_pinned_recipe_validation_does_not_trust_a_recomputed_import_hash(package):
    changed = deepcopy(package["candidates"][0])
    changed["procedure"] = "A chemically different preparation"
    changed["procedure_sha256"] = digest(changed["procedure"])
    with pytest.raises(ValueError, match="pinned original"):
        validate_candidates([changed], require_full_grid=False)


def test_credential_shaped_text_cannot_enter_saved_campaign(tmp_path, package):
    service, cid = setup_service(tmp_path, package)
    original = service.export(cid)
    with pytest.raises(ValueError, match="credential-shaped"):
        service.update_config(
            cid,
            {"llm": {"forward_system_message": "mistaken pasted key: sk-" + "x" * 30}},
        )
    assert service.export(cid) == original


def test_auto_refresh_coalesces_repeated_measurement_submission(tmp_path, package):
    service, cid = setup_service(
        tmp_path, package, auto_suggest=True, new_measurement_budget=3
    )
    sid = suggest(service, cid)
    service.reserve(cid, sid)
    service.measure(cid, sid, measurement(10), request_id="autorefresh-1")
    if "thread" in service.jobs[cid]:
        service.jobs[cid]["thread"].join(8)
    service.measure(cid, sid, measurement(10), request_id="autorefresh-1")
    state = service.get(cid)
    assert len(state["suggestions"]) == 2 and len(state["observations"]) == 4
    assert state["rng_state"]["suggestion_sequence"] == 2


def test_synthetic_bundle_resume_cannot_contact_provider_or_measure_into_real_state(
    tmp_path, package, monkeypatch
):
    from boicl import campaign
    from boicl.moc_demo import demo_runner

    monkeypatch.setattr(
        campaign,
        "run_text_engine",
        lambda *a, **k: pytest.fail("Synthetic campaign called real provider path"),
    )
    service = CampaignService(tmp_path)
    cid = service.create(
        "moc_llm",
        package,
        {"auto_suggest": False, "llm": {"shortlist_size": 2, "fetch_k": 10}},
        synthetic_demo=True,
    )
    sid = suggest(service, cid)
    assert (
        service.get(cid)["suggestions"][-1]["engine_result"]["synthetic_demo"] is True
    )
    service.reserve(cid, sid)
    service.measure(cid, sid, measurement(20), refresh=False)
    assert service.get(cid)["observations"][-1]["synthetic"] is True
    restored = CampaignService(tmp_path)
    suggest(restored, cid)
    real = service.create("moc_llm", package, {"auto_suggest": False})
    service.runner = demo_runner
    with pytest.raises(ValueError, match="explicitly marked synthetic"):
        service.start_suggestion(real, background=False)


@pytest.mark.parametrize(
    "acquisition,units",
    [
        ("expected_improvement", "percentage points"),
        ("probability_of_improvement", "probability"),
        ("upper_confidence_bound", "wt%"),
    ],
)
def test_suggestion_summary_exposes_target_diagnostics_and_actual_acquisition(
    tmp_path, package, acquisition, units
):
    def runner(snapshot, eligible, *_):
        cid = eligible[0]["candidate_id"]
        return {
            "selected_candidate_id": cid,
            "predictions": {cid: {"mean": 70, "std": 2, "score": 0.5}},
            "target": {
                "multiplier_draw": 1.2,
                "raw_target": 100.56,
                "resolved_target": 100,
                "bounds_applied": True,
            },
            "diagnostics": {
                "acceptance_rate": 0.3,
                "warnings": ["single chain"],
                "log_hyperparameter_trace": [[1] * 7],
            },
        }

    service, cid = setup_service(tmp_path, package, llm={"acquisition": acquisition})
    service.runner = runner
    suggest(service, cid)
    result = service.summary(cid)["suggestions"][-1]
    assert result["acquisition_units"] == units and result["acquisition"] == acquisition
    assert (
        result["inverse_target"]["raw_target"] == 100.56
        and result["inverse_target"]["resolved_target"] == 100
    )
    assert result["sampler_diagnostics"]["acceptance_rate"] == 0.3
    assert (
        "log_hyperparameter_trace" not in result["sampler_diagnostics"]
        and "engine_result" not in result
    )
