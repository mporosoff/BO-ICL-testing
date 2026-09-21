"""Scientific inclusion revisions and effective initial-cohort comparisons."""
from copy import deepcopy
import json

import pytest

from boicl.campaign import CampaignService
from boicl.campaign_plot import comparison_compatibility, plot_payload
from boicl.measurement_quality import effective_initial_cohort
from boicl.moc_import import load_moc_package


MASS = {"quantification_method": "gsas_ii_mass_fraction", "normalization": "all phases"}
AREA = {
    "quantification_method": "xrd_area_fraction",
    "normalization": "integrated pattern",
}


def first(snapshot, eligible, *_):
    return {"candidate_id": eligible[0]["candidate_id"], "score": 1}


def generic(service):
    return service.create_generic(
        [
            {
                "candidate_id": f"c{i}",
                "procedure": f"Synthetic recipe {i}",
                "x": i,
                "value": (i + 1) * 10 if i < 3 else None,
            }
            for i in range(8)
        ],
        [{"column": "x"}],
        preset="generic_llm",
        bounds=[0, 100],
        overrides={"auto_suggest": False},
    )


def revise(service, cid, definition=MASS, policy="retain_with_justification"):
    return service.revise_measurement_definition(
        cid, definition, policy, "Explicit scientific decision for this test cohort"
    )


def measure(service, cid, value, definition=MASS):
    service.start_suggestion(cid, background=False)
    suggestion = service.get(cid)["suggestions"][-1]["suggestion_id"]
    service.reserve(cid, suggestion)
    service.measure(
        cid,
        suggestion,
        {"value": value, "measurement_quality": definition},
        refresh=False,
    )
    return service.get(cid)["observations"][-1]["observation_id"]


def test_historical_exclusion_can_be_justifiably_restored_without_reviving_archive(
    tmp_path,
):
    package = load_moc_package()
    service = CampaignService(tmp_path)
    a = service.create("moc_gp", package, {"auto_suggest": False})
    b = service.create("moc_llm", package, {"auto_suggest": False})
    original = service.export(a)
    revise(service, a, policy="exclude")
    excluded = service.export(a)
    assert service.active(excluded) == []
    revise(service, a)
    revise(service, b)
    restored = service.export(a)
    assert [r["moc_wt_pct"] for r in service.active(restored)] == [72.1, 83.8, 23.4]
    assert all(
        r["measurement_quality"]["quantification_method"] == "historical_unspecified"
        for r in service.active(restored)
    )
    assert restored["initial_observations"] == original["initial_observations"]
    assert (
        restored["initialization_fingerprint"] == original["initialization_fingerprint"]
    )
    assert restored["archive"] == original["archive"] and len(restored["archive"]) == 5
    assert all(not r["training_included"] for r in restored["archive"])
    for row, prior in zip(restored["observations"], excluded["observations"]):
        history = row["training_inclusion_history"]
        assert [decision["action"] for decision in history] == ["excluded", "restored"]
        assert history[-1]["previous_exclusion"] == prior["training_exclusion"]
        assert "training_exclusion" not in row
    assert comparison_compatibility(restored, service.get(b))["compatible"]
    assert service.comparison(a, b)["shared_later_outcomes"] is False
    restarted = CampaignService(tmp_path)
    assert restarted.export(a) == restored
    archive = json.loads(json.dumps(restored))
    imported_service = CampaignService(tmp_path / "portable")
    imported = imported_service.import_bundle(archive)
    assert imported_service.export(imported) == restored
    checkpoint = restarted.save_checkpoint(a, "Restored historical inclusion")
    copy = restarted.restore_checkpoint(a, checkpoint["checkpoint_id"])
    assert effective_initial_cohort(restarted.get(copy)) == effective_initial_cohort(
        restored
    )


def test_switch_back_restores_only_latest_definition_excluded_records(tmp_path):
    service = CampaignService(tmp_path, runner=first)
    cid = generic(service)
    revise(service, cid)
    original_id = measure(service, cid, 40)
    service.refine(
        cid, original_id, {"value": 41}, "Corrected same physical experiment"
    )
    latest_id = service.get(cid)["observations"][-1]["observation_id"]
    revise(service, cid, AREA, "exclude")
    state = service.export(cid)
    assert service.active(state) == []
    unrelated = state["observations"][0]
    unrelated["training_exclusion"] = {
        "schema_version": 1,
        "source": "operator_invalid_sample",
        "reason": "Specimen was contaminated; not a definition decision",
        "at": "test",
    }
    legacy = state["observations"][1]
    legacy["training_exclusion"].pop("source")
    resumed = CampaignService(tmp_path / "resume", runner=first)
    cid = resumed.import_bundle(state)
    revise(resumed, cid, MASS)
    restored = resumed.export(cid)
    included = {r["observation_id"] for r in resumed.active(restored)}
    assert latest_id in included and original_id not in included
    assert unrelated["observation_id"] not in included
    assert legacy["observation_id"] in included
    original = next(
        r for r in restored["observations"] if r["observation_id"] == original_id
    )
    assert (
        original["record_status"] == "superseded_refinement"
        and not original["training_included"]
    )
    assert (
        next(
            r
            for r in restored["observations"]
            if r["observation_id"] == unrelated["observation_id"]
        )["training_exclusion"]
        == unrelated["training_exclusion"]
    )
    assert resumed.summary(cid)["counts"]["new_measurements"] == 1
    assert CampaignService(tmp_path / "resume").export(cid) == restored


def test_zero_versus_three_effective_initial_records_are_not_matched(tmp_path):
    service = CampaignService(tmp_path)
    full = generic(service)
    revise(service, full)
    old_bug = service.export(full)
    old_bug["campaign_id"] = "a" * 32
    for row in old_bug["observations"]:
        row["training_included"] = False
    zero = service.import_bundle(old_bug)
    result = comparison_compatibility(service.get(full), service.get(zero))
    assert result["mismatches"] == ["effective_initialization"]
    with pytest.raises(ValueError, match="matched initial conditions"):
        service.comparison(full, zero)
    with pytest.raises(ValueError, match="Random control"):
        plot_payload(service.get(full), random_campaign=service.get(zero))


def test_refinement_inclusion_history_does_not_mutate_its_superseded_record(tmp_path):
    service = CampaignService(tmp_path)
    cid = generic(service)
    revise(service, cid, policy="exclude")
    revise(service, cid)
    original = deepcopy(service.get(cid)["observations"][0])
    service.refine(
        cid, original["observation_id"], {"value": 11}, "Reanalyzed initial specimen"
    )
    revised_id = service.get(cid)["observations"][-1]["observation_id"]
    revise(service, cid, policy="exclude")
    revise(service, cid)
    rows = {row["observation_id"]: row for row in service.get(cid)["observations"]}
    assert (
        rows[original["observation_id"]]["training_inclusion_history"]
        == original["training_inclusion_history"]
    )
    assert rows[original["observation_id"]]["record_status"] == "superseded_refinement"
    assert not rows[original["observation_id"]]["training_included"]
    assert rows[revised_id]["training_included"]
    assert len(rows[revised_id]["training_inclusion_history"]) == 4


def test_initial_refinement_identity_values_and_noise_are_compared_not_later_outcomes(
    tmp_path,
):
    service = CampaignService(tmp_path, runner=first)
    a, b = generic(service), generic(service)
    for cid in (a, b):
        revise(service, cid)
    control = service.create_control(a)
    measure(service, a, 40)
    measure(service, b, 50)
    measure(service, b, 60)
    payload = plot_payload(
        service.get(a),
        comparisons=[service.get(b)],
        random_campaign=service.get(control),
    )
    assert payload["comparison_diagnostics"] == []
    assert service.comparison(a, b)["shared_later_outcomes"] is False
    initial = service.get(b)["observations"][0]
    service.refine(
        b,
        initial["observation_id"],
        {"value": 11},
        "New refinement of an initial specimen",
    )
    mismatch = comparison_compatibility(service.get(a), service.get(b))
    assert mismatch["mismatches"] == ["effective_initialization"]
    matching_copy = deepcopy(service.get(b))
    matching_copy["campaign_id"] = "b" * 32
    for row in matching_copy["observations"]:
        row["recorded_at"] = "different incidental save timestamp"
    assert comparison_compatibility(service.get(b), matching_copy)["compatible"]
    for key, value in (
        ("value", 12),
        ("objective_sigma", 2),
        ("refinement_version", "another-refinement"),
    ):
        changed = deepcopy(matching_copy)
        changed["observations"][-1][key] = value
        assert comparison_compatibility(service.get(b), changed)["mismatches"] == [
            "effective_initialization"
        ]


@pytest.mark.parametrize("included", [(False, False), (False, True), (True, False)])
def test_import_rejects_duplicate_latest_physical_records_regardless_of_inclusion(
    tmp_path, included
):
    service = CampaignService(tmp_path)
    cid = generic(service)
    before = service.export(cid)
    invalid = deepcopy(before)
    first_row, second_row = invalid["observations"][:2]
    second_row["physical_measurement_id"] = first_row["physical_measurement_id"]
    first_row["training_included"], second_row["training_included"] = included
    with pytest.raises(ValueError, match="Multiple latest records"):
        service.import_bundle(invalid)
    assert service.export(cid) == before
    assert len(service.campaigns) == 1


@pytest.mark.parametrize("exclude_latest", [False, True])
def test_import_accepts_superseded_record_with_one_valid_latest_refinement(
    tmp_path, exclude_latest
):
    service = CampaignService(tmp_path)
    cid = generic(service)
    old = service.get(cid)["observations"][0]["observation_id"]
    service.refine(cid, old, {"value": 11}, "Documented replacement of one measurement")
    if exclude_latest:
        revise(service, cid, policy="exclude")
    state = service.export(cid)
    imported_service = CampaignService(tmp_path / "imported")
    imported = imported_service.import_bundle(state)
    assert imported_service.export(imported) == state
    assert CampaignService(tmp_path / "imported").export(imported) == state
