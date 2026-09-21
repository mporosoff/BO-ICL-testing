"""Physical effort remains visible when the training definition changes."""
from boicl.campaign import CampaignService
from boicl.campaign_plot import plot_payload

import pytest


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_service_budget_plot_archive_and_refinement_share_physical_positions(
    tmp_path, direction
):
    def runner(snapshot, eligible, *_):
        # Distinct unmeasured fixtures; excluded historical candidates are not chosen.
        seen = {row["candidate_id"] for row in snapshot["observations"]}
        return {
            "candidate_id": next(
                row["candidate_id"]
                for row in eligible
                if row["candidate_id"] not in seen
            ),
            "score": 1,
        }

    service = CampaignService(tmp_path / "source", runner=runner)
    cid = service.create_generic(
        [
            {
                "candidate_id": str(i),
                "procedure": f"Recipe {i}",
                "x": i,
                "value": -2 if i == 0 else None,
            }
            for i in range(4)
        ],
        [{"column": "x"}],
        preset="generic_llm",
        bounds=[-5, 5],
        direction=direction,
        overrides={"new_measurement_budget": 2, "auto_suggest": False},
    )
    mass = {
        "quantification_method": "gsas_ii_mass_fraction",
        "normalization": "all refined phases",
    }
    area = {
        "quantification_method": "xrd_area_fraction",
        "normalization": "integrated pattern",
    }

    def measure(value, definition):
        assert service.start_suggestion(cid, background=False)["status"] == "suggested"
        sid = service.get(cid)["suggestions"][-1]["suggestion_id"]
        service.reserve(cid, sid)
        service.measure(cid, sid, {"value": value, **definition}, refresh=False)
        return service.get(cid)["observations"][-1]["observation_id"]

    service.revise_measurement_definition(
        cid, mass, "exclude", "Unknown seed method excluded"
    )
    first_id = measure(4, mass)
    service.revise_measurement_definition(
        cid, area, "exclude", "Use an area-based definition"
    )
    measure(1, area)
    with pytest.raises(ValueError, match="budget"):
        service.start_suggestion(cid, background=False)
    payload = plot_payload(service.get(cid))
    assert (
        service.summary(cid)["counts"]["new_measurements"]
        == payload["plot_counts"]["new_completed"]
        == 2
    )
    assert payload["live_observation_points"][0]["index"] == 3
    assert payload["live_observation_points"][0]["axis_label"] == "2"
    assert payload["best_trace"][-1]["best"] == 1
    assert [
        point["axis_label"] for point in payload["excluded_measurement_points"]
    ] == ["i1", "1"]

    restarted = CampaignService(tmp_path / "source")
    assert plot_payload(restarted.get(cid)) == payload
    restored_service = CampaignService(tmp_path / "import")
    restored_id = restored_service.import_bundle(service.export(cid))
    assert (
        plot_payload(restored_service.get(restored_id))["plot_counts"]
        == payload["plot_counts"]
    )
    # A documented same-experiment revision can join the new basis without renumbering.
    service.refine(
        cid,
        first_id,
        {"value": 2, **area},
        "Validated revised quantification for the same physical experiment",
    )
    refined = plot_payload(service.get(cid))
    assert refined["plot_counts"]["new_completed"] == 2
    assert [
        (point["axis_label"], point["value"])
        for point in refined["live_observation_points"]
    ] == [("1", 2), ("2", 1)]
