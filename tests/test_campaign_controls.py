from copy import deepcopy

import pytest

from boicl.campaign import CampaignService
from boicl.campaign_controls import (
    clear_random_control,
    comparison_campaigns,
    random_control_campaign,
    random_control_state,
    record_random_control,
    start_random_control,
)


@pytest.fixture
def paired(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Random control must not call a model runner or oracle")

    service = CampaignService(tmp_path, runner=forbidden)
    pair = service.create_pair(
        overrides={"new_measurement_budget": 2, "auto_suggest": False}
    )
    return service, pair


def test_random_read_is_nonmutating_and_start_reserves_once(paired, monkeypatch):
    service, pair = paired
    parent = service.get(pair["llm"])
    initial_ids = [row["observation_id"] for row in parent["observations"]]
    assert random_control_state(service, pair["llm"]) == {}
    assert len(service.list()) == 2
    state = start_random_control(service, pair["llm"], {"target_count": 2})
    assert state["status"] == "waiting_for_result"
    assert (
        state["observations"] == [] and len(state["initialization_observations"]) == 3
    )
    assert state["current_candidate"]["suggestion_id"]
    control = service.get(state["campaign_id"])
    assert [row["observation_id"] for row in control["observations"]] == initial_ids
    assert service.get(pair["llm"]) == parent
    again = start_random_control(service, pair["llm"], {"target_count": 2})
    assert again["current_candidate"] == state["current_candidate"]
    assert len(service.get(state["campaign_id"])["suggestions"]) == 1

    def no_full_pool_read(*args, **kwargs):
        pytest.fail(
            "Read-only graph/control helpers must not copy full candidate pools"
        )

    monkeypatch.setattr(service, "get", no_full_pool_read)
    assert (
        random_control_state(service, pair["llm"])["current_candidate"]
        == state["current_candidate"]
    )
    assert comparison_campaigns(service, pair["llm"])[0]["campaign_id"] == pair["gp"]


def test_random_measurement_is_independent_zero_valid_idempotent_and_resumable(
    paired, tmp_path
):
    service, pair = paired
    state = start_random_control(service, pair["llm"], {"target_count": 2})
    proposal = state["current_candidate"]
    payload = {
        "suggestion_id": proposal["suggestion_id"],
        "candidate_id": proposal["candidate_id"],
        "value": 0,
        "uncertainty": 0.4,
    }
    after = record_random_control(service, pair["llm"], payload)
    assert len(after["observations"]) == 1 and after["observations"][0]["value"] == 0
    assert after["observations"][0]["uncertainty"] == 0.4
    assert after["current_candidate"]["candidate_id"] != proposal["candidate_id"]
    assert len(service.get(pair["llm"])["observations"]) == 3
    saved = deepcopy(service.get(after["campaign_id"]))
    replay = record_random_control(service, pair["llm"], payload)
    assert replay["current_candidate"] == after["current_candidate"]
    assert service.get(after["campaign_id"]) == saved
    resumed = CampaignService(tmp_path)
    assert (
        random_control_state(resumed, pair["llm"])["current_candidate"]
        == after["current_candidate"]
    )
    second = {"suggestion_id": after["current_candidate"]["suggestion_id"], "value": 85}
    finished = record_random_control(resumed, pair["llm"], second)
    assert finished["status"] == "complete" and finished["current_candidate"] is None
    assert len(finished["observations"]) == 2
    assert finished["observations"][-1]["uncertainty"] is None


def test_random_missing_id_budget_and_candidate_mismatch_do_not_mutate(paired):
    service, pair = paired
    with pytest.raises(ValueError, match="exceeds"):
        start_random_control(service, pair["llm"], {"target_count": 3})
    assert random_control_state(service, pair["llm"]) == {}
    state = start_random_control(service, pair["llm"], {"target_count": 1})
    before = service.get(state["campaign_id"])
    with pytest.raises(ValueError, match="suggestion_id"):
        record_random_control(service, pair["llm"], {"value": 45})
    with pytest.raises(ValueError, match="differs"):
        record_random_control(
            service,
            pair["llm"],
            {
                "suggestion_id": state["current_candidate"]["suggestion_id"],
                "candidate_id": "wrong",
                "value": 45,
            },
        )
    assert service.get(state["campaign_id"]) == before


def test_clear_preserves_old_random_history_and_releases_reservation(paired):
    service, pair = paired
    state = start_random_control(service, pair["llm"], {"target_count": 2})
    old_id = state["campaign_id"]
    cleared = clear_random_control(service, pair["llm"])
    assert cleared["campaign_id"] != old_id
    assert cleared["current_candidate"] is None and cleared["observations"] == []
    assert service.get(old_id)["suggestions"][-1]["status"] == "cancelled"
    assert (
        random_control_campaign(service, pair["llm"])["campaign_id"]
        == cleared["campaign_id"]
    )
    assert comparison_campaigns(service, pair["llm"])[0]["campaign_id"] == pair["gp"]


def test_zero_target_starts_no_pending_physical_experiment(paired):
    service, pair = paired
    state = start_random_control(service, pair["llm"], {"target_count": 0})
    assert state["status"] == "complete" and state["current_candidate"] is None
    assert service.get(state["campaign_id"])["suggestions"] == []


def test_generic_control_uses_same_helpers_and_preserves_zero_uncertainty(tmp_path):
    from boicl.campaign_plot import plot_payload

    service = CampaignService(tmp_path)
    records = [
        {"candidate_id": str(index), "temperature": index, "cost": value}
        for index, value in enumerate([10, 20, None, None])
    ]
    cid = service.create_generic(
        records,
        [{"column": "temperature", "transform": "linear"}],
        objective="cost",
        direction="minimize",
        bounds=None,
        overrides={"new_measurement_budget": 1, "auto_suggest": False},
    )
    state = start_random_control(service, cid, {"target_count": 1})
    state = record_random_control(
        service,
        cid,
        {
            "suggestion_id": state["current_candidate"]["suggestion_id"],
            "value": -5,
            "uncertainty": 0,
        },
    )
    assert state["observations"][0]["value"] == -5
    assert state["observations"][0]["uncertainty"] == 0
    assert state["trace"][-1]["best"] == -5
    payload = plot_payload(
        service.get(cid), random_campaign=random_control_campaign(service, cid)
    )
    assert payload["best_trace"][-1]["best"] == 10
    assert payload["live_random_walk_trace"][-1]["best"] == -5
