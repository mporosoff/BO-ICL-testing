"""Each inverse preview must describe its own upcoming, isolated request path."""
from copy import deepcopy
import hashlib
import socket

import pytest

from boicl.campaign import CampaignService


def create_campaign(
    service, *, manual=None, direction="maximize", measured=2, synthetic=True
):
    return service.create_generic(
        [
            dict(
                candidate_id=f"c{i}",
                procedure=f"Synthetic recipe {i}",
                setting=i,
                value=[-2, 1][i] if i < measured else None,
            )
            for i in range(6)
        ],
        [{"column": "setting"}],
        preset="generic_llm",
        bounds=[-10, 10],
        direction=direction,
        synthetic_demo=synthetic,
        overrides={
            "auto_suggest": False,
            "llm": {
                "manual_inverse_target": manual,
                "inverse_proposal_count": 3,
                "shortlist_size": 2,
            },
        },
    )


def files(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def inverse_request(result):
    return next(
        log["request"] for log in result["request_log"] if log["role"] == "inverse"
    )


@pytest.fixture(autouse=True)
def prohibit_providers(monkeypatch):
    import openai
    from boicl.embedding_cache import EmbeddingCache

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Request preview tests must not construct providers or prepare embeddings"
        )

    monkeypatch.setattr(openai, "OpenAI", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(EmbeddingCache, "prepare", forbidden)


@pytest.mark.parametrize("manual", [None, 0])
@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_two_inverse_modes_match_execution_through_independent_sequences(
    tmp_path, monkeypatch, manual, direction
):
    from boicl.moc_demo import DemoChat

    sent = []
    original_create = DemoChat.create

    def track(self, **request):
        sent.append(deepcopy(request))
        return original_create(self, **request)

    monkeypatch.setattr(DemoChat, "create", track)
    service = CampaignService(tmp_path)
    cid = create_campaign(service, manual=manual, direction=direction)
    for sequence in range(2):
        before, disk, calls = service.export(cid), files(service.root), len(sent)
        bo = service.request_preview(cid, role="bo_inverse")
        standalone = service.request_preview(cid, role="standalone_inverse")
        assert bo["status"] == standalone["status"] == "exact"
        assert bo["preview_label"] == "Next BO inverse request"
        assert standalone["preview_label"] == "Standalone inverse proposal"
        assert bo["request"]["n"] == 1 and standalone["request"]["n"] == 3
        assert bo["request_schedule"] == dict(
            sequence_kind="bo_suggestion",
            sequence=sequence,
            seed=616 + sequence,
            inverse_count=1,
        )
        assert standalone["request_schedule"] == dict(
            sequence_kind="standalone_inverse",
            sequence=sequence,
            seed=1000616 + sequence,
            inverse_count=3,
        )
        assert (
            service.export(cid) == before
            and files(service.root) == disk
            and len(sent) == calls
        )
        if manual is None:
            assert (
                bo["target"]["multiplier_draw"]
                != standalone["target"]["multiplier_draw"]
            )
            assert (
                bo["target"]["resolved_target"]
                != standalone["target"]["resolved_target"]
            )
        else:
            assert (
                bo["target"]["resolved_target"]
                == standalone["target"]["resolved_target"]
                == 0
            )
            assert bo["target"]["multiplier_draw"] is None

        # A standalone call must not advance or otherwise alter the next BO request.
        assert (
            service.start_inverse_proposals(cid, background=False)["status"]
            == "proposed"
        )
        proposal = service.get(cid)["inverse_proposals"][-1]["engine_result"]
        assert standalone["request"] == inverse_request(proposal)
        assert standalone["target"] == proposal["target"]
        assert (
            service.request_preview(cid, role="bo_inverse")["request"] == bo["request"]
        )

        next_standalone = service.request_preview(cid, role="standalone_inverse")
        assert service.start_suggestion(cid, background=False)["status"] == "suggested"
        suggestion = service.get(cid)["suggestions"][-1]["engine_result"]
        assert bo["request"] == inverse_request(suggestion)
        assert bo["target"] == suggestion["target"]
        assert (
            service.request_preview(cid, role="standalone_inverse")["request"]
            == next_standalone["request"]
        )

        # Resume retains both schedules, including generated but unreserved steps.
        service = CampaignService(tmp_path)
        assert service.get(cid)["rng_state"]["suggestion_sequence"] == sequence + 1
        assert len(service.get(cid)["inverse_proposals"]) == sequence + 1


def test_recorded_requests_ignore_current_mode_settings_and_schedule(tmp_path):
    service = CampaignService(tmp_path)
    cid = create_campaign(service)
    bo = service.request_preview(cid, role="bo_inverse")
    standalone = service.request_preview(cid, role="standalone_inverse")
    assert (
        service.request_preview(cid, role="inverse")["request"] == standalone["request"]
    )
    service.start_suggestion(cid, background=False)
    service.start_inverse_proposals(cid, background=False)
    data = service.get(cid)
    records = [
        (data["suggestions"][-1]["suggestion_id"], bo["request"]),
        (data["inverse_proposals"][-1]["proposal_id"], standalone["request"]),
    ]
    service.update_config(
        cid,
        {
            "llm": {
                "manual_inverse_target": 0,
                "inverse_proposal_count": 2,
                "inverse_system_message": "Changed system",
            }
        },
    )
    before, disk = service.export(cid), files(service.root)
    for record_id, request in records:
        for role in ("bo_inverse", "standalone_inverse", "inverse"):
            preview = service.request_preview(cid, role=role, suggestion_id=record_id)
            assert preview["source"] == "recorded" and preview["status"] == "exact"
            assert preview["preview_label"] == "Recorded inverse request"
            assert preview["request"] == request and preview["request_schedule"] is None
    assert service.export(cid) == before and files(service.root) == disk


def test_missing_cache_and_unselected_forward_stay_unresolved_without_calls(tmp_path):
    service = CampaignService(tmp_path)
    cid = create_campaign(service, synthetic=False)
    before, disk = service.export(cid), files(service.root)
    for role, count in (("bo_inverse", 1), ("standalone_inverse", 3)):
        preview = service.request_preview(cid, role=role)
        assert preview["status"] == "unresolved" and preview["request"] is None
        assert preview["request_parameters"]["n"] == count
        assert "cached" in preview["reason"]
    forward = service.request_preview(cid)
    assert forward["status"] == "unresolved" and forward["request"] is None
    assert "shortlist depends" in forward["reason"]
    assert service.export(cid) == before and files(service.root) == disk


def test_initial_design_and_filled_budget_do_not_claim_next_bo_inverse_call(tmp_path):
    service = CampaignService(tmp_path)
    single = create_campaign(service, measured=1)
    preview = service.request_preview(single, role="bo_inverse")
    assert preview["status"] == "unresolved" and preview["request"] is None
    assert "initial design" in preview["reason"]
    assert (
        service.request_preview(single, role="standalone_inverse")["status"] == "exact"
    )
    cid = create_campaign(service)
    service.update_config(cid, {"new_measurement_budget": 0})
    preview = service.request_preview(cid, role="bo_inverse")
    assert preview["status"] == "unresolved" and preview["request"] is None
    assert "budget" in preview["reason"]


def test_unknown_preview_role_rejected_without_writes(tmp_path):
    service = CampaignService(tmp_path)
    cid = create_campaign(service)
    before, disk = service.export(cid), files(service.root)
    with pytest.raises(ValueError, match="preview role"):
        service.request_preview(cid, role="inverse_typo")
    assert service.export(cid) == before and files(service.root) == disk
