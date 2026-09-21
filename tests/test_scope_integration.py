"""Independent cross-view contract checks for the clarified toolkit scope."""
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import pytest

from boicl.campaign import CampaignService
from boicl.local_app import LocalAppHandler, LocalBOState


@pytest.fixture
def shared_views(tmp_path):
    calls = []

    def runner(snapshot, eligible, cancel, progress):
        calls.append(snapshot["campaign_id"])
        return dict(
            candidate_id=eligible[0]["candidate_id"],
            status="suggested",
            score=2.0,
            selection_reason="synthetic integration fixture",
            acquisition_units="test units",
            prediction=dict(
                mean=70.0,
                sd=3.0,
                lower95=64.0,
                upper95=76.0,
                uncertainty_type="mock latent interval",
            ),
        )

    service = CampaignService(tmp_path / "shared", runner=runner)
    cid = service.create(
        "moc_gp", overrides={"auto_suggest": False}, synthetic_demo=True
    )

    class Handler(LocalAppHandler):
        def log_message(self, *args):
            pass

    Handler.state = LocalBOState(tmp_path)
    Handler.moc_service = service
    Handler.moc_demo = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, payload=None):
        raw = None if payload is None else json.dumps(payload).encode()
        with urlopen(
            Request(
                base + path, data=raw, headers={"Content-Type": "application/json"}
            ),
            timeout=30,
        ) as response:
            return json.load(response)

    yield service, cid, calls, request, Handler, tmp_path
    server.shutdown()
    server.server_close()
    thread.join()


def test_main_and_focused_share_lifecycle_graph_settings_and_resume(shared_views):
    service, cid, calls, request, handler, tmp_path = shared_views
    main = request("/api/toolkit/state?campaign=" + cid)
    focus = request("/api/moc/state?id=" + cid)
    assert main["shared_campaign"]["campaign_id"] == focus["campaign_id"] == cid
    assert main["shared_campaign"]["counts"]["measured"] == 3
    assert [row["index"] for row in main["live_observation_points"]] == [1, 2, 3]
    assert [row["axis_label"] for row in main["live_observation_points"]] == [
        "i1",
        "i2",
        "i3",
    ]
    assert main["best_trace"][0]["index"] == 1
    assert main["best_trace"][0]["axis_label"] == "i1"
    assert calls == []
    request("/api/toolkit/action", dict(campaign=cid, action="suggest"))
    service.jobs[cid]["thread"].join(timeout=20)
    focus = request("/api/moc/state?id=" + cid)
    proposed = focus["suggestions"][-1]
    request("/api/moc/reserve", dict(id=cid, suggestion_id=proposed["suggestion_id"]))
    main = request("/api/toolkit/state?campaign=" + cid)
    assert main["shared_campaign"]["counts"]["pending"] == 1
    assert len(main["live_observation_points"]) == 3
    request(
        "/api/toolkit/action",
        dict(
            campaign=cid,
            action="measure",
            suggestion_id=proposed["suggestion_id"],
            request_id="same-browser-save",
            values=dict(
                moc_wt_pct=0,
                moc_wt_pct_sigma=1.5,
                gof=0.9,
                closure_gap_wt_pct=0,
                closure_gap_origin="synthetic explicit override",
            ),
        ),
    )
    focus = request("/api/moc/state?id=" + cid)
    assert focus["counts"]["measured"] == 4 and focus["counts"]["pending"] == 0
    assert focus["observations"][-1]["moc_wt_pct"] == 0
    main = request("/api/toolkit/state?campaign=" + cid)
    assert main["live_observation_points"][-1]["index"] == 4
    assert main["live_observation_points"][-1]["optimization_step"] == 1
    assert main["best_trace"][-1]["index"] == 4
    assert main["best_trace"][-1]["axis_label"] == "1"
    assert main["best_trace"][-1]["best"] == 83.8
    request("/api/moc/config", dict(id=cid, changes={"llm": {"uncertainty_scalar": 0}}))
    main = request("/api/toolkit/state?campaign=" + cid)
    assert main["config"]["llm_uncertainty_calibration"] == 0
    assert len(calls) == 1 and len(service.list()) == 1
    exported = request("/api/moc/export?id=" + cid)
    handler.moc_service = CampaignService(tmp_path / "shared", runner=service.runner)
    restored = request("/api/toolkit/state?campaign=" + cid)
    assert restored["shared_campaign"]["campaign_id"] == cid
    assert restored["shared_campaign"]["counts"]["measured"] == 4
    assert restored["shared_config"]["llm"]["uncertainty_scalar"] == 0
    assert (
        exported["history_revision"] == restored["shared_campaign"]["history_revision"]
    )
    assert calls == [cid]


def test_view_reads_do_not_create_comparison_arms_or_provider_requests(shared_views):
    service, cid, calls, request, _, _ = shared_views
    pair = service.create_pair(synthetic_demo=True)
    for arm in (cid, pair["gp"], pair["llm"]):
        for path in ("/api/toolkit/state?campaign=", "/api/moc/state?id="):
            request(path + arm)
    assert len(service.list()) == 3
    assert calls == []
    assert service.comparison(pair["gp"], pair["llm"])["shared_later_outcomes"] is False
