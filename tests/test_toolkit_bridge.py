"""Main-toolkit HTTP projection shares one campaign ledger and model contract."""
from copy import deepcopy
import http.client
from http.server import ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import pytest

from boicl.campaign import CampaignService
from boicl.generic_import import load_generic_package
from boicl.local_app import LocalAppHandler, LocalBOState
from boicl.toolkit_bridge import config_changes, legacy_config, project


@pytest.fixture
def toolkit_server(tmp_path, monkeypatch):
    import openai

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: pytest.fail(
            "Bridge regression unexpectedly constructed provider client"
        ),
    )

    class Handler(LocalAppHandler):
        moc_service = None
        moc_init_lock = threading.Lock()
        moc_demo = False

        def log_message(self, *args):
            pass

    Handler.state = LocalBOState(tmp_path / "legacy")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, payload=None, method="POST"):
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=30
        )
        connection.request(
            method,
            path,
            body=None if method == "GET" else json.dumps(payload or {}),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        status = response.status
        data = json.loads(response.read())
        connection.close()
        return status, data

    yield request, Handler
    server.shutdown()
    server.server_close()
    thread.join(5)


def payload(direction="minimize"):
    return {
        "records": [
            {
                "candidate_id": f"c{i}",
                "x": i,
                "loss": value,
                "sigma": 0.4,
                "text": f"Run design {i}",
            }
            for i, value in enumerate([-12, -5, None, None, None])
        ],
        "feature_spec": [{"column": "x", "transform": "linear", "bounds": [0, 4]}],
        "objective": "loss",
        "direction": direction,
        "bounds": [-20, 20],
        "units": "J",
        "procedure_column": "text",
        "sigma_column": "sigma",
        "name": "Generic shared fixture",
    }


def wait(handler, cid):
    thread = handler.moc_service.jobs[cid].get("thread")
    if thread:
        thread.join(10)
        assert not thread.is_alive()
    assert (
        handler.moc_service.jobs[cid]["status"] == "suggested"
    ), handler.moc_service.jobs[cid]


@pytest.mark.parametrize(
    "direction,expected,value", [("minimize", -12, -14), ("maximize", -5, 0)]
)
def test_main_http_generic_max_min_quality_and_same_pending_ledger(
    toolkit_server, direction, expected, value
):
    request, handler = toolkit_server
    status, result = request("/api/toolkit/create-generic", payload(direction))
    assert status == 200, result
    cid = result["campaign_id"]
    status, state = request(
        "/api/toolkit/action",
        {
            "campaign": cid,
            "action": "config",
            "values": {
                "auto_suggest": False,
                "structured_gp": {
                    "burn_in": 4,
                    "retained_draws": 12,
                    "predict_thin": 2,
                },
            },
        },
    )
    assert status == 200 and state["shared_campaign"]["best"] == expected
    assert (
        state["config"]["objective_direction"] == direction
        and state["config"]["objective_lower_bound"] == -20
    )
    assert all(not row["objectives"] for row in state["candidates"])
    status, _ = request("/api/toolkit/action", {"campaign": cid, "action": "suggest"})
    assert status == 200
    wait(handler, cid)
    state = request("/api/toolkit/state?campaign=" + cid, method="GET")[1]
    suggestion = state["suggestions"][0]
    assert (
        -20
        <= suggestion["prediction"]["lower95"]
        <= suggestion["mean"]
        <= suggestion["prediction"]["upper95"]
        <= 20
    )
    status, reserved = request(
        "/api/toolkit/action",
        {
            "campaign": cid,
            "action": "reserve",
            "suggestion_id": suggestion["suggestion_id"],
        },
    )
    assert status == 200 and reserved["shared_campaign"]["counts"]["pending"] == 1
    direct = request("/api/moc/state?id=" + cid, method="GET")[1]
    assert (
        direct["suggestions"][-1]["status"] == "pending"
    )  # both HTTP views read the same ID
    quality = {
        "value": value,
        "objective_sigma": 0,
        "gof": 0,
        "closure_gap": -0.7,
        "closure_gap_origin": "user supplied residual",
    }
    status, measured = request(
        "/api/toolkit/action",
        {
            "campaign": cid,
            "action": "measure",
            "suggestion_id": suggestion["suggestion_id"],
            "values": quality,
            "request_id": "bridge-quality",
        },
    )
    assert status == 200, measured
    row = measured["observations"][-1]
    assert (
        row["value"] == value
        and row["uncertainty"] == 0
        and row["gof"] == 0
        and row["closure_gap"] == -0.7
    )
    assert measured["shared_campaign"]["counts"]["new_measurements"] == 1
    assert (
        handler.state.observations == []
    )  # legacy mutable state never acquires a duplicate history
    assert handler.moc_service.get(cid)["observations"][-1]["closure_gap"] == -0.7


def test_main_config_roundtrip_preserves_managed_null_custom_empty_zero_and_noop(
    toolkit_server,
):
    request, handler = toolkit_server
    cid = request("/api/toolkit/create-generic", payload())[1]["campaign_id"]
    service = handler.moc_service
    initial = service.export(cid)
    visible = legacy_config(initial["config"])
    visible["selector_mode"] = initial["config"]["llm"]["selector_mode"]
    mapped = config_changes(visible, initial["config"])
    service.update_config(cid, mapped)
    assert service.export(cid) == initial
    service.update_config(
        cid,
        {
            "llm": {
                "forward_system_message": "",
                "inverse_system_message": "custom inverse",
                "uncertainty_scalar": 0,
            }
        },
    )
    current = service.export(cid)
    projected = legacy_config(current["config"])
    projected["selector_mode"] = current["config"]["llm"]["selector_mode"]
    status, state = request(
        "/api/toolkit/action",
        {"campaign": cid, "action": "config", "values": projected},
    )
    assert status == 200, state
    assert service.export(cid) == current
    assert state["shared_config"]["llm"]["forward_system_message"] == ""
    assert state["shared_config"]["llm"]["uncertainty_scalar"] == 0
    assert (
        state["shared_prompt_preview"]["forward"]
        and "MoC" not in state["shared_prompt_preview"]["forward"]
    )


def test_partial_bound_edit_retains_other_endpoint_and_rejects_invalid_existing_outcome(
    toolkit_server,
):
    request, handler = toolkit_server
    cid = request("/api/toolkit/create-generic", payload())[1]["campaign_id"]
    status, state = request(
        "/api/toolkit/action",
        {"campaign": cid, "action": "config", "values": {"objective_lower_bound": -30}},
    )
    assert status == 200 and state["shared_config"]["bounds"] == [-30, 20]
    prior = handler.moc_service.export(cid)
    status, error = request(
        "/api/toolkit/action",
        {"campaign": cid, "action": "config", "values": {"objective_upper_bound": -15}},
    )
    assert status == 400
    assert handler.moc_service.export(cid) == prior


def test_readonly_projection_uses_actual_validated_cache_model_coverage(
    tmp_path, monkeypatch
):
    from boicl import moc_http

    service = CampaignService(tmp_path / "shared")
    spec = [{"column": "x", "transform": "linear"}]
    cid = service.create_generic(
        payload()["records"], spec, objective="loss", bounds=[-20, 20]
    )
    service.update_config(
        cid,
        {
            "engine": "gpr_embeddings",
            "embedding_gp": {"embedding_model": "text-embedding-3-small"},
        },
    )
    cache = SimpleNamespace(
        spec=SimpleNamespace(model="text-embedding-3-small"),
        coverage=lambda rows: {
            "requested": len(rows),
            "hit_count": len(rows) - 1,
            "missing_ids": [rows[-1]["candidate_id"]],
        },
    )
    monkeypatch.setattr(moc_http, "cache_for", lambda *args: cache)
    legacy = LocalBOState(tmp_path / "legacy")
    before = service.export(cid)
    result = project(service, cid, legacy)
    assert result["embedding_cache"] == {
        "total_count": 5,
        "cached_count": 4,
        "ready": False,
        "model": "text-embedding-3-small",
    }
    assert service.export(cid) == before and legacy.observations == []


@pytest.mark.parametrize("procedure_column", ["loss", "sigma", "gof"])
def test_objective_or_quality_procedure_inputs_rejected_at_http_and_library(
    toolkit_server, procedure_column
):
    request, handler = toolkit_server
    body = payload()
    body["procedure_column"] = procedure_column
    status, error = request("/api/toolkit/create-generic", body)
    assert status == 400, error
    assert handler.moc_service.list() == []
    with pytest.raises(ValueError, match="procedure inputs"):
        load_generic_package(
            body["records"],
            body["feature_spec"],
            objective="loss",
            procedure_column=procedure_column,
            sigma_column="sigma",
        )


def test_generic_llm_prompt_adapter_and_minimization_without_moc_assumptions():
    from boicl.llm_engine import LLMEngine, render_messages, score_responses

    engine = LLMEngine(
        {
            "prompt_style": "generic",
            "objective_name": "adsorption energy",
            "objective_units": "eV",
            "maximize": False,
            "objective_bounds": [-30, 10],
            "inverse_jitter": 0,
        }
    )
    assert engine.config["maximize"] is False and engine.config["objective_bounds"] == [
        -30,
        10,
    ]
    rows = [
        {
            "observation_id": "one",
            "candidate_id": "a",
            "procedure": "Reference recipe",
            "value": -12,
        }
    ]
    messages = render_messages(
        "forward",
        rows,
        "Candidate recipe",
        include_phase_context=False,
        prompt_style="generic",
        objective_name="adsorption energy",
        objective_units="eV",
    )
    assert "MoC" not in json.dumps(messages) and "adsorption energy (eV)" in json.dumps(
        messages
    )
    scored = score_responses(
        "a", ["-20", "-10", "-20", "-10", "-20"], -12, engine.config
    )
    assert scored["acquisition"] == pytest.approx(4.8)


def test_compact_view_snapshots_are_defensive_keep_recipe_text_and_cannot_be_imported(
    tmp_path,
):
    service = CampaignService(tmp_path)
    cid = service.create_generic(
        payload()["records"],
        [{"column": "x"}],
        objective="loss",
        bounds=[-20, 20],
        overrides={
            "auto_suggest": False,
            "structured_gp": {"burn_in": 2, "retained_draws": 4, "predict_thin": 1},
        },
    )
    service.start_suggestion(cid, background=False)
    original = service.export(cid)
    compact = service.view_snapshot(cid)
    assert len(compact["candidates"]) == 3  # two observed designs and one proposed
    assert len(service.view_snapshot(cid, all_candidates=True)["candidates"]) == 5
    assert all("engine_result" not in r for r in compact["suggestions"])
    assert original["suggestions"][0]["engine_result"]["scores"]
    compact["config"]["name"] = "mutated"
    compact["candidates"][0]["procedure"] = "mutated"
    summary = service.summary(cid)
    summary["suggestions"][0]["candidate"]["procedure"] = "mutated"
    summary["observations"][0]["value"] = 999
    assert service.export(cid) == original
    with pytest.raises(ValueError, match="Unknown campaign bundle fields"):
        service.import_bundle(service.view_snapshot(cid))
    row = service.list()[0]
    assert row["candidate_count"] == 5 and row["observation_count"] == 2
