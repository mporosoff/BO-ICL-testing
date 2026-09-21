"""Real loopback HTTP contracts; every provider is forbidden or mocked."""
from copy import deepcopy
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import threading

import pytest

from boicl.local_app import LocalAppHandler, LocalBOState
from boicl.moc_cli import offline_demo


@pytest.fixture
def http_server(tmp_path, monkeypatch):
    import openai

    def forbidden(*args, **kwargs):
        raise AssertionError("HTTP lifecycle must not construct a provider client")

    monkeypatch.setattr(openai, "OpenAI", forbidden)
    running = []

    def start(demo=True):
        class Handler(LocalAppHandler):
            moc_service = None
            moc_init_lock = threading.Lock()

            def log_message(self, *args):
                pass

        Handler.state = LocalBOState(tmp_path / f"server-{len(running)}")
        Handler.moc_demo = demo
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        running.append((server, thread))

        def request(action, payload=None, method="POST", raw=None):
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=30
            )
            body = (
                raw
                if raw is not None
                else None
                if method == "GET"
                else json.dumps(payload or {})
            )
            connection.request(
                method,
                action if action.startswith("/") else "/api/moc/" + action,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            data = response.read().decode("utf-8")
            headers = dict(response.getheaders())
            status = response.status
            connection.close()
            return status, json.loads(data), headers

        return request, Handler

    yield start
    for server, thread in running:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def wait_job(handler, cid):
    job = handler.moc_service.jobs[cid]
    if job.get("thread"):
        job["thread"].join(timeout=30)
        assert not job["thread"].is_alive(), "Mocked background job did not complete"
    return job


def small_shared_llm(request):
    status, created, _ = request(
        "/api/toolkit/create-generic",
        {
            "records": [
                {
                    "candidate_id": f"c{i}",
                    "procedure": f"Prepare design {i}",
                    "x": i,
                    "yield": i + 1 if i < 2 else None,
                }
                for i in range(5)
            ],
            "feature_spec": [{"column": "x", "transform": "linear", "bounds": [0, 4]}],
            "objective": "yield",
            "units": "units",
            "direction": "maximize",
            "bounds": [0, 20],
            "procedure_column": "procedure",
        },
    )
    assert status == 200
    cid = created["campaign_id"]
    assert (
        request(
            "config", {"id": cid, "changes": {"engine": "llm", "auto_suggest": False}}
        )[0]
        == 200
    )
    return cid


def test_http_manual_target_full_preview_and_separate_inverse_records(http_server):
    request, handler = http_server()
    cid = small_shared_llm(request)
    status, projected, _ = request(
        "/api/toolkit/action",
        {
            "campaign": cid,
            "action": "config",
            "values": {
                "optimizer": "llm",
                "inverse_target_value": "0",
                "inverse_design_count": 3,
                "prediction_system_message": "Exact custom forward system.",
                "inverse_system_message": "Exact custom inverse system.",
            },
        },
    )
    assert status == 200
    assert projected["shared_config"]["llm"]["manual_inverse_target"] == 0
    assert projected["config"]["inverse_target_value"] == 0
    assert "shared_prompt_preview" not in projected
    before = handler.moc_service.export(cid)
    inverse = request("request-preview", {"id": cid, "role": "inverse"})[1]
    forward = request(
        "request-preview", {"id": cid, "role": "forward", "candidate_id": "c2"}
    )[1]
    assert inverse["status"] == forward["status"] == "exact"
    assert inverse["request"]["n"] == 3 and forward["request"]["n"] == 5
    assert (
        inverse["request"]["messages"][0]["content"] == "Exact custom inverse system."
    )
    assert (
        forward["request"]["messages"][0]["content"] == "Exact custom forward system."
    )
    assert "Prepare design 2" in forward["request"]["messages"][-1]["content"]
    assert handler.moc_service.export(cid) == before
    assert not handler.moc_service.jobs
    status, started, _ = request("inverse-proposal", {"id": cid})
    assert status == 200
    assert wait_job(handler, cid)["status"] == "proposed"
    after = handler.moc_service.export(cid)
    for key in ("candidates", "observations", "suggestions", "rng_state"):
        assert after[key] == before[key]
    proposal = after["inverse_proposals"][-1]
    assert proposal["returned_count"] == 3
    assert proposal["target"]["resolved_target"] == 0
    recorded = request(
        "request-preview",
        {"id": cid, "role": "inverse", "suggestion_id": proposal["proposal_id"]},
    )[1]
    assert recorded["status"] == "exact" and recorded["source"] == "recorded"
    assert recorded["request"] == inverse["request"]
    checkpoint = request(
        "checkpoint", {"id": cid, "name": "Manual zero and proposals"}
    )[1]
    copy_id = request(
        "restore-checkpoint", {"id": cid, "checkpoint_id": checkpoint["checkpoint_id"]}
    )[1]["campaign_id"]
    restored = request("state?id=" + copy_id, method="GET")[1]
    assert restored["config"]["llm"]["manual_inverse_target"] == 0
    assert restored["counts"]["inverse_proposals"] == 1
    assert (
        request(
            "/api/toolkit/action",
            {
                "campaign": cid,
                "action": "config",
                "values": {"inverse_target_value": ""},
            },
        )[0]
        == 200
    )
    assert (
        handler.moc_service.get(cid)["config"]["llm"]["manual_inverse_target"] is None
    )
    assert (
        request(
            "/api/toolkit/action",
            {
                "campaign": cid,
                "action": "config",
                "values": {"inverse_target_value": 21},
            },
        )[0]
        == 400
    )


def test_http_quantification_decision_and_refinement_preserve_source_values(
    http_server,
):
    request, handler = http_server()
    cid = small_shared_llm(request)
    before = deepcopy(handler.moc_service.get(cid)["observations"])
    status, result, _ = request(
        "measurement-definition",
        {
            "id": cid,
            "definition": {
                "quantification_method": "gsas_ii_mass_fraction",
                "normalization": "all refined phases",
            },
            "historical_policy": "retain_with_justification",
            "reason": "Offline fixture: operator documents a reviewed comparability decision.",
        },
    )
    assert status == 200, result
    assert [r["value"] for r in handler.moc_service.get(cid)["observations"]] == [
        r["value"] for r in before
    ]
    assert all(
        r["measurement_quality"]["quantification_method"] == "historical_unspecified"
        for r in handler.moc_service.get(cid)["observations"]
    )
    request("suggest", {"id": cid})
    assert wait_job(handler, cid)["status"] == "suggested"
    suggestion = request("state?id=" + cid, method="GET")[1]["suggestions"][-1]
    request("reserve", {"id": cid, "suggestion_id": suggestion["suggestion_id"]})
    values = dict(
        value=3,
        quantification_method="gsas_ii_mass_fraction",
        normalization="all refined phases",
        source_file="fixture.gpx",
        source_identifier="sample-c",
        refinement_id="refinement-v1",
        uncertainty_method="reported covariance",
        definition_note="Offline provenance fixture",
    )
    status, measured, _ = request(
        "measure",
        {"id": cid, "suggestion_id": suggestion["suggestion_id"], "values": values},
    )
    assert status == 200, measured
    observation = measured["observations"][-1]
    assert observation["measurement_quality"]["schema_version"] == 1
    assert observation["measurement_quality"]["source_file"] == "fixture.gpx"
    bad = {**values, "quantification_method": "xrd_area_fraction"}
    assert (
        request(
            "refine",
            {
                "id": cid,
                "observation_id": observation["observation_id"],
                "values": bad,
                "reason": "Incompatible method must reject",
            },
        )[0]
        == 400
    )
    status, changed, _ = request(
        "measurement-definition",
        {
            "id": cid,
            "definition": {
                "quantification_method": "xrd_area_fraction",
                "normalization": "all refined phases",
            },
            "historical_policy": "exclude",
            "reason": "Offline fixture: exclude incompatible and unknown data explicitly.",
        },
    )
    assert status == 200, changed
    status, main_state, _ = request("/api/toolkit/state?campaign=" + cid, method="GET")
    assert status == 200 and len(main_state["observations"]) == 3
    assert all(r["training_included"] is False for r in main_state["observations"])
    assert [r["value"] for r in main_state["observations"]] == [1, 2, 3]


def test_live_request_preview_reports_missing_selector_without_any_provider_or_write(
    http_server,
):
    request, handler = http_server(demo=False)
    cid = small_shared_llm(request)
    before = handler.moc_service.export(cid)
    root = handler.moc_service.root
    saved = {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    status, preview, _ = request(
        "request-preview", {"id": cid, "role": "forward", "candidate_id": "c2"}
    )
    assert status == 200 and preview["status"] == "unresolved"
    assert (
        preview["request"] is None
        and preview["request_parameters"]["model"] == "gpt-4o"
    )
    assert preview["reason"] and "Prepare design 2" in preview["query_message"]
    assert handler.moc_service.export(cid) == before and not handler.moc_service.jobs
    assert {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == saved


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
def test_request_preview_controls_never_attach_unrelated_candidate_to_recorded_step(
    tmp_path,
):
    from boicl.moc_ui import MOC_HTML
    import re

    focused_handler = re.search(
        r"\$\('requestPreview'\)\.onclick=.*?;\n", MOC_HTML
    ).group()
    focus_path = tmp_path / "focus-preview.js"
    focus_path.write_text(focused_handler, encoding="utf-8")
    main_path = Path(__file__).resolve().parents[1] / "boicl/toolkit_main.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map(),el=id=>{if(!elements.has(id))elements.set(id,{value:'',addEventListener(){}});return elements.get(id);};
let ready,captured;
const context={URLSearchParams,TextDecoder,location:{search:'?campaign=arm',pathname:'/'},
 document:{addEventListener(n,fn){ready=fn;},createElement(){return {};},head:{append(){}}},
 $:el,state:{},window:{},renderError(m){throw Error(m);}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);ready();
context.toolkitFetch=async(path,payload)=>{captured=payload;return {status:'exact',source:'recorded'};};
el('requestPreviewSource').value='recorded';el('requestPreviewRole').value='forward';
el('requestPreviewCandidate').value='unrelated-current-candidate';el('sharedReplayStep').value='selected-step';
(async()=>{
 await el('requestPreview').onclick();
 assert.equal(captured.candidate_id,null);assert.equal(captured.suggestion_id,'selected-step');
 el('requestPreviewSource').value='current';await el('requestPreview').onclick();
 assert.equal(captured.candidate_id,'unrelated-current-candidate');assert.equal(captured.suggestion_id,null);
 const focus={$:el,current:'arm',guarded:fn=>fn(),api:async(path,payload)=>{captured=payload;return {status:'exact'};}};
 vm.createContext(focus);vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),focus);
 el('requestPreviewSource').value='recorded';el('replayStep').value='focused-step';
 await el('requestPreview').onclick();
 assert.equal(captured.candidate_id,null);assert.equal(captured.suggestion_id,'focused-step');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(main_path), str(focus_path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("preset", ["moc_gp", "moc_llm", "moc_embedding_gp"])
def test_http_demo_all_engines_measure_reserve_replay_and_resume(http_server, preset):
    request, handler = http_server()
    status, created, _ = request("create", {"preset": preset, "budget": 3})
    assert status == 200
    cid = created["campaign_id"]
    assert not handler.moc_service.jobs
    status, summary, _ = request("state?id=" + cid, method="GET")
    assert status == 200 and summary["counts"]["available"] == 7773
    assert summary["best"] == 83.8 and summary["synthetic_demo"]
    changes = {
        "auto_suggest": False,
        "structured_gp": {"burn_in": 2, "retained_draws": 4, "predict_thin": 1},
    }
    assert request("config-preview", {"id": cid, "changes": changes})[0] == 200
    assert not handler.moc_service.jobs
    assert request("config", {"id": cid, "changes": changes})[0] == 200
    assert request("suggest", {"id": cid})[0] == 200
    assert wait_job(handler, cid)["status"] == "suggested"
    summary = request("state?id=" + cid, method="GET")[1]
    first = summary["suggestions"][-1]
    assert first["candidate_id"] and first["prediction"]
    assert first["candidate_id"] not in {
        row["candidate_id"] for row in summary["observations"]
    }
    assert (
        request("reserve", {"id": cid, "suggestion_id": first["suggestion_id"]})[1][
            "counts"
        ]["pending"]
        == 1
    )
    values = dict(
        moc_wt_pct=0,
        moc_wt_pct_sigma=0,
        gof=1,
        closure_gap_wt_pct=0,
        closure_gap_origin="synthetic test override",
        source_note="SYNTHETIC HTTP fixture",
    )
    status, measured, _ = request(
        "measure",
        {
            "id": cid,
            "suggestion_id": first["suggestion_id"],
            "values": values,
            "request_id": "test-measure",
        },
    )
    assert status == 200
    assert measured["counts"]["measured"] == 4 and measured["counts"]["pending"] == 0
    assert measured["observations"][-1]["moc_wt_pct"] == 0
    assert measured["observations"][-1]["synthetic"]
    assert (
        request("replay", {"id": cid, "suggestion_id": first["suggestion_id"]})[0]
        == 200
    )
    assert request("suggest", {"id": cid})[0] == 200
    assert wait_job(handler, cid)["status"] == "suggested"
    second = request("state?id=" + cid, method="GET")[1]["suggestions"][-1]
    assert second["candidate_id"] != first["candidate_id"]
    request("reserve", {"id": cid, "suggestion_id": second["suggestion_id"]})
    status, bundle, headers = request("export?id=" + cid, method="GET")
    assert status == 200 and "attachment" in headers["Content-Disposition"]
    assert "offline-test-not-a-real-key" not in json.dumps(bundle)
    assert request("import", {"bundle": bundle})[0] == 200
    restored = request("import", {"bundle": bundle})[1]["campaign_id"]
    counts = request("state?id=" + restored, method="GET")[1]["counts"]
    assert counts["measured"] == 4 and counts["pending"] == 1
    assert (
        request("release", {"id": restored, "suggestion_id": second["suggestion_id"]})[
            1
        ]["counts"]["pending"]
        == 0
    )


def test_http_import_config_errors_and_export_never_access_provider(http_server):
    request, handler = http_server(demo=False)
    status, pair, _ = request("pair", {"budget": 0})
    assert status == 200 and pair["manifest"]["shared"]["new_measurement_budget"] == 0
    assert not handler.moc_service.jobs
    cid = pair["llm"]
    assert request("suggest", {"id": cid})[0] == 400
    preview = request(
        "config-preview", {"id": cid, "changes": {"llm": {"uncertainty_scalar": 0}}}
    )
    assert preview[0] == 200 and preview[1]["after"]["llm"]["uncertainty_scalar"] == 0
    assert (
        request("state?id=" + cid, method="GET")[1]["config"]["llm"][
            "uncertainty_scalar"
        ]
        == 1
    )
    bundle = request("export?id=" + cid, method="GET")[1]
    assert request("import", {"bundle": bundle})[0] == 200
    dirty = deepcopy(bundle)
    dirty["credentials"] = {"api_key": "must-not-be-returned"}
    status, error, _ = request("import", {"bundle": dirty})
    assert status == 400 and "must-not-be-returned" not in json.dumps(error)
    synthetic = deepcopy(bundle)
    synthetic["synthetic_demo"] = True
    assert request("import", {"bundle": synthetic})[0] == 400
    assert request("state?id=unknown", method="GET")[0] == 400
    assert request("missing", {})[0] == 400
    status, error, _ = request("create", raw="{bad json")
    assert status == 400 and "error" in error
    assert request("create", raw="[]")[0] == 400
    assert request("create", {"workbook": "invalid-base64"})[0] == 400
    assert not handler.moc_service.jobs


def test_cache_prepare_callback_names_and_no_provider_on_cache_hits(
    http_server, monkeypatch
):
    from boicl import embedding_cache

    seen = []

    def prepare(self, records, embedder, batch_size=64, cancelled=None, progress=None):
        assert callable(cancelled) and not cancelled()
        seen.append((self.spec, len(records)))
        report = dict(
            generated=0,
            errors=[],
            cancelled=False,
            batches=0,
            requested=len(records),
            hit_count=len(records),
            missing_ids=[],
            validated_hits=[],
        )
        progress(report)
        return report

    class ValidatedCacheFixture:
        def __init__(self, directory, spec):
            self.spec = spec

    ValidatedCacheFixture.prepare = prepare
    monkeypatch.setattr(
        embedding_cache, "create_embedding_cache", ValidatedCacheFixture
    )
    request, handler = http_server(demo=False)
    for preset, model, prefix in [
        ("moc_llm", "text-embedding-3-large", "experimental procedure: "),
        ("moc_embedding_gp", "text-embedding-ada-002", ""),
    ]:
        cid = request("create", {"preset": preset})[1]["campaign_id"]
        assert request("cache-prepare", {"id": cid})[0] == 200
        job = wait_job(handler, cid)
        assert job["status"] == "complete" and job["provider_attempts"] == []
        assert (
            seen[-1][0].model == model
            and seen[-1][0].format("recipe") == prefix + "recipe"
        )
        assert seen[-1][1] == 7776
    cid = request("create", {"preset": "moc_gp"})[1]["campaign_id"]
    assert request("cache-prepare", {"id": cid})[0] == 400


def test_offline_cli_demo_exports_matched_resumable_campaigns(tmp_path, monkeypatch):
    import openai

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: pytest.fail("Demo must not construct a provider"),
    )
    report = offline_demo(
        tmp_path / "demo",
        sampler_settings={"burn_in": 2, "retained_draws": 4, "predict_thin": 1},
    )
    assert report["synthetic_demo"] and report["provider_calls"] == 0
    assert not report["laboratory_validation"]
    for arm in ("gp", "llm"):
        assert report["arms"][arm]["replay_verified"]
        assert report["arms"][arm]["counts_after_resume"]["measured"] == 4
        assert report["arms"][arm]["counts_after_resume"]["pending"] == 1
        assert (tmp_path / "demo" / f"{arm}-campaign.json.gz").exists()


def test_raw_json_import_preserves_large_rng_state_and_replay(http_server):
    request, handler = http_server()
    assert request("list", method="GET")[0] == 200
    cid = handler.moc_service.create_generic(
        [
            dict(candidate_id=f"c{i}", procedure=f"Set knob to {i}", value=value)
            for i, value in enumerate([-2, 0, None])
        ],
        [],
        preset="generic_llm",
        bounds=[-10, 10],
        synthetic_demo=True,
        overrides={"auto_suggest": False},
    )
    assert request("suggest", {"id": cid})[0] == 200
    assert wait_job(handler, cid)["status"] == "suggested"
    bundle = request("export?id=" + cid, method="GET")[1]
    proposal = bundle["suggestions"][-1]
    rng = proposal["engine_result"]["rng_state_after"]
    assert rng["state"]["state"] > 2**53
    original_text = json.dumps(bundle, ensure_ascii=False)
    status, imported, _ = request("import", {"bundle_json": original_text})
    assert status == 200, imported
    imported_id = imported["campaign_id"]
    roundtrip = request("export?id=" + imported_id, method="GET")[1]
    assert roundtrip["suggestions"][-1]["engine_result"]["rng_state_after"] == rng
    assert (
        request(
            "replay", {"id": imported_id, "suggestion_id": proposal["suggestion_id"]}
        )[0]
        == 200
    )
    for malformed in (
        {"bundle_json": []},
        {"bundle_json": "[]"},
        {"bundle_json": "{bad"},
        {"bundle_json": original_text, "bundle": bundle},
    ):
        assert request("import", malformed)[0] == 400


@pytest.mark.skipif(
    shutil.which("node") is None,
    reason="Node is unavailable for browser import contract",
)
def test_main_browser_import_forwards_original_json_bytes(tmp_path):
    source = Path(__file__).resolve().parents[1] / "boicl" / "toolkit_main.js"
    raw = '{"bundle_version":1,"state":123456789012345678901234567890123456789}'
    archive = tmp_path / "original.json"
    archive.write_text(raw, encoding="utf-8")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const raw = fs.readFileSync(process.argv[2], 'utf8');
let captured;
const context = {URLSearchParams, TextDecoder, location:{search:'', pathname:'/'},
  document:{addEventListener(){}}, state:{}, setBusy(){},
  renderError(message){throw new Error(message);}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
context.toolkitFetch = async (path, payload) => {assert.equal(path, '/api/moc/import');captured=payload;return {campaign_id:'imported'};};
context.chooseSharedCampaign = async () => {};
context.scheduleSharedPoll = () => {};
const bytes = new TextEncoder().encode(raw);
context.toolkitRequest('/api/import-campaign-archive', {body:bytes.buffer}).then(() => {
  assert.deepStrictEqual(Object.keys(captured), ['bundle_json']);
  assert.equal(captured.bundle_json, raw);
}).catch(error => {console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(source), str(archive)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_three_concurrent_campaigns_and_checkpoint_fork_route_independently(
    http_server,
):
    from boicl.campaign import CampaignService

    request, handler = http_server()
    ids = [
        request("create", {"preset": preset})[1]["campaign_id"]
        for preset in ("moc_gp", "moc_llm", "moc_gp")
    ]
    all_started, release = threading.Event(), threading.Event()
    calls = []

    def runner(snapshot, eligible, cancel, progress):
        calls.append(snapshot["campaign_id"])
        if len(calls) == 3:
            all_started.set()
        assert release.wait(20), "Concurrent HTTP requests did not all start"
        return dict(
            candidate_id=eligible[0]["candidate_id"],
            status="suggested",
            score=1,
            selection_reason="Synthetic concurrent-route fixture",
            prediction=dict(mean=80, sd=2, lower95=76, upper95=84),
        )

    handler.moc_service.runner = runner
    try:
        for index, cid in enumerate(ids):
            assert (
                request("config", {"id": cid, "changes": {"auto_suggest": False}})[0]
                == 200
            )
            path, payload = (
                ("/api/toolkit/action", {"campaign": cid, "action": "suggest"})
                if index % 2 == 0
                else ("suggest", {"id": cid})
            )
            assert request(path, payload)[0] == 200
        assert all_started.wait(20)
        assert all(handler.moc_service.jobs[cid]["status"] == "running" for cid in ids)
        assert set(calls) == set(ids)
    finally:
        release.set()
    for cid in ids:
        assert wait_job(handler, cid)["status"] == "suggested"
    proposals = [
        request("state?id=" + cid, method="GET")[1]["suggestions"][-1] for cid in ids
    ]
    assert (
        request(
            "/api/toolkit/action",
            dict(
                campaign=ids[0],
                action="reserve",
                suggestion_id=proposals[0]["suggestion_id"],
            ),
        )[0]
        == 200
    )
    assert (
        request(
            "reserve", dict(id=ids[1], suggestion_id=proposals[1]["suggestion_id"])
        )[0]
        == 200
    )
    status, checkpoint, _ = request(
        "checkpoint", {"id": ids[0], "name": "Before measurement"}
    )
    assert status == 200 and checkpoint["pending_count"] == 1
    points = request("checkpoints?id=" + ids[0], method="GET")[1]["checkpoints"]
    assert checkpoint in points and len(points) >= 3
    values = dict(
        moc_wt_pct=90,
        moc_wt_pct_sigma=1,
        gof=1,
        closure_gap_wt_pct=0,
        closure_gap_origin="test fixture",
    )
    assert (
        request(
            "measure",
            dict(id=ids[0], suggestion_id=proposals[0]["suggestion_id"], values=values),
        )[0]
        == 200
    )
    status, fork, _ = request(
        "restore-checkpoint",
        {"id": ids[0], "checkpoint_id": checkpoint["checkpoint_id"]},
    )
    assert status == 200 and fork["campaign_id"] not in ids
    assert (
        request(
            "restore-checkpoint",
            {"id": ids[1], "checkpoint_id": checkpoint["checkpoint_id"]},
        )[0]
        == 400
    )
    handler.moc_service = CampaignService(handler.moc_service.root, runner=runner)
    counts = [
        request("state?id=" + cid, method="GET")[1]["counts"]
        for cid in ids + [fork["campaign_id"]]
    ]
    assert [row["measured"] for row in counts] == [4, 3, 3, 3]
    assert [row["pending"] for row in counts] == [0, 1, 0, 1]
    assert len(handler.moc_service.jobs) == 0
    restored = request(
        "/api/toolkit/state?campaign=" + fork["campaign_id"], method="GET"
    )[1]
    assert restored["shared_campaign"]["campaign_id"] == fork["campaign_id"]
    assert (
        restored["shared_config"]
        == request("export?id=" + fork["campaign_id"], method="GET")[1]["config"]
    )
    assert len(request("list", method="GET")[1]["campaigns"]) == 4


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node is unavailable for UI routing checks"
)
def test_main_browser_new_tab_and_checkpoint_actions_use_selected_identity():
    source = Path(__file__).resolve().parents[1] / "boicl" / "toolkit_main.js"
    script = r"""
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const elements=new Map();
const element=id=>{if(!elements.has(id))elements.set(id,{value:'',addEventListener(){}});return elements.get(id);};
let ready, opened, captured, chosen;
const context={URLSearchParams,TextDecoder,location:{search:'?campaign=active-arm',pathname:'/'},
 document:{addEventListener(name,fn){ready=fn;},createElement(){return {};},head:{append(){}}},
 $:element,state:{},window:{open(...args){opened=args;}},renderNotice(){},renderError(message){throw Error(message);}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);ready();
element('savedCampaign').value='shared:third-arm';element('openCampaignTab').onclick();
assert.deepStrictEqual(opened,['/?campaign=third-arm','_blank','noopener']);
context.toolkitFetch=async(path,payload)=>{captured={path,payload};return {campaign_id:'independent-copy'};};
context.chooseSharedCampaign=async id=>{chosen=id;};
element('checkpointSelect').value='checkpoint-one';
element('restoreCheckpoint').onclick().then(()=>{
 assert.equal(captured.path,'/api/moc/restore-checkpoint');
 assert.equal(captured.payload.id,'active-arm');
 assert.equal(captured.payload.checkpoint_id,'checkpoint-one');
 assert.equal(chosen,'independent-copy');
}).catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(source)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node is unavailable for UI isolation checks"
)
def test_focused_campaign_switch_discards_another_arms_preview_and_drafts(tmp_path):
    from boicl.moc_ui import MOC_HTML
    import re

    load_source = re.search(
        r"async function load\(\).*?(?=\nfunction render\()", MOC_HTML, re.S
    ).group()
    source = tmp_path / "focused-load.js"
    source.write_text(load_source, encoding="utf-8")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
const element=id=>{if(!elements.has(id))elements.set(id,{value:'A draft',textContent:'A preview',innerHTML:'A pending',checked:true,dataset:{campaign:'A'},classList:{add(){},remove(){}}});return elements.get(id);};
const context={$:element,localStorage:{setItem(){}},history:{replaceState(){}},
 api:async()=>({campaign_id:'B'}),render(){},fillQualityValues(){}};
vm.createContext(context);
vm.runInContext("let current='B', state=null, settings={engine:'llm'}, settingsCampaign='A';"+fs.readFileSync(process.argv[1],'utf8'),context);
(async()=>{
 await context.load();
 assert.equal(vm.runInContext('settings',context),null);
 assert.equal(vm.runInContext('settingsCampaign',context),null);
 assert.equal(element('configPreview').textContent,'');
 assert.equal(element('value').value,'');
 assert.equal(element('pending').innerHTML,'');
 assert.equal(element('checkpointPanel').dataset.campaign,'B');
 element('value').value='B draft';await context.load();
 assert.equal(element('value').value,'B draft');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(source)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
