"""Versioned preset controls use complete backend settings, never stale UI fields."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from boicl.campaign import CampaignService
from boicl.campaign_config import preset_catalog, preview_preset
from boicl.local_app import INDEX_HTML
from test_moc_http import http_server


def test_http_five_point_catalog_pair_and_saved_overrides(http_server):
    request, handler = http_server()
    status, catalog, _ = request("presets", method="GET")
    assert status == 200
    assert handler.moc_service is None
    choices = {row["preset"]: row for row in catalog["presets"]}
    assert (
        choices["moc_five_gp"]["name"] == "MoC five-point comparison — six-variable GP"
    )
    assert choices["moc_five_llm"]["name"] == "MoC five-point comparison — BO-ICL LLM"
    assert choices["moc_five_gp"]["version"] == "1.0.0"
    assert {
        "moc_gp",
        "generic_gp",
        "generic_llm",
        "generic_embedding_gp",
    } <= choices.keys()
    for preset, threshold in (("moc_five_gp", 3), ("moc_gp", 10)):
        status, preview, _ = request("preset-preview", {"preset": preset})
        assert status == 200
        assert (
            preview["config"]["structured_gp"]["ei_after_unique_measured_designs"]
            == threshold
        )
        assert preview["config"]["preset_version"] == "1.0.0"
        assert request("list", method="GET")[1]["campaigns"] == []
    status, pair, _ = request("pair", {"study": "five_point"})
    assert status == 200, pair
    service = handler.moc_service
    gp, llm = service.get(pair["gp"]), service.get(pair["llm"])
    assert gp["initial_observations"] == llm["initial_observations"]
    assert len(gp["initial_observations"]) == 3
    assert len(gp["candidates"]) == 7776
    assert service.jobs == {}
    for data, preset in ((gp, "moc_five_gp"), (llm, "moc_five_llm")):
        assert data["config"]["preset"] == preset
        assert data["config"]["new_measurement_budget"] == 5
        assert data["config"]["auto_suggest"] is False
        assert data["suggestions"] == []
    original_observations = deepcopy(gp["observations"])
    status, main, _ = request(
        "/api/toolkit/action",
        {
            "campaign": pair["gp"],
            "action": "config",
            "values": {
                "iterations_per_trial": 4,
                "auto_suggest": True,
                "structured_gp": {"ei_after_unique_measured_designs": 7},
            },
        },
    )
    assert status == 200
    assert (
        main["shared_config"]["structured_gp"]["ei_after_unique_measured_designs"] == 7
    )
    assert main["config"]["preset_version"] == "1.0.0"
    saved = deepcopy(main["shared_config"])
    assert request("state?id=" + pair["gp"], method="GET")[1]["config"] == saved
    status, bundle, _ = request("export?id=" + pair["gp"], method="GET")
    assert status == 200 and bundle["config"] == saved
    status, imported, _ = request("import", {"bundle_json": json.dumps(bundle)})
    assert status == 200
    assert service.get(imported["campaign_id"])["config"] == saved
    # The same browser-facing service restores explicit overrides from disk.
    handler.moc_service = CampaignService(service.root, runner=service.runner)
    assert request("state?id=" + pair["gp"], method="GET")[1]["config"] == saved
    status, preview, _ = request(
        "preset-preview", {"id": pair["gp"], "preset": "moc_five_gp"}
    )
    assert status == 200 and preview["config"]["new_measurement_budget"] == 5
    assert handler.moc_service.get(pair["gp"])["config"] == saved
    status, applied, _ = request(
        "preset-apply", {"id": pair["gp"], "preset": "moc_five_gp"}
    )
    assert status == 200
    assert applied["config"]["structured_gp"]["ei_after_unique_measured_designs"] == 3
    assert applied["config"]["new_measurement_budget"] == 5
    assert applied["config"]["auto_suggest"] is False
    assert handler.moc_service.get(pair["gp"])["observations"] == original_observations
    assert handler.moc_service.jobs == {}


def test_http_generic_preset_apply_keeps_dataset_and_rejects_cross_schema(http_server):
    request, handler = http_server()
    status, created, _ = request(
        "/api/toolkit/create-generic",
        {
            "records": [
                {"procedure": "Synthetic A", "x": 0, "loss": -2},
                {"procedure": "Synthetic B", "x": 1, "loss": None},
            ],
            "feature_spec": [{"column": "x", "transform": "linear", "bounds": [0, 1]}],
            "objective": "loss",
            "units": "arbitrary units",
            "direction": "minimize",
            "bounds": None,
            "procedure_column": "procedure",
        },
    )
    assert status == 200, created
    cid = created["campaign_id"]
    assert (
        request(
            "config",
            {
                "id": cid,
                "changes": {
                    "new_measurement_budget": 17,
                    "llm": {"forward_system_message": "Keep this generic prompt"},
                },
            },
        )[0]
        == 200
    )
    before = handler.moc_service.get(cid)
    for action in ("preset-preview", "preset-apply"):
        assert request(action, {"id": cid, "preset": "moc_five_llm"})[0] == 400
        assert handler.moc_service.get(cid) == before
    status, preview, _ = request("preset-preview", {"id": cid, "preset": "generic_llm"})
    assert status == 200
    expected = preview["config"]
    assert expected["objective"] == "loss" and expected["units"] == "arbitrary units"
    assert expected["direction"] == "minimize" and expected["bounds"] is None
    assert expected["new_measurement_budget"] is None
    assert expected["llm"]["forward_system_message"] == "Keep this generic prompt"
    assert "MoC" not in json.dumps(expected["structured_gp"]["feature_spec"])
    status, applied, _ = request("preset-apply", {"id": cid, "preset": "generic_llm"})
    assert status == 200 and applied["config"] == expected
    assert handler.moc_service.get(cid)["observations"] == before["observations"]
    assert handler.moc_service.jobs == {}
    # Omitted budget uses factory five; an explicit zero remains a deliberate override.
    status, created, _ = request("create", {"preset": "moc_five_llm", "budget": 0})
    assert status == 200
    assert (
        handler.moc_service.get(created["campaign_id"])["config"][
            "new_measurement_budget"
        ]
        == 0
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
def test_main_preset_preview_and_creation_do_not_reuse_stale_settings(tmp_path):
    main = Path(__file__).resolve().parents[1] / "boicl/toolkit_main.js"
    data = tmp_path / "presets.json"
    data.write_text(
        json.dumps(
            {
                "catalog": preset_catalog(),
                "previews": {
                    key: preview_preset(key)
                    for key in ("moc_five_gp", "moc_five_llm", "moc_gp", "generic_llm")
                },
            }
        ),
        encoding="utf-8",
    )
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const fixtures=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),elements=new Map(),calls=[],loaded=[];
const el=id=>{if(!elements.has(id))elements.set(id,{value:'',checked:false,disabled:false,dataset:{},textContent:'',addEventListener(){},classList:{toggle(){}}});return elements.get(id);};
const context={$:el,URLSearchParams,location:{search:'',pathname:'/'},document:{addEventListener(){}},state:{shared_campaign:null},
 escapeHtml:String,setBusy(){},renderError(m){throw Error(m);}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
context.chooseSharedCampaign=async id=>{loaded.push(id);vm.runInContext('sharedCampaignId='+JSON.stringify(id),context);};
context.toolkitFetch=async(path,body)=>{calls.push({path,body});
 if(path.endsWith('/presets'))return {presets:fixtures.catalog};
 if(path.endsWith('/preset-preview')){if(body.id&&body.preset.startsWith('moc_'))throw Error('Cannot apply MoC to generic data');return fixtures.previews[body.preset];}
 if(path.endsWith('/pair'))return {gp:'new-gp',llm:'new-llm'};
 return {campaign_id:'new-campaign'};};
(async()=>{
 el('iterationsPerTrial').value='999';el('autoSuggest').checked=true;el('predictionModel').value='stale-model';el('predictionSystemMessage').value='stale prompt';
 el('mocPreset').value='moc_five_gp';el('presetAction').value='create';await context.loadMainPresetCatalog();
 assert(el('mocPreset').innerHTML.includes('moc_gp'));assert(el('mocPreset').innerHTML.includes('generic_llm'));
 assert.equal(JSON.parse(el('presetPreviewConfig').textContent).config.new_measurement_budget,5);
 await context.loadMainPreset();assert.deepEqual(calls.at(-1),{path:'/api/moc/create',body:{preset:'moc_five_gp'}});
 assert.deepEqual(loaded,['new-campaign']);
 el('presetAction').value='pair';await context.previewMainPreset();
 const pair=JSON.parse(el('presetPreviewConfig').textContent);assert.equal(pair.length,2);
 assert(pair.every(p=>p.config.new_measurement_budget===5&&p.config.auto_suggest===false));
 await context.loadMainPreset();assert.deepEqual(calls.at(-1),{path:'/api/moc/pair',body:{study:'five_point'}});
 el('presetAction').value='create';el('mocPreset').value='generic_llm';await context.previewMainPreset();
 assert.equal(el('loadMocPreset').disabled,true);const count=calls.length;await context.loadMainPreset();assert.equal(calls.length,count);
 vm.runInContext("sharedCampaignId='generic-arm'",context);el('presetAction').value='apply';el('mocPreset').value='moc_five_llm';
 await context.previewMainPreset();assert(el('presetPreviewSummary').textContent.includes('Cannot apply'));
 assert.equal(vm.runInContext('mainPresetPreview',context),null);
 el('mocPreset').value='generic_llm';await context.previewMainPreset();await context.loadMainPreset();
 assert.deepEqual(calls.at(-1),{path:'/api/moc/preset-apply',body:{id:'generic-arm',preset:'generic_llm'}});
 // A slow earlier preview cannot replace a later selected preset.
 let release;context.toolkitFetch=async(path,body)=>body.preset==='moc_five_gp'?new Promise(resolve=>release=resolve):fixtures.previews[body.preset];
 el('presetAction').value='create';el('mocPreset').value='moc_five_gp';const earlier=context.previewMainPreset();
 el('mocPreset').value='moc_five_llm';await context.previewMainPreset();release(fixtures.previews.moc_five_gp);await earlier;
 assert.equal(JSON.parse(el('presetPreviewConfig').textContent).preset,'moc_five_llm');
 assert.equal(vm.runInContext('mainPresetPreview.preset',context),'moc_five_llm');
 // Duplicate submissions cannot create a second campaign or apply twice, and a
 // completed request cannot replace a campaign selected while it was pending.
 for(const action of ['create','apply']){
  let finish,writes=0;const navigations=loaded.length;
  context.toolkitFetch=async(path,body)=>path.endsWith('/preset-preview')?fixtures.previews[body.preset]:(writes++,new Promise(resolve=>finish=resolve));
  vm.runInContext("sharedCampaignId='original-arm'",context);
  el('presetAction').value=action;el('mocPreset').value=action==='create'?'moc_five_gp':'generic_llm';
  await context.previewMainPreset();const pending=context.loadMainPreset();await context.loadMainPreset();assert.equal(writes,1);
  vm.runInContext("sharedCampaignId='newly-selected-arm'",context);finish({campaign_id:'created-arm'});await pending;
  assert.equal(loaded.length,navigations);assert.equal(vm.runInContext('sharedCampaignId',context),'newly-selected-arm');
  assert.equal(vm.runInContext('mainPresetSubmitting',context),false);
 }
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(main), str(data)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
def test_main_structured_gp_threshold_is_sent_to_shared_config(tmp_path):
    assert 'id="gpEIThreshold"' in INDEX_HTML
    main = Path(__file__).resolve().parents[1] / "boicl/toolkit_main.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');const elements=new Map();
const el=id=>{if(!elements.has(id))elements.set(id,{value:'',dataset:{}});return elements.get(id);};let submitted;
const context={$:el,URLSearchParams,location:{search:'?campaign=gp',pathname:'/'},document:{addEventListener(){}},setBusy(){},renderError(m){throw Error(m);},
 state:{shared_campaign:{campaign_id:'gp'},shared_config:{engine:'gpr_features',llm:{}}}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
context.renderShared=()=>{};context.scheduleSharedPoll=()=>{};context.toolkitAction=async(action,body)=>{submitted=body.values;return context.state;};
el('gpEIThreshold').value='3';el('gpBurnIn').value='1000';el('gpDraws').value='4000';el('gpThin').value='20';
(async()=>{await context.toolkitRequest('/api/config',{body:JSON.stringify({optimizer:'gpr_features'})});
 assert.deepEqual(JSON.parse(JSON.stringify(submitted.structured_gp)),{burn_in:1000,retained_draws:4000,predict_thin:20,ei_after_unique_measured_designs:3});
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(main)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
