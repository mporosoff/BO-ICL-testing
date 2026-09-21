"""Execute the focused view against deferred requests and real resolved presets."""
import json
import re
import shutil
import subprocess

import pytest

from boicl.campaign_config import preset_catalog, preview_preset
from boicl.moc_ui import MOC_HTML


def run_focused(tmp_path, assertions):
    source = re.search(r"<script>(.*?)</script>", MOC_HTML, re.S).group(1)
    source = source.replace(
        "guarded(async()=>{await loadPresets();await catalog();if(current)await load();});",
        "",
    )
    source_path = tmp_path / "focused.js"
    source_path.write_text(source, encoding="utf-8")
    presets = {
        p["preset"]: preview_preset(p["preset"])
        for p in preset_catalog()
        if p["data_schema"] == "moc"
    }
    fixture_path = tmp_path / "presets.json"
    fixture_path.write_text(
        json.dumps({"presets": preset_catalog(), "resolved": presets}), encoding="utf-8"
    )
    script = (
        r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const fixtures=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const elements=new Map(),calls=[],timers=new Map();let nextTimer=0;
const el=id=>{
 if(!elements.has(id)){
  const classes=new Set();let value='';
  elements.set(id,{dataset:{},files:[],checked:false,disabled:false,textContent:'',innerHTML:'',
   get value(){return value;},set value(v){value=String(v);},
   classList:{add(c){classes.add(c);},remove(c){classes.delete(c);},toggle(c,on){if(on)classes.add(c);else classes.delete(c);},contains(c){return classes.has(c);}}});
 }
 return elements.get(id);
};
const clone=x=>JSON.parse(JSON.stringify(x));
function campaign(id='a',preset='moc_five_llm'){
 const config=clone(fixtures.resolved[preset].config);config.llm.manual_inverse_target=90;
 return {campaign_id:id,config,engine_label:preset,history_revision:0,best:83.8,
 counts:{measured:3,pending:0,available:7773,unique_measured:3},observations:[],archive:[],suggestions:[],provenance:{},progress:{status:'running'}};
}
let response=async(action,body)=>{
 if(action==='presets')return {presets:fixtures.presets};
 if(action==='preset-preview'){
  const value=clone(fixtures.resolved[body.preset]);Object.assign(value.config,body.overrides||{});return value;
 }
 throw Error('Unexpected request '+action);
};
const context={URLSearchParams,location:{search:'',pathname:'/moc'},document:{getElementById:el},window:{},
 localStorage:{getItem(){return '';},setItem(){}},history:{replaceState(){}},
 setTimeout(fn){timers.set(++nextTimer,fn);return nextTimer;},clearTimeout(id){timers.delete(id);},
 fetch:async(path,options)=>{const action=path.replace('/api/moc/',''),body=options.body?JSON.parse(options.body):{};calls.push({action,body});const data=await response(action,body);return {ok:true,json:async()=>data};}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const evaluate=code=>vm.runInContext(code,context);
function show(saved){context.fixture=clone(saved);evaluate('state=fixture;current=fixture.campaign_id;render();');}
async function settle(){await new Promise(resolve=>setImmediate(resolve));}
(async()=>{
"""
        + assertions
        + r"""
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    )
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(source_path), str(fixture_path)],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")


def test_focused_all_settings_drafts_survive_running_poll_and_discard(tmp_path):
    run_focused(
        tmp_path,
        r"""
const saved=campaign();show(saved);
const edits={method:'gp',representation:'gpr_embeddings',auto:true,forwardModel:'custom-forward',inverseModel:'custom-inverse',samples:'9',manualTarget:'',inverseCount:'3',scalar:'0',examples:'all',forwardPrompt:'Unsaved prediction',inversePrompt:'Unsaved inverse',threshold:'4'};
for(const [id,value] of Object.entries(edits)){
 if(id==='auto')el(id).checked=value;else el(id).value=value;
 el(id).oninput();
}
response=async()=>saved;
const poll=[...timers.values()].at(-1);await poll();await settle();
for(const [id,value] of Object.entries(edits))assert.equal(id==='auto'?el(id).checked:el(id).value,value,id+' draft survived polling');
assert(el('suggest').disabled);assert(el('settingsDraftStatus').textContent.includes('Unsaved'));
el('discardSettings').onclick();assert.equal(el('manualTarget').value,'90');assert.equal(el('forwardPrompt').value,'');
assert.equal(el('threshold').value,String(saved.config.structured_gp.ei_after_unique_measured_designs));assert(!evaluate('settingsDirty()'));
// An intentional switch discards all drafts, including A -> B -> A.
el('forwardPrompt').value='A draft';el('forwardPrompt').oninput();evaluate("switchCampaign('b')");assert.equal(evaluate('state'),null);assert(el('active').classList.contains('hidden'));show(campaign('b','moc_gp'));
assert.equal(el('forwardPrompt').value,'');assert.equal(el('threshold').value,'10');
el('inverseModel').value='B draft';evaluate("switchCampaign('a')");show(saved);assert.equal(el('inverseModel').value,saved.config.llm.inverse_model);
""",
    )


def test_focused_settings_save_and_request_order_protect_drafts(tmp_path):
    run_focused(
        tmp_path,
        r"""
const original=campaign();show(original);
el('manualTarget').value='';el('manualTarget').oninput();el('forwardPrompt').value='Saved custom';el('forwardPrompt').oninput();
let oldReply,saveReply,latest=clone(original);
response=async(action,body)=>{
 if(action==='config-preview')return {after:{...latest.config,...body.changes}};
 if(action==='config')return new Promise(resolve=>{saveReply=resolve;latest=clone(original);Object.assign(latest.config.llm,body.changes.llm);});
 if(action.startsWith('state?'))return new Promise(resolve=>{oldReply=resolve;});
 throw Error(action);
};
const oldPoll=context.load();await settle();await el('previewSettings').onclick();
assert(!el('applySettings').classList.contains('hidden'));
const save=el('applySettings').onclick();await settle();
assert.equal(calls.find(x=>x.action==='config').body.changes.llm.manual_inverse_target,null);
response=async()=>latest;saveReply(latest);await save;oldReply(original);await oldPoll;
assert.equal(el('manualTarget').value,'');assert.equal(el('forwardPrompt').value,'Saved custom');assert(!evaluate('settingsDirty()'));
assert.equal(evaluate('state.config.llm.manual_inverse_target'),null);
// A second edit made during save remains a draft after the accepted first edit.
el('forwardPrompt').value='First edit';el('forwardPrompt').oninput();response=async(action,body)=>{
 if(action==='config-preview')return {after:latest.config};
 if(action==='config')return new Promise(resolve=>{saveReply=resolve;latest=clone(latest);Object.assign(latest.config.llm,body.changes.llm);});
 return latest;
};
await el('previewSettings').onclick();const saving=el('applySettings').onclick();await settle();
el('forwardPrompt').value='New draft during save';el('forwardPrompt').oninput();saveReply(latest);await saving;
assert.equal(el('forwardPrompt').value,'New draft during save');assert(evaluate('settingsDirty()'));
// An outdated preview cannot expose Apply after a new input.
let releasePreview;response=async()=>new Promise(resolve=>{releasePreview=resolve;});
const preview=el('previewSettings').onclick();await settle();el('inversePrompt').value='Later input';el('inversePrompt').oninput();releasePreview({after:latest.config});await preview;
assert(el('applySettings').classList.contains('hidden'));
// A -> B -> A during a read cannot accept the old A snapshot.
const delayed=context.load();await settle();const releaseA=releasePreview;
evaluate("switchCampaign('b');switchCampaign('a')");show(latest);releaseA(original);await delayed;
assert.equal(evaluate('state.config.llm.forward_system_message'),'First edit');
""",
    )


def test_focused_resolved_presets_defaults_explicit_overrides_and_reset(tmp_path):
    run_focused(
        tmp_path,
        r"""
el('preset').value='moc_five_gp';await context.loadPresets();
assert.equal(el('budget').value,'5');assert(el('creationPresetLabel').textContent.includes('1.0.0'));
let selected=JSON.parse(el('creationConfig').textContent);assert.equal(selected.config.structured_gp.ei_after_unique_measured_designs,3);
assert.equal(selected.config.new_measurement_budget,5);assert.equal(selected.config.auto_suggest,false);
const baseResponse=response;let created;
response=async(action,body)=>{
 if(action==='create'||action==='pair'){created=body;return {campaign_id:'new',gp:'new',llm:'other'};}
 if(action==='list')return {campaigns:[{campaign_id:'new',name:'New',data_schema:'moc'}]};
 if(action.startsWith('state?'))return campaign('new',body.preset||'moc_five_gp');
 return baseResponse(action,body);
};
await context.create(false);assert.equal(created.preset,'moc_five_gp');assert(!Object.hasOwn(created,'budget'),'Untouched default is not replaced by null');
await context.create(true);assert.equal(created.study,'five_point');assert(!Object.hasOwn(created,'budget'));
el('budget').value='0';el('budget').oninput();await el('budget').onchange();await context.create(true);assert.equal(created.budget,0);
el('budget').value='';el('budget').oninput();await el('budget').onchange();await context.create(false);assert.equal(created.budget,null,'Deliberate unlimited override survives');
el('preset').value='moc_gp';await el('preset').onchange();selected=JSON.parse(el('creationConfig').textContent);assert.equal(selected.config.structured_gp.ei_after_unique_measured_designs,10);assert.equal(el('budget').value,'');
el('preset').value='moc_five_llm';await el('preset').onchange();selected=JSON.parse(el('creationConfig').textContent);assert.equal(el('budget').value,'5');assert.equal(selected.config.llm.forward_model,'gpt-4o');assert.equal(selected.config.llm.forward_system_message,null);
show(campaign('new','moc_gp'));el('resetPreset').value='moc_five_gp';await el('previewPresetReset').onclick();assert.equal(JSON.parse(el('configPreview').textContent).preset,'moc_five_gp');
let resetCalls=0;response=async(action,body)=>{
 if(action==='preset-apply'){resetCalls++;assert.deepEqual(body,{id:'new',preset:'moc_five_gp'});return campaign('new','moc_five_gp');}
 return campaign('new','moc_five_gp');
};
assert.equal(resetCalls,0,'Preview never writes');await el('applySettings').onclick();assert.equal(resetCalls,1);assert.equal(el('threshold').value,'3');
assert(el('savedPresetLabel').textContent.includes('1.0.0'));
""",
    )


def test_focused_stale_reads_switch_save_and_failed_save_are_isolated(tmp_path):
    run_focused(
        tmp_path,
        r"""
const a=campaign();show(a);let releases=[];
response=async()=>new Promise(resolve=>releases.push(resolve));
const older=context.load(),newer=context.load();await settle();
const fresh=clone(a);fresh.config.llm.manual_inverse_target=70;
releases[1](fresh);await newer;releases[0](a);await older;
assert.equal(el('manualTarget').value,'70','Older same-campaign response cannot overwrite newer state');
el('manualTarget').value='0';el('manualTarget').oninput();
response=async(action)=>{if(action==='config-preview')return {after:fresh.config};throw Error('Save rejected');};
await el('previewSettings').onclick();await el('applySettings').onclick();
assert.equal(el('manualTarget').value,'0');assert(evaluate('settingsDirty()'));assert(el('notice').textContent.includes('Save rejected'));
let finish;response=async(action)=>action==='config-preview'?{after:fresh.config}:new Promise(resolve=>{finish=resolve;});
await el('previewSettings').onclick();const saving=el('applySettings').onclick();await settle();
evaluate("switchCampaign('b')");show(campaign('b'));el('forwardPrompt').value='B owns this draft';el('forwardPrompt').oninput();
finish(fresh);await saving;assert.equal(el('forwardPrompt').value,'B owns this draft');assert.equal(evaluate('state.campaign_id'),'b');
""",
    )


def test_focused_preset_preview_ignores_outdated_selection_response(tmp_path):
    run_focused(
        tmp_path,
        r"""
let replies=[];response=async(action,body)=>new Promise(resolve=>replies.push(()=>resolve(clone(fixtures.resolved[body.preset]))));
el('preset').value='moc_gp';const first=context.previewCreation(true);await settle();
el('preset').value='moc_five_gp';const second=context.previewCreation(true);await settle();
for(const reply of replies.slice(3))reply();await second;
for(const reply of replies.slice(0,3))reply();await first;
const result=JSON.parse(el('creationConfig').textContent);assert.equal(result.preset,'moc_five_gp');assert.equal(el('budget').value,'5');
assert.equal(result.config.structured_gp.ei_after_unique_measured_designs,3);
""",
    )


def test_focused_discard_invalidates_pending_unchanged_preview(tmp_path):
    run_focused(
        tmp_path,
        r"""
show(campaign());let finish;
response=async()=>new Promise(resolve=>{finish=resolve;});
const preview=el('previewSettings').onclick();await settle();
el('discardSettings').onclick();finish({after:campaign().config});await preview;
assert(el('applySettings').classList.contains('hidden'));assert.equal(el('configPreview').textContent,'');
""",
    )


@pytest.mark.parametrize("paired", [False, True])
def test_focused_creation_latch_covers_preview_file_and_post(tmp_path, paired):
    run_focused(
        tmp_path,
        r"""
show(campaign());el('preset').value='moc_five_gp';
let previewReplies=[],fileReply,postReply,created=0;
el('workbook').files=[{arrayBuffer:()=>new Promise(resolve=>{fileReply=resolve;})}];
context.btoa=s=>Buffer.from(s).toString('base64');
response=async(action,body)=>{
 if(action==='preset-preview')return new Promise(resolve=>previewReplies.push(()=>resolve(clone(fixtures.resolved[body.preset]))));
 if(action==='create'||action==='pair'){created++;return new Promise(resolve=>{postReply=resolve;});}
 throw Error('Unexpected '+action);
};
const paired=PAIRED;
const pending=context.create(paired);await settle();
assert(el('create').disabled&&el('pair').disabled);
await assert.rejects(()=>context.create(!paired),/already in progress/);
assert.equal(previewReplies.length,3);
for(const reply of previewReplies)reply();await settle();
assert(fileReply);assert(el('create').disabled&&el('pair').disabled,'Preset validation cannot re-enable creation while reading workbook');
await assert.rejects(()=>context.create(paired),/already in progress/);
fileReply(new Uint8Array([1,2,3]).buffer);await settle();
assert.equal(created,1);assert(el('create').disabled&&el('pair').disabled);
await assert.rejects(()=>context.create(paired),/already in progress/);
// Do not navigate away from a campaign the user deliberately opened during POST.
evaluate("switchCampaign('b')");show(campaign('b'));
postReply({campaign_id:'new',gp:'new',llm:'second'});await pending;
assert.equal(evaluate('current'),'b');assert.equal(evaluate('state.campaign_id'),'b');
assert.equal(created,1);assert(!el('create').disabled&&!el('pair').disabled);
// A failed subsequent request releases the latch for retry.
el('workbook').files=[];response=async()=>{throw Error('Validation unavailable');};
await assert.rejects(()=>context.create(paired),/Validation unavailable/);
assert.equal(evaluate('creationBusy'),false);
""".replace(
            "PAIRED", json.dumps(paired)
        ),
    )


def test_focused_paid_and_cache_actions_require_clean_completed_settings(tmp_path):
    run_focused(
        tmp_path,
        r"""
const saved=campaign();show(saved);
const actionIds=['generateInverse','cachePrepare','cacheImport','suggest'];
el('forwardModel').value='Draft model';el('forwardModel').oninput();
for(const id of actionIds){assert(el(id).disabled);await el(id).onclick();}
assert.equal(calls.length,0,'Dirty drafts must prevent all model/cache requests');
assert(el('notice').textContent.includes('Apply reviewed settings or discard'));
el('discardSettings').onclick();
let finishSave;
response=async(action,body)=>{
 if(action==='config-preview')return {after:saved.config};
 if(action==='config')return new Promise(resolve=>{finishSave=resolve;});
 if(action.startsWith('state?'))return saved;
 throw Error(action);
};
// A no-op reviewed save still has an in-flight boundary that actions must respect.
await el('previewSettings').onclick();const saving=el('applySettings').onclick();await settle();
const before=calls.length;
for(const id of actionIds){assert(el(id).disabled);await el(id).onclick();}
assert.equal(calls.length,before,'In-flight save must finish before any action can use its settings');
assert(el('notice').textContent.includes('Wait for the settings save'));
finishSave(saved);await saving;
for(const id of ['generateInverse','cachePrepare','cacheImport'])assert(!el(id).disabled);
assert.equal(evaluate('settingsSavePending'),false);
""",
    )


@pytest.mark.parametrize(
    "saved_mode,saved_count,selected_mode,expected_count",
    [
        ("all", None, "nearest", 5),
        ("nearest", 7, "nearest", 7),
        ("nearest", 7, "all", None),
    ],
)
def test_focused_example_mode_payload_passes_service_preview_and_save(
    tmp_path, saved_mode, saved_count, selected_mode, expected_count
):
    from boicl.campaign import CampaignService

    case = json.dumps(
        {
            "saved_mode": saved_mode,
            "saved_count": saved_count,
            "selected_mode": selected_mode,
        }
    )
    output = run_focused(
        tmp_path,
        r"""
const testcase=TESTCASE,initial=campaign();
initial.config.llm.selector_mode=testcase.saved_mode;initial.config.llm.selector_k=testcase.saved_count;
show(initial);el('examples').value=testcase.selected_mode;el('examples').onchange();
let changes;
response=async(action,body)=>{
 assert.equal(action,'config-preview');changes=body.changes;
 return {after:initial.config};
};
await el('previewSettings').onclick();assert(changes);assert(!el('applySettings').classList.contains('hidden'));
console.log(JSON.stringify(changes));
""".replace(
            "TESTCASE", case
        ),
    )
    changes = json.loads(output)
    service = CampaignService(tmp_path / "campaigns")
    cid = service.create(
        "moc_five_llm",
        overrides={
            "llm": {
                "selector_mode": saved_mode,
                "selector_k": saved_count,
                "manual_inverse_target": 90,
            }
        },
        synthetic_demo=True,
    )
    observations = service.get(cid)["observations"]
    preview = service.update_config(cid, changes, apply=False)
    assert preview["after"]["llm"]["selector_mode"] == selected_mode
    assert preview["after"]["llm"]["selector_k"] == expected_count
    assert service.get(cid)["config"]["llm"]["selector_mode"] == saved_mode
    saved = service.update_config(cid, changes)
    assert saved["config"]["llm"]["selector_mode"] == selected_mode
    assert saved["config"]["llm"]["selector_k"] == expected_count
    assert service.get(cid)["observations"] == observations
    resumed = CampaignService(tmp_path / "campaigns")
    assert resumed.get(cid)["config"]["llm"]["selector_k"] == expected_count
