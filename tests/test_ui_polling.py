"""DOM event and request-order regression for shared settings drafts."""
from pathlib import Path
import json
import re
import shutil
import subprocess

import pytest

from boicl.local_app import INDEX_HTML
from boicl.moc_ui import MOC_HTML


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
def test_saved_campaign_picker_preserves_selection_until_active_campaign_changes(
    tmp_path,
):
    source = re.search(
        r"    function renderCampaigns\(\) \{.*?(?=\n    function renderConfig)",
        INDEX_HTML,
        re.S,
    ).group()
    path = tmp_path / "campaign-picker.js"
    path.write_text(source, encoding="utf-8")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
for(const prefix of ['shared:','']){
 const picker={value:'',dataset:{},set innerHTML(value){this.markup=value;this.value='';}},name={value:''};
 const campaigns=['a','b','c'].map(id=>({id:prefix+id,name:'Campaign '+id}));
 const context={$:id=>id==='savedCampaign'?picker:name,escapeHtml:String,state:{campaign:campaigns[0],campaigns}};
 vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
 context.renderCampaigns();assert.equal(picker.value,prefix+'a');
 picker.value=prefix+'b';context.renderCampaigns();assert.equal(picker.value,prefix+'b','Poll A must preserve the selection of B');
 assert.equal(context.state.campaign.id,prefix+'a','Selection alone must not load B');
 context.state.campaign=campaigns[1];context.renderCampaigns();assert.equal(picker.value,prefix+'b','Loaded B must be selected');
 picker.value=prefix+'a';context.state.campaign=campaigns[2];context.renderCampaigns();assert.equal(picker.value,prefix+'c','Switch to C replaces the former draft');
 picker.value=prefix+'b';context.state.campaigns=[campaigns[0],campaigns[2]];context.renderCampaigns();assert.equal(picker.value,prefix+'c','Removed draft selection falls back to the active campaign');
 picker.value='';context.renderCampaigns();assert.equal(picker.value,'','An intentionally cleared choice stays cleared during polling');
 context.state.campaign={};context.renderCampaigns();assert.equal(picker.value,'','An unsaved campaign has no stale saved selection');
}
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
def test_shared_comparison_controls_explain_hidden_saved_random_arm(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "boicl/toolkit_main.js").read_text(
        encoding="utf-8"
    )
    source = source[
        source.index("function sharedHiddenCurves()") : source.index(
            "function renderShared()"
        )
    ]
    path = tmp_path / "comparison-controls.js"
    path.write_text(source, encoding="utf-8")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
function exercise(graphOnly){
 const choices={innerHTML:''},storage=new Map();let plotRenders=0,tableRenders=0,checkboxes=[];
 const context={graphOnly,sharedCampaignId:'parent',$:()=>choices,
  escapeHtml:s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),
  localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v)},
  renderPlot:()=>plotRenders++,renderBenchmarkRuns:()=>tableRenders++,
  document:{querySelectorAll:()=>{checkboxes=[...choices.innerHTML.matchAll(/data-curve="([^"]+)"/g)].map(m=>({dataset:{curve:m[1]},checked:true}));return checkboxes;}},
  state:{benchmark_runs:[],shared_control_id:'saved-control',live_random_walk:{comparison_compatibility:{compatible:true}}}};
 vm.createContext(context);vm.runInContext(source,context);
 context.renderSharedComparisons();assert(choices.innerHTML.includes('data-curve="random-control"'));
 checkboxes[0].checked=false;checkboxes[0].onchange();assert.equal(plotRenders,1);assert.equal(tableRenders,1);
 assert.deepEqual(JSON.parse(storage.get('boicl_plot_visibility:parent')),['random-control']);
 context.state.live_random_walk.comparison_compatibility={compatible:false};
 context.state.comparison_diagnostics=[{campaign_id:'saved-control',kind:'random_control',compatible:false,mismatches:['effective_initialization','measurement_definition']}];
 const before=JSON.stringify(context.state);context.renderSharedComparisons();
 assert(!choices.innerHTML.includes('data-curve="random-control"'));
 assert(choices.innerHTML.includes('Independent random control remains saved'));
 assert(choices.innerHTML.includes('graph is hidden because the campaigns differ'));
 assert(choices.innerHTML.includes('initial training cohort, measurement definition'));
 assert(choices.innerHTML.includes('/?campaign=saved-control'));
 assert(!choices.innerHTML.includes('Create a matched pair'));assert.equal(JSON.stringify(context.state),before);
 const hiddenMarkup=choices.innerHTML;
 // Once compatibility returns, restore the checkbox while preserving user visibility settings.
 context.state.live_random_walk.comparison_compatibility={compatible:true};context.state.comparison_diagnostics=[];
 context.renderSharedComparisons();assert(choices.innerHTML.includes('data-curve="random-control"'));
 assert(!choices.innerHTML.includes('remains saved'));assert(!choices.innerHTML.includes(' checked'));
 // Diagnostic strings remain inert even when imported from a campaign bundle.
 context.state.comparison_diagnostics=[{campaign_id:'other&arm',mismatches:['<script>bad</script>']}];
 context.renderSharedComparisons();assert(!choices.innerHTML.includes('<script>'));assert(choices.innerHTML.includes('&lt;script&gt;'));
 return hiddenMarkup;
}
assert.equal(exercise(false),exercise(true)); // Main and focused graph iframe use this same renderer.
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
@pytest.mark.parametrize("view", ["main", "focused"])
def test_preview_controls_distinguish_bo_and_standalone_inverse_requests(
    tmp_path, view
):
    for page in (INDEX_HTML, MOC_HTML):
        assert '<option value="bo_inverse">Next BO inverse request</option>' in page
        assert (
            '<option value="standalone_inverse">Standalone inverse proposal</option>'
            in page
        )
    if view == "main":
        source = (
            Path(__file__).resolve().parents[1] / "boicl/toolkit_main.js"
        ).read_text(encoding="utf-8")
        handler = re.search(
            r'  \$\("requestPreview"\)\.onclick = async \(\) => \{.*?\n  \};',
            source,
            re.S,
        ).group()
    else:
        handler = next(
            line
            for line in MOC_HTML.splitlines()
            if "$('requestPreview').onclick=" in line
        )
    path = tmp_path / "preview-ui.js"
    path.write_text(handler, encoding="utf-8")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map(),calls=[];const el=id=>{if(!elements.has(id))elements.set(id,{value:'',textContent:''});return elements.get(id);};
const reply=async(path,body)=>{calls.push({path,body});return {status:'exact',preview_kind:body.role};};
const context={$:el,current:'campaign',sharedCampaignId:'campaign',guarded:fn=>fn(),api:reply,toolkitFetch:reply,renderError(m){throw Error(m);}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
(async()=>{
 el('requestPreviewCandidate').value='selected-candidate';el('replayStep').value=el('sharedReplayStep').value='recorded-id';
 for(const role of ['forward','bo_inverse','standalone_inverse']){
  el('requestPreviewRole').value=role;
  for(const source of ['current','recorded']){
   el('requestPreviewSource').value=source;await el('requestPreview').onclick();
   const {path,body}=calls.at(-1);assert(path.endsWith('request-preview'));
   assert.equal(body.id,'campaign');assert.equal(body.role,role);
   assert.equal(body.suggestion_id,source==='recorded'?'recorded-id':null);
   assert.equal(body.candidate_id,source==='recorded'?null:'selected-candidate');
   assert.equal(JSON.parse(el('requestPreviewResult').textContent).preview_kind,role);
  }
 }
 assert.equal(calls.length,6); // Preview does not save settings or launch a job.
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
@pytest.mark.parametrize("view", ["main", "focused"])
def test_definition_policy_restores_saved_state_without_overwriting_drafts(
    tmp_path, view
):
    from boicl.campaign import CampaignService

    service = CampaignService(tmp_path / "campaigns")
    cid = service.create_generic(
        [
            {"candidate_id": "a", "procedure": "Synthetic A", "x": 0, "value": 1},
            {"candidate_id": "b", "procedure": "Synthetic B", "x": 1, "value": None},
        ],
        [{"column": "x"}],
        synthetic_demo=True,
    )
    unresolved = service.summary(cid)["config"]["measurement_definition"]
    definition = {
        "quantification_method": "gsas_ii_mass_fraction",
        "normalization": "reported phases",
    }
    service.revise_measurement_definition(
        cid, definition, "retain_with_justification", "Synthetic justified retention"
    )
    # The UI consumes configuration restored from disk, not a hand-built UI fixture.
    restored = CampaignService(tmp_path / "campaigns")
    retained = restored.summary(cid)["config"]["measurement_definition"]
    restored.revise_measurement_definition(
        cid, definition, "exclude", "Synthetic exclusion"
    )
    excluded = restored.summary(cid)["config"]["measurement_definition"]
    fixture = tmp_path / "definitions.json"
    fixture.write_text(
        json.dumps(
            {"retained": retained, "excluded": excluded, "unresolved": unresolved}
        ),
        encoding="utf-8",
    )
    if view == "main":
        source = (
            Path(__file__).resolve().parents[1] / "boicl/toolkit_main.js"
        ).read_text(encoding="utf-8")
        assert (
            "syncDefinitionForm(config.measurement_definition || {}, sharedCampaignId)"
            in source
        )
        handler = re.search(
            r'  \$\("applyDefinition"\)\.onclick = async \(\) => \{.*?\n  \};',
            source,
            re.S,
        ).group()
    else:
        source = MOC_HTML
        assert "syncDefinitionForm(c.measurement_definition||{},current)" in source
        handler = next(
            line
            for line in source.splitlines()
            if "$('applyDefinition').onclick=" in line
        )
    helper = re.search(r"function syncDefinitionForm\(.*?\n\}", source, re.S).group()
    path = tmp_path / "definition-ui.js"
    path.write_text(helper + "\n" + handler, encoding="utf-8")
    for page in (INDEX_HTML, MOC_HTML):
        assert (
            '<option value="unresolved">Unresolved — choose a policy</option>' in page
        )
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8'),saved=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const elements=new Map();const el=id=>{if(!elements.has(id))elements.set(id,{value:'',dataset:{}});return elements.get(id);};
let submitted,releaseSave,refreshes=0;
const context={$:el,current:'a',sharedCampaignId:'a',guarded:fn=>fn(),notice(){},renderNotice(){},renderError(m){throw Error(m);},
 api:async(action,body)=>{submitted=body;return new Promise(resolve=>{releaseSave=resolve;});},
 toolkitFetch:async(path,body)=>{submitted=body;return new Promise(resolve=>{releaseSave=resolve;});}};
context.refresh=context.load=async()=>{refreshes++;context.syncDefinitionForm(saved.excluded,context.current);};
vm.createContext(context);vm.runInContext(source,context);
(async()=>{
 // Reloaded persisted retention must not look like an exclusion decision.
 context.syncDefinitionForm(saved.retained,'a');assert.equal(el('definitionHistoricalPolicy').value,'retain_with_justification');
 assert.equal(el('definitionNormalization').value,'reported phases');
 // A draft policy, normalization or reason survives a same-campaign poll.
 el('definitionHistoricalPolicy').value='exclude';el('definitionNormalization').value='unsaved basis';el('definitionReason').value='Draft scientific reason';
 context.syncDefinitionForm(saved.retained,'a');
 assert.equal(el('definitionHistoricalPolicy').value,'exclude');assert.equal(el('definitionNormalization').value,'unsaved basis');
 assert.equal(el('definitionReason').value,'Draft scientific reason');
 // Switching replaces the former campaign's draft with that campaign's saved policy.
 context.syncDefinitionForm(saved.excluded,'b');assert.equal(el('definitionHistoricalPolicy').value,'exclude');
 assert.equal(el('definitionNormalization').value,'reported phases');assert.equal(el('definitionReason').value,'');
 context.syncDefinitionForm(saved.retained,'a');assert.equal(el('definitionHistoricalPolicy').value,'retain_with_justification');
 // A revision saved in another view appears during polling when no local draft exists.
 context.syncDefinitionForm(saved.excluded,'a');assert.equal(el('definitionHistoricalPolicy').value,'exclude');
 context.syncDefinitionForm(saved.retained,'a');el('definitionReason').value='Reason alone is a draft';
 context.syncDefinitionForm(saved.excluded,'a');assert.equal(el('definitionHistoricalPolicy').value,'retain_with_justification');
 // The real decision handler submits the visible saved policy and resynchronizes after success.
 const pending=el('applyDefinition').onclick();assert.equal(submitted.historical_policy,'retain_with_justification');
 assert.equal(submitted.id,'a');releaseSave({});await pending;
 assert.equal(refreshes,1);assert.equal(el('definitionHistoricalPolicy').value,'exclude');assert.equal(el('definitionReason').value,'');
 // A campaign switch during that save must not reset the new campaign's draft.
 el('definitionReason').value='Another decision';const switched=el('applyDefinition').onclick();
 context.current=context.sharedCampaignId='b';context.syncDefinitionForm(saved.retained,'b');
 el('definitionHistoricalPolicy').value='exclude';el('definitionReason').value='New campaign draft';
 releaseSave({});await switched;assert.equal(refreshes,1);assert.equal(el('definitionReason').value,'New campaign draft');
 // An unresolved backend policy has an explicit choice, not an implied saved exclusion.
 context.syncDefinitionForm(saved.unresolved,'c');assert.equal(el('definitionHistoricalPolicy').value,'unresolved');
 assert.equal(el('definitionMethod').value,'');assert.equal(el('definitionReason').value,'');
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(path), str(fixture)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
def test_main_action_events_require_saved_settings_and_campaign_ownership(tmp_path):
    """Execute the real click handlers through their asynchronous request boundaries."""
    handlers = []
    for button in (
        "prepareEmbeddings",
        "inverseDesign",
        "runBenchmark",
        "clearAndRunBenchmark",
    ):
        handlers.append(
            re.search(
                rf"    \$\('{button}'\)\.addEventListener\('click', async \(\) => \{{.*?\n    \}}\);",
                INDEX_HTML,
                re.S,
            ).group()
        )
    handlers.append(
        re.search(
            r"    async function updateSuggestions\(\) \{.*?\n    \}",
            INDEX_HTML,
            re.S,
        ).group()
    )
    handlers.append("$('suggest').addEventListener('click', updateSuggestions);")
    path = tmp_path / "action-handlers.js"
    path.write_text("\n".join(handlers), encoding="utf-8")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const actions={prepareEmbeddings:['/api/precompute-embeddings'],inverseDesign:['/api/inverse-design'],
 runBenchmark:['/api/run-benchmark'],clearAndRunBenchmark:['/api/clear-benchmarks','/api/run-benchmark'],suggest:['/api/suggest']};
async function exercise(button,scenario,campaign='arm-a'){
 const elements=new Map(),calls=[];let resolveConfig,resolveClear;
 const el=id=>{if(!elements.has(id))elements.set(id,{value:id==='inverseTargetValue'?'0':'3',
  addEventListener(event,fn){this[event]=fn;}});return elements.get(id);};
 const context={sharedCampaignId:campaign,$:el,window:{confirm:()=>true},payloadConfig:()=>({inverse_target_value:'0'}),
  request:async(path,options)=>{calls.push({path,body:options.body?JSON.parse(options.body):null,campaign:context.sharedCampaignId});
   if(path==='/api/config')return new Promise(resolve=>{resolveConfig=resolve;});
   if(path==='/api/clear-benchmarks')return new Promise(resolve=>{resolveClear=resolve;});
   return {saved:true};}};
 vm.createContext(context);vm.runInContext(source,context);
 const pending=el(button).click();
 assert.deepEqual(calls.map(c=>c.path),['/api/config']);
 if(scenario==='switch-config')context.sharedCampaignId='arm-b';
 resolveConfig(scenario==='reject-config'?null:{saved:true});
 // Let the config continuation reach clear, but do not resolve clear yet.
 await new Promise(resolve=>setImmediate(resolve));
 if(resolveClear){
  assert.deepEqual(calls.map(c=>c.path),['/api/config','/api/clear-benchmarks']);
  if(scenario==='switch-clear')context.sharedCampaignId='arm-b';
  resolveClear(scenario==='reject-clear'?null:{cleared:true});
 }
 await pending;
 const expected=['/api/config'];
 if(!['reject-config','switch-config'].includes(scenario)){
  expected.push(actions[button][0]);
  if(button==='clearAndRunBenchmark'&&!['reject-clear','switch-clear'].includes(scenario))expected.push('/api/run-benchmark');
 }
 assert.deepEqual(calls.map(c=>c.path),expected,button+': '+scenario);
 assert(calls.every(c=>c.campaign===campaign),'No request may target a newly selected campaign');
 if(button==='inverseDesign'&&scenario==='success')assert.deepEqual(calls[1].body,{target_value:'0',count:'3'});
}
(async()=>{
 for(const button of Object.keys(actions)){
  for(const scenario of ['reject-config','switch-config','success'])await exercise(button,scenario);
  // Preserve the generic/legacy workflow whose shared identity is empty.
  await exercise(button,'success','');
 }
 for(const scenario of ['reject-clear','switch-clear'])await exercise('clearAndRunBenchmark',scenario);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
def test_focused_inverse_event_keeps_initiating_campaign_across_awaits(tmp_path):
    path = tmp_path / "focused-inverse-handler.js"
    path.write_text(
        next(
            line
            for line in MOC_HTML.splitlines()
            if "$('generateInverse').onclick=" in line
        ),
        encoding="utf-8",
    )
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
async function exercise(scenario){
 const elements=new Map(),calls=[],errors=[];let resolveConfig,rejectConfig,resolveProposal,loads=0;
 const el=id=>{if(!elements.has(id))elements.set(id,{value:id==='manualTarget'?'0':'3'});return elements.get(id);};
 const context={current:'arm-a',$:el,guarded:async fn=>{try{await fn();}catch(e){errors.push(e.message);}},
  api:async(action,body)=>{calls.push({action,body});
   if(action==='config')return new Promise((resolve,reject)=>{resolveConfig=resolve;rejectConfig=reject;});
   return new Promise(resolve=>{resolveProposal=resolve;});},load:async()=>{loads+=1;}};
 vm.createContext(context);vm.runInContext(source,context);
 const pending=el('generateInverse').onclick();
 assert.equal(calls.length,1);assert.equal(calls[0].body.id,'arm-a');
 assert.equal(calls[0].body.changes.llm.manual_inverse_target,0);
 assert.equal(calls[0].body.changes.llm.inverse_proposal_count,3);
 if(scenario==='switch-config')context.current='arm-b';
 if(scenario==='reject-config')rejectConfig(Error('Target outside bounds'));else resolveConfig({saved:true});
 await new Promise(resolve=>setImmediate(resolve));
 if(resolveProposal){
  assert.equal(calls[1].action,'inverse-proposal');assert.equal(calls[1].body.id,'arm-a');
  if(scenario==='switch-proposal')context.current='arm-b';
  resolveProposal({started:true});
 }
 await pending;
 assert.equal(calls.length,['reject-config','switch-config'].includes(scenario)?1:2);
 assert.equal(loads,scenario==='success'?1:0);
 assert.equal(errors.length,scenario==='reject-config'?1:0);
}
(async()=>{for(const scenario of ['reject-config','switch-config','switch-proposal','success'])await exercise(scenario);})()
 .catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node unavailable")
def test_blank_manual_target_survives_blur_poll_and_stale_pre_save_response(tmp_path):
    payload_source = re.search(
        r"function payloadConfig\(\).*?(?=\n    function latestPartialBenchmarkRun)",
        INDEX_HTML,
        re.S,
    ).group()
    payload_path = tmp_path / "payload-config.js"
    payload_path.write_text(payload_source, encoding="utf-8")
    main_path = Path(__file__).resolve().parents[1] / "boicl/toolkit_main.js"
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map(),listeners=new Map();
const el=id=>{if(!elements.has(id))elements.set(id,{value:'',checked:false,tagName:'INPUT',addEventListener(){},closest(){return el('toolkitSettings');}});return elements.get(id);};
let timer,submitted,releaseOld;
const initial={config:{inverse_target_value:0},shared_config:{engine:'llm',llm:{forward_system_message:null,inverse_system_message:null}},shared_campaign:{campaign_id:'arm'}};
const context={URLSearchParams,TextDecoder,location:{search:'?campaign=arm',pathname:'/'},state:initial,busy:false,
 document:{activeElement:null,addEventListener(n,fn){listeners.set(n,fn);},createElement(){return {};},head:{append(){}}},
 $:el,window:{},setTimeout(fn){timer=fn;return 1;},clearTimeout(){},renderPlot(){},renderProgress(){},renderShared(){},
 setBusy(v){context.busy=v;},renderError(m){throw Error(m);},render(){el('inverseTargetValue').value=context.state.config.inverse_target_value??'';}};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);listeners.get('DOMContentLoaded')();
context.renderShared=()=>{};
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
el('inverseTargetValue').value='0';
context.toolkitFetch=async()=>initial;
(async()=>{
 // Input becomes blank, then Apply takes focus before a poll response arrives.
 el('inverseTargetValue').value='';listeners.get('input')({target:el('inverseTargetValue')});
 context.document.activeElement={tagName:'BUTTON'};
 context.scheduleSharedPoll();await timer();
 assert.equal(el('inverseTargetValue').value,'');
 assert.equal(context.payloadConfig().inverse_target_value,'');
 // A pre-save poll cannot restore its old server value after successful Apply.
 context.toolkitFetch=async(path,payload)=>{
   if(path.startsWith('/api/toolkit/state'))return new Promise(resolve=>{releaseOld=resolve;});
   submitted=payload.values;
   return {...initial,config:{inverse_target_value:null},shared_config:{...initial.shared_config,llm:{...initial.shared_config.llm,manual_inverse_target:null}}};
 };
 context.scheduleSharedPoll();const pending=timer();
 await context.toolkitRequest('/api/config',{body:JSON.stringify(context.payloadConfig())});
 assert.equal(submitted.inverse_target_value,'');
 releaseOld(initial);await pending;
 assert.equal(el('inverseTargetValue').value,'');
 assert.equal(context.state.shared_config.llm.manual_inverse_target,null);
 assert.equal(vm.runInContext('sharedConfigDirty',context),false);
})().catch(error=>{console.error(error);process.exitCode=1;});
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(main_path), str(payload_path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
