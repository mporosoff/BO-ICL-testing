"""DOM event and request-order regression for shared settings drafts."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from boicl.local_app import INDEX_HTML
from boicl.moc_ui import MOC_HTML


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
