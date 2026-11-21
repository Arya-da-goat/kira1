import {getMeta,setMeta,activities,read,openDB} from './db.js';
// A single controller owns both the discovery and training workers. Chat is served
// by an independent inference worker that only receives immutable weight snapshots.
const discovery=new Worker(new URL('./discovery-worker.js',import.meta.url),{type:'module'});
const trainer=new Worker(new URL('./training-worker.js',import.meta.url),{type:'module'});
const inference=new Worker(new URL('./inference-worker.js',import.meta.url),{type:'module'});
let state={ready:false,modelReady:false,status:'initializing',active:false,enabled:true,paused:false,
  discovery:{},training:{},activity:[],error:'',backend:'Pure JavaScript CPU',storageUsed:0,leader:true};
const listeners=new Set(),requests=new Map();let rpcId=0, initialized=false,releaseLock=null;
let settings={storageMB:80,intervalMs:3200,throttleMs:400,learningRate:.001};
const workerReady=new Set();let pendingAutostart=false;
const notify=()=>listeners.forEach(fn=>{try{fn({...state,settings:{...settings}});}catch(e){console.warn(e);}});
export const subscribe=fn=>{listeners.add(fn);fn({...state,settings:{...settings}});return()=>listeners.delete(fn);};
export function snapshot(){return {...state,settings:{...settings}};}
async function activity(message,kind='info',at=Date.now()) {
  state.activity=[{message,kind,at},...state.activity].slice(0,45);notify();
}
function capture(worker) {
  worker.onerror=event=>{state.error=`Worker error: ${event.message||'unknown'}`;activity(state.error,'error');};
  worker.onmessage=async({data})=>{
    const {type}=data;
    if(type==='error') {state.error=data.message;activity(data.message,'error');return;}
    if(type==='activity'){activity(data.message,data.kind,data.at);return;}
    if(type==='ready'&&(worker===discovery||worker===trainer)){
      workerReady.add(worker);
      if(pendingAutostart&&workerReady.size===2){pendingAutostart=false;run('start',false).catch(e=>activity(e.message,'error'));}
    }
    if(worker===discovery){
      if(type==='stats')state.discovery=data;
      else if(type==='tokenizer_ready')trainer.postMessage({type:'tokenizer_ready'});
      else if(type==='article_ready')trainer.postMessage({type:'article_ready',id:data.id});
      else if(type==='paused_storage') {state.status='paused (storage limit)';state.paused=true;trainer.postMessage({type:'pause'});}
      else if(type==='state'&&!state.paused)state.status=data.state;
    }else if(worker===trainer){
      if(type==='stats')state.training=data;
      else if(type==='checkpoint')inference.postMessage({type:'checkpoint',checkpoint:data.checkpoint});
      else if(type==='state'&&state.active)state.status=data.state;
    }else if(worker===inference){
      if(type==='ready'){state.ready=true;state.modelReady=Boolean(data.modelReady);state.training={...state.training,parameters:data.parameters,step:data.step};}
      if(type==='result'){
        const deferred=requests.get(data.id);if(deferred){requests.delete(data.id);data.error?deferred.reject(new Error(data.error)):deferred.resolve(data.response);}
      }
    }
    notify();
  };
}
capture(discovery);capture(trainer);capture(inference);
inference.postMessage({type:'init'});
async function acquireLock() {
  if(!navigator.locks)return true;
  return new Promise(resolve=>{
    navigator.locks.request('kira-continuous-training-v1',{ifAvailable:true},async lock=>{
      if(!lock){resolve(false);return;}
      resolve(true);await new Promise(release=>{releaseLock=release;});
    }).catch(e=>{console.error(e);resolve(false);});
  });
}
async function init(){
  try{
    await openDB();
    settings={...settings,...await getMeta('settings',{})};
    state.enabled=await getMeta('autonomyEnabled',true);
    state.activity=await activities(35);
    state.leader=await acquireLock();
    if(!state.leader){
      state.status='Learning active in another tab';state.paused=true;
      state.error='Another Kira tab owns the training lock; close it to enable training here.';
      const latest=await read('checkpoints','latest');
      if(latest?.value)inference.postMessage({type:'checkpoint',checkpoint:latest.value});
    }
    if(state.leader){
      discovery.postMessage({type:'init',config:settings});
      trainer.postMessage({type:'init',config:settings});
    }
    initialized=true;notify();
    // Restore the persisted setting, never discard weights on restart.
    if(state.enabled&&state.leader){
      if(workerReady.size===2)run('start',false);else pendingAutostart=true;
    } else state.status=state.leader?'stopped':'another tab active';
    refreshStorage();
  }catch(e){state.error=`Storage/worker initialization failed: ${e.message}`;state.status='error';notify();}
}
async function refreshStorage(){
  try{const info=await navigator.storage?.estimate?.();state.storageUsed=info?.usage||0;state.storageQuota=info?.quota||0;notify();}
  catch{}
}
export async function setSettings(next){
  settings={...settings,...next};
  settings.storageMB=Math.max(10,Math.min(500,Number(settings.storageMB)||80));
  settings.intervalMs=Math.max(2600,Math.min(20000,Number(settings.intervalMs)||3200));
  settings.throttleMs=Math.max(250,Math.min(15000,Number(settings.throttleMs)||400));
  settings.learningRate=Math.max(.00001,Math.min(.005,Number(settings.learningRate)||.001));
  await setMeta('settings',settings);
  discovery.postMessage({type:'config',config:settings});
  trainer.postMessage({type:'config',config:settings});
  notify();return settings;
}
async function run(command,persist=true){
  if(!initialized)throw new Error('The browser database is still opening.');
  if(!state.leader)throw new Error('Learning is owned by another Kira tab. Close that tab and reload.');
  if(persist)await setMeta('autonomyEnabled',command==='start'||command==='resume'||command==='pause');
  state.enabled=command==='start'||command==='resume'||command==='pause';
  state.paused=command==='pause';state.active=command==='start'||command==='resume';
  state.status=command==='pause'?'paused':command==='stop'?'stopped':'running';
  discovery.postMessage({type:command,config:settings});
  trainer.postMessage({type:command,config:settings});
  notify();
}
export const start=()=>run('start');
export const pause=()=>run('pause');
export const resume=()=>run('resume');
export const stop=()=>run('stop');
export async function api(path,body){
  if(path==='/api/status'){
    const {ready,active,training,status,discovery}=state;
    return {ready,model_ready:state.modelReady,training_active:active,busy:false,chat_mode:'wikipedia_grounded',detail:state.error||status,
      training_step:training.step||0,parameters:training.parameters||0,
      hardware:{device:state.backend},config:{max_seq_len:48,num_layers:1,num_attention_heads:2,num_kv_heads:2,ffn_type:'dense',tie_word_embeddings:false},
      memory_estimate:{formatted_total:`${((training.modelBytes||0)/1024).toFixed(0)} KiB weights + optimizer`},
      autonomy_enabled:state.enabled,discovery};
  }
  if(path==='/api/metrics')return {history:state.training.history||[]};
  if(path==='/api/chat'){
    if(!state.ready)throw new Error('The chat worker is still initializing. Please retry.');
    const id=++rpcId;
    trainer.postMessage({type:'chat_priority',value:true});
    try{return await new Promise((resolve,reject)=>{
      const timeout=setTimeout(()=>{requests.delete(id);reject(new Error('Local inference took too long. Reduce the token count.'));},90000);
      requests.set(id,{resolve:x=>{clearTimeout(timeout);resolve(x);},reject:e=>{clearTimeout(timeout);reject(e);}});
      inference.postMessage({type:'generate',id,request:body});
    });}finally{trainer.postMessage({type:'chat_priority',value:false});}
  }
  if(path==='/api/tools/search'){
    const search=new URL('https://en.wikipedia.org/w/api.php');
    for(const [k,v] of Object.entries({action:'query',list:'search',srsearch:String(body?.query||'').slice(0,300),srlimit:6,format:'json',origin:'*'}))search.searchParams.set(k,v);
    const result=await fetch(search,{signal:AbortSignal.timeout(16000)});
    if(!result.ok)throw new Error(`Wikipedia search HTTP ${result.status}`);
    const raw=await result.json();
    if(raw.error)throw new Error(raw.error.info||'Wikipedia search unavailable');
    const doc=new DOMParser();
    return {results:(raw.query?.search||[]).map(x=>({title:x.title,
      url:`https://en.wikipedia.org/wiki/${encodeURIComponent(x.title.replace(/ /g,'_'))}`,
      snippet:doc.parseFromString(x.snippet||'','text/html').body.textContent.slice(0,1200)}))};
  }
  throw new Error(`Unsupported local API route ${path}`);
}
if(typeof document!=='undefined'){
  document.addEventListener('visibilitychange',()=>trainer.postMessage({type:'low_power',value:document.hidden}));
  setInterval(refreshStorage,30000);
}
init();
