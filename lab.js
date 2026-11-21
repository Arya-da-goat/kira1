import {api,subscribe,start,pause,resume,stop,setSettings,snapshot} from './controller.js';
import {plotLoss} from './chart.js';
export {api};
const $=id=>document.getElementById(id);
let latest=snapshot(),lastRendered='',callback=null;
const fmt=n=>Number.isFinite(n)?Number(n).toLocaleString():'—';
const num=(n,d=3)=>Number.isFinite(n)?Number(n).toFixed(d):'—';
function facts(details){
  const el=$('trainingMetrics');el.replaceChildren();
  for(const [title,value] of Object.entries(details)){
    const div=document.createElement('div');div.className='fact';
    const dt=document.createElement('dt'),dd=document.createElement('dd');
    dt.textContent=title;dd.textContent=String(value);div.append(dt,dd);el.append(div);
  }
}
function render(state){
  latest=state;
  const d=state.discovery,t=state.training;
  $('trainingStatus').textContent=`${state.status.toUpperCase()} · ${state.modelReady?'Transformer ready':'Building tokenizer / Transformer'} · ${state.backend}${state.error?' · '+state.error:''}`;
  const current={
    'Current Wikipedia article':d.article||'Waiting for discovery',
    'Subject/category':d.subject||'Not discovered',
    'Articles discovered':fmt(d.discovered),
    'Articles downloaded':fmt(d.downloaded),
    'Articles used in training':fmt(t.trainedArticles),
    'Article tokens indexed':fmt(d.tokensIndexed),
    'Training tokens processed':fmt(t.tokensProcessed),
    'Gradient steps':fmt(t.step),
    'Training loss':num(t.trainLoss,4),
    'Validation loss':num(t.validationLoss,4),
    'Perplexity':num(t.perplexity,2),
    'Parameters':fmt(t.parameters),
    'Learning rate':t.learningRate?Number(t.learningRate).toExponential(2):'—',
    'Model + optimizer':t.modelBytes?`${(t.modelBytes/1024).toFixed(1)} KiB`:'—',
    'Site storage estimate':`${(state.storageUsed/1048576).toFixed(1)} MB`,
    'Articles/minute':fmt(d.articlesPerMinute)
  };
  const key=JSON.stringify(current);
  if(key!==lastRendered){lastRendered=key;facts(current);}
  $('trainingRuns').replaceChildren();
  for(const record of state.activity.slice(0,35)){
    const entry=document.createElement('p');entry.className='feed-entry';
    const time=document.createElement('time');time.textContent=new Date(record.at).toLocaleTimeString();
    entry.append(time,document.createTextNode(` ${record.message}`));
    if(record.kind==='error'||record.kind==='warning')entry.classList.add('feed-error');
    $('trainingRuns').append(entry);
  }
  plotLoss(t.history||[],$('trainingChart'),$('trainingLegend'));
  const blocked=!state.leader;
  $('autoStart').disabled=blocked||state.active;
  $('autoPause').disabled=blocked||!state.active;
  $('autoResume').disabled=blocked||state.active||!state.enabled;
  $('autoStop').disabled=blocked||(!state.enabled&&!state.active);
  $('connectionStatus').textContent=`Engine: ${state.status}. Transformer: ${state.modelReady?'ready':'initializing'}. ${(state.storageUsed/1048576).toFixed(1)} MB site storage used.`;
}
export function initLab(refreshModel,isGenerating){
  callback=refreshModel;
  $('openConnection').onclick=()=>$('connectionDialog').showModal();
  $('closeConnection').onclick=()=>$('connectionDialog').close();
  $('openTraining').onclick=()=>$('trainingDialog').showModal();
  $('closeTraining').onclick=()=>$('trainingDialog').close();
  const handle=async action=>{
    try{$('trainingError').textContent='';await action();await refreshModel();}
    catch(e){$('trainingError').textContent=e.message;}
  };
  $('autoStart').onclick=()=>handle(start);
  $('autoPause').onclick=()=>handle(pause);
  $('autoResume').onclick=()=>handle(resume);
  $('autoStop').onclick=()=>handle(stop);
  $('trainingOptions').onsubmit=event=>{
    event.preventDefault();
    handle(()=>setSettings({storageMB:Number($('autoStorage').value),intervalMs:Number($('autoInterval').value),
      throttleMs:Number($('autoThrottle').value),learningRate:Number($('autoRate').value)}));
  };
  const settings=latest.settings||{};
  $('autoStorage').value=settings.storageMB||80;
  $('autoInterval').value=settings.intervalMs||3200;
  $('autoThrottle').value=settings.throttleMs||400;
  $('autoRate').value=settings.learningRate||.001;
  subscribe(state=>{
    render(state);
    // Keep the original status pill and chat controls in sync without rebuilding messages.
    if(!isGenerating()&&(!window.__kiraStatusTs||Date.now()-window.__kiraStatusTs>3200)){
      window.__kiraStatusTs=Date.now();refreshModel();
    }
  });
  refreshModel();
}
