import {openDB,read,write,all,keys,getMeta,setMeta,addActivity} from './db.js';
import {ByteBPE} from './bpe.js';
import {TinyTransformer,CONFIG} from './transformer.js';
let model=null,tokenizer=null,active=false,paused=false,stopped=true,working=false,chatPriority=false;
let config={learningRate:.001,batchSize:1,stepsPerArticle:2,throttleMs:400,storageMB:80};
let stats={step:0,tokensProcessed:0,trainLoss:null,validationLoss:null,perplexity:null,accuracy:null,gradientNorm:null,learningRate:.001,parameters:0,modelBytes:0,trainedArticles:0,bestValidationLoss:null,history:[]};
let catalogs=new Map(), catalogKeys=new Set(), recent=[],trainedIds=new Set(),plateau=0,checkpointTimer=0,setupInFlight=null;
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const msg=(type,payload={})=>postMessage({type,...payload});
async function activity(message,type='info'){await addActivity(message,type);msg('activity',{message,kind:type,at:Date.now()});}
async function reloadCatalog() {
  catalogs=new Map();catalogKeys=new Set();
  const ids=await keys('samples',50000);
  // Only the id metadata from existing samples is needed, and only changed data gets loaded.
  for(const id of ids){const record=await read('samples',id);if(!record)continue;
    catalogKeys.add(id);const subject=record.subject||'Uncategorized';
    if(!catalogs.has(subject))catalogs.set(subject,[]);
    catalogs.get(subject).push(id);
  }
}
async function refreshOne(id) {
  if(catalogKeys.has(id))return;
  const record=await read('samples',id);if(!record)return;
  catalogKeys.add(id);
  if(!catalogs.has(record.subject))catalogs.set(record.subject,[]);
  catalogs.get(record.subject).push(id);
  recent.push(id);if(recent.length>70)recent.shift();
}
async function setupCore() {
  if(model)return true;
  const saved=await getMeta('tokenizer');
  if(!saved)return false;
  tokenizer=new ByteBPE(saved.merges);
  const latest=await read('checkpoints','latest');
  model=new TinyTransformer(tokenizer.vocabSize,latest?.value||null);
  const old=await getMeta('trainingStats');
  if(old){stats={...stats,...old};stats.step=model.step;}
  stats.parameters=model.parameters;stats.modelBytes=model.parameters*4*3;
  await reloadCatalog();
  for(const id of catalogKeys){const article=await read('articles',id);if(article?.trainCount>0)trainedIds.add(id);}
  stats.trainedArticles=trainedIds.size;
  if(!latest)await saveCheckpoint('initial');
  else msg('checkpoint',{checkpoint:{...model.snapshot(),merges:tokenizer.merges}});
  msg('stats',{...stats});msg('ready',{modelReady:true});
  return true;
}
async function setup(){
  if(model)return true;
  if(setupInFlight)return setupInFlight;
  setupInFlight=setupCore();
  try{return await setupInFlight;}finally{setupInFlight=null;}
}
async function saveCheckpoint(reason='periodic') {
  if(!model)return;
  const value={...model.snapshot(),merges:tokenizer.merges,createdAt:Date.now(),reason};
  await write('checkpoints',{key:'latest',value});
  await setMeta('trainingStats',stats);
  msg('checkpoint',{checkpoint:value});
  if(reason!=='initial')await activity(`Checkpoint saved — step ${model.step} (${reason})`);
}
function chooseSampleId() {
  if(!catalogs.size)return null;
  if(recent.length&&Math.random()<.4)return recent[Math.floor(Math.random()*recent.length)];
  const subjects=[...catalogs.keys()];
  const chosen=subjects[Math.floor(Math.random()*subjects.length)];
  const entries=catalogs.get(chosen);
  return entries[Math.floor(Math.random()*entries.length)];
}
async function sample(isValidation=false) {
  if(!catalogs.size)return null;
  for(let attempt=0;attempt<35;attempt++){
    const id=chooseSampleId(),item=await read('samples',id);
    if(!item||!item.windows?.length||Boolean(item.validation)!==isValidation)continue;
    return {id,subject:item.subject,window:item.windows[Math.floor(Math.random()*item.windows.length)]};
  }
  return null;
}
async function evaluate() {
  const results=[];
  for(let i=0;i<5;i++){
    const item=await sample(true);
    if(!item)break;
    results.push(model.loss(item.window.slice(0,-1),item.window.slice(1)));
  }
  if(!results.length){stats.validationLoss=null;stats.perplexity=null;return;}
  stats.validationLoss=results.reduce((s,x)=>s+x.loss,0)/results.length;
  stats.perplexity=Math.exp(Math.min(50,stats.validationLoss));
  const oldBest=stats.bestValidationLoss;
  if(oldBest===null||stats.validationLoss<oldBest-0.001){
    plateau=0;stats.bestValidationLoss=stats.validationLoss;
    await write('checkpoints',{key:'best',value:{...model.snapshot(),merges:tokenizer.merges,createdAt:Date.now(),reason:'best validation'}});
    await activity(`New best validation loss: ${stats.validationLoss.toFixed(3)}`);
  }else if(++plateau>=3){
    plateau=0;stats.learningRate=Math.max(.00001,stats.learningRate*.7);
    await activity(`Validation plateau: learning rate reduced to ${stats.learningRate.toExponential(2)}`);
  }
}
async function cycle(){
  if(working)return;working=true;
  try{
    while(active&&!stopped){
      if(paused||chatPriority||!model){await sleep(300);continue;}
      const item=await sample(false);
      if(!item){msg('state',{state:'waiting for training documents'});await sleep(1800);continue;}
      const batch=[item];
      const cores=navigator.hardwareConcurrency||2;
      const targetBatch=globalThis.lowPower||chatPriority?1:Math.max(1,Math.min(3,Math.floor(cores/4)));
      for(let b=1;b<targetBatch;b++){const next=await sample(false);if(next)batch.push(next);}
      const began=performance.now();
      const measured=model.trainBatch(batch.map(x=>x.window),stats.learningRate);
      if(!Number.isFinite(measured.loss))throw new Error('Non-finite training loss; training paused');
      stats.step=model.step;
      stats.trainLoss=measured.loss;
      stats.accuracy=measured.accuracy;
      stats.gradientNorm=measured.gradient_norm;
      stats.tokensProcessed+=batch.reduce((total,x)=>total+x.window.length-1,0);
      let newlyTrained=false;
      for(const sampled of batch)if(!trainedIds.has(sampled.id)){
        newlyTrained=true;
        trainedIds.add(sampled.id);
        stats.trainedArticles++;
        const article=await read('articles',sampled.id);
        if(article){article.trainCount=(article.trainCount||0)+1;await write('articles',article);}
      }
      msg('stats',{...stats,computeMs:performance.now()-began});
      if(stats.step===1||stats.step%10===0)await activity(`Training: Step ${stats.step} · loss ${measured.loss.toFixed(3)}`);
      if(stats.step%20===0){await evaluate();msg('stats',{...stats});}
      if(stats.step%10===0){stats.history.push({step:stats.step,train_loss:stats.trainLoss,validation_loss:stats.validationLoss});if(stats.history.length>180)stats.history.shift();}
      if(newlyTrained){await saveCheckpoint('new article');checkpointTimer=Date.now();}
      else if(stats.step%5===0){await saveCheckpoint('periodic');checkpointTimer=Date.now();}
      // Yield after *every* actual gradient step, to keep chat & low-end devices responsive.
      const delay=Math.max(150,config.throttleMs)+(globalThis.lowPower?800:0);
      await sleep(delay);
    }
  }catch(e){paused=true;msg('error',{message:e.message});await activity(`Training paused: ${e.message}`,'error').catch(()=>{});}
  finally{working=false;if(model)await saveCheckpoint(stopped?'stopped':'paused').catch(e=>msg('error',{message:e.message}));}
}
onmessage=async event=>{
  const {type}=event.data;
  try{
    if(type==='init'){
      config={...config,...event.data.config};stats.learningRate=config.learningRate;
      await openDB();
      await setup();
      msg('ready',{modelReady:!!model});
    }else if(type==='tokenizer_ready'){
      if(!model)await setup();
    }else if(type==='article_ready'){
      await refreshOne(event.data.id);
    }else if(type==='start'||type==='resume'){
      config={...config,...event.data.config};active=true;paused=false;stopped=false;
      if(!model)await setup();
      msg('state',{state:model?'running':'waiting for tokenizer'});cycle();
    }else if(type==='pause'){
      paused=true;msg('state',{state:'paused'});
      if(!working&&model)await saveCheckpoint('paused');
    }else if(type==='stop'){
      stopped=true;active=false;paused=false;msg('state',{state:'stopped'});
      if(!working&&model)await saveCheckpoint('stopped');
    }else if(type==='chat_priority')chatPriority=Boolean(event.data.value);
    else if(type==='low_power')globalThis.lowPower=Boolean(event.data.value);
    else if(type==='config'){
      config={...config,...event.data.config};
      if(Number.isFinite(config.learningRate))stats.learningRate=config.learningRate;
    }
  }catch(e){msg('error',{message:e.message});}
};
