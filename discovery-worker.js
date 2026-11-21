import {openDB, read, write, all, count, getMeta, setMeta, addActivity} from './db.js';
import {ByteBPE, makeWindows} from './bpe.js';
const API='https://en.wikipedia.org/w/api.php';
let active=false,paused=false,stopped=true, aborter=null, exploring=false;
let config={storageMB:80, intervalMs:3200, context:48};
let stats={discovered:0,downloaded:0,trainedArticles:0,tokensIndexed:0,subject:'',article:'',articlesPerMinute:0,storageBytes:0};
let graph={categories:[],links:[],knownTitles:[],visitedCategories:{},seenSubjects:{}};
let tokenizer=null, waitMs=3200, downloadTimes=[];
const msg=(type,payload={})=>postMessage({type,...payload});
const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function record(message,type='info'){
  await addActivity(message,type);msg('activity',{message,kind:type,at:Date.now()});
}
async function fetchJSON(params) {
  aborter=new AbortController();
  const url=new URL(API);
  Object.entries({action:'query',format:'json',formatversion:'2',origin:'*',...params}).forEach(([k,v])=>url.searchParams.set(k,String(v)));
  const response=await fetch(url,{mode:'cors',cache:'no-store',signal:AbortSignal.any([aborter.signal,AbortSignal.timeout(14000)])});
  if(!response.ok) {const e=new Error(`Wikipedia API HTTP ${response.status}`);e.status=response.status;throw e;}
  const data=await response.json();
  if(data.error)throw new Error(`Wikipedia API: ${data.error.info||data.error.code}`);
  return data;
}
function choose(arr){return arr[Math.floor(Math.random()*arr.length)];}
function chooseFresh(pages){return choose(pages.filter(x=>!graph.knownTitles.includes(x.title)).length?pages.filter(x=>!graph.knownTitles.includes(x.title)):pages);}
function pushBounded(arr,item,max){if(!arr.includes(item))arr.push(item);if(arr.length>max)arr.shift();}
function considerCategory(name) {
  if(!name||/maintenance|stub articles|articles with|pages using|cs1|wikipedia|wikidata|citation|short description|dated |all articles|births|deaths|living people|use dmy|use mdy|template/i.test(name))return;
  if(name.length>115)return;
  pushBounded(graph.categories,name,1400);
}
async function discover() {
  // Dynamic graph exploration. Random discovery keeps the traversal from being trapped in
  // any preconfigured subject list; discovered links/categories expand the frontier.
  const pick=Math.random(), available=graph.categories.length+graph.links.length;
  if(pick<.30||available===0){
    const data=await fetchJSON({list:'random',rnnamespace:0,rnlimit:12});
    const pages=(data.query?.random||[]).filter(p=>p.ns===0&&p.title);
    if(!pages.length)throw new Error('Wikipedia returned no random articles');
    return {title:chooseFresh(pages).title,source:'random'};
  }
  if(pick<.78&&graph.categories.length) {
    const candidates=[...graph.categories].sort((a,b)=>(graph.visitedCategories[a]||0)-(graph.visitedCategories[b]||0));
    const category=choose(candidates.slice(0,Math.min(30,candidates.length)));
    graph.visitedCategories[category]=(graph.visitedCategories[category]||0)+1;
    const data=await fetchJSON({list:'categorymembers',cmtitle:`Category:${category}`,cmtype:'page|subcat',cmlimit:25});
    const pages=[];
    for(const item of data.query?.categorymembers||[]) {
      if(item.ns===14)considerCategory(item.title.replace(/^Category:/,''));
      if(item.ns===0)pages.push(item);
    }
    if(pages.length)return {title:chooseFresh(pages).title,source:`Category: ${category}`};
  }
  while(graph.links.length){
    const title=graph.links.splice(Math.floor(Math.random()*graph.links.length),1)[0];
    if(!graph.knownTitles.includes(title))return {title,source:'Article links'};
  }
  const random=await fetchJSON({list:'random',rnnamespace:0,rnlimit:10});
  return {title:chooseFresh(random.query.random).title,source:'random'};
}
async function loadArticle(title) {
  const data=await fetchJSON({prop:'extracts|categories|links',titles:title,explaintext:1,exsectionformat:'plain',exlimit:1,cllimit:35,pllimit:40});
  const page=data.query?.pages?.[0];
  if(!page||page.missing||!page.pageid||!page.extract)throw new Error('Article has no readable extract');
  const content=page.extract.replace(/\n{4,}/g,'\n\n').trim().slice(0,16000);
  if(content.length<260)throw new Error('Article too short for training');
  const categories=(page.categories||[]).map(x=>x.title.replace(/^Category:/,''));
  const filtered=categories.filter(x=>!/maintenance|stub|wikipedia|articles with|pages using|cs1|short description|living people|dated |births|deaths/i.test(x));
  const subject=filtered[0]||'Uncategorized';
  for(const category of filtered.slice(0,25))considerCategory(category);
  for(const item of page.links||[])if(item.ns===0&&item.title&&Math.random()<.45)pushBounded(graph.links,item.title,1000);
  return {id:page.pageid,title:page.title,subject,categories:filtered.slice(0,20),content,url:`https://en.wikipedia.org/wiki/${encodeURIComponent(page.title.replace(/ /g,'_'))}`,fetchedAt:Date.now(),status:'downloaded',tokenCount:0,trainCount:0};
}
async function persistGraph(){await setMeta('graph',graph);await setMeta('discoveryStats',stats);}
async function encodeArticle(article) {
  if(!tokenizer)return;
  const tokens=tokenizer.encode(article.content);
  const windows=makeWindows([...tokens,256],config.context,Math.max(8,config.context-3),160);
  if(!windows.length)return;
  await write('samples',{id:article.id,subject:article.subject,title:article.title,windows,tokenCount:tokens.length,validation:article.validation===true});
  article.tokenCount=tokens.length;article.status='indexed';
  await write('articles',article);
  stats.tokensIndexed+=tokens.length;
  await record(`Tokenizing: ${tokens.length.toLocaleString()} tokens — ${article.title}`);
  msg('article_ready',{id:article.id,title:article.title,tokens:tokens.length,subject:article.subject});
}
async function bootstrapTokenizer() {
  if(tokenizer)return;
  const existing=await all('articles',24);
  if(existing.length<3)return;
  await record('Fitting byte-pair tokenizer to the first Wikipedia documents');
  tokenizer=ByteBPE.fit(existing.slice(0,5).map(a=>a.content),94);
  await setMeta('tokenizer',{merges:tokenizer.merges,vocabSize:tokenizer.vocabSize});
  msg('tokenizer_ready',{vocabSize:tokenizer.vocabSize});
  await record(`Tokenizer fitted: ${tokenizer.vocabSize} token ids (${tokenizer.merges.length} merges)`);
  for(const article of existing)if(article.status!=='indexed')await encodeArticle(article);
}
async function checkCapacity() {
  const estimate=await navigator.storage?.estimate?.().catch(()=>null);
  if(estimate?.usage!==undefined)stats.storageBytes=estimate.usage;
  if(estimate?.quota && estimate.usage/estimate.quota>.90)throw new Error('Browser storage exceeds 90% of available quota; collection is paused');
  if(estimate?.usage && estimate.usage>config.storageMB*1024*1024)throw new Error(`Kira storage cap (${config.storageMB} MB) reached; existing data remains available for training`);
}
async function loop() {
  if(exploring)return;exploring=true;
  while(active&&!stopped) {
    if(paused){await sleep(450);continue;}
    if(!navigator.onLine){msg('error',{message:'Offline: Wikipedia discovery will retry when connected'});await sleep(9000);continue;}
    try {
      await checkCapacity();
      const found=await discover();
      if(graph.knownTitles.includes(found.title)){await sleep(config.intervalMs);continue;}
      stats.article=found.title;stats.subject=found.source;
      msg('stats',{...stats});
      await record(`Discovering: ${found.title}`);
      const article=await loadArticle(found.title);
      const existing=await read('articles',article.id);
      if(existing){waitMs=config.intervalMs;await sleep(waitMs);continue;}
      stats.discovered++;
      pushBounded(graph.knownTitles,article.title,10000);
      article.validation=stats.downloaded%5===4;
      graph.seenSubjects[article.subject]=(graph.seenSubjects[article.subject]||0)+1;
      stats.article=article.title;stats.subject=article.subject;
      await write('articles',article);
      stats.downloaded++;
      downloadTimes.push(Date.now());
      downloadTimes=downloadTimes.filter(x=>x>Date.now()-60000);
      stats.articlesPerMinute=downloadTimes.length;
      await record(`Article downloaded and indexed: ${article.title} — ${article.subject}`);
      if(!tokenizer)await bootstrapTokenizer();
      else await encodeArticle(article);
      await persistGraph();
      msg('stats',{...stats});
      waitMs=config.intervalMs;
    } catch(error) {
      if(error.name==='AbortError'){continue;}
      const fatal=/storage|quota/i.test(error.message);
      await record(error.message,fatal?'warning':'error').catch(()=>{});
      msg('error',{message:error.message,storage:fatal});
      if(fatal){paused=true;msg('paused_storage');}
      waitMs=Math.min(60000,Math.max(5000,waitMs*1.6));
    }
    await sleep(waitMs+(Math.random()*600));
  }
  exploring=false;
}
async function initialize(settings={}) {
  config={...config,...settings};
  await openDB();
  graph={...graph,...(await getMeta('graph',{}))};
  stats=await getMeta('discoveryStats',stats);
  const saved=await getMeta('tokenizer');
  if(saved){tokenizer=new ByteBPE(saved.merges);msg('tokenizer_ready',{vocabSize:tokenizer.vocabSize});}
  if(tokenizer){const articles=await all('articles',150);for(const article of articles)if(article.status!=='indexed')await encodeArticle(article);}
  msg('stats',{...stats});
  msg('ready');
}
onmessage=async event=>{
  const {type}=event.data;
  try {
    if(type==='init')await initialize(event.data.config);
    else if(type==='start'||type==='resume'){
      config={...config,...event.data.config};active=true;stopped=false;paused=false;waitMs=config.intervalMs;
      msg('state',{state:'running'});loop();
    } else if(type==='pause') {paused=true;aborter?.abort();msg('state',{state:'paused'});}
    else if(type==='stop'){stopped=true;active=false;paused=false;aborter?.abort();await persistGraph();msg('state',{state:'stopped'});}
    else if(type==='config')config={...config,...event.data.config};
  }catch(e){msg('error',{message:e.message});}
};
