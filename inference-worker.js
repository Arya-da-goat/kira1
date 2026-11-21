import {ByteBPE,EOS} from './bpe.js';
import {TinyTransformer,CONFIG} from './transformer.js';
import {wikiAnswer} from './wiki-answer.js';
import {all} from './db.js';

// Grounded Wikipedia chat is available immediately. Experimental neural text
// generation remains opt-in, and has its own immutable training snapshot.
let model=null,tokenizer=null,generating=false,pendingSnapshot=null;
const msg=(type,body={})=>postMessage({type,...body});
function ready(){msg('ready',{parameters:model?.parameters||0,step:model?.step||0,modelReady:!!model});}
function install(next){
  const nextTokenizer=new ByteBPE(next.merges);
  const nextModel=new TinyTransformer(nextTokenizer.vocabSize,next);
  tokenizer=nextTokenizer;model=nextModel;ready();
}
async function generateExperimental(request={}) {
  if (!model) return {text:'The experimental Transformer has not initialized yet. It needs at least three downloaded Wikipedia articles to train. Wikipedia-grounded chat works immediately; select it in Settings.',
    mode:'experimental',generated_tokens:0,prompt_tokens:0,seconds:0,tokens_per_second:0,stop_reason:'no_checkpoint'};
  const start=performance.now();
  const history=(request.messages||[]).slice(-8).map(m=>`${m.role}: ${m.content}`).join('\n');
  const context=(request.context||[]).join('\n').slice(-1200);
  const prompt=[request.memory||'',history,context,request.prompt].filter(Boolean).join('\n').slice(-2500);
  const prefix=tokenizer.encode(prompt),contextTokens=prefix.slice(-CONFIG.context);
  if(!contextTokens.length)contextTokens.push(EOS);
  const generated=[];
  const cap=Math.max(1,Math.min(80,Number(request.max_new_tokens)||36));
  let stop_reason='max_new_tokens';
  for(let i=0;i<cap;i++){
    const next=model.sampleNext(contextTokens,request);
    if(next===EOS){stop_reason='end_of_text';break;}
    generated.push(next);contextTokens.push(next);
    if(contextTokens.length>CONFIG.context)contextTokens.shift();
    if(i%8===0)await new Promise(resolve=>setTimeout(resolve,0));
  }
  const seconds=(performance.now()-start)/1000;
  const raw=tokenizer.decode(generated);
  // Invalid generated UTF-8 is evidence of an undertrained BYTE model. Do not
  // present garbled replacement characters as if they are a successful answer.
  const invalid=/[\uFFFD\u0000-\u0008\u000B\u000C\u000E-\u001F]/.test(raw);
  const text=invalid || !raw.trim()
    ? 'The experimental Transformer produced invalid or incomplete text. This tiny model is still learning and cannot reliably generate sentences. Switch Answer mode to Wikipedia-grounded for readable answers.'
    : raw;
  return {text,mode:'experimental',generated_tokens:generated.length,prompt_tokens:prefix.length,
    seconds,tokens_per_second:generated.length/Math.max(seconds,.001),stop_reason:invalid?'invalid_utf8':stop_reason};
}
onmessage=async ({data})=>{
  const {type}=data;
  if(type==='init'){ready();return;}
  if(type==='checkpoint'){
    try{
      if(generating){pendingSnapshot=data.checkpoint;return;}
      install(data.checkpoint);
    }catch(e){msg('error',{message:`Checkpoint not usable: ${e.message}`});}
  }else if(type==='generate'){
    const {id,request}=data;
    generating=true;
    try{
      const experimental=request?.response_mode==='experimental';
      const response=experimental
        ? await generateExperimental(request)
        : await wikiAnswer(request,{cachedArticles:()=>all('articles',300)});
      msg('result',{id,response});
    }catch(e){msg('result',{id,error:e.message});}
    finally{
      generating=false;
      if(pendingSnapshot){
        try{install(pendingSnapshot);}catch(e){msg('error',{message:e.message});}
        pendingSnapshot=null;
      }
    }
  }
};
