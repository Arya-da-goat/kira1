// Small, genuine decoder-only 2-head causal Transformer with pre-layer norms,
// token+learned position embeddings, feed-forward block and softmax language head.
// Derivatives below are analytical gradients through attention/softmax/layernorm.
// The optimizer is AdamW; no fake training metrics or random 'learning' animations.
export const CONFIG = Object.freeze({dim:32, heads:2, context:48, hidden:64});
const EPS=1e-5;
const rand = seed => { let x = seed | 0; return () => { x ^= x << 13; x ^= x >>> 17; x ^= x << 5; return ((x>>>0)/4294967296); }; };
const normal = random => (random()+random()+random()+random()-2)*1.73205;
function tensor(size, rnd, scale=.03, zero=false) {
  const value = new Float32Array(size);
  if (!zero) for(let k=0;k<size;k++) value[k]=normal(rnd)*scale;
  return value;
}
function norm(x, T,D) {
  const y=new Float32Array(T*D), inv=new Float32Array(T);
  for(let t=0;t<T;t++){
    let mean=0,v=0; const start=t*D;
    for(let d=0;d<D;d++) mean+=x[start+d]; mean/=D;
    for(let d=0;d<D;d++) v+=(x[start+d]-mean)**2;
    inv[t]=1/Math.sqrt(v/D+EPS);
    for(let d=0;d<D;d++) y[start+d]=(x[start+d]-mean)*inv[t];
  }
  return {y,inv};
}
function normBackward(grad, cached, T,D) {
  const out=new Float32Array(T*D), {y,inv}=cached;
  for(let t=0;t<T;t++) {
    let sum=0,sumXY=0; const base=t*D;
    for(let j=0;j<D;j++) { sum+=grad[base+j]; sumXY+=grad[base+j]*y[base+j]; }
    for(let j=0;j<D;j++) out[base+j]=(grad[base+j]-sum/D-y[base+j]*sumXY/D)*inv[t];
  }
  return out;
}
function linear(x,T,inn,out,w,bias) {
  const output=new Float32Array(T*out);
  for(let t=0;t<T;t++) for(let j=0;j<out;j++) {
    let sum=bias?.[j]||0;
    for(let k=0;k<inn;k++) sum+=x[t*inn+k]*w[k*out+j];
    output[t*out+j]=sum;
  }
  return output;
}
function linearGrad(x,dY,w,gW,gB,T,inn,out) {
  const dx=new Float32Array(T*inn);
  for(let t=0;t<T;t++) for(let j=0;j<out;j++) {
    const grad=dY[t*out+j];
    if(gB) gB[j]+=grad;
    for(let k=0;k<inn;k++) {
      gW[k*out+j]+=x[t*inn+k]*grad;
      dx[t*inn+k]+=w[k*out+j]*grad;
    }
  }
  return dx;
}
function add(x,y) { const out=new Float32Array(x.length); for(let i=0;i<x.length;i++)out[i]=x[i]+y[i];return out; }
function attention(Q,K,V,T,D,heads) {
  const A=new Float32Array(T*D), probs=new Float32Array(heads*T*T),hDim=D/heads;
  for(let h=0;h<heads;h++) for(let t=0;t<T;t++) {
    let max=-Infinity;
    for(let s=0;s<=t;s++) {
      let score=0;
      for(let k=0;k<hDim;k++) score+=Q[t*D+h*hDim+k]*K[s*D+h*hDim+k];
      score/=Math.sqrt(hDim); probs[h*T*T+t*T+s]=score; max=Math.max(max,score);
    }
    let denom=0;
    for(let s=0;s<=t;s++){const p=Math.exp(probs[h*T*T+t*T+s]-max);probs[h*T*T+t*T+s]=p;denom+=p;}
    for(let s=0;s<=t;s++) {
      const p=probs[h*T*T+t*T+s]/denom;
      probs[h*T*T+t*T+s]=p;
      for(let k=0;k<hDim;k++) A[t*D+h*hDim+k]+=p*V[s*D+h*hDim+k];
    }
  }
  return {A,probs};
}
function attentionGrad(Q,K,V,dA,probs,T,D,heads) {
  const dQ=new Float32Array(T*D),dK=new Float32Array(T*D),dV=new Float32Array(T*D);
  const hDim=D/heads, scale=1/Math.sqrt(hDim);
  for(let h=0;h<heads;h++) for(let t=0;t<T;t++){
    const dP=new Float32Array(t+1);
    let weighted=0;
    for(let s=0;s<=t;s++) {
      let dot=0;
      for(let k=0;k<hDim;k++){
        const col=h*hDim+k;
        dot+=dA[t*D+col]*V[s*D+col];
        dV[s*D+col]+=probs[h*T*T+t*T+s]*dA[t*D+col];
      }
      dP[s]=dot;weighted+=dot*probs[h*T*T+t*T+s];
    }
    for(let s=0;s<=t;s++) {
      const dScore=probs[h*T*T+t*T+s]*(dP[s]-weighted)*scale;
      for(let k=0;k<hDim;k++) {
        const col=h*hDim+k;
        dQ[t*D+col]+=dScore*K[s*D+col];
        dK[s*D+col]+=dScore*Q[t*D+col];
      }
    }
  }
  return {dQ,dK,dV};
}
export class TinyTransformer {
  constructor(vocab, checkpoint=null) {
    this.vocab=vocab;this.config=CONFIG;this.step=checkpoint?.step||0;
    const D=CONFIG.dim,H=CONFIG.hidden,L=CONFIG.context,rnd=rand(83171);
    const specs={embedding:[vocab*D,.06],position:[L*D,.025],
      wq:[D*D,.16],wk:[D*D,.16],wv:[D*D,.16],wo:[D*D,.14],
      w1:[D*H,.13],b1:[H,0],w2:[H*D,.11],b2:[D,0],
      head:[D*vocab,.1],bh:[vocab,0]};
    this.weights={};this.m={};this.v={};
    for(const [key,[size,scale]] of Object.entries(specs)) {
      this.weights[key]=tensor(size,rnd,scale,!scale);
      if(checkpoint?.weights?.[key]) {
        if(checkpoint.weights[key].length!==size)throw new Error('Checkpoint architecture differs from tokenizer.');
        this.weights[key].set(checkpoint.weights[key]);
      }
      this.m[key]=new Float32Array(checkpoint?.m?.[key]||size);
      this.v[key]=new Float32Array(checkpoint?.v?.[key]||size);
    }
  }
  get parameters() {return Object.values(this.weights).reduce((a,p)=>a+p.length,0);}
  forward(ids,keepCache=true) {
    const T=ids.length,D=CONFIG.dim,H=CONFIG.hidden,V=this.vocab,w=this.weights;
    if(T>CONFIG.context || !T)throw new Error('Invalid context length');
    const X=new Float32Array(T*D);
    for(let t=0;t<T;t++)for(let k=0;k<D;k++)X[t*D+k]=w.embedding[ids[t]*D+k]+w.position[t*D+k];
    const N1=norm(X,T,D),Q=linear(N1.y,T,D,D,w.wq),K=linear(N1.y,T,D,D,w.wk),Value=linear(N1.y,T,D,D,w.wv);
    const attentionState=attention(Q,K,Value,T,D,CONFIG.heads);
    const projected=linear(attentionState.A,T,D,D,w.wo), R=add(X,projected), N2=norm(R,T,D);
    const pre=linear(N2.y,T,D,H,w.w1,w.b1),act=new Float32Array(pre.length);
    for(let j=0;j<pre.length;j++)act[j]=Math.max(0,pre[j]);
    const ff=linear(act,T,H,D,w.w2,w.b2),Y=add(R,ff),N3=norm(Y,T,D);
    const logits=linear(N3.y,T,D,V,w.head,w.bh);
    if(!keepCache)return {logits};
    return {logits,cache:{X,N1,Q,K,Value,attentionState,R,N2,pre,act,N3,T}};
  }
  loss(ids, targets) {
    const T=ids.length,V=this.vocab, {logits}=this.forward(ids,false);
    let total=0,accuracy=0;
    for(let t=0;t<T;t++) {
      let maximum=-Infinity,sum=0,best=-1;
      for(let k=0;k<V;k++)if(logits[t*V+k]>maximum){maximum=logits[t*V+k];best=k;}
      for(let k=0;k<V;k++)sum+=Math.exp(logits[t*V+k]-maximum);
      total+=Math.log(sum)+maximum-logits[t*V+targets[t]];
      if(best===targets[t])accuracy++;
    }
    return {loss:total/T, accuracy:accuracy/T};
  }
  gradients(tokens) {
    const ids=tokens.slice(0,-1),targets=tokens.slice(1),T=ids.length,D=CONFIG.dim,H=CONFIG.hidden,V=this.vocab;
    const w=this.weights,grads=Object.fromEntries(Object.entries(w).map(([key,p])=>[key,new Float32Array(p.length)]));
    const {logits,cache:c}=this.forward(ids,true);
    const dl=new Float32Array(T*V);
    let loss=0,accuracy=0;
    for(let t=0;t<T;t++){
      let max=-Infinity,best=0,total=0;
      for(let j=0;j<V;j++)if(logits[t*V+j]>max){max=logits[t*V+j];best=j;}
      for(let j=0;j<V;j++) {const prob=Math.exp(logits[t*V+j]-max);dl[t*V+j]=prob;total+=prob;}
      for(let j=0;j<V;j++)dl[t*V+j]=((dl[t*V+j]/total)-(j===targets[t]?1:0))/T;
      loss+=(Math.log(total)+max-logits[t*V+targets[t]])/T;
      if(best===targets[t])accuracy++;
    }
    const dN3=linearGrad(c.N3.y,dl,w.head,grads.head,grads.bh,T,D,V);
    const dY=normBackward(dN3,c.N3,T,D);
    const dAct=linearGrad(c.act,dY,w.w2,grads.w2,grads.b2,T,H,D);
    const dPre=new Float32Array(dAct.length);
    for(let i=0;i<dPre.length;i++)dPre[i]=c.pre[i]>0?dAct[i]:0;
    const dN2=linearGrad(c.N2.y,dPre,w.w1,grads.w1,grads.b1,T,D,H);
    const dR=add(dY,normBackward(dN2,c.N2,T,D));
    const dAttention=linearGrad(c.attentionState.A,dR,w.wo,grads.wo,null,T,D,D);
    const {dQ,dK,dV}=attentionGrad(c.Q,c.K,c.Value,dAttention,c.attentionState.probs,T,D,CONFIG.heads);
    const dNq=linearGrad(c.N1.y,dQ,w.wq,grads.wq,null,T,D,D);
    const dNk=linearGrad(c.N1.y,dK,w.wk,grads.wk,null,T,D,D);
    const dNv=linearGrad(c.N1.y,dV,w.wv,grads.wv,null,T,D,D);
    const dN=new Float32Array(T*D);
    for(let i=0;i<T*D;i++)dN[i]=dNq[i]+dNk[i]+dNv[i];
    const dX=add(dR,normBackward(dN,c.N1,T,D));
    for(let t=0;t<T;t++)for(let d=0;d<D;d++) {
      grads.embedding[ids[t]*D+d]+=dX[t*D+d];
      grads.position[t*D+d]+=dX[t*D+d];
    }
    if(!Number.isFinite(loss))throw new Error('Non-finite gradient loss');
    return {grads,loss,accuracy:accuracy/T};
  }
  train(tokens,lr=.001,wd=.01) {return this.trainBatch([tokens],lr,wd);}
  trainBatch(sequences,lr=.001,wd=.01) {
    if(!Array.isArray(sequences)||!sequences.length)throw new Error('Training batch is empty');
    const w=this.weights,grads=Object.fromEntries(Object.entries(w).map(([key,p])=>[key,new Float32Array(p.length)]));
    let loss=0,accuracy=0;
    for(const tokens of sequences){
      const one=this.gradients(tokens);loss+=one.loss;accuracy+=one.accuracy;
      for(const key of Object.keys(grads)){
        const target=grads[key],source=one.grads[key];
        for(let j=0;j<target.length;j++)target[j]+=source[j]/sequences.length;
      }
    }
    loss/=sequences.length;accuracy/=sequences.length;
    let sum=0;
    for(const g of Object.values(grads))for(const e of g)sum+=e*e;
    const gradNorm=Math.sqrt(sum),clip=Math.min(1,1/(gradNorm+1e-8));
    this.step++;
    const b1=.9,b2=.999,eps=1e-8,correct1=1-b1**this.step,correct2=1-b2**this.step;
    for(const key of Object.keys(w)) {
      const p=w[key],m=this.m[key],v=this.v[key],g=grads[key];
      for(let i=0;i<p.length;i++){
        const z=g[i]*clip;
        m[i]=b1*m[i]+(1-b1)*z;v[i]=b2*v[i]+(1-b2)*z*z;
        const decay=key==='b1'||key==='b2'||key==='bh'?0:wd;
        p[i]-=lr*(m[i]/correct1/(Math.sqrt(v[i]/correct2)+eps)+decay*p[i]);
      }
    }
    if(!Number.isFinite(loss))throw new Error('Training divergence: non-finite loss');
    return {loss,accuracy,gradient_norm:gradNorm,batch_size:sequences.length};
  }
  snapshot() {
    return {step:this.step,vocab:this.vocab,config:CONFIG,
      weights:Object.fromEntries(Object.entries(this.weights).map(([k,x])=>[k,new Float32Array(x)])),
      m:Object.fromEntries(Object.entries(this.m).map(([k,x])=>[k,new Float32Array(x)])),
      v:Object.fromEntries(Object.entries(this.v).map(([k,x])=>[k,new Float32Array(x)]))};
  }
  sampleNext(context,{temperature=.8,top_k=40,top_p=.95,repetition_penalty=1}={}) {
    const ids=context.slice(-CONFIG.context),logits=this.forward(ids,false).logits,last=logits.slice((ids.length-1)*this.vocab);
    const previous=new Set(ids);
    for(let i=0;i<last.length;i++){
      if(repetition_penalty>1&&previous.has(i))last[i]/=repetition_penalty;
      last[i]/=Math.max(.01,temperature);
    }
    if(temperature===0)return last.indexOf(Math.max(...last));
    const scored=Array.from(last,(score,id)=>({id,score})).sort((a,b)=>b.score-a.score);
    const limit=Math.max(1,Math.min(this.vocab,top_k||this.vocab));scored.length=limit;
    const maximum=scored[0].score;
    let total=0;for(const item of scored){item.prob=Math.exp(item.score-maximum);total+=item.prob;}
    let cumul=0;const retained=[];
    for(const item of scored) {const p=item.prob/total;cumul+=p;retained.push({...item,prob:p});if(cumul>=top_p)break;}
    const r=Math.random()*retained.reduce((s,x)=>s+x.prob,0);let progress=0;
    for(const item of retained){progress+=item.prob;if(progress>=r)return item.id;}
    return retained.at(-1).id;
  }
}
