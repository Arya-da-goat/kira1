import {ByteBPE,makeWindows} from './bpe.js';
import {TinyTransformer} from './transformer.js';
import assert from 'node:assert/strict';
const source=[
  'Science describes the world and the natural causes of events. Quantum mechanics is a theory of matter.',
  'History describes human civilization, governments, and events from the past.',
  'Mathematics uses logical structures and relationships between numbers and geometry.'
];
const tok=ByteBPE.fit(source,94);
for(const text of source)assert.equal(tok.decode(tok.encode(text)),text,'BPE must round-trip Unicode text');
assert.equal(tok.decode(tok.encode('தமிழ் உலகம், français, 日本語')),'தமிழ் உலகம், français, 日本語');
const network=new TinyTransformer(tok.vocabSize);
const windows=source.flatMap(s=>makeWindows(tok.encode(s.repeat(3)),48,40,2));
assert.ok(windows.length>=2);
const [a,b]=windows;
const before=network.loss(a.slice(0,-1),a.slice(1));
for(let step=0;step<35;step++)network.trainBatch([a,b],.002);
const after=network.loss(a.slice(0,-1),a.slice(1));
assert.ok(after.loss<before.loss-.6,`Training loss must decrease ${before.loss} -> ${after.loss}`);
const restored=new TinyTransformer(tok.vocabSize,network.snapshot());
assert.equal(restored.step,network.step);
assert.ok(Math.abs(restored.loss(a.slice(0,-1),a.slice(1)).loss-after.loss)<1e-9);
assert.ok(Number.isInteger(network.sampleNext(tok.encode('Science'),{temperature:.8,top_k:15,top_p:.9})));
console.log(JSON.stringify({result:'PASS',vocab:tok.vocabSize,params:network.parameters,steps:network.step,before:before.loss,after:after.loss,restored:true}));
