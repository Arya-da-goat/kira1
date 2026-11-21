import assert from 'node:assert/strict';
import {wikiAnswer,queryFromMessage,mediaWikiFetch,readableExtract} from './wiki-answer.js';
const wiki = async params => {
  if(params.list === 'search') {
    assert.equal(params.srsearch,'photosynthesis');
    return {query:{search:[{title:'Photosynthesis',pageid:1234}]}};
  }
  assert.equal(params.exintro,1);
  assert.equal(params.exlimit,3);
  return {query:{pages:[{pageid:1234,title:'Photosynthesis',extract:'Photosynthesis is the process by which plants and some other organisms convert light energy into chemical energy.\n\nIn most cases, oxygen is also released into the atmosphere.'}]}};
};
const answer=await wikiAnswer({prompt:'What is photosynthesis?'},{getJSON:wiki});
assert.match(answer.text,/Photosynthesis is the process/);
assert.match(answer.text,/en\.wikipedia\.org\/wiki\/Photosynthesis/);
assert.equal(answer.mode,'wikipedia');
assert.equal(answer.generated_tokens,0);
assert.equal(queryFromMessage({prompt:'Who was Ada Lovelace?'}).query,'Ada Lovelace');
assert.equal(queryFromMessage({prompt:'Explain black holes'}).query,'black holes');
assert.equal(queryFromMessage({prompt:'Tell me more',messages:[{role:'assistant',content:answer.text}]}).query,'Photosynthesis');
assert.match(readableExtract('A '.repeat(800)),/A/);
let calls=0;
const hello=await wikiAnswer({prompt:'hello'},{getJSON:async()=>{calls++;throw new Error('no')}});
assert.match(hello.text,/Kira/);assert.equal(calls,0);
const cached=await wikiAnswer({prompt:'What is photosynthesis?'},{getJSON:async()=>{throw new Error('offline');},cachedArticles:async()=>[{title:'Photosynthesis',content:'Photosynthesis is the process by which plants create chemical energy from sunlight.',url:'https://en.wikipedia.org/wiki/Photosynthesis'}]});
assert.equal(cached.mode,'cached');assert.match(cached.text,/Saved article/);
const noData=await wikiAnswer({prompt:'what is quantum yogurt?'},{getJSON:async()=>{throw new Error('offline');}});
assert.equal(noData.mode,'unavailable');assert.doesNotMatch(noData.text,/\uFFFD/);
const callsToFetch=[];
await mediaWikiFetch({list:'search',srsearch:'physics'},async (url,opts)=>{
  callsToFetch.push(new URL(url));
  return {ok:true,json:async()=>({query:{search:[]}})};
});
assert.equal(callsToFetch[0].searchParams.get('origin'),'*');
assert.equal(callsToFetch[0].searchParams.get('srsearch'),'physics');
console.log(JSON.stringify({result:'PASS',checks:['Wikipedia response','source link','query extraction','follow-up','greetings without network','offline cached response','unavailable response','CORS parameters']}));
