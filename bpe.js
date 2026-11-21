// Byte-level BPE: vocabulary = 0..255 bytes, 256=end-of-text, 257=unknown (reserved),
// then merge ids 258+. Merges are fitted to actual Wikipedia bytes and frozen with weights.
export const EOS = 256;
export const FIRST_MERGE = 258;
const enc = new TextEncoder();
const dec = new TextDecoder();
export class ByteBPE {
  constructor(merges = []) {
    this.merges = merges.map(([a, b]) => [a, b]);
    this.dictionary = new Map(this.merges.map(([a, b], i) => [`${a},${b}`, FIRST_MERGE + i]));
    this.expansions = new Map(this.merges.map(([a,b], i) => [FIRST_MERGE+i, [a,b]]));
  }
  static fit(documents, numMerges = 94) {
    // Limit the bootstrap sample so the actual pair-counting procedure remains phone-friendly.
    const initial = documents.map(s => Array.from(enc.encode(String(s).slice(0, 11000)))).filter(x => x.length);
    const merges = [];
    for (let n = 0; n < numMerges; n++) {
      const freqs = new Map();
      for (const arr of initial) for (let j = 0; j + 1 < arr.length; j++) {
        const key = `${arr[j]},${arr[j+1]}`;
        freqs.set(key, (freqs.get(key) || 0) + 1);
      }
      let winner = null, frequency = 2;
      for (const [key, count] of freqs) if (count > frequency) { winner = key; frequency = count; }
      if (!winner) break;
      const [a,b] = winner.split(',').map(Number), newId = FIRST_MERGE + merges.length;
      merges.push([a,b]);
      for (let k = 0; k < initial.length; k++) {
        const prev = initial[k], next = [];
        for (let j=0; j<prev.length; j++) {
          if (j+1<prev.length && prev[j]===a && prev[j+1]===b) { next.push(newId); j++; }
          else next.push(prev[j]);
        }
        initial[k] = next;
      }
    }
    return new ByteBPE(merges);
  }
  get vocabSize() { return FIRST_MERGE + this.merges.length; }
  encode(text) {
    let tokens = Array.from(enc.encode(String(text)));
    for (let m = 0; m < this.merges.length; m++) {
      const [a,b] = this.merges[m], out=[];
      for (let i=0; i<tokens.length; i++) {
        if (i+1<tokens.length && tokens[i]===a && tokens[i+1]===b) { out.push(FIRST_MERGE+m); i++; }
        else out.push(tokens[i]);
      }
      tokens=out;
    }
    return tokens;
  }
  decode(tokens) {
    const bytes=[];
    const expand = id => {
      if (id < 256) bytes.push(id);
      else if (this.expansions.has(id)) for(const item of this.expansions.get(id)) expand(item);
    };
    for (const token of tokens) expand(token);
    return dec.decode(new Uint8Array(bytes));
  }
}
export function makeWindows(tokens, length=20, stride=18, maxWindows=180) {
  const windows=[];
  if (tokens.length <= length) return windows;
  for(let i=0; i+length<tokens.length && windows.length<maxWindows; i+=stride) {
    windows.push(Uint16Array.from(tokens.slice(i,i+length+1)));
  }
  return windows;
}
