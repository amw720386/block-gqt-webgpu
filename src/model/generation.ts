import {Decoder} from "./decoder.js";
export function topK(logits:Float32Array,k=10){return Array.from(logits,(_,id)=>id).sort((a,b)=>logits[b]-logits[a]||a-b).slice(0,k).map(id=>({id,logit:logits[id]}));}
export function greedy(logits:Float32Array){let id=0;for(let i=1;i<logits.length;i++)if(logits[i]>logits[id])id=i;return id;}
export async function generate(model:Decoder,prompt:Uint32Array,maxTokens:number,onToken?:(tokens:number[])=>void){
  const begin=performance.now();let {logits}=await model.prefill(prompt);const prefillMs=performance.now()-begin,firstLogits=logits.slice(),tokens:number[]=[],decodeMs:number[]=[];let pendingStart:number|undefined,ttftMs=0;
  for(let i=0;i<maxTokens;i++){
    const token=greedy(logits);const now=performance.now();if(pendingStart!==undefined)decodeMs.push(now-pendingStart);else ttftMs=now-begin;
    tokens.push(token);onToken?.(tokens);if(model.weights.adapter.eosTokenIds.includes(token)||i===maxTokens-1)break;
    pendingStart=performance.now();({logits}=await model.decode(token));
  }
  return {tokens,prefillMs,ttftMs,decodeMs,firstLogits,firstTopK:topK(firstLogits)};
}
