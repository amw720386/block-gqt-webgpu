import {Weights} from "../../src/model/weights.js";
import {Calibration} from "../../src/model/calibration.js";
import {Decoder} from "../../src/model/decoder.js";
import {Tokenizer} from "../../src/model/tokenizer.js";
import {compare} from "./compare.js";
import {generate} from "../../src/model/generation.js";
export async function run(){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error("No adapter");const device=await adapter.requestDevice(),errors:string[]=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
  const weights=await Weights.load(device),calibration=await Calibration.load(device),tokenizer=await Tokenizer.load(),cases=[];
  try{for(const compressed of [false,true]){
    const tokens=tokenizer.encode('A library contains books about science, history, art, and language. Readers can learn new skills by studying examples and asking questions. A careful explanation gives enough detail to make each step clear.').slice(0,40);
    const chunked=await Decoder.create(weights,64,32,compressed?calibration:undefined),incremental=await Decoder.create(weights,64,1,compressed?calibration:undefined);
    try{
      console.log(`Live append ${compressed?'compressed':'FP16'}`);const a=await chunked.prefill(tokens);let b;for(const token of tokens)b=await incremental.decode(token);
      const error=compare(a.logits,b!.logits,[0.001,0.0001]);
      if(chunked.length!==tokens.length||incremental.length!==tokens.length)throw new Error('Incorrect append length');
      if(compressed&&(chunked.encodedRows!==tokens.length*90||incremental.encodedRows!==tokens.length*90))throw new Error('Encode count mismatch');
      cases.push({compressed,chunked_vs_single_token:error,length:tokens.length});
    }finally{chunked.destroy();incremental.destroy();}
    for(const prompt of ['What is 2 + 2?','Name one planet in our solar system.']){
      const ids=tokenizer.encode(tokenizer.chat(prompt)),model=await Decoder.create(weights,ids.length+12,32,compressed?calibration:undefined);
      try{const result=await generate(model,ids,12);cases.push({compressed,prompt,tokens:result.tokens,text:tokenizer.decode(result.tokens)});}finally{model.destroy();}
    }
  }return {passed:errors.length===0,errors,cases};}finally{calibration.destroy();weights.destroy();device.destroy();}
}
