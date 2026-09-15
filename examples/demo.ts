import {Weights} from "../src/model/weights.js";
import {Calibration} from "../src/model/calibration.js";
import {Decoder} from "../src/model/decoder.js";
import {Tokenizer} from "../src/model/tokenizer.js";
import {generate} from "../src/model/generation.js";
import {modelTarget} from "../src/model/targets.js";
let loaded:{id:string;weights:Weights;tokenizer:Tokenizer;calibration?:Calibration}|undefined;
const status=document.querySelector('#status')!,output=document.querySelector('#output')!,metrics=document.querySelector('#metrics')!,button=document.querySelector<HTMLButtonElement>('#run')!;
button.onclick=async()=>{
  button.disabled=true;status.textContent='Preparing model…';output.textContent='';metrics.textContent='';let model:Decoder|undefined;
  try{
    const target=modelTarget(document.querySelector<HTMLSelectElement>('#model')!.value);if(loaded?.id!==target.id){const previous=loaded;loaded=undefined;previous?.calibration?.destroy();previous?.weights.destroy();previous?.weights.device.destroy();status.textContent=`Loading ${target.label}…`;
      const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw new Error('WebGPU unavailable');const device=await adapter.requestDevice({requiredLimits:{maxBufferSize:adapter.limits.maxBufferSize,maxStorageBufferBindingSize:adapter.limits.maxStorageBufferBindingSize}});device.lost.then(info=>{if(loaded?.weights.device===device)status.textContent=`GPU device lost: ${info.message}`;});
      let weights:Weights|undefined;try{weights=await Weights.load(device,target.base);const tokenizer=await Tokenizer.load(target.base,weights.adapter.chat,weights.adapter.vocabSize);loaded={id:target.id,weights,tokenizer};}catch(e){weights?.destroy();device.destroy();throw e;}}
    const {weights,tokenizer}=loaded,compressed=document.querySelector<HTMLSelectElement>('#mode')!.value==='compressed';if(!compressed&&loaded.calibration){loaded.calibration.destroy();loaded.calibration=undefined;}if(compressed&&!loaded.calibration)loaded.calibration=await Calibration.load(weights.device,weights.adapter,target.calibrationBase,weights.manifest.revision);
    const prompt=tokenizer.encode(tokenizer.chat(document.querySelector<HTMLTextAreaElement>('#prompt')!.value));if(prompt.length+32>8192)throw new Error('Prompt exceeds model context limit');
    model=await Decoder.create(weights,prompt.length+32,32,compressed?loaded.calibration:undefined);status.textContent=`Prefilling ${prompt.length} tokens…`;output.textContent='';
    const result=await generate(model,prompt,32,tokens=>{status.textContent=`Generated ${tokens.length} tokens`;output.textContent=tokenizer.decode(tokens);});
    const byRole:Record<string,number>={};for(const a of model.allocations)byRole[a.role]=(byRole[a.role]??0)+a.bytes;
    metrics.textContent=JSON.stringify({model:target.label,path:compressed?'Block-GTQ K / FP16 V':'FP16 K/V',prompt_tokens:prompt.length,generated_tokens:result.tokens.length,prefill_ms:result.prefillMs,
      decode_tokens_per_second:result.decodeMs.length?1000*result.decodeMs.length/result.decodeMs.reduce((a,b)=>a+b,0):null,explicit_gpu_buffer_bytes:byRole,
      transient_uniform_bytes:model.maxTransientUniformBytes,encoded_key_rows:model.encodedRows},null,2);status.textContent='Complete. Weights remain loaded for the next prompt.';
  }catch(e){status.textContent=String(e);}finally{model?.destroy();button.disabled=false;}
};
