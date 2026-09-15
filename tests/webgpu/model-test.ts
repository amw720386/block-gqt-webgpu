import {Weights} from "../../src/model/weights.js";
import {Decoder} from "../../src/model/decoder.js";
import {Tokenizer} from "../../src/model/tokenizer.js";
import {loadFixture} from "./fixture.js";
import {compare,exact} from "./compare.js";
import {gpuMetadata} from "./phase2/phase2-test.js";
import {capture} from "./capture.js";
export async function run(){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error("No adapter");const device=await adapter.requestDevice(),errors:string[]=[];device.addEventListener("uncapturederror",e=>errors.push(e.error.message));
  console.log('Loading pinned FP16 model');const weights=await Weights.load(device),tokenizer=await Tokenizer.load();let model:Decoder|undefined;
  try{
    const index=await (await fetch('/fixtures/model/index.json')).json();for(const t of index.tokenizer_tests)exact(tokenizer.encode(t.text),t.ids,"tokenizer");
    model=await Decoder.create(weights,64,32);const cases=[];
    for(const name of index.cases){
      const f=await loadFixture(`/fixtures/model/${name}.json`);console.log(`FP16 ${name}`);
      const actual=await model.forward(f.u32("input_ids"),true),logits=compare(actual.logits,f.f32("logits"));
      try{compare(actual.logits,f.f32("logits"),[0.003,0.0003]);}catch(e){errors.push(`${name} logits: ${e}`);}
      const localThresholdExceeded:string[]=[];
      const attention=actual.layers.map((a,i)=>{const metric=compare(a,f.f32(`attention_${i}`));try{compare(a,f.f32(`attention_${i}`),[0.001,0.0003]);}catch(e){localThresholdExceeded.push(`${name} layer ${i}: ${e}`);}return metric;});
      let selected=0;for(let i=1;i<actual.logits.length;i++)if(actual.logits[i]>actual.logits[selected])selected=i;
      if(selected!==Number((f.manifest as any).selected_token))errors.push(`Greedy token differs: ${selected}`);
      const arrays:Record<string,Float32Array>={};for(let i=0;i<30;i++){
        arrays[`attention_${i}`]=actual.layers[i];arrays[`q_${i}`]=actual.attentionInputs[i].q;
        arrays[`k_bits_${i}`]=Float32Array.from(actual.attentionInputs[i].k);arrays[`v_bits_${i}`]=Float32Array.from(actual.attentionInputs[i].v);
      }
      cases.push({name,logits,attention,selected,length:model.length,rows:f.u32("input_ids").length,localThresholdExceeded,capture:capture(arrays)});
    }
    return {passed:errors.length===0,requires_cpu_attention_gate:true,errors,cases,environment:gpuMetadata(adapter,device),tokenizer_tests:index.tokenizer_tests.length,greedy_text:index.greedy_text,allocations:model.allocations,transient_uniform_bytes:model.maxTransientUniformBytes};
  }finally{model?.destroy();weights.destroy();device.destroy();}
}
