import {Weights} from "../../src/model/weights.js";
import {Calibration} from "../../src/model/calibration.js";
import {Decoder} from "../../src/model/decoder.js";
import {readBuffer} from "../../src/webgpu/buffers.js";
import {loadFixture} from "./fixture.js";
import {compare,exact} from "./compare.js";
import {capture} from "./capture.js";
import {gpuMetadata} from "./phase2/phase2-test.js";
export async function run(){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error("No adapter");const device=await adapter.requestDevice(),errors:string[]=[];device.addEventListener("uncapturederror",e=>errors.push(e.error.message));
  const weights=await Weights.load(device),calibration=await Calibration.load(device);let model:Decoder|undefined;
  try{
    model=await Decoder.create(weights,64,32,calibration);const index=await (await fetch('/fixtures/model/compressed-index.json')).json(),cases=[];
    const packed=(model as unknown as {packed:{codes:GPUBuffer;norms:GPUBuffer}[]}).packed;
    let previous:Uint8Array[]=[];let previousNorms:Uint16Array[]=[];let previousLength=0;
    for(let step=0;step<index.cases.length;step++){
      const name=index.cases[step],f=await loadFixture(`/fixtures/model/${name}.json`),base=await loadFixture(`/fixtures/model/fp16_step${step}.json`);console.log(name);
      const actual=await model.forward(f.u32('input_ids'),true),arrays:Record<string,Float32Array>={};
      const codes=await Promise.all(packed.map(async p=>new Uint8Array(await readBuffer(device,p.codes))));
      const norms=await Promise.all(packed.map(async p=>new Uint16Array(await readBuffer(device,p.norms))));
      for(let i=0;i<90;i++){
        if(previous.length)exact(codes[i].subarray(0,previousLength*calibration.heads[i].rowBytes),previous[i].subarray(0,previousLength*calibration.heads[i].rowBytes),'historical packed K unchanged');
        if(previousNorms.length)exact(norms[i].subarray(0,previousLength*calibration.heads[i].normStride),previousNorms[i].subarray(0,previousLength*calibration.heads[i].normStride),'historical norms unchanged');
        arrays[`codes_${i}`]=Float32Array.from(codes[i]);arrays[`norms_${i}`]=Float32Array.from(norms[i]);
      }
      for(let i=0;i<30;i++){arrays[`attention_${i}`]=actual.layers[i];arrays[`q_${i}`]=actual.attentionInputs[i].q;arrays[`v_bits_${i}`]=Float32Array.from(actual.attentionInputs[i].v);arrays[`new_k_${i}`]=actual.attentionInputs[i].newK!;}
      previous=codes;previousNorms=norms;previousLength=model.length;
      if(model.encodedRows!==90*model.length)throw new Error("Historical keys recompressed");
      let selected=0;for(let i=1;i<actual.logits.length;i++)if(actual.logits[i]>actual.logits[selected])selected=i;
      let provisionalLogitThresholdExceeded:string|null=null;try{compare(actual.logits,f.f32('logits'),[0.03,0.003]);}catch(e){provisionalLogitThresholdExceeded=String(e);}
      cases.push({name,rows:f.u32('input_ids').length,length:model.length,selected,reference_selected:(f.manifest as any).selected_token,provisionalLogitThresholdExceeded,
        logit_port_error:compare(actual.logits,f.f32('logits')),quantization_error:compare(f.f32('logits'),base.f32('logits')),
        total_model_error:compare(actual.logits,base.f32('logits')),attention:actual.layers.map((a,i)=>compare(a,f.f32(`attention_${i}`))),capture:capture(arrays)});
    }
    return {passed:errors.length===0,errors,requires_cpu_attention_gate:true,cases,environment:gpuMetadata(adapter,device),encodedRows:model.encodedRows,allocations:model.allocations,calibration_sha256:calibration.sha256};
  }finally{model?.destroy();calibration.destroy();weights.destroy();device.destroy();}
}
