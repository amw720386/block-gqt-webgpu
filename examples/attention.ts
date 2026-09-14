import {AttentionEngine} from "../src/attention/attention.js";
import {FP16Keys} from "../src/runtime/keys.js";
import {CompressedKeys} from "../src/runtime/compressed-keys.js";
import {Values} from "../src/runtime/values.js";
import {readBuffer} from "../src/webgpu/buffers.js";
// This example uses a saved synthetic fixture, not a model or live tokenizer.
import {attentionFixture} from "../tests/webgpu/attention-fixture.js";

async function run(){
  const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw new Error("WebGPU unavailable");
  const device=await adapter.requestDevice(),{fixture:f,m,parameters}=await attentionFixture("t128_d64");
  const engine=await AttentionEngine.create(device),v=new Values(device,m.dim,m.context_length);
  v.append(f.f32("v"));const results=[];
  try{
    for(const keys of [new FP16Keys(device,m.dim,m.context_length),await CompressedKeys.create(device,parameters,m.context_length)]){
      const session=engine.session(keys,v,m.queries);
      try{
        await keys.append(f.halfBits("k"));
        const output=await session.attend(f.f32("q"),f.u32("positions"),true);
        const values=new Float32Array(await readBuffer(device,output,m.queries*m.dim*4));
        results.push({path:keys.kind,first_query_first_eight_values:Array.from(values.slice(0,8)),persistent_k_bytes:keys.allocations.filter(x=>x.role==="persistent-k").reduce((s,x)=>s+x.bytes,0)});
      }finally{session.destroy();keys.destroy();}
    }
    return results;
  }finally{v.destroy();device.destroy();}
}
void run().then(r=>document.querySelector("pre")!.textContent=JSON.stringify(r,null,2)).catch(e=>document.querySelector("pre")!.textContent=String(e));
