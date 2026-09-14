import {AttentionEngine} from "../../src/attention/attention.js";
import {OptimizedAttention} from "../../src/attention/optimized.js";
import {CompressedKeys} from "../../src/runtime/compressed-keys.js";
import {PackedKeys} from "../../src/runtime/packed-keys.js";
import {Values} from "../../src/runtime/values.js";
import {readBuffer} from "../../src/webgpu/buffers.js";
import {attentionFixture} from "./attention-fixture.js";
import {compare,exact} from "./compare.js";
import {capture} from "./capture.js";
import {gpuMetadata} from "./phase2/phase2-test.js";

export async function run(){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error("No adapter");const device=await adapter.requestDevice(),errors:string[]=[];
  device.addEventListener("uncapturederror",e=>errors.push(e.error.message));
  try{
    const index=await (await fetch("/fixtures/attention/index.json")).json(),engine=await AttentionEngine.create(device),cases=[];
    for(const c of index.cases){
      const {fixture:f,m,parameters}=await attentionFixture(c.name),values=new Values(device,m.dim,m.context_length);
      const baseline=await CompressedKeys.create(device,parameters,m.context_length),packed=await PackedKeys.create(device,parameters,m.context_length);
      const phase3=engine.session(baseline,values,m.queries),optimized=await OptimizedAttention.create(device,packed,values,m.queries);
      try{
        const k=f.halfBits("k"),v=f.f32("v");
        for(const keys of [baseline,packed]){await keys.append(k.subarray(0,3*m.dim));await keys.append(k.subarray(3*m.dim));}
        values.append(v);
        const b=baseline as unknown as {packed:GPUBuffer;norms:GPUBuffer;decoded:GPUBuffer};
        exact(new Uint8Array(await readBuffer(device,packed.packed)),new Uint8Array(await readBuffer(device,b.packed)),"packed bytes");
        exact(new Uint8Array(await readBuffer(device,packed.norms)),new Uint8Array(await readBuffer(device,b.norms)),"norm bits");
        const captured:Record<string,Float32Array>={},comparisons=[];
        for(const causal of [false,true]){
          await phase3.attend(f.f32("q"),f.u32("positions"),causal);await optimized.attend(f.f32("q"),f.u32("positions"),causal);
          const a=phase3.inspect(),o=optimized.inspect(),prefix=causal?"causal_":"",metrics:Record<string,unknown>={};
          for(const stage of ["logits","probabilities","output"] as const){
            const size=stage==="output"?m.queries*m.dim*4:m.queries*m.context_length*4;
            const expected=new Float32Array(await readBuffer(device,a[stage],size)),actual=new Float32Array(await readBuffer(device,o[stage],size));
            // Use existing strict baseline tolerances for optimization-induced error.
            metrics[stage]={optimization_error:compare(actual,expected,m.tolerances.baseline[stage]),
              compressed_port_error:compare(actual,f.f32(`${prefix}compressed_${stage}`)),
              quantization_error:compare(f.f32(`${prefix}compressed_${stage}`),f.f32(`${prefix}baseline_${stage}`)),
              total_error_vs_fp16:compare(actual,f.f32(`${prefix}baseline_${stage}`))};
            captured[`${causal}_${stage}`]=actual;
            if(stage!=="logits")compare(actual,f.f32(`${prefix}compressed_${stage}`),m.tolerances.port[stage]);
          }
          if(c.name==="t128_d64"){
            await optimized.attend(f.f32("q"),f.u32("positions"),causal);
            exact(new Float32Array(await readBuffer(device,optimized.output,m.queries*m.dim*4)),captured[`${causal}_output`],"repeat determinism");
          }
          comparisons.push({causal,metrics});
        }
        captured.reconstructed_k=new Float32Array(await readBuffer(device,b.decoded));
        if(packed.allocations.some(a=>a.role==="scratch"&&a.bytes>4))throw new Error("Unexpected K scratch allocation");
        cases.push({name:c.name,upstream_commit:m.upstream_commit,fixture_sha256:m.sha256,packed_bytes_exact:true,norm_bits_exact:true,comparisons,capture:capture(captured)});
      }catch(e){throw new Error(`${c.name}: ${e}`);}finally{phase3.destroy();optimized.destroy();baseline.destroy();packed.destroy();values.destroy();}
    }
    const contracts=await appendContracts(device);
    if(errors.length)throw new Error(errors.join("\n"));return {passed:true,environment:gpuMetadata(adapter,device),cases,contracts};
  }finally{device.destroy();}
}
async function appendContracts(device:GPUDevice){
  const {fixture:f,m,parameters}=await attentionFixture("t128_d64"),keys=await PackedKeys.create(device,parameters,3),values=new Values(device,m.dim,3);
  const s=await OptimizedAttention.create(device,keys,values,1);
  try{
    const k=f.halfBits("k"),v=f.f32("v"),q=new Float32Array(m.dim);
    for(const end of [1,3]){
      const begin=keys.length;await keys.append(k.subarray(begin*m.dim,end*m.dim));values.append(v.subarray(begin*m.dim,end*m.dim));
      await s.attend(q,new Uint32Array([end-1]),true);
      const expected=Float32Array.from({length:m.dim},(_,d)=>{let n=0;for(let i=0;i<end;i++)n+=v[i*m.dim+d];return n/end;});
      compare(new Float32Array(await readBuffer(device,s.output)),expected,[1e-6,1e-6]);
    }
    for(const fail of [()=>keys.append(k.subarray(0,m.dim)),()=>s.setQueries(q,new Uint32Array([3])),()=>s.setQueries(new Float32Array(m.dim-1),new Uint32Array([0]))]){
      let rejected=false;try{await fail();}catch{rejected=true;}if(!rejected)throw new Error("Missing contract rejection");
    }
    const changed={...parameters,rotation:parameters.rotation.slice()};
    const other=parameters.groupOf.findIndex(g=>g!==parameters.groupOf[0]);changed.rotation[other*m.dim]=0.01;
    let rejected=false;try{const bad=await PackedKeys.create(device,changed,3);bad.destroy();}catch{rejected=true;}
    if(!rejected)throw new Error("Off-group transform was silently truncated");
    return {append_prefix:true,capacity_bounds:true,query_bounds:true,off_group_transform_rejected:true};
  }finally{s.destroy();keys.destroy();values.destroy();}
}
