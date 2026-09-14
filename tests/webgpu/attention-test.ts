import {AttentionEngine,type AttentionSession} from "../../src/attention/attention.js";
import {FP16Keys,type KeyStore} from "../../src/runtime/keys.js";
import {CompressedKeys} from "../../src/runtime/compressed-keys.js";
import {Values} from "../../src/runtime/values.js";
import {readBuffer} from "../../src/webgpu/buffers.js";
import {gpuMetadata} from "./phase2/phase2-test.js";
import {attentionFixture} from "./attention-fixture.js";
import {compare,exact} from "./compare.js";
import {capture} from "./capture.js";
import {runtimeContracts} from "./runtime-contracts.js";

export async function intermediateReadback(device:GPUDevice,session:AttentionSession){
  const buffers=session.inspect(),count=buffers.queries*buffers.tokens;
  const [logits,probabilities,output]=await Promise.all([
    readBuffer(device,buffers.logits,count*4),readBuffer(device,buffers.probabilities,count*4),
    readBuffer(device,buffers.output,buffers.queries*session.keys.dim*4)]);
  return {logits:new Float32Array(logits),probabilities:new Float32Array(probabilities),output:new Float32Array(output)};
}
async function mustReject(fn:()=>unknown|Promise<unknown>,label:string){
  try{await fn();}catch{return;}throw new Error(`Expected rejection: ${label}`);
}
export async function validateAttention(device:GPUDevice,compressed:boolean){
  const engine=await AttentionEngine.create(device),index=await (await fetch("/fixtures/attention/index.json")).json();
  const results=[];
  for(const c of index.cases){
    const {fixture:f,m,parameters}=await attentionFixture(c.name);
    const values=new Values(device,m.dim,m.context_length);
    const paths:KeyStore[]=[new FP16Keys(device,m.dim,m.context_length)];
    if(compressed)paths.push(await CompressedKeys.create(device,parameters,m.context_length));
    const k=f.halfBits("k"),v=f.f32("v");
    try{
      for(const keys of paths){
        // An odd row boundary exercises shared packed words across append calls.
        await keys.append(k.subarray(0,3*m.dim));
        const incomplete=engine.session(keys,values,m.queries);
        try{await mustReject(()=>incomplete.setQueries(f.f32("q"),f.u32("positions")),"mismatched K/V lengths");}
        finally{incomplete.destroy();}
        await keys.append(k.subarray(3*m.dim));
      }
      values.append(v.subarray(0,3*m.dim));values.append(v.subarray(3*m.dim));
      const comparisons=[];const captured:Record<string,Float32Array>={};
      for(const keys of paths){
        const session=engine.session(keys,values,m.queries),label=keys.kind==="fp16"?"baseline":"compressed";
        try{
          let logitBound:Float64Array|undefined;
          if(keys.kind==="block-gtq"){
            const encoder=device.createCommandEncoder(),view=keys.prepare(encoder);device.queue.submit([encoder.finish()]);
            const decoded=new Float32Array(await readBuffer(device,view.buffer,m.context_length*m.dim*4));
            captured.reconstructed_k=decoded;
            // Test-only FP32 reconstruction readback, never packed-K CPU transfer.
            // Near a LUT boundary the CPU and GPU encoder can select different
            // codes. Keep that port error visible; bound its effect on QK exactly.
            const reference=f.f32("reconstructed_k"),q=f.f32("q");logitBound=new Float64Array(m.queries*m.context_length);
            for(let row=0;row<m.queries;row++)for(let token=0;token<m.context_length;token++){
              let bound=0;for(let d=0;d<m.dim;d++)bound+=Math.abs(q[row*m.dim+d]*(decoded[token*m.dim+d]-reference[token*m.dim+d]));
              logitBound[row*m.context_length+token]=bound/Math.sqrt(m.dim);
            }
          }
          await mustReject(()=>keys.append(new Uint16Array(m.dim)),"K capacity overflow");
          await mustReject(()=>session.setQueries(new Float32Array(m.dim-1),new Uint32Array([0])),"query shape");
          await mustReject(()=>session.setQueries(f.f32("q"),new Uint32Array([0,0,m.context_length])),"query position");
          for(const causal of [false,true]){
            await session.attend(f.f32("q"),f.u32("positions"),causal);
            const actual=await intermediateReadback(device,session),prefix=causal?"causal_":"";
            const errors={} as Record<string,ReturnType<typeof compare>>;
            const quantization={} as Record<string,ReturnType<typeof compare>>;
            for(const stage of ["logits","probabilities","output"] as const){
              const reference=f.f32(`${prefix}${label}_${stage}`);
              errors[stage]=compare(actual[stage],reference,m.tolerances[label==="baseline"?"baseline":"port"][stage],stage==="logits"?logitBound:undefined);
              quantization[stage]=compare(f.f32(`${prefix}compressed_${stage}`),f.f32(`${prefix}baseline_${stage}`));
              captured[`${label}_${causal}_${stage}`]=actual[stage];
            }
            for(let row=0;row<m.queries;row++){
              let sum=0;for(let t=0;t<m.context_length;t++){
                const p=actual.probabilities[row*m.context_length+t];if(p<0)throw new Error("Negative probability");sum+=p;
                if(causal&&t>f.u32("positions")[row]&&p!==0)throw new Error("Causal probability is not zero");
              }
              if(Math.abs(sum-1)>0.0001)throw new Error("Softmax does not sum to one");
            }
            if(causal)exact(actual.output.subarray(0,m.dim),v.subarray(0,m.dim),"first causal query equals V0");
            if(c.name==="t128_d64"){
              await session.attend(f.f32("q"),f.u32("positions"),causal);const repeat=await intermediateReadback(device,session);
              for(const stage of ["logits","probabilities","output"] as const)exact(repeat[stage],actual[stage],"deterministic attention");
            }
            comparisons.push({path:label,causal,numerical_error:errors,quantization_error:quantization,
              output_actual:Array.from(actual.output),output_reference:Array.from(f.f32(`${prefix}${label}_output`)),
              total_output_error_vs_fp16:compare(actual.output,f.f32(`${prefix}baseline_output`)),allocations:[...keys.allocations,...values.allocations,...session.allocations]});
          }
        }finally{session.destroy();}
      }
      results.push({name:c.name,context_length:m.context_length,head_dimension:m.dim,fixture_sha256:m.sha256,upstream_commit:m.upstream_commit,comparisons,capture:capture(captured)});
    }catch(e){throw new Error(`${c.name}: ${e}`);}finally{paths.forEach(k=>k.destroy());values.destroy();}
  }
  return results;
}
export async function runAttention(compressed:boolean){
  const adapter=await navigator.gpu?.requestAdapter();if(!adapter)throw new Error("No GPU adapter");
  const device=await adapter.requestDevice(),errors:string[]=[];device.addEventListener("uncapturederror",e=>errors.push(e.error.message));
  try{
    const cases=await validateAttention(device,compressed);const contracts=compressed?await runtimeContracts(device):undefined;await device.queue.onSubmittedWorkDone();
    if(errors.length)throw new Error(errors.join("\n"));
    return {passed:true,mode:compressed?"fp16-and-compressed":"fp16-only",environment:gpuMetadata(adapter,device),cuda_encoder_validated:false,contracts,cases};
  }finally{device.destroy();}
}
