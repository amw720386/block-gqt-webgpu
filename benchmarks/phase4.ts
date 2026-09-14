import {AttentionEngine} from "../src/attention/attention.js";
import {OptimizedAttention} from "../src/attention/optimized.js";
import {CompressedKeys} from "../src/runtime/compressed-keys.js";
import {PackedKeys} from "../src/runtime/packed-keys.js";
import {FP16Keys} from "../src/runtime/keys.js";
import {Values} from "../src/runtime/values.js";
import {readBuffer} from "../src/webgpu/buffers.js";
import {attentionFixture} from "../tests/webgpu/attention-fixture.js";
import {gpuMetadata} from "../tests/webgpu/phase2/phase2-test.js";
import {compare} from "../tests/webgpu/compare.js";
import {timePasses} from "./timing.js";

export async function run(final=false){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter?.features.has("timestamp-query"))throw new Error("Timestamps required");
  const device=await adapter.requestDevice({requiredFeatures:["timestamp-query"]}),errors:string[]=[];
  device.addEventListener("uncapturederror",e=>errors.push(e.error.message));
  try{
    const engine=await AttentionEngine.create(device),cases=[];
    for(const dim of (final?[64,128]:[128]))for(const tokens of (final?[512,1024,2048,4096]:[4096])){
      const {fixture:f,m,parameters}=await attentionFixture(`t${tokens}_d${dim}`),values=new Values(device,dim,tokens);
      const baseline=new FP16Keys(device,dim,tokens),materialized=await CompressedKeys.create(device,parameters,tokens),packed=await PackedKeys.create(device,parameters,tokens);
      const b=engine.session(baseline,values,3),p3=engine.session(materialized,values,3),p4=await OptimizedAttention.create(device,packed,values,3);
      const paths=[{name:"fp16",keys:baseline,session:b,count:3},{name:"phase3",keys:materialized,session:p3,count:4},{name:"phase4",keys:packed,session:p4,count:3}];
      try{
        values.append(f.f32("v"));for(const p of paths)await p.keys.append(f.halfBits("k"));
        const data:Record<string,Record<string,Float32Array>>={};
        for(const p of paths){
          await p.session.attend(f.f32("q"),f.u32("positions"),false);const view=p.session.inspect();data[p.name]={};
          for(const stage of ["logits","probabilities","output"] as const){
            const size=stage==="output"?3*dim*4:3*tokens*4;
            data[p.name][stage]=new Float32Array(await readBuffer(device,view[stage],size));
          }
        }
        const ordered:typeof paths=[...paths.slice(cases.length%3),...paths.slice(0,cases.length%3)];const measurements=[];
        for(const p of ordered){
          const numerical:Record<string,unknown>={};
          for(const stage of ["logits","probabilities","output"] as const){
            const actual=data[p.name][stage],ref=f.f32(`${p.name==="fp16"?"baseline":"compressed"}_${stage}`);
            if(p.name==="fp16")compare(actual,ref,m.tolerances.baseline[stage]);
            else if(stage!=="logits")compare(actual,ref,m.tolerances.port[stage]);
            numerical[stage]={error_vs_own_reference:compare(actual,ref),quantization_error:compare(f.f32(`compressed_${stage}`),f.f32(`baseline_${stage}`)),
              optimization_error:p.name==="phase4"?compare(actual,data.phase3[stage],m.tolerances.baseline[stage]):null,
              total_error_vs_fp16:compare(actual,f.f32(`baseline_${stage}`))};
          }
          const allocations=[...p.keys.allocations,...values.allocations,...p.session.allocations];
          const bytes=(role:string)=>allocations.filter(a=>a.role===role).reduce((s,a)=>s+a.bytes,0);
          measurements.push({path:p.name,timing:await timePasses(device,e=>p.session.record(e),p.count),numerical,allocations,
            persistent_k_bytes:bytes("persistent-k"),shared_metadata_bytes:bytes("shared"),scratch_bytes:bytes("scratch"),persistent_v_bytes:bytes("persistent-v"),
            total_explicit_buffer_bytes:allocations.reduce((s,a)=>s+a.bytes,0),
            pass_labels:p.name==="phase3"?["decode_inverse","qk","softmax","v_accumulation"]:p.name==="phase4"?["fused_decode_inverse_qk","softmax","v_accumulation"]:["qk","softmax","v_accumulation"]});
        }
        cases.push({name:`t${tokens}_d${dim}`,context_length:tokens,head_dimension:dim,queries:3,causal:false,capacity:tokens,upstream_commit:m.upstream_commit,fixture_sha256:m.sha256,measurements});
      }finally{paths.forEach(p=>{p.session.destroy();p.keys.destroy();});values.destroy();}
    }
    if(errors.length)throw new Error(errors.join("\n"));return {passed:true,environment:gpuMetadata(adapter,device),cases,
      measurement:"GPU timestamps span all passes, excluding append, uploads, pipeline setup, command construction, readback. Raw per-pass samples and full-chain samples retained. Three paths coexist and share V; sequential timings use rotating path order. GPU workgroup scratch is 1024 bytes per active fused-dot workgroup, not a full-context GPUBuffer and not in explicit buffer ledger. Query/driver/pipeline sizes unobservable; timestamp buffers reported separately. No explicit hot-path clears. Not peak GPU memory or model throughput."};
  }finally{device.destroy();}
}
