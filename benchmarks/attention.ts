import {AttentionEngine,type AttentionSession} from "../src/attention/attention.js";
import {FP16Keys,type KeyStore} from "../src/runtime/keys.js";
import {CompressedKeys} from "../src/runtime/compressed-keys.js";
import {Values} from "../src/runtime/values.js";
import {readBuffer} from "../src/webgpu/buffers.js";
import {attentionFixture} from "../tests/webgpu/attention-fixture.js";
import {gpuMetadata} from "../tests/webgpu/phase2/phase2-test.js";
import {compare} from "../tests/webgpu/compare.js";

const warmups=5,repetitions=20;
function statistics(samples:number[]){
  const sorted=[...samples].sort((a,b)=>a-b),mean=samples.reduce((a,b)=>a+b,0)/samples.length;
  return {median:(sorted[9]+sorted[10])/2,mean,stddev:Math.sqrt(samples.reduce((s,x)=>s+(x-mean)**2,0)/samples.length),min:sorted[0],max:sorted.at(-1)};
}
async function measure(device:GPUDevice,session:AttentionSession){
  const query=device.createQuerySet({type:"timestamp",count:2});
  const resolve=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
  const staging=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const samples:number[]=[];
  try{
    for(let i=-warmups;i<repetitions;i++){
      const encoder=device.createCommandEncoder();session.record(encoder,query);
      encoder.resolveQuerySet(query,0,2,resolve,0);encoder.copyBufferToBuffer(resolve,0,staging,0,16);
      device.queue.submit([encoder.finish()]);await staging.mapAsync(GPUMapMode.READ);
      const stamps=new BigUint64Array(staging.getMappedRange()),us=Number(stamps[1]-stamps[0])/1000;
      staging.unmap();if(!Number.isFinite(us)||us<=0)throw new Error("Invalid GPU timestamps");if(i>=0)samples.push(us);
    }
    return {unit:"us",warmup_count:warmups,repetition_count:repetitions,samples,...statistics(samples),timestamp_buffer_bytes:32};
  }finally{query.destroy();resolve.destroy();staging.destroy();}
}
export async function benchmark(){
  const adapter=await navigator.gpu?.requestAdapter();if(!adapter?.features.has("timestamp-query"))throw new Error("GPU timestamps required");
  const device=await adapter.requestDevice({requiredFeatures:["timestamp-query"]}),errors:string[]=[];
  device.addEventListener("uncapturederror",e=>errors.push(e.error.message));
  try{
    const engine=await AttentionEngine.create(device);
    const index=await (await fetch("/fixtures/attention/bench-index.json")).json();
    if(index.cases.length!==12)throw new Error("Run python -m tests.oracles.export_attention --bench before attention benchmarks");
    const cases=[];
    for(const [caseIndex,c] of index.cases.entries()){
      const {fixture:f,m,parameters}=await attentionFixture(c.name),values=new Values(device,m.dim,m.context_length);
      values.append(f.f32("v"));const paths:KeyStore[]=[new FP16Keys(device,m.dim,m.context_length),await CompressedKeys.create(device,parameters,m.context_length)];
      const measurements=[];
      // Alternate path order across workloads; all paths use the same device/Q/V.
      if(caseIndex%2)paths.reverse();
      try{for(const keys of paths){
        await keys.append(f.halfBits("k"));const session=engine.session(keys,values,m.queries);
        try{
          await session.attend(f.f32("q"),f.u32("positions"),false);
          const actual=new Float32Array(await readBuffer(device,session.output,m.queries*m.dim*4));
          const label=keys.kind==="fp16"?"baseline":"compressed";
          const port=compare(actual,f.f32(`${label}_output`),m.tolerances[label==="baseline"?"baseline":"port"].output);
          const allocations=[...keys.allocations,...values.allocations,...session.allocations];
          const bytes=(role:string)=>allocations.filter(a=>a.role===role).reduce((s,a)=>s+a.bytes,0);
          measurements.push({path:label,latency:await measure(device,session),allocations,persistent_k_bytes:bytes("persistent-k"),shared_metadata_bytes:bytes("shared"),
            explicit_runtime_buffer_bytes:allocations.reduce((s,a)=>s+a.bytes,0),
            numerical_error_vs_own_reference:port,quantization_error:compare(f.f32("compressed_output"),f.f32("baseline_output")),
            total_output_error_vs_fp16:compare(actual,f.f32("baseline_output"))});
        }finally{session.destroy();}
      }}finally{paths.forEach(k=>k.destroy());values.destroy();}
      cases.push({name:c.name,context_length:m.context_length,head_dimension:m.dim,queries:m.queries,causal:false,capacity:m.context_length,
        fixture_sha256:m.sha256,upstream_commit:m.upstream_commit,measurements});
    }
    await device.queue.onSubmittedWorkDone();if(errors.length)throw new Error(errors.join("\n"));
    return {passed:true,environment:gpuMetadata(adapter,device),cases};
  }finally{device.destroy();}
}
