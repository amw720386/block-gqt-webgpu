import {loadFixture} from "../fixture.js";
import {mixedBuffers,mixedPipelines,type Kernel} from "./mixed.js";
import {gpuMetadata,validateMixed} from "./phase2-test.js";

export function summary(samples:number[]) {
  const ordered=[...samples].sort((a,b)=>a-b),mean=samples.reduce((a,b)=>a+b,0)/samples.length;
  return {median:(ordered[(ordered.length-1)>>1]+ordered[ordered.length>>1])/2,mean,
    stddev:Math.sqrt(samples.reduce((s,x)=>s+(x-mean)**2,0)/samples.length),min:ordered[0],max:ordered.at(-1)!};
}
export async function benchmark() {
  const adapter=await navigator.gpu?.requestAdapter();if(!adapter) throw new Error("No WebGPU adapter");
  if(!adapter.features.has("timestamp-query")) throw new Error("Timestamp queries required; no host-timing substitution");
  const device=await adapter.requestDevice({requiredFeatures:["timestamp-query"]});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>errors.push(e.error.message));
  try {
    // Gate diagnostics on the same device/configuration's correctness tests.
    const correctness=await validateMixed(device);
    const compiled=await mixedPipelines(device);
    const index=await (await fetch("/fixtures/mixed/index.json")).json();
    const records=[];
    const warmup=10,repetitions=30,batch=20,rows=256,capacity=512;
    for(const c of index.cases.filter((x:{name:string})=>/^(low|mixed)_d/.test(x.name))) {
      const f=await loadFixture(`/fixtures/mixed/${c.name}.json`),m=f.manifest;
      const b=mixedBuffers(device,f,compiled,rows,capacity);
      const baseline=device.createBuffer({size:capacity*m.dim*2,usage:GPUBufferUsage.STORAGE});
      const query=device.createQuerySet({type:"timestamp",count:2});
      const resolve=device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      const readback=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try {
        let init=device.createCommandEncoder();b.dispatch(init,"quantize");device.queue.submit([init.finish()]);await device.queue.onSubmittedWorkDone();
        const timings=[];
        for(const kernel of ["quantize","decode"] as Kernel[]) {
          const samples=[];
          for(let rep=-warmup;rep<repetitions;rep++) {
            const encoder=device.createCommandEncoder();
            b.dispatch(encoder,kernel,{querySet:query,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1},batch);
            encoder.resolveQuerySet(query,0,2,resolve,0);encoder.copyBufferToBuffer(resolve,0,readback,0,16);
            device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
            const times=new BigUint64Array(readback.getMappedRange()),nanoseconds=Number(times[1]-times[0]);
            readback.unmap();if(rep>=0)samples.push(nanoseconds/1000/batch);
          }
          timings.push({kernel,environment_kind:"WebGPU GPU timestamps",unit:"microseconds_per_dispatch",
            measurement_boundary:"compute pass timestamps / dispatches_per_sample; excludes clears, uploads, readback, compilation and queue submission",
            warmup_count:warmup,repetition_count:repetitions,dispatches_per_sample:batch,samples,statistics:summary(samples)});
        }
        const persistent=b.ledger.filter(x=>x.role==="persistent").reduce((s,x)=>s+x.bytes,0);
        const metadata=b.ledger.filter(x=>x.role==="shared_metadata").reduce((s,x)=>s+x.bytes,0);
        records.push({fixture:c.name,fixture_sha256:m.sha256,upstream_commit:m.upstream_commit,
          workload:{rows,capacity,dim:m.dim,allocation:Array.from(f.u32("allocation")),average_bits:m.average_bits,
            row_bytes:m.row_bytes,logical_row_bytes:m.logical_row_bytes,norm_stride:m.norm_stride,coordinate_space:m.coordinate_space,
            distribution:"repeat the five deterministic fixture rows",capacity_utilization:rows/capacity},
          memory:{allocated_buffers:b.ledger,baseline_fp16_bytes:baseline.size,persistent_bytes:persistent,shared_metadata_bytes:metadata,
            effective_compression_ratio:baseline.size/(persistent+metadata),persistent_only_ratio:baseline.size/persistent,
            bytes_per_active_vector:(persistent+metadata)/rows,
            timing_buffers:[{name:"query resolve",bytes:resolve.size},{name:"timestamp readback",bytes:readback.size}],
            note:"Buffer sizes are actual GPUBuffer.size. Driver allocations/query-set backing are not observable; no GPU peak-memory claim."},timings});
      } finally {b.destroy();baseline.destroy();query.destroy();resolve.destroy();readback.destroy();}
    }
    if(errors.length) throw new Error(errors.join("\n"));
    return {passed:true,evidence_kind:"isolated_kernel_development_microbenchmark",environment:gpuMetadata(adapter,device),
      correctness_passed:correctness.length,records};
  } finally {device.destroy();}
}
