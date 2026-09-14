export function stats(samples:number[]){
  const sorted=[...samples].sort((a,b)=>a-b),n=sorted.length,mean=samples.reduce((s,x)=>s+x,0)/n;
  return {samples,median:(sorted[Math.floor((n-1)/2)]+sorted[Math.floor(n/2)])/2,mean,
    stddev:Math.sqrt(samples.reduce((s,x)=>s+(x-mean)**2,0)/n),min:sorted[0],max:sorted[n-1],unit:"us"};
}
// Timestamp unchanged passes through an encoder proxy; no baseline source edits.
export async function timePasses(device:GPUDevice,record:(e:GPUCommandEncoder)=>void,passes:number,warmups=5,repetitions=20){
  const query=device.createQuerySet({type:"timestamp",count:passes*2});
  const resolved=device.createBuffer({size:passes*16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
  const mapped=device.createBuffer({size:passes*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const samples=Array.from({length:passes},()=>[] as number[]),total:number[]=[],host:number[]=[];
  try{
    for(let r=-warmups;r<repetitions;r++){
      const encoder=device.createCommandEncoder();let index=0;
      const proxy=new Proxy(encoder,{get(target,key){
        if(key==="beginComputePass")return (descriptor:GPUComputePassDescriptor={})=>{
          const i=index++;return target.beginComputePass({...descriptor,timestampWrites:{querySet:query,beginningOfPassWriteIndex:i*2,endOfPassWriteIndex:i*2+1}});
        };
        const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;
      }});
      const start=performance.now();record(proxy);const construction=(performance.now()-start)*1000;
      if(index!==passes)throw new Error(`Expected ${passes} passes, got ${index}`);
      encoder.resolveQuerySet(query,0,passes*2,resolved,0);encoder.copyBufferToBuffer(resolved,0,mapped,0,passes*16);
      device.queue.submit([encoder.finish()]);await mapped.mapAsync(GPUMapMode.READ);
      const t=new BigUint64Array(mapped.getMappedRange());
      if(r>=0){for(let i=0;i<passes;i++)samples[i].push(Number(t[i*2+1]-t[i*2])/1000);total.push(Number(t[passes*2-1]-t[0])/1000);host.push(construction);}
      mapped.unmap();
    }
    return {warmup_count:warmups,repetition_count:repetitions,passes:samples.map(stats),total:stats(total),host_record:stats(host),measurement_buffers_bytes:passes*32};
  }finally{query.destroy();resolved.destroy();mapped.destroy();}
}
