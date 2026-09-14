import { loadFixture } from "../fixture.js";
export type Fixture = Awaited<ReturnType<typeof loadFixture>>;
export type Kernel = "quantize" | "decode" | "decode_f32" | "convert_norms";

export async function mixedPipelines(device: GPUDevice) {
  const shader = device.createShaderModule({ code: await (await fetch("/tests/webgpu/phase2/mixed.wgsl")).text() });
  const errors = (await shader.getCompilationInfo()).messages.filter(x=>x.type==="error");
  if (errors.length) throw new Error(errors.map(x=>`${x.lineNum}: ${x.message}`).join("\n"));
  const layout = device.createBindGroupLayout({ entries: Array.from({length:8},(_,binding)=>({
    binding, visibility: 4, buffer: { type: binding < 3 ? "read-only-storage" as const : "storage" as const },
  })) });
  const pipelineLayout=device.createPipelineLayout({ bindGroupLayouts:[layout] });
  const pipelines = {} as Record<Kernel,GPUComputePipeline>;
  for (const entryPoint of ["quantize","decode","decode_f32","convert_norms"] as Kernel[])
    pipelines[entryPoint]=await device.createComputePipelineAsync({layout:pipelineLayout,compute:{module:shader,entryPoint}});
  return {layout,pipelines};
}

export function mixedBuffers(device: GPUDevice, fixture: Fixture,
  compiled: Awaited<ReturnType<typeof mixedPipelines>>, rows=fixture.manifest.rows, capacity=fixture.manifest.capacity) {
  const m=fixture.manifest, d=m.dim;
  if (m.version!==2 || m.packing!=="production-k-nibble-byte-v1" || d>128 || d%2 || m.n_groups>8 ||
      !Number.isInteger(rows) || rows<1 || capacity<rows) throw new Error("Unsupported mixed fixture/workload");
  // Upstream setup already groups dimensions; the secondary pack sort is identity.
  if (fixture.u32("pack_perm").some((x,i)=>x!==i)) throw new Error("Nonidentity secondary packing requires another path");
  const meta=[rows,d,m.n_groups,m.row_bytes,m.norm_stride,m.nopack_start,m.max_centroids,capacity,0,0,0,0];
  for (const [slot,key] of [[8,"head_perm"],[9,"group_of"],[10,"pos_to_cb"],[11,"code_lut"]] as const) {
    meta[slot]=meta.length;
    const values=key==="code_lut"?fixture.u8(key):fixture.u32(key);
    for (const v of values) meta.push(v);
  }
  const floats=Float32Array.from(["rotation","centroids","lut_offsets","lut_inv_scales"].flatMap(k=>Array.from(fixture.f32(k))));
  const original=fixture.f32("original"), input=new Float32Array(rows*d);
  for(let row=0;row<rows;row++) input.set(original.subarray((row%m.rows)*d,(row%m.rows+1)*d),row*d);
  const buffers: GPUBuffer[]=[];
  const ledger: {name:string;role:string;bytes:number}[]=[];
  function make(name:string,role:string,size:number) {
    const b=device.createBuffer({size:Math.ceil(size/4)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    buffers.push(b);ledger.push({name,role,bytes:b.size});return b;
  }
  function write(b:GPUBuffer,data:ArrayBufferView) {
    const bytes=new Uint8Array(Math.ceil(data.byteLength/4)*4);
    bytes.set(new Uint8Array(data.buffer,data.byteOffset,data.byteLength)); device.queue.writeBuffer(b,0,bytes);
  }
  const metadata=make("integer tables","shared_metadata",meta.length*4);write(metadata,new Uint32Array(meta));
  const tables=make("rotation/codebooks/LUT parameters","shared_metadata",floats.byteLength);write(tables,floats);
  const x=make("FP32 input","input",input.byteLength);write(x,input);
  const codes=make("debug indices","scratch",rows*d*4);
  const packed=make("reserved K bytes","persistent",capacity*m.row_bytes);
  const norms=make("reserved FP16 corrected norms","persistent",capacity*m.norm_stride*2);
  const scales=make("FP32 corrected scales","scratch",rows*m.norm_stride*4);
  const output=make("FP32 reconstruction","output",rows*d*4);
  const group=device.createBindGroup({layout:compiled.layout,entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
  function dispatch(encoder:GPUCommandEncoder,kernel:Kernel,timestampWrites?:GPUComputePassTimestampWrites,batch=1) {
    // Clearing precedes the timestamped pass; it is excluded from kernel timing.
    if(kernel==="quantize") {encoder.clearBuffer(packed);encoder.clearBuffer(norms);encoder.clearBuffer(scales);}
    if(kernel==="convert_norms") encoder.clearBuffer(norms);
    const pass=encoder.beginComputePass(timestampWrites?{timestampWrites}:{});
    pass.setPipeline(compiled.pipelines[kernel]);pass.setBindGroup(0,group);
    for(let i=0;i<batch;i++) pass.dispatchWorkgroups(rows);pass.end();
  }
  function golden() {
    if(rows!==m.rows) throw new Error("Golden upload only for fixture-sized workloads");
    write(packed,fixture.u8("cache_packed"));write(norms,fixture.halfBits("cache_norms"));write(scales,fixture.f32("corrected_f32"));
  }
  async function read() {
    const sources=[codes,packed,norms,scales,output];
    const targets=sources.map(b=>device.createBuffer({size:b.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}));
    try {
      const encoder=device.createCommandEncoder();sources.forEach((b,i)=>encoder.copyBufferToBuffer(b,0,targets[i],0,b.size));
      device.queue.submit([encoder.finish()]);await Promise.all(targets.map(b=>b.mapAsync(GPUMapMode.READ)));
      const data=targets.map(b=>b.getMappedRange().slice(0));
      return {codes:new Uint32Array(data[0]),packed:new Uint8Array(data[1]),norms:new Uint16Array(data[2]),scales:new Float32Array(data[3]),values:new Float32Array(data[4])};
    } finally {targets.forEach(b=>b.destroy());}
  }
  return {dispatch,golden,read,ledger,rows,capacity,writeScales:(data:Float32Array)=>write(scales,data),destroy:()=>buffers.forEach(b=>b.destroy())};
}
