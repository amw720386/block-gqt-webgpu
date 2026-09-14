export type Allocation = {name:string; role:"persistent-k"|"persistent-v"|"shared"|"scratch"; bytes:number};

export function storage(device:GPUDevice, bytes:number, label:string):GPUBuffer {
  const size=Math.ceil(bytes/4)*4;
  if(!Number.isSafeInteger(size)||size<4||size>device.limits.maxBufferSize||size>device.limits.maxStorageBufferBindingSize)
    throw new RangeError(`${label}: unsupported buffer size ${size}`);
  return device.createBuffer({label,size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
}
export function upload(device:GPUDevice,buffer:GPUBuffer,values:ArrayBufferView,offset=0):void {
  const data=new Uint8Array(Math.ceil(values.byteLength/4)*4);
  data.set(new Uint8Array(values.buffer,values.byteOffset,values.byteLength));
  if(offset%4||offset+data.byteLength>buffer.size) throw new RangeError("Unaligned/out-of-range buffer upload");
  device.queue.writeBuffer(buffer,offset,data);
}
export function checkFinite(values:Float32Array,label:string):void {
  if(values.some(x=>!Number.isFinite(x))) throw new RangeError(`${label}: finite inputs required`);
}
export async function readBuffer(device:GPUDevice,buffer:GPUBuffer,bytes=buffer.size):Promise<ArrayBuffer> {
  if(bytes%4||bytes>buffer.size)throw new RangeError("Invalid readback length");
  const readback=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try {
    const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(buffer,0,readback,0,bytes);
    device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
    return readback.getMappedRange().slice(0);
  } finally {readback.destroy();}
}
export async function shader(device:GPUDevice,url:URL,half=false):Promise<GPUShaderModule> {
  async function text(url:URL){const r=await fetch(url);if(!r.ok)throw new Error(`Shader HTTP ${r.status}: ${url}`);return r.text();}
  const code=(half?await text(new URL("./half.wgsl",import.meta.url)):"")+"\n"+await text(url);
  const module=device.createShaderModule({label:url.pathname,code});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  return module;
}
