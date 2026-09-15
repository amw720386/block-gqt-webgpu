import {storage,upload} from "../webgpu/buffers.js";
export type Weight={buffer:GPUBuffer;shape:number[];bytes:number};
export class Weights {
  readonly tensors=new Map<string,Weight>();bytes=0;
  private constructor(readonly device:GPUDevice,readonly manifest:Record<string,unknown>){}
  static async load(device:GPUDevice,base="/models/smollm2"){
    const manifest=await (await fetch(`${base}/manifest.json`)).json();
    const response=await fetch(`${base}/fp16.safetensors`);if(!response.ok)throw new Error("Run model setup first");const bytes=await response.arrayBuffer();
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),b=>b.toString(16).padStart(2,"0")).join("");
    if(digest!==manifest.sha256['fp16.safetensors'])throw new Error("Model hash mismatch");
    const view=new DataView(bytes);const length=Number(view.getBigUint64(0,true));
    const header=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,8,length))),weights=new Weights(device,manifest);
    try{for(const [name,entry] of Object.entries(header)){
      if(name==="__metadata__")continue;const t=entry as {dtype:string;shape:number[];data_offsets:number[]};
      if(t.dtype!=="F16")throw new Error("Expected exported FP16 weights");const [begin,end]=t.data_offsets;
      if(end-begin!==t.shape.reduce((a,b)=>a*b,2))throw new Error("Invalid tensor length");
      const buffer=storage(device,end-begin,name);upload(device,buffer,new Uint8Array(bytes,8+length+begin,end-begin));
      weights.tensors.set(name,{buffer,shape:t.shape,bytes:buffer.size});weights.bytes+=buffer.size;
    }}catch(e){weights.destroy();throw e;}return weights;
  }
  get(name:string){const w=this.tensors.get(name);if(!w)throw new Error(`Missing weight ${name}`);return w.buffer;}
  destroy(){this.tensors.forEach(t=>t.buffer.destroy());}
}
