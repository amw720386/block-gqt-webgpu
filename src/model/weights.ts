import {storage,upload} from "../webgpu/buffers.js";
import {modelAdapter,type ModelAdapter} from "./adapter.js";
export type Weight={buffer:GPUBuffer;shape:number[];bytes:number};
export class Weights {
  readonly tensors=new Map<string,Weight>();bytes=0;
  private constructor(readonly device:GPUDevice,readonly manifest:Record<string,any>,readonly adapter:ModelAdapter){}
  static async load(device:GPUDevice,base="/models/smollm2"){
    const [manifest,config]=await Promise.all([fetch(`${base}/manifest.json`).then(r=>{if(!r.ok)throw new Error("Model manifest unavailable");return r.json();}),fetch(`${base}/config.json`).then(r=>{if(!r.ok)throw new Error("Model config unavailable");return r.json();})]);
    const adapter=modelAdapter(config);
    const response=await fetch(`${base}/fp16.safetensors`);if(!response.ok)throw new Error("Run model setup first");const bytes=await response.arrayBuffer();
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),b=>b.toString(16).padStart(2,"0")).join("");
    if(digest!==manifest.sha256['fp16.safetensors'])throw new Error("Model hash mismatch");
    const view=new DataView(bytes);const length=Number(view.getBigUint64(0,true));
    const header=JSON.parse(new TextDecoder().decode(new Uint8Array(bytes,8,length))),weights=new Weights(device,manifest,adapter);
    try{for(const [name,entry] of Object.entries(header)){
      if(name==="__metadata__")continue;const t=entry as {dtype:string;shape:number[];data_offsets:number[]};
      if(t.dtype!=="F16")throw new Error("Expected exported FP16 weights");const [begin,end]=t.data_offsets;
      if(end-begin!==t.shape.reduce((a,b)=>a*b,2))throw new Error("Invalid tensor length");
      const buffer=storage(device,end-begin,name);upload(device,buffer,new Uint8Array(bytes,8+length+begin,end-begin));
      weights.tensors.set(name,{buffer,shape:t.shape,bytes:buffer.size});weights.bytes+=buffer.size;
    }}catch(e){weights.destroy();throw e;}weights.validate();return weights;
  }
  get(name:string){const w=this.tensors.get(name);if(!w)throw new Error(`Missing weight ${name}`);return w.buffer;}
  private validate(){
    const a=this.adapter,shape=(name:string,expected:number[])=>{const actual=this.tensors.get(name)?.shape;if(!actual||actual.length!==expected.length||actual.some((v,i)=>v!==expected[i]))throw new Error(`Invalid weight shape ${name}`);};
    shape("model.embed_tokens.weight",[a.vocabSize,a.hiddenSize]);shape("model.norm.weight",[a.hiddenSize]);
    for(let layer=0;layer<a.layers;layer++){
      const p=`model.layers.${layer}.`;shape(p+"input_layernorm.weight",[a.hiddenSize]);shape(p+"post_attention_layernorm.weight",[a.hiddenSize]);
      shape(p+"self_attn.q_proj.weight",[a.queryHeads*a.headDim,a.hiddenSize]);shape(p+"self_attn.k_proj.weight",[a.kvHeads*a.headDim,a.hiddenSize]);shape(p+"self_attn.v_proj.weight",[a.kvHeads*a.headDim,a.hiddenSize]);shape(p+"self_attn.o_proj.weight",[a.hiddenSize,a.hiddenSize]);
      shape(p+"mlp.gate_proj.weight",[a.intermediateSize,a.hiddenSize]);shape(p+"mlp.up_proj.weight",[a.intermediateSize,a.hiddenSize]);shape(p+"mlp.down_proj.weight",[a.hiddenSize,a.intermediateSize]);
      for(const projection of ["q_proj","k_proj","v_proj"] as const){const bias=a.attentionBias(p,projection);if(bias)shape(bias,[projection==="q_proj"?a.queryHeads*a.headDim:a.kvHeads*a.headDim]);}
    }
  }
  destroy(){this.tensors.forEach(t=>t.buffer.destroy());}
}
