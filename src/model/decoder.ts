import {storage,upload,readBuffer} from "../webgpu/buffers.js";
import {Weights} from "./weights.js";
import {ModelOps} from "./ops.js";
import {Calibration} from "./calibration.js";
export type ModelAllocation={name:string;role:"weights"|"k"|"v"|"tables"|"scratch";bytes:number};
// Fixed, validated SmolLM2-135M architecture. No generic model abstraction.
export class Decoder {
  length=0;readonly allocations:ModelAllocation[]=[];private owned:GPUBuffer[]=[];private busy=false;
  private caches:{k:GPUBuffer;v:GPUBuffer}[]=[];
  private packed:{codes:GPUBuffer;norms:GPUBuffer}[]=[];encodedRows=0;
  private h:GPUBuffer[];private norm:GPUBuffer;private q:GPUBuffer;private k:GPUBuffer;private v:GPUBuffer;private qr:GPUBuffer;private kr:GPUBuffer;
  private attention:GPUBuffer;private projection:GPUBuffer;private gate:GPUBuffer;private up:GPUBuffer;private gated:GPUBuffer;
  private scores:GPUBuffer;private probabilities:GPUBuffer;private ids:GPUBuffer;private last:GPUBuffer;readonly logits:GPUBuffer;private dummy:GPUBuffer;
  maxTransientUniformBytes=0;
  private constructor(readonly weights:Weights,private ops:ModelOps,readonly capacity:number,readonly chunkSize:number,readonly calibration?:Calibration){
    const device=weights.device;if(!Number.isInteger(capacity)||capacity<1||capacity>8192||!Number.isInteger(chunkSize)||chunkSize<1||chunkSize>32)throw new RangeError("Unsupported capacity/chunk size");
    const make=(name:string,role:ModelAllocation['role'],bytes:number)=>{const b=storage(device,bytes,name);this.owned.push(b);this.allocations.push({name,role,bytes:b.size});return b;};
    this.allocations.push({name:"FP16 model weights (shared)",role:"weights",bytes:weights.bytes});
    if(calibration)this.allocations.push(...calibration.allocations);
    for(let l=0;l<30;l++)this.caches.push({k:make(`layer ${l} ${calibration?'unused K binding':'FP16 K'}`,calibration?"scratch":"k",calibration?4:3*capacity*64*2),v:make(`layer ${l} FP16 V`,"v",3*capacity*64*2)});
    if(calibration)for(let i=0;i<90;i++)this.packed.push({codes:make(`head ${i} packed K`,"k",capacity*calibration.heads[i].rowBytes),norms:make(`head ${i} corrected norms`,"k",capacity*calibration.heads[i].normStride*2)});
    const hidden=(name:string)=>make(name,"scratch",chunkSize*576*4);
    this.h=[hidden("hidden A"),hidden("hidden B")];this.norm=hidden("norm");this.q=hidden("Q projection");this.qr=hidden("Q RoPE");
    this.k=make("K projection","scratch",chunkSize*192*4);this.kr=make("K RoPE","scratch",chunkSize*192*4);this.v=make("V projection","scratch",chunkSize*192*4);
    this.attention=hidden("attention output");this.projection=hidden("output/down projection");
    this.gate=make("MLP gate","scratch",chunkSize*1536*4);this.up=make("MLP up","scratch",chunkSize*1536*4);this.gated=make("MLP product","scratch",chunkSize*1536*4);
    this.scores=make("attention logits","scratch",chunkSize*9*capacity*4);this.probabilities=make("attention probabilities","scratch",chunkSize*9*capacity*4);
    this.ids=make("token IDs","scratch",chunkSize*4);this.last=make("last hidden","scratch",576*4);this.logits=make("vocabulary logits","scratch",49152*4);this.dummy=make("unused binding","scratch",4);
  }
  static async create(weights:Weights,capacity:number,chunkSize=32,calibration?:Calibration){return new Decoder(weights,await ModelOps.create(weights.device),capacity,chunkSize,calibration);}
  async forward(tokens:Uint32Array,trace=false){
    if(this.busy)throw new Error("Concurrent model execution unsupported");if(!tokens.length||tokens.length>this.chunkSize||this.length+tokens.length>this.capacity||tokens.some(t=>t>=49152))throw new RangeError("Invalid tokens/capacity");
    this.busy=true;const device=this.weights.device,rows=tokens.length,start=this.length,cmd=this.ops.commands(),debug:GPUBuffer[]=[];let h=0;
    try{
      upload(device,this.ids,tokens);
      const standard=(name:string,p:number[],input:GPUBuffer,weight:GPUBuffer,output:GPUBuffer,x:number,y=1,extra=this.dummy)=>cmd.dispatch(name,p,[input,weight,output,extra],x,y);
      const mat=(input:GPUBuffer,weight:string,output:GPUBuffer,n:number,k:number,r=rows)=>standard("matmul",[r,n,k],input,this.weights.get(weight),output,Math.ceil(n/16),Math.ceil(r/16));
      standard("embedding",[rows,576],this.ids,this.weights.get("model.embed_tokens.weight"),this.h[h],Math.ceil(rows*576/128));
      for(let layer=0;layer<30;layer++){
        const prefix=`model.layers.${layer}.`,cache=this.caches[layer];
        standard("norm",[rows,576],this.h[h],this.weights.get(prefix+"input_layernorm.weight"),this.norm,rows);
        mat(this.norm,prefix+"self_attn.q_proj.weight",this.q,576,576);mat(this.norm,prefix+"self_attn.k_proj.weight",this.k,192,576);mat(this.norm,prefix+"self_attn.v_proj.weight",this.v,192,576);
        standard("rope",[rows,0,0,0,start,9],this.q,this.dummy,this.qr,Math.ceil(rows*9*32/128));
        standard("rope",[rows,0,0,0,start,3],this.k,this.dummy,this.kr,Math.ceil(rows*3*32/128));
        if(this.calibration){for(let head=0;head<3;head++){
          const index=layer*3+head,c=this.calibration.heads[index],packed=this.packed[index];
          cmd.dispatch("encode_k",[rows,start,head,start+rows],[c.config,c.table,this.kr,packed.codes,packed.norms],rows);
        }}else cmd.dispatch("append",[rows,start,this.capacity],[this.kr,cache.k],Math.ceil(rows*96/128));
        cmd.dispatch("append",[rows,start,this.capacity],[this.v,cache.v],Math.ceil(rows*96/128));
        const attentionBuffers=[this.qr,cache.k,cache.v,this.scores,this.probabilities,this.attention],p=[rows,start+rows,this.capacity,start];
        if(this.calibration){for(let head=0;head<3;head++){
          const index=layer*3+head,c=this.calibration.heads[index],packed=this.packed[index];
          cmd.dispatch("compressed_dot",[rows,start,head,start+rows],[c.config,c.table,this.qr,packed.codes,packed.norms,c.ranges,this.scores],start+rows);
        }}else cmd.dispatch("dot",p,attentionBuffers,start+rows,3);
        cmd.dispatch("softmax",p,attentionBuffers,rows*9);cmd.dispatch("weighted",p,attentionBuffers,rows*576);
        if(trace){
          for(const source of [this.attention,this.qr]){const b=storage(device,rows*576*4,`trace layer ${layer}`);debug.push(b);cmd.encoder.copyBufferToBuffer(source,0,b,0,b.size);}
          for(const source of [cache.k,cache.v]){
            if(this.calibration&&source===cache.k){const b=storage(device,rows*192*4,"test-only new post-RoPE K");debug.push(b);cmd.encoder.copyBufferToBuffer(this.kr,0,b,0,b.size);continue;}
            const b=storage(device,3*(start+rows)*64*2,`trace cache ${layer}`);debug.push(b);
            for(let head=0;head<3;head++)cmd.encoder.copyBufferToBuffer(source,head*this.capacity*64*2,b,head*(start+rows)*64*2,(start+rows)*64*2);
          }
        }
        mat(this.attention,prefix+"self_attn.o_proj.weight",this.projection,576,576);
        standard("element",[rows,576,0,0],this.projection,this.dummy,this.h[1-h],Math.ceil(rows*576/128),1,this.h[h]);h=1-h;
        standard("norm",[rows,576],this.h[h],this.weights.get(prefix+"post_attention_layernorm.weight"),this.norm,rows);
        mat(this.norm,prefix+"mlp.gate_proj.weight",this.gate,1536,576);mat(this.norm,prefix+"mlp.up_proj.weight",this.up,1536,576);
        standard("element",[rows,1536,0,1],this.gate,this.dummy,this.gated,Math.ceil(rows*1536/128),1,this.up);
        mat(this.gated,prefix+"mlp.down_proj.weight",this.projection,576,1536);
        standard("element",[rows,576,0,0],this.projection,this.dummy,this.h[1-h],Math.ceil(rows*576/128),1,this.h[h]);h=1-h;
      }
      standard("norm",[rows,576],this.h[h],this.weights.get("model.norm.weight"),this.norm,rows);
      cmd.encoder.copyBufferToBuffer(this.norm,(rows-1)*576*4,this.last,0,576*4);mat(this.last,"model.embed_tokens.weight",this.logits,49152,576,1);
      this.maxTransientUniformBytes=Math.max(this.maxTransientUniformBytes,cmd.uniformBytes);await cmd.submit();this.length+=rows;if(this.calibration)this.encodedRows+=rows*90;
      const logits=new Float32Array(await readBuffer(device,this.logits));const snapshots=await Promise.all(debug.map(b=>readBuffer(device,b)));
      const layers=Array.from({length:trace?30:0},(_,i)=>new Float32Array(snapshots[i*4]));
      const attentionInputs=Array.from({length:trace?30:0},(_,i)=>({q:new Float32Array(snapshots[i*4+1]),k:new Uint16Array(snapshots[i*4+2]),v:new Uint16Array(snapshots[i*4+3]),newK:this.calibration?new Float32Array(snapshots[i*4+2]):undefined}));
      return {logits,layers,attentionInputs};
    }finally{debug.forEach(b=>b.destroy());this.busy=false;}
  }
  async prefill(tokens:Uint32Array){let result;for(let i=0;i<tokens.length;i+=this.chunkSize)result=await this.forward(tokens.subarray(i,i+this.chunkSize));if(!result)throw new Error("Empty prompt");return result;}
  async decode(token:number){return this.forward(new Uint32Array([token]));}
  destroy(){this.owned.forEach(b=>b.destroy());}
}
