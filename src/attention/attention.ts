import {storage,upload,shader,checkFinite,type Allocation} from "../webgpu/buffers.js";
import type {KeyStore} from "../runtime/keys.js";
import type {Values} from "../runtime/values.js";

type Pipelines={layout:GPUBindGroupLayout;dot:GPUComputePipeline;softmax:GPUComputePipeline;weighted_sum:GPUComputePipeline};
export class AttentionEngine {
  private constructor(readonly device:GPUDevice,private pipelines:Pipelines){}
  static async create(device:GPUDevice):Promise<AttentionEngine>{
    const module=await shader(device,new URL("./attention.wgsl",import.meta.url),true);
    const layout=device.createBindGroupLayout({entries:Array.from({length:8},(_,binding)=>({binding,visibility:4,
      buffer:{type:binding===0?"uniform" as const:binding<5?"read-only-storage" as const:"storage" as const}}))});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
    const pipelines=await Promise.all(["dot","softmax","weighted_sum"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
    return new AttentionEngine(device,{layout,dot:pipelines[0],softmax:pipelines[1],weighted_sum:pipelines[2]});
  }
  session(keys:KeyStore,values:Values,maxQueries=3):AttentionSession{return new AttentionSession(this.device,this.pipelines,keys,values,maxQueries);}
}

export class AttentionSession {
  readonly allocations:Allocation[]=[];readonly output:GPUBuffer;
  private readonly buffers:GPUBuffer[]=[];private readonly q:GPUBuffer;private readonly positions:GPUBuffer;
  private readonly params:GPUBuffer;private readonly logits:GPUBuffer;private readonly probabilities:GPUBuffer;
  private queries=0;private causal=false;
  constructor(readonly device:GPUDevice,private pipelines:Pipelines,readonly keys:KeyStore,readonly values:Values,readonly maxQueries:number){
    if(keys.device!==device||values.device!==device||keys.dim!==values.dim||keys.capacity!==values.capacity||
        !Number.isInteger(maxQueries)||maxQueries<1)throw new RangeError("Incompatible attention storage");
    const make=(name:string,size:number)=>{const b=storage(device,size,name);this.buffers.push(b);this.allocations.push({name,role:"scratch",bytes:b.size});return b;};
    this.q=make("Q",maxQueries*keys.dim*4);this.positions=make("query positions",maxQueries*4);
    this.logits=make("attention logits",maxQueries*keys.capacity*4);
    this.probabilities=make("attention probabilities",maxQueries*keys.capacity*4);
    this.output=make("attention output",maxQueries*keys.dim*4);
    this.params=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.buffers.push(this.params);this.allocations.push({name:"attention parameters",role:"scratch",bytes:32});
  }
  setQueries(q:Float32Array,positions:Uint32Array,causal=false):void{
    if(this.keys.length!==this.values.length||!this.keys.length)throw new RangeError("K/V lengths must match and be nonzero");
    const count=q.length/this.keys.dim;
    if(!Number.isInteger(count)||count<1||count>this.maxQueries||positions.length!==count||positions.some(x=>x>=this.keys.length))
      throw new RangeError("Invalid query shape or causal positions");
    checkFinite(q,"Q");upload(this.device,this.q,q);upload(this.device,this.positions,positions);this.queries=count;this.causal=causal;
  }
  // GPU-only operation. K.prepare may materialize K today; a fused consumer can
  // replace this boundary without changing append/V ownership or Q semantics.
  record(encoder:GPUCommandEncoder,timestamps?:GPUQuerySet):void{
    if(!this.queries||this.keys.length!==this.values.length)throw new Error("Queries/storage not ready");
    const start=timestamps?{querySet:timestamps,beginningOfPassWriteIndex:0}:undefined;
    const keys=this.keys.prepare(encoder,this.keys.kind==="block-gtq"?start:undefined);
    upload(this.device,this.params,new Uint32Array([this.keys.length,this.keys.dim,this.queries,+this.causal,+(keys.format==="f32"),0,0,0]));
    const group=this.device.createBindGroup({layout:this.pipelines.layout,entries:[this.params,this.q,keys.buffer,this.values.buffer,this.positions,this.logits,this.probabilities,this.output]
      .map((buffer,binding)=>({binding,resource:{buffer}}))});
    const stages=[{pipeline:this.pipelines.dot,groups:Math.ceil(this.queries*this.keys.length/64),stamp:this.keys.kind==="fp16"?start:undefined},
      {pipeline:this.pipelines.softmax,groups:this.queries,stamp:undefined},
      {pipeline:this.pipelines.weighted_sum,groups:Math.ceil(this.queries*this.keys.dim/64),stamp:timestamps?{querySet:timestamps,endOfPassWriteIndex:1}:undefined}];
    for(const stage of stages){
      if(stage.groups>this.device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError("Attention dispatch exceeds device limits");
      const pass=encoder.beginComputePass(stage.stamp?{timestampWrites:stage.stamp}:{});
      pass.setPipeline(stage.pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(stage.groups);pass.end();
    }
  }
  async attend(q:Float32Array,positions:Uint32Array,causal=false):Promise<GPUBuffer>{
    this.setQueries(q,positions,causal);const encoder=this.device.createCommandEncoder();this.record(encoder);
    this.device.queue.submit([encoder.finish()]);await this.device.queue.onSubmittedWorkDone();return this.output;
  }
  // Optional inspection adapter: these are real algorithm intermediates, not
  // persistent K storage. Normal callers only consume attend()'s GPU output.
  inspect():{logits:GPUBuffer;probabilities:GPUBuffer;output:GPUBuffer;queries:number;tokens:number}{
    return {logits:this.logits,probabilities:this.probabilities,output:this.output,queries:this.queries,tokens:this.keys.length};
  }
  destroy():void{this.buffers.forEach(b=>b.destroy());}
}
