import {storage,upload,shader,checkFinite,type Allocation} from "../webgpu/buffers.js";
import {PackedKeys} from "../runtime/packed-keys.js";
import {Values} from "../runtime/values.js";

export class OptimizedAttention {
  readonly allocations:Allocation[]=[];readonly output:GPUBuffer;
  private buffers:GPUBuffer[]=[];private q:GPUBuffer;private positions:GPUBuffer;private logits:GPUBuffer;private probabilities:GPUBuffer;private params:GPUBuffer;
  private queries=0;private causal=false;private dotGroup:GPUBindGroup;private tailGroup:GPUBindGroup;
  private constructor(readonly device:GPUDevice,readonly keys:PackedKeys,readonly values:Values,readonly maxQueries:number,
    private dot:GPUComputePipeline,private softmax:GPUComputePipeline,private accumulate:GPUComputePipeline,readonly tailMode:"serial"|"parallel"){
    if(keys.device!==device||values.device!==device||keys.dim!==values.dim||keys.capacity!==values.capacity||!Number.isInteger(maxQueries)||maxQueries<1)throw new RangeError("Incompatible attention storage");
    const make=(name:string,size:number)=>{const b=storage(device,size,name);this.buffers.push(b);this.allocations.push({name,role:"scratch",bytes:b.size});return b;};
    this.q=make("Q",maxQueries*keys.dim*4);this.positions=make("positions",maxQueries*4);
    this.logits=make("logits",maxQueries*keys.capacity*4);this.probabilities=make("probabilities",maxQueries*keys.capacity*4);
    this.output=make("output",maxQueries*keys.dim*4);
    this.params=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.buffers.push(this.params);this.allocations.push({name:"parameters",role:"scratch",bytes:32});
    this.dotGroup=device.createBindGroup({layout:dot.getBindGroupLayout(0),entries:[this.params,keys.integers,keys.floats,keys.packed,keys.norms,this.q,this.positions,this.logits,keys.ranges].map((buffer,binding)=>({binding,resource:{buffer}}))});
    this.tailGroup=device.createBindGroup({layout:softmax.getBindGroupLayout(0),entries:[this.params,this.q,keys.packed,values.buffer,this.positions,this.logits,this.probabilities,this.output].map((buffer,binding)=>({binding,resource:{buffer}}))});
  }
  static async create(device:GPUDevice,keys:PackedKeys,values:Values,maxQueries=3,tailMode:"serial"|"parallel"="parallel"){
    const fused=await shader(device,new URL("./fused-dot.wgsl",import.meta.url),true),tail=await shader(device,new URL(tailMode==="serial"?"./attention.wgsl":"./parallel-tail.wgsl",import.meta.url),true);
    const dot=await device.createComputePipelineAsync({layout:"auto",compute:{module:fused,entryPoint:"dot"}});
    const layout=device.createBindGroupLayout({entries:Array.from({length:8},(_,binding)=>({binding,visibility:4,buffer:{type:binding===0?"uniform" as const:binding<5?"read-only-storage" as const:"storage" as const}}))});
    const pl=device.createPipelineLayout({bindGroupLayouts:[layout]});
    const [softmax,accumulate]=await Promise.all(["softmax","weighted_sum"].map(entryPoint=>device.createComputePipelineAsync({layout:pl,compute:{module:tail,entryPoint}})));
    return new OptimizedAttention(device,keys,values,maxQueries,dot,softmax,accumulate,tailMode);
  }
  setQueries(q:Float32Array,positions:Uint32Array,causal=false){
    this.keys.ready();const n=q.length/this.keys.dim;
    if(this.keys.length!==this.values.length||!Number.isInteger(n)||n<1||n>this.maxQueries||positions.length!==n||positions.some(x=>x>=this.keys.length))throw new RangeError("Invalid Q/K/V shape or positions");
    checkFinite(q,"Q");upload(this.device,this.q,q);upload(this.device,this.positions,positions);this.queries=n;this.causal=causal;
  }
  record(e:GPUCommandEncoder){
    this.keys.ready();if(!this.queries||this.keys.length!==this.values.length)throw new Error("Attention not ready");
    upload(this.device,this.params,new Uint32Array([this.keys.length,this.keys.dim,this.queries,+this.causal,0,0,0,0]));
    const accumulationGroups=this.tailMode==="serial"?Math.ceil(this.queries*this.keys.dim/64):this.queries*this.keys.dim;
    for(const [pipeline,group,count] of [[this.dot,this.dotGroup,this.keys.length],[this.softmax,this.tailGroup,this.queries],[this.accumulate,this.tailGroup,accumulationGroups]] as const){
      if(count>this.device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError("Dispatch limit");
      const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(count);p.end();
    }
  }
  async attend(q:Float32Array,positions:Uint32Array,causal=false){this.setQueries(q,positions,causal);const e=this.device.createCommandEncoder();this.record(e);this.device.queue.submit([e.finish()]);await this.device.queue.onSubmittedWorkDone();return this.output;}
  inspect(){return {logits:this.logits,probabilities:this.probabilities,output:this.output,queries:this.queries,tokens:this.keys.length};}
  destroy(){this.buffers.forEach(b=>b.destroy());}
}
