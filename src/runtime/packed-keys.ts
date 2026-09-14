import {storage,upload,shader,type Allocation} from "../webgpu/buffers.js";
import {tables,type BlockGTQParameters} from "../webgpu/layout.js";
import {appendCount} from "./keys.js";

// Phase 4 compressed storage. Uses the unchanged Phase 3 encoder, but allocates
// only its unused four-byte output binding, never a reconstructed K tensor.
export class PackedKeys {
  readonly dim:number;length=0;private pending=false;readonly allocations:Allocation[]=[];
  readonly integers:GPUBuffer;readonly floats:GPUBuffer;readonly packed:GPUBuffer;readonly norms:GPUBuffer;readonly ranges:GPUBuffer;
  private readonly dummy:GPUBuffer;private readonly buffers:GPUBuffer[]=[];
  private constructor(readonly device:GPUDevice,readonly parameters:BlockGTQParameters,readonly capacity:number,
    private layout:GPUBindGroupLayout,private pipeline:GPUComputePipeline){
    if(!Number.isInteger(capacity)||capacity<1||capacity>device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError("Invalid capacity");
    this.dim=parameters.dim;const t=tables(parameters),d=this.dim;
    // Prove that skipping off-group entries preserves this exported transform.
    const ranges=new Uint32Array(d*2);
    for(let j=0;j<d;j++){
      const members=Array.from({length:d},(_,i)=>i).filter(i=>parameters.groupOf[i]===parameters.groupOf[j]);
      const start=members[0],end=members.at(-1)!+1;
      if(end-start!==members.length)throw new Error("Non-contiguous transform group");
      for(let i=0;i<d;i++)if((i<start||i>=end)&&parameters.rotation[i*d+j]!==0)throw new Error("Off-group transform is nonzero");
      ranges[j*2]=start;ranges[j*2+1]=end;
    }
    const make=(name:string,role:Allocation['role'],bytes:number)=>{const b=storage(device,bytes,name);this.buffers.push(b);this.allocations.push({name,role,bytes:b.size});return b;};
    this.integers=make("K integer tables","shared",t.integer.byteLength);upload(device,this.integers,t.integer);
    this.floats=make("K float tables","shared",t.floating.byteLength);upload(device,this.floats,t.floating);
    this.ranges=make("inverse group bounds","shared",ranges.byteLength);upload(device,this.ranges,ranges);
    this.packed=make("K nibble/byte capacity","persistent-k",capacity*parameters.rowBytes);
    this.norms=make("K FP16 scale capacity","persistent-k",capacity*parameters.normStride*2);
    this.dummy=make("unused encoder output","scratch",4);
  }
  static async create(device:GPUDevice,p:BlockGTQParameters,capacity:number){
    tables(p);const module=await shader(device,new URL("../webgpu/mixed.wgsl",import.meta.url),true);
    const layout=device.createBindGroupLayout({entries:Array.from({length:7},(_,binding)=>({binding,visibility:4,
      buffer:{type:binding===6?"uniform" as const:binding<3?"read-only-storage" as const:"storage" as const}}))});
    const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"encode"}});
    return new PackedKeys(device,p,capacity,layout,pipeline);
  }
  async append(values:Uint16Array){
    if(this.pending)throw new Error("Concurrent K appends unsupported");const count=appendCount(values,this.dim,this.length,this.capacity);
    this.pending=true;let input:GPUBuffer|undefined,params:GPUBuffer|undefined;
    try{
      input=storage(this.device,values.byteLength,"K append input");upload(this.device,input,values);
      params=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});upload(this.device,params,new Uint32Array([count,this.length,0,0]));
      const group=this.device.createBindGroup({layout:this.layout,entries:[this.integers,this.floats,input,this.packed,this.norms,this.dummy,params].map((buffer,binding)=>({binding,resource:{buffer}}))});
      const e=this.device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(this.pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(count);pass.end();
      this.device.queue.submit([e.finish()]);await this.device.queue.onSubmittedWorkDone();this.length+=count;
    }finally{input?.destroy();params?.destroy();this.pending=false;}
  }
  ready(){if(this.pending||!this.length)throw new Error("K storage not ready");}
  destroy(){this.buffers.forEach(b=>b.destroy());}
}
