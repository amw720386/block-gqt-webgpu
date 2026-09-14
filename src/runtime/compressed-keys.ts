import {storage,upload,shader,type Allocation} from "../webgpu/buffers.js";
import {tables,type BlockGTQParameters} from "../webgpu/layout.js";
import {appendCount,type KeyStore,type KeyView} from "./keys.js";

export class CompressedKeys implements KeyStore {
  readonly kind="block-gtq" as const;readonly dim:number;length=0;private pending=false;
  readonly allocations:Allocation[]=[];
  private readonly buffers:GPUBuffer[]=[];
  private readonly integers:GPUBuffer;private readonly floats:GPUBuffer;
  private readonly packed:GPUBuffer;private readonly norms:GPUBuffer;private readonly decoded:GPUBuffer;
  private readonly decodeParams:GPUBuffer;private readonly dummy:GPUBuffer;
  private constructor(readonly device:GPUDevice,readonly parameters:BlockGTQParameters,readonly capacity:number,
    private readonly layout:GPUBindGroupLayout,private readonly encodePipeline:GPUComputePipeline,private readonly decodePipeline:GPUComputePipeline){
    if(!Number.isInteger(capacity)||capacity<1||capacity>device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError("Unsupported K capacity");
    this.dim=parameters.dim;const t=tables(parameters);
    const make=(name:string,role:Allocation["role"],bytes:number)=>{const b=storage(device,bytes,name);this.buffers.push(b);this.allocations.push({name,role,bytes:b.size});return b;};
    this.integers=make("K integer tables","shared",t.integer.byteLength);upload(device,this.integers,t.integer);
    this.floats=make("K float tables","shared",t.floating.byteLength);upload(device,this.floats,t.floating);
    this.packed=make("K nibble/byte capacity","persistent-k",capacity*parameters.rowBytes);
    this.norms=make("K FP16 scale capacity","persistent-k",capacity*parameters.normStride*2);
    this.decoded=make("materialized FP32 K","scratch",capacity*this.dim*4);
    this.dummy=make("unused decode input","scratch",4);
    this.decodeParams=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.buffers.push(this.decodeParams);this.allocations.push({name:"decode parameters",role:"scratch",bytes:16});
    // WebGPU initializes new buffers to zero. Rows are append-only, never overwritten.
  }
  static async create(device:GPUDevice,p:BlockGTQParameters,capacity:number):Promise<CompressedKeys>{
    tables(p);
    const module=await shader(device,new URL("../webgpu/mixed.wgsl",import.meta.url),true);
    const layout=device.createBindGroupLayout({entries:Array.from({length:7},(_,binding)=>({binding,visibility:4,
      buffer:{type:binding===6?"uniform" as const:binding<3?"read-only-storage" as const:"storage" as const}}))});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
    const pipelines=await Promise.all(["encode","decode"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
    return new CompressedKeys(device,p,capacity,layout,pipelines[0],pipelines[1]);
  }
  private bindings(input:GPUBuffer,params:GPUBuffer):GPUBindGroup{
    return this.device.createBindGroup({layout:this.layout,entries:[this.integers,this.floats,input,this.packed,this.norms,this.decoded,params].map((buffer,binding)=>({binding,resource:{buffer}}))});
  }
  async append(values:Uint16Array):Promise<void>{
    if(this.pending)throw new Error("Concurrent K appends are not supported");
    const count=appendCount(values,this.dim,this.length,this.capacity);this.pending=true;
    const input=storage(this.device,values.byteLength,"K append input");upload(this.device,input,values);
    const params=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    upload(this.device,params,new Uint32Array([count,this.length,0,0]));
    try{
      const encoder=this.device.createCommandEncoder(),pass=encoder.beginComputePass();
      pass.setPipeline(this.encodePipeline);pass.setBindGroup(0,this.bindings(input,params));pass.dispatchWorkgroups(count);pass.end();
      this.device.queue.submit([encoder.finish()]);await this.device.queue.onSubmittedWorkDone();this.length+=count;
    }finally{input.destroy();params.destroy();this.pending=false;}
  }
  prepare(encoder:GPUCommandEncoder,start?:GPUComputePassTimestampWrites):KeyView{
    if(this.pending)throw new Error("K append is still pending");
    upload(this.device,this.decodeParams,new Uint32Array([this.length,0,0,0]));
    const pass=encoder.beginComputePass(start?{timestampWrites:start}:{});
    pass.setPipeline(this.decodePipeline);pass.setBindGroup(0,this.bindings(this.dummy,this.decodeParams));pass.dispatchWorkgroups(this.length);pass.end();
    return {buffer:this.decoded,format:"f32"};
  }
  destroy():void{this.buffers.forEach(b=>b.destroy());}
}
