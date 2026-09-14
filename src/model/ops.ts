import {shader} from "../webgpu/buffers.js";
export class ModelOps {
  private constructor(readonly device:GPUDevice,readonly kernels:Record<string,GPUComputePipeline>){}
  static async create(device:GPUDevice){
    const kernels:Record<string,GPUComputePipeline>={};
    for(const [file,names,types] of [
      ["kernels.wgsl",["embedding","matmul","norm","element","rope"],["uniform","read-only-storage","read-only-storage","storage","read-only-storage"]],
      ["attention.wgsl",["dot","softmax","weighted"],["uniform","read-only-storage","read-only-storage","read-only-storage","storage","storage","storage"]],
      ["append.wgsl",["append"],["uniform","read-only-storage","storage"]],
      ["codec.wgsl",["encode_k"],["uniform","read-only-storage","read-only-storage","read-only-storage","storage","storage"]],
      ["compressed-dot.wgsl",["compressed_dot"],["uniform","read-only-storage","read-only-storage","read-only-storage","read-only-storage","read-only-storage","read-only-storage","storage"]]
    ] as const){
      const module=await shader(device,new URL(file,import.meta.url),true);
      const layout=device.createPipelineLayout({bindGroupLayouts:[device.createBindGroupLayout({entries:types.map((type,binding)=>({binding,visibility:4,buffer:{type}}))})]});
      for(const name of names)kernels[name]=await device.createComputePipelineAsync({layout,compute:{module,entryPoint:name}});
    }
    return new ModelOps(device,kernels);
  }
  commands(){return new Commands(this);}
}
export class Commands {
  readonly encoder:GPUCommandEncoder;private uniforms:GPUBuffer[]=[];uniformBytes=0;
  constructor(private ops:ModelOps){this.encoder=ops.device.createCommandEncoder();}
  dispatch(name:string,parameters:number[],buffers:GPUBuffer[],x:number,y=1){
    const device=this.ops.device;if(x>device.limits.maxComputeWorkgroupsPerDimension||y>device.limits.maxComputeWorkgroupsPerDimension)throw new RangeError("Model dispatch limit");
    const params=device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});this.uniforms.push(params);this.uniformBytes+=params.size;
    device.queue.writeBuffer(params,0,Uint32Array.from({length:8},(_,i)=>parameters[i]??0));
    const pipeline=this.ops.kernels[name],group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[params,...buffers].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const pass=this.encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(x,y);pass.end();
  }
  async submit(){try{this.ops.device.queue.submit([this.encoder.finish()]);await this.ops.device.queue.onSubmittedWorkDone();}finally{this.uniforms.forEach(b=>b.destroy());}}
}
