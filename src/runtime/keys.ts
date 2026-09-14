import {storage,upload,type Allocation} from "../webgpu/buffers.js";

export type KeyView={buffer:GPUBuffer;format:"f16"|"f32"};
export interface KeyStore {
  readonly device:GPUDevice;readonly dim:number;readonly capacity:number;readonly length:number;
  readonly allocations:Allocation[];readonly kind:"fp16"|"block-gtq";
  append(values:Uint16Array):Promise<void>;
  prepare(encoder:GPUCommandEncoder,start?:GPUComputePassTimestampWrites):KeyView;
  destroy():void;
}
export function appendCount(values:Uint16Array,dim:number,length:number,capacity:number):number {
  if(!values.length||values.length%dim)throw new RangeError("K append must contain whole nonempty rows");
  if(values.some(x=>(x&0x7c00)===0x7c00))throw new RangeError("Finite FP16 K required");
  const count=values.length/dim;if(length+count>capacity)throw new RangeError("K capacity exceeded");return count;
}
export class FP16Keys implements KeyStore {
  readonly kind="fp16" as const;readonly allocations:Allocation[];readonly buffer:GPUBuffer;length=0;
  constructor(readonly device:GPUDevice,readonly dim:number,readonly capacity:number){
    if(!Number.isInteger(dim)||dim<2||dim%2||!Number.isInteger(capacity)||capacity<1)throw new RangeError("Invalid K dimensions");
    this.buffer=storage(device,capacity*dim*2,"FP16 K");this.allocations=[{name:"FP16 K",role:"persistent-k",bytes:this.buffer.size}];
  }
  async append(values:Uint16Array):Promise<void>{
    const count=appendCount(values,this.dim,this.length,this.capacity);upload(this.device,this.buffer,values,this.length*this.dim*2);this.length+=count;
  }
  prepare():KeyView{return {buffer:this.buffer,format:"f16"};}
  destroy():void{this.buffer.destroy();}
}
