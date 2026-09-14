import {storage,upload,checkFinite,type Allocation} from "../webgpu/buffers.js";

export class Values {
  readonly buffer:GPUBuffer;readonly allocations:Allocation[];length=0;
  constructor(readonly device:GPUDevice,readonly dim:number,readonly capacity:number){
    if(!Number.isInteger(dim)||dim<1||!Number.isInteger(capacity)||capacity<1)throw new RangeError("Invalid V dimensions");
    this.buffer=storage(device,capacity*dim*4,"FP32 V");this.allocations=[{name:"FP32 V",role:"persistent-v",bytes:this.buffer.size}];
  }
  append(values:Float32Array):void{
    if(!values.length||values.length%this.dim)throw new RangeError("V append must contain whole rows");
    checkFinite(values,"V");const count=values.length/this.dim;
    if(this.length+count>this.capacity)throw new RangeError("V capacity exceeded");
    upload(this.device,this.buffer,values,this.length*this.dim*4);this.length+=count;
  }
  destroy():void{this.buffer.destroy();}
}
