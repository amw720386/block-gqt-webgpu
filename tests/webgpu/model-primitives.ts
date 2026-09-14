import {ModelOps} from "../../src/model/ops.js";
import {storage,upload,readBuffer} from "../../src/webgpu/buffers.js";
import {loadFixture} from "./fixture.js";
import {compare} from "./compare.js";
export async function run(){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error("No adapter");const device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));const ops=await ModelOps.create(device),f=await loadFixture('/fixtures/model/rope.json'),cases=[];
  try{for(const c of (f.manifest as any).cases){
    const input=f.f32(c.name+'_x'),x=storage(device,input.byteLength,'RoPE input'),y=storage(device,input.byteLength,'RoPE output'),dummy=storage(device,4,'unused');
    try{
      upload(device,x,input);const cmd=ops.commands();cmd.dispatch('rope',[c.rows,0,0,0,c.start,c.heads],[x,dummy,y,dummy],Math.ceil(c.rows*c.heads*32/128));await cmd.submit();
      const actual=new Float32Array(await readBuffer(device,y));cases.push({name:c.name,error:compare(actual,f.f32(c.name+'_y'))});
      compare(actual,f.f32(c.name+'_y'),[0.001,0.0001]);
    }finally{x.destroy();y.destroy();dummy.destroy();}
  }return {passed:errors.length===0,errors,cases};}finally{device.destroy();}
}
