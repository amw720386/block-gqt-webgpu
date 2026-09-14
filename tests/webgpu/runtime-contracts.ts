import {AttentionEngine} from "../../src/attention/attention.js";
import {CompressedKeys} from "../../src/runtime/compressed-keys.js";
import {FP16Keys,type KeyStore} from "../../src/runtime/keys.js";
import {Values} from "../../src/runtime/values.js";
import {readBuffer} from "../../src/webgpu/buffers.js";
import {attentionFixture} from "./attention-fixture.js";
import {compare} from "./compare.js";

export async function runtimeContracts(device:GPUDevice){
  const {fixture:f,m,parameters}=await attentionFixture("t128_d64"),engine=await AttentionEngine.create(device);
  const stores:KeyStore[]=[new FP16Keys(device,m.dim,3),await CompressedKeys.create(device,parameters,3)];
  for(const keys of stores){
    const values=new Values(device,m.dim,3),session=engine.session(keys,values,1);
    try{
      const k=f.halfBits("k"),v=f.f32("v"),q=new Float32Array(m.dim);
      await keys.append(k.subarray(0,m.dim));values.append(v.subarray(0,m.dim));
      await session.attend(q,new Uint32Array([0]));
      compare(new Float32Array(await readBuffer(device,session.output,m.dim*4)),v.subarray(0,m.dim),[0,0]);
      await keys.append(k.subarray(m.dim,3*m.dim));values.append(v.subarray(m.dim,3*m.dim));
      await session.attend(q,new Uint32Array([2]));
      const expected=Float32Array.from({length:m.dim},(_,d)=>(v[d]+v[m.dim+d]+v[2*m.dim+d])/3);
      compare(new Float32Array(await readBuffer(device,session.output,m.dim*4)),expected,[0.000001,0.000001]);
      if(keys.length!==3||values.length!==3)throw new Error("Append lengths incorrect");
      let rejected=false;try{values.append(new Float32Array(m.dim));}catch{rejected=true;}
      if(!rejected||values.length!==3)throw new Error("Overflow mutated V state");
    }finally{session.destroy();values.destroy();keys.destroy();}
  }
  return {prefix_then_append:true,capacity_is_not_active_length:true,values_overflow_rejected:true};
}
