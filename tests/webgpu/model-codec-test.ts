import {ModelOps} from "../../src/model/ops.js";
import {Calibration} from "../../src/model/calibration.js";
import {PackedKeys} from "../../src/runtime/packed-keys.js";
import {storage,upload,readBuffer} from "../../src/webgpu/buffers.js";
import {loadFixture} from "./fixture.js";
import {exact} from "./compare.js";
export async function run(){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error("No adapter");const device=await adapter.requestDevice(),errors:string[]=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
  const ops=await ModelOps.create(device),calibration=await Calibration.load(device),f=await loadFixture('/fixtures/model/calibration.json'),rope=await loadFixture('/fixtures/model/rope.json');
  const data=rope.f32('h3_p4095_y'),x=storage(device,data.byteLength,'post-RoPE input'),halves=storage(device,3*3*64*2,'FP16 append reference');upload(device,x,data);
  try{
    const cmd=ops.commands();cmd.dispatch('append',[3,0,3],[x,halves],3);await cmd.submit();const half=new Uint16Array(await readBuffer(device,halves)),cases=[];
    for(let h=0;h<90;h++){
      const c=f.u32(`h${h}_config`),t=f.f32(`h${h}_table`),maxc=c[5],off=4096+64*maxc,head=h%3;
      const p={dim:64,groups:c[1],rowBytes:c[2],normStride:c[3],nibbleDims:c[4],maxCentroids:maxc,permutation:c.slice(c[6],c[6]+64),groupOf:c.slice(c[7],c[7]+64),positionToCodebook:c.slice(c[8],c[8]+64),codeLut:Uint8Array.from(c.slice(c[9])),rotation:t.slice(0,4096),centroids:t.slice(4096,off),offsets:t.slice(off,off+64),inverseScales:t.slice(off+64,off+128)};
      const reference=await PackedKeys.create(device,p,3),packed=storage(device,3*c[2],'model codes'),norms=storage(device,3*c[3]*2,'model norms');
      try{
        await reference.append(half.subarray(head*3*64,(head+1)*3*64));const a=calibration.heads[h],e=ops.commands();
        e.dispatch('encode_k',[3,0,head,3],[a.config,a.table,x,packed,norms],3);await e.submit();
        exact(new Uint8Array(await readBuffer(device,packed)),new Uint8Array(await readBuffer(device,reference.packed)),'Phase 4 code bytes');
        exact(new Uint16Array(await readBuffer(device,norms)),new Uint16Array(await readBuffer(device,reference.norms)),'Phase 4 FP16 scale bits');cases.push({head:h,packed_exact:true,norms_exact:true});
      }finally{reference.destroy();packed.destroy();norms.destroy();}
    }
    return {passed:errors.length===0,errors,cases,convention:'post-RoPE coordinates, position 4095..4097; exact against unchanged Phase 4 GPU encoder'};
  }finally{x.destroy();halves.destroy();calibration.destroy();device.destroy();}
}
