import {storage,upload} from "../webgpu/buffers.js";
import type {ModelAdapter} from "./adapter.js";
export type CalibrationHead={config:GPUBuffer;table:GPUBuffer;ranges:GPUBuffer;rowBytes:number;normStride:number};
export class Calibration {
  readonly heads:CalibrationHead[]=[];readonly allocations:{name:string;role:"tables";bytes:number}[]=[];private buffers:GPUBuffer[]=[];
  private constructor(readonly device:GPUDevice,readonly sha256:string){}
  static async load(device:GPUDevice,adapter?:ModelAdapter,base='/fixtures/model',modelRevision?:string){
    const m=await (await fetch(`${base}/calibration.json`)).json(),payload=await (await fetch(`${base}/${m.binary}`)).arrayBuffer();
    const sha=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',payload)),b=>b.toString(16).padStart(2,'0')).join('');
    const expected=adapter?adapter.layers*adapter.kvHeads:90;
    const expectedRevision=modelRevision??(adapter?undefined:"12fd25f77366fa6b3b4b768ec3050bf629380bac");
    if(sha!==m.sha256||m.heads.length!==expected||(expectedRevision&&m.model_revision!==expectedRevision)||m.upstream_commit!=="3fe14e7d4b8a6c85402818f7e22052527ed42e93")throw new Error("Calibration mismatch");
    const c=new Calibration(device,sha);
    try{for(let i=0;i<expected;i++){
      const read=(name:string)=>{const a=m.arrays[`h${i}_${name}`];if(!a||a.offset%4||a.offset+a.nbytes>payload.byteLength)throw new Error("Bad calibration descriptor");return payload.slice(a.offset,a.offset+a.nbytes);};
      const config=new Uint32Array(read('config')),table=new Float32Array(read('table')),ranges=new Uint32Array(read('ranges'));
      if(config[0]!==64||config[1]>8||ranges.length!==128||table.some(x=>!Number.isFinite(x)))throw new Error("Unsupported calibration");
      for(let j=0;j<64;j++)for(let k=0;k<64;k++)if((k<ranges[j*2]||k>=ranges[j*2+1])&&table[k*64+j]!==0)throw new Error("Off-group rotation");
      const make=(name:string,a:ArrayBufferView)=>{const b=storage(device,a.byteLength,`head ${i} ${name}`);upload(device,b,a);c.buffers.push(b);c.allocations.push({name:`head ${i} ${name}`,role:"tables",bytes:b.size});return b;};
      c.heads.push({config:make('config',config),table:make('table',table),ranges:make('ranges',ranges),rowBytes:config[2],normStride:config[3]});
    }}catch(e){c.destroy();throw e;}return c;
  }
  destroy(){this.buffers.forEach(b=>b.destroy());}
}
