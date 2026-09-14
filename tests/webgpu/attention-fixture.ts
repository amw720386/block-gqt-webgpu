import {loadFixture,type Manifest} from "./fixture.js";
import type {BlockGTQParameters} from "../../src/webgpu/layout.js";
export type AttentionManifest=Manifest&{context_length:number;queries:number;layout:string;layout_sha256:string;
  tolerances:Record<"baseline"|"port",Record<"logits"|"probabilities"|"output",[number,number]>>};
export async function attentionFixture(name:string){
  const fixture=await loadFixture(`/fixtures/attention/${name}.json`);
  const m=fixture.manifest as AttentionManifest;
  const layout=await loadFixture(`/fixtures/mixed/${m.layout}.json`),p=layout.manifest;
  if(p.sha256!==m.layout_sha256||m.coordinate_space!=="synthetic-rope-free"||p.dim!==m.dim)throw new Error("Attention calibration mismatch");
  if(layout.u32("pack_perm").some((x,i)=>x!==i))throw new Error("Unsupported secondary permutation");
  const parameters:BlockGTQParameters={dim:p.dim,groups:p.n_groups,rowBytes:p.row_bytes,normStride:p.norm_stride,nibbleDims:p.nopack_start,maxCentroids:p.max_centroids,
    permutation:layout.u32("head_perm"),groupOf:layout.u32("group_of"),positionToCodebook:layout.u32("pos_to_cb"),codeLut:layout.u8("code_lut"),
    rotation:layout.f32("rotation"),centroids:layout.f32("centroids"),offsets:layout.f32("lut_offsets"),inverseScales:layout.f32("lut_inv_scales")};
  const check=(key:string,shape:number[])=>{
    if(JSON.stringify(m.arrays[key]?.shape)!==JSON.stringify(shape))throw new Error(`Invalid attention array shape: ${key}`);
  };
  check("q",[m.queries,m.dim]);check("k",[m.context_length,m.dim]);check("v",[m.context_length,m.dim]);check("positions",[m.queries]);
  return {fixture,m,parameters};
}
