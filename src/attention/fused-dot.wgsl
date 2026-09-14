// One K vector/workgroup. Never writes reconstructed K to device storage.
// Each lane reconstructs a dimension, preserving within-group summation order.
struct Params {tokens:u32,dim:u32,queries:u32,causal:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> config:array<u32>;
@group(0) @binding(2) var<storage,read> table:array<f32>;
@group(0) @binding(3) var<storage,read> packed:array<u32>;
@group(0) @binding(4) var<storage,read> norms:array<u32>;
@group(0) @binding(5) var<storage,read> q:array<f32>;
@group(0) @binding(6) var<storage,read> positions:array<u32>;
@group(0) @binding(7) var<storage,read_write> logits:array<f32>;
@group(0) @binding(8) var<storage,read> ranges:array<u32>;
var<workgroup> centroids:array<f32,128>;
var<workgroup> products:array<f32,128>;
@compute @workgroup_size(128)
fn dot(@builtin(workgroup_id) block:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let token=block.x;let d=p.dim;
  if(lane<d){
    var a=token*config[2];var shift=0u;var mask=255u;
    if(lane<config[4]){a+=lane/2u;shift=(lane%2u)*4u;mask=15u;}else{a+=config[4]/2u+lane-config[4];}
    let code=(packed[a/4u]>>((a%4u)*8u+shift))&mask;
    let h=token*config[3]+config[config[7]+lane];
    let scale=half_value((norms[h/2u]>>((h%2u)*16u))&65535u);
    centroids[lane]=table[d*d+lane*config[5]+code]*scale;
  }
  workgroupBarrier();var reconstructed=0.0;
  if(lane<d){for(var i=ranges[lane*2u];i<ranges[lane*2u+1u];i++){reconstructed+=centroids[i]*table[i*d+lane];}}
  for(var row=0u;row<p.queries;row++){
    // Uniform branch across the workgroup: no barrier divergence.
    if(p.causal!=0u&&token>positions[row]){if(lane==0u){var inf=0xff800000u;logits[row*p.tokens+token]=bitcast<f32>(inf);}continue;}
    products[lane]=0.0;if(lane<d){products[lane]=reconstructed*q[row*d+config[config[6]+lane]];}
    workgroupBarrier();
    for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){products[lane]+=products[lane+stride];}workgroupBarrier();}
    if(lane==0u){logits[row*p.tokens+token]=products[0]*inverseSqrt(f32(d));}
    workgroupBarrier();
  }
}
