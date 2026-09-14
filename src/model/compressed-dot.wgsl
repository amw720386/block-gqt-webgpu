// Phase 4 group-aware inverse transform; one key tile serves all GQA queries.
struct P {rows:u32,start:u32,head:u32,tokens:u32}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> config:array<u32>;
@group(0) @binding(2) var<storage,read> table:array<f32>;
@group(0) @binding(3) var<storage,read> q:array<f32>;
@group(0) @binding(4) var<storage,read> packed:array<u32>;
@group(0) @binding(5) var<storage,read> norms:array<u32>;
@group(0) @binding(6) var<storage,read> ranges:array<u32>;
@group(0) @binding(7) var<storage,read_write> logits:array<f32>;
var<workgroup> centroids:array<f32,64>;
var<workgroup> key:array<f32,64>;
@compute @workgroup_size(128)
fn compressed_dot(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let token=group.x;
  if(lane<64u){
    var a=token*config[2];var shift=0u;var mask=255u;
    if(lane<config[4]){a+=lane/2u;shift=(lane%2u)*4u;mask=15u;}else{a+=config[4]/2u+lane-config[4];}
    let code=(packed[a/4u]>>((a%4u)*8u+shift))&mask;
    let h=token*config[3]+config[config[7]+lane];let scale=half_value((norms[h/2u]>>((h%2u)*16u))&65535u);
    centroids[lane]=table[4096u+lane*config[5]+code]*scale;
  }
  workgroupBarrier();
  if(lane<64u){var sum=0.0;for(var i=ranges[lane*2u];i<ranges[lane*2u+1u];i++){sum+=centroids[i]*table[i*64u+lane];}key[config[config[6]+lane]]=sum;}
  workgroupBarrier();
  if(lane<p.rows*3u){
    let row=lane/3u;let head=p.head*3u+lane%3u;let index=(row*9u+head)*p.tokens+token;
    if(token>p.start+row){var inf=0xff800000u;logits[index]=bitcast<f32>(inf);return;}
    var sum=0.0;for(var d=0u;d<64u;d++){sum+=q[(row*9u+head)*64u+d]*key[d];}logits[index]=sum*0.125;
  }
}
