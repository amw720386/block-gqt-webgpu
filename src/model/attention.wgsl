// Model cache: [KV head, capacity, 64], FP16. Q is [chunk,9,64].
struct P {rows:u32,tokens:u32,capacity:u32,start:u32}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> q:array<f32>;
@group(0) @binding(2) var<storage,read> k:array<u32>;
@group(0) @binding(3) var<storage,read> v:array<u32>;
@group(0) @binding(4) var<storage,read_write> logits:array<f32>;
@group(0) @binding(5) var<storage,read_write> probabilities:array<f32>;
@group(0) @binding(6) var<storage,read_write> output:array<f32>;
var<workgroup> key:array<f32,64>;
var<workgroup> partial:array<f32,128>;
@compute @workgroup_size(128)
fn dot(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let token=group.x;let kv=group.y;
  if(lane<64u){let i=(kv*p.capacity+token)*64u+lane;key[lane]=half_value((k[i/2u]>>((i%2u)*16u))&65535u);}
  workgroupBarrier();
  if(lane<p.rows*3u){
    let row=lane/3u;let head=kv*3u+lane%3u;let index=(row*9u+head)*p.tokens+token;
    if(token>p.start+row){var inf=0xff800000u;logits[index]=bitcast<f32>(inf);return;}
    var sum=0.0;for(var d=0u;d<64u;d++){sum+=q[(row*9u+head)*64u+d]*key[d];}logits[index]=sum*0.125;
  }
}
@compute @workgroup_size(128)
fn softmax(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let row=group.x;let offset=row*p.tokens;let end=min(p.tokens,p.start+row/9u+1u);var maximum=-3.4e38;
  for(var t=lane;t<end;t+=128u){maximum=max(maximum,logits[offset+t]);}partial[lane]=maximum;workgroupBarrier();
  for(var s=64u;s>0u;s/=2u){if(lane<s){partial[lane]=max(partial[lane],partial[lane+s]);}workgroupBarrier();}
  maximum=partial[0];workgroupBarrier();var sum=0.0;
  for(var t=lane;t<p.tokens;t+=128u){var value=0.0;if(t<end){value=exp(logits[offset+t]-maximum);}probabilities[offset+t]=value;sum+=value;}
  partial[lane]=sum;workgroupBarrier();for(var s=64u;s>0u;s/=2u){if(lane<s){partial[lane]+=partial[lane+s];}workgroupBarrier();}
  for(var t=lane;t<p.tokens;t+=128u){probabilities[offset+t]/=partial[0];}
}
@compute @workgroup_size(128)
fn weighted(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let index=group.x;let row=index/64u;let d=index%64u;let kv=(row%9u)/3u;let end=min(p.tokens,p.start+row/9u+1u);var sum=0.0;
  for(var t=lane;t<end;t+=128u){let i=(kv*p.capacity+t)*64u+d;let value=half_value((v[i/2u]>>((i%2u)*16u))&65535u);sum+=probabilities[row*p.tokens+t]*value;}
  partial[lane]=sum;workgroupBarrier();for(var s=64u;s>0u;s/=2u){if(lane<s){partial[lane]+=partial[lane+s];}workgroupBarrier();}
  if(lane==0u){output[index]=partial[0];}
}
