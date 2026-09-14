// Single head, row-major Q[queries,D], K[T,D], V[T,D]. Accumulate in FP32.
struct Params {tokens:u32,dim:u32,queries:u32,causal:u32,key_f32:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read> q:array<f32>;
@group(0) @binding(2) var<storage,read> k:array<u32>;
@group(0) @binding(3) var<storage,read> v:array<f32>;
@group(0) @binding(4) var<storage,read> positions:array<u32>;
@group(0) @binding(5) var<storage,read_write> logits:array<f32>;
@group(0) @binding(6) var<storage,read_write> probabilities:array<f32>;
@group(0) @binding(7) var<storage,read_write> output:array<f32>;
fn allowed(row:u32,token:u32)->bool{return p.causal==0u||token<=positions[row];}
fn key_at(i:u32)->f32{
  if(p.key_f32!=0u){return bitcast<f32>(k[i]);}
  return half_value((k[i/2u]>>((i%2u)*16u))&65535u);
}
@compute @workgroup_size(64)
fn dot(@builtin(global_invocation_id) id:vec3<u32>){
  let index=id.x;if(index>=p.queries*p.tokens){return;}
  let row=index/p.tokens;let token=index%p.tokens;
  if(!allowed(row,token)){var infinityBits=0xff800000u;logits[index]=bitcast<f32>(infinityBits);return;}
  var sum=0.0;
  for(var d=0u;d<p.dim;d++){sum+=q[row*p.dim+d]*key_at(token*p.dim+d);}
  logits[index]=sum*inverseSqrt(f32(p.dim));
}
@compute @workgroup_size(1)
fn softmax(@builtin(global_invocation_id) id:vec3<u32>){
  let row=id.x;if(row>=p.queries){return;}let offset=row*p.tokens;
  var maximum=-3.4e38;
  for(var t=0u;t<p.tokens;t++){if(allowed(row,t)){maximum=max(maximum,logits[offset+t]);}}
  var denominator=0.0;
  for(var t=0u;t<p.tokens;t++){
    var value=0.0;if(allowed(row,t)){value=exp(logits[offset+t]-maximum);}
    probabilities[offset+t]=value;denominator+=value;
  }
  for(var t=0u;t<p.tokens;t++){probabilities[offset+t]/=denominator;}
}
@compute @workgroup_size(64)
fn weighted_sum(@builtin(global_invocation_id) id:vec3<u32>){
  let index=id.x;if(index>=p.queries*p.dim){return;}let row=index/p.dim;let d=index%p.dim;
  var value=0.0;
  for(var t=0u;t<p.tokens;t++){value+=probabilities[row*p.tokens+t]*v[t*p.dim+d];}
  output[index]=value;
}
