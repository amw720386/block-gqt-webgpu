// Same intermediates/semantics as Phase 3. Parallel reduction changes FP32 order.
struct Params {tokens:u32,dim:u32,queries:u32,causal:u32,key_f32:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(3) var<storage,read> v:array<f32>;
@group(0) @binding(4) var<storage,read> positions:array<u32>;
@group(0) @binding(5) var<storage,read_write> logits:array<f32>;
@group(0) @binding(6) var<storage,read_write> probabilities:array<f32>;
@group(0) @binding(7) var<storage,read_write> output:array<f32>;
var<workgroup> scratch:array<f32,128>;
@compute @workgroup_size(128)
fn softmax(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let row=group.x;let base=row*p.tokens;var maximum=-3.4e38;
  for(var t=lane;t<p.tokens;t+=128u){if(p.causal==0u||t<=positions[row]){maximum=max(maximum,logits[base+t]);}}
  scratch[lane]=maximum;workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){scratch[lane]=max(scratch[lane],scratch[lane+stride]);}workgroupBarrier();}
  maximum=scratch[0];workgroupBarrier();var sum=0.0;
  for(var t=lane;t<p.tokens;t+=128u){
    var value=0.0;if(p.causal==0u||t<=positions[row]){value=exp(logits[base+t]-maximum);}
    probabilities[base+t]=value;sum+=value;
  }
  scratch[lane]=sum;workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){scratch[lane]+=scratch[lane+stride];}workgroupBarrier();}
  let denominator=scratch[0];
  for(var t=lane;t<p.tokens;t+=128u){probabilities[base+t]/=denominator;}
}
@compute @workgroup_size(128)
fn weighted_sum(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
  let index=group.x;let row=index/p.dim;let d=index%p.dim;var sum=0.0;
  for(var t=lane;t<p.tokens;t+=128u){sum+=probabilities[row*p.tokens+t]*v[t*p.dim+d];}
  scratch[lane]=sum;workgroupBarrier();
  for(var stride=64u;stride>0u;stride/=2u){if(lane<stride){scratch[lane]+=scratch[lane+stride];}workgroupBarrier();}
  if(lane==0u){output[index]=scratch[0];}
}
