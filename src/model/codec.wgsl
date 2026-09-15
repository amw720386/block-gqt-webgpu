// Same production arithmetic/packing as Phase 4, fed directly from GPU RoPE.
struct P {rows:u32,start:u32,head:u32,tokens:u32}
@group(0) @binding(0) var<uniform> p:P;
@group(0) @binding(1) var<storage,read> config:array<u32>;
@group(0) @binding(2) var<storage,read> table:array<f32>;
@group(0) @binding(3) var<storage,read> input:array<f32>;
@group(0) @binding(4) var<storage,read_write> packed:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> norms:array<atomic<u32>>;
fn grp(i:u32)->u32{return config[config[7]+i];}
@compute @workgroup_size(1)
fn encode_k(@builtin(global_invocation_id) id:vec3<u32>){
  let row=id.x;let destination=p.start+row;let d=64u;var x:array<f32,64>;var lengths:array<f32,8>;var centroidLengths:array<f32,8>;var codes:array<u32,64>;
  for(var i=0u;i<d;i++){
    let at=(row*p.tokens+p.head)*d+config[config[6]+i];x[i]=half_value(half_bits(input[at]));lengths[grp(i)]+=x[i]*x[i];
  }
  for(var g=0u;g<config[1];g++){lengths[g]=sqrt(lengths[g]+1e-30);}
  for(var i=0u;i<d;i++){let n=lengths[grp(i)];x[i]/=select(1.0,n,n>1e-10);}
  let offset=d*d+d*config[5];
  for(var j=0u;j<d;j++){
    var y=0.0;for(var i=0u;i<d;i++){y+=x[i]*table[j*d+i];}
    let bin=u32(clamp((y-table[offset+j])*table[offset+d+j],0.0,255.0));
    let code=config[config[9]+config[config[8]+j]*256u+bin];codes[j]=code;
    let c=table[d*d+j*config[5]+code];centroidLengths[grp(j)]+=c*c;
  }
  for(var g=0u;g<config[1];g++){
    let scale=lengths[g]/max(sqrt(centroidLengths[g]),1e-10);let h=destination*config[3]+g;
    atomicOr(&norms[h/2u],half_bits(scale)<<((h%2u)*16u));
  }
  for(var i=0u;i<d;i++){
    var a=destination*config[2];var shift=0u;if(i<config[4]){a+=i/2u;shift=(i%2u)*4u;}else{a+=config[4]/2u+i-config[4];}
    atomicOr(&packed[a/4u],codes[i]<<((a%4u)*8u+shift));
  }
}
