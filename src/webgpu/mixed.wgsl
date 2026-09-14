// Production K: widths <=4 occupy nibbles, >4 raw bytes; rows have byte strides.
// Params: count and destination row for encode. Decode uses count, destination=0.
struct Params {count:u32, destination:u32, pad0:u32, pad1:u32}
@group(0) @binding(0) var<storage,read> config:array<u32>;
@group(0) @binding(1) var<storage,read> table:array<f32>;
@group(0) @binding(2) var<storage,read> input:array<u32>;
@group(0) @binding(3) var<storage,read_write> packed:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> norms:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> output:array<f32>;
@group(0) @binding(6) var<uniform> p:Params;
fn grp(i:u32)->u32{return config[config[7]+i];}
fn centroid(i:u32,code:u32)->f32{return table[config[0]*config[0]+i*config[5]+code];}
fn byte_at(a:u32)->u32{return (atomicLoad(&packed[a/4u])>>((a%4u)*8u))&255u;}
@compute @workgroup_size(1)
fn encode(@builtin(global_invocation_id) id:vec3<u32>){
  let r=id.x;let d=config[0];if(r>=p.count){return;}let destination=p.destination+r;
  var x:array<f32,128>;var lengths:array<f32,8>;var centroidLengths:array<f32,8>;var codes:array<u32,128>;
  for(var i=0u;i<d;i++){
    let at=r*d+config[config[6]+i];x[i]=half_value((input[at/2u]>>((at%2u)*16u))&65535u);
    lengths[grp(i)]+=x[i]*x[i];
  }
  for(var g=0u;g<config[1];g++){lengths[g]=sqrt(lengths[g]+1e-30);}
  for(var i=0u;i<d;i++){let n=lengths[grp(i)];x[i]/=select(1.0,n,n>1e-10);}
  let offset=d*d+d*config[5];
  for(var j=0u;j<d;j++){
    var y=0.0;for(var i=0u;i<d;i++){y+=x[i]*table[j*d+i];}
    let bin=u32(clamp((y-table[offset+j])*table[offset+d+j],0.0,255.0));
    let code=config[config[9]+config[config[8]+j]*256u+bin];codes[j]=code;
    let c=centroid(j,code);centroidLengths[grp(j)]+=c*c;
  }
  for(var g=0u;g<config[1];g++){
    let scale=lengths[g]/max(sqrt(centroidLengths[g]),1e-10);let h=destination*config[3]+g;
    atomicOr(&norms[h/2u],half_bits(scale)<<((h%2u)*16u));
  }
  // Appends only target never-written rows. Atomic OR protects shared word tails.
  for(var i=0u;i<d;i++){
    var a=destination*config[2];var shift=0u;
    if(i<config[4]){a+=i/2u;shift=(i%2u)*4u;}else{a+=config[4]/2u+i-config[4];}
    atomicOr(&packed[a/4u],codes[i]<<((a%4u)*8u+shift));
  }
}
@compute @workgroup_size(1)
fn decode(@builtin(global_invocation_id) id:vec3<u32>){
  let r=id.x;let d=config[0];if(r>=p.count){return;}var values:array<f32,128>;
  for(var i=0u;i<d;i++){
    var code=0u;
    if(i<config[4]){code=(byte_at(r*config[2]+i/2u)>>((i%2u)*4u))&15u;}
    else{code=byte_at(r*config[2]+config[4]/2u+i-config[4]);}
    let h=r*config[3]+grp(i);let scale=half_value((atomicLoad(&norms[h/2u])>>((h%2u)*16u))&65535u);
    values[i]=centroid(i,code)*scale;
  }
  for(var j=0u;j<d;j++){
    var sum=0.0;for(var i=0u;i<d;i++){sum+=values[i]*table[i*d+j];}
    output[r*d+config[config[6]+j]]=sum;
  }
}
