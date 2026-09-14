// Header: rows,D,groups,rowBytes,normStride,nibbleDims,maxCents,capacity,
// offsets for headPerm,groupOf,posToCb,codeLut. Tables are immutable.
@group(0) @binding(0) var<storage, read> config: array<u32>;
// rotation[D,D], centroids[D,maxCents], LUT offsets[D], inverse scales[D].
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read> input: array<f32>;
@group(0) @binding(3) var<storage, read_write> codes: array<u32>;
@group(0) @binding(4) var<storage, read_write> packed: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> half_norms: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> scales: array<f32>;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;

fn group_of(i: u32) -> u32 { return config[config[9] + i]; }
fn centroid(i: u32, code: u32) -> f32 { return table[config[1]*config[1] + i*config[6] + code]; }
fn byte_at(address: u32) -> u32 {
  return (atomicLoad(&packed[address/4u]) >> ((address%4u)*8u)) & 255u;
}
// Explicit IEEE round-to-nearest-even, including subnormals and overflow.
// WGSL native float conversion may choose a different allowed rounding mode.
fn half_bits(value: f32) -> u32 {
  let bits=bitcast<u32>(value); let sign=(bits>>16u)&32768u;
  let e=i32((bits>>23u)&255u)-112; let fraction=bits&8388607u;
  if(e>=31) {return sign|31744u;}
  if(e < -10) {return sign;}
  var mantissa=fraction;var shift=13u;var exponent=u32(max(e,0))<<10u;
  if(e<=0) {mantissa|=8388608u;shift=u32(14-e);exponent=0u;}
  let base=mantissa>>shift;let remainder=mantissa&((1u<<shift)-1u);let halfway=1u<<(shift-1u);
  let increment=select(0u,1u,remainder>halfway||(remainder==halfway&&(base&1u)!=0u));
  return sign|(exponent+base+increment);
}
fn half_value(h: u32) -> f32 {
  let sign=select(1.0,-1.0,(h&32768u)!=0u);let e=(h>>10u)&31u;let mantissa=h&1023u;
  if(e==0u) {return sign*f32(mantissa)*5.960464477539063e-8;}
  if(e==31u) {return bitcast<f32>(((h&32768u)<<16u)|2139095040u|(mantissa<<13u));}
  return bitcast<f32>(((h&32768u)<<16u)|((e+112u)<<23u)|(mantissa<<13u));
}
fn stored_scale(row: u32, g: u32) -> f32 {
  let h = row*config[4]+g;
  return half_value((atomicLoad(&half_norms[h/2u])>>((h%2u)*16u))&65535u);
}

@compute @workgroup_size(1)
fn quantize(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x; let d = config[1]; let ng = config[2];
  if (row >= config[0]) { return; }
  var x: array<f32,128>;
  var norms: array<f32,8>;
  var lengths: array<f32,8>;
  var local_codes: array<u32,128>;
  for (var i=0u; i<d; i++) {
    x[i] = input[row*d+config[config[8]+i]];
    norms[group_of(i)] += x[i]*x[i];
  }
  for (var g=0u; g<ng; g++) { norms[g] = sqrt(norms[g]+1e-30); }
  for (var i=0u; i<d; i++) {
    let n=norms[group_of(i)]; x[i] /= select(1.0,n,n>1e-10);
  }
  let lo = d*d+d*config[6];
  for (var j=0u; j<d; j++) {
    var y=0.0;
    for (var i=0u; i<d; i++) { y += x[i]*table[j*d+i]; }
    let bin=u32(clamp((y-table[lo+j])*table[lo+d+j],0.0,255.0));
    let code=config[config[11]+config[config[10]+j]*256u+bin];
    local_codes[j]=code; codes[row*d+j]=code;
    let c=centroid(j,code); lengths[group_of(j)] += c*c;
  }
  for (var g=0u; g<ng; g++) {
    let scale=norms[g]/max(sqrt(lengths[g]),1e-10);
    scales[row*config[4]+g]=scale;
    let h=row*config[4]+g;
    // Two adjacent rows/groups may share a word: atomic OR avoids lost bytes.
    atomicOr(&half_norms[h/2u],half_bits(scale)<<((h%2u)*16u));
  }
  for (var i=0u; i<d; i++) {
    var address=row*config[3]; var shift=0u;
    if (i<config[5]) { address+=i/2u; shift=(i%2u)*4u; }
    else { address+=config[5]/2u+i-config[5]; }
    atomicOr(&packed[address/4u],local_codes[i] << ((address%4u)*8u+shift));
  }
}

fn reconstruct(row: u32, use_half: bool) {
  let d=config[1];
  if (row>=config[0]) { return; }
  var values: array<f32,128>;
  for (var i=0u; i<d; i++) {
    var code=0u;
    if (i<config[5]) { code=(byte_at(row*config[3]+i/2u)>>((i%2u)*4u))&15u; }
    else { code=byte_at(row*config[3]+config[5]/2u+i-config[5]); }
    var scale=scales[row*config[4]+group_of(i)];
    if (use_half) { scale=stored_scale(row,group_of(i)); }
    values[i]=centroid(i,code)*scale;
  }
  for (var j=0u; j<d; j++) {
    var sum=0.0;
    for (var i=0u; i<d; i++) { sum+=values[i]*table[i*d+j]; }
    output[row*d+config[config[8]+j]]=sum;
  }
}
@compute @workgroup_size(1)
fn decode(@builtin(global_invocation_id) id: vec3<u32>) { reconstruct(id.x,true); }
@compute @workgroup_size(1)
fn decode_f32(@builtin(global_invocation_id) id: vec3<u32>) { reconstruct(id.x,false); }
@compute @workgroup_size(1)
fn convert_norms(@builtin(global_invocation_id) id: vec3<u32>) {
  let row=id.x;if(row>=config[0]) {return;}
  for(var g=0u;g<config[4];g++) {
    let h=row*config[4]+g;
    atomicOr(&half_norms[h/2u],half_bits(scales[h])<<((h%2u)*16u));
  }
}

