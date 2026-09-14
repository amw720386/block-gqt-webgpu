// IEEE half conversion for finite inputs; preserved Phase 2 RNE/subnormal behavior.
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
