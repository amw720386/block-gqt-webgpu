// Immutable, exported calibration tables. No RNG or oracle imports in the runtime.
export type BlockGTQParameters = {
  dim:number; groups:number; rowBytes:number; normStride:number; nibbleDims:number; maxCentroids:number;
  permutation:Uint32Array; groupOf:Uint32Array; positionToCodebook:Uint32Array; codeLut:Uint8Array;
  rotation:Float32Array; centroids:Float32Array; offsets:Float32Array; inverseScales:Float32Array;
};

export function validateLayout(p:BlockGTQParameters):void {
  const d=p.dim;
  if(![d,p.groups,p.rowBytes,p.normStride,p.nibbleDims,p.maxCentroids].every(Number.isInteger)||
      d<2||d>128||d%2||p.groups<1||p.groups>8||p.normStride<p.groups||p.nibbleDims<0||p.nibbleDims>d||p.nibbleDims%2||
      p.rowBytes<p.nibbleDims/2+d-p.nibbleDims||p.maxCentroids<2||p.maxCentroids>256)
    throw new RangeError("Invalid Block-GTQ layout");
  if(p.permutation.length!==d||new Set(p.permutation).size!==d||p.permutation.some(i=>i>=d)||
      p.groupOf.length!==d||p.groupOf.some(g=>g>=p.groups)||p.positionToCodebook.length!==d||
      p.rotation.length!==d*d||p.centroids.length!==d*p.maxCentroids||p.offsets.length!==d||p.inverseScales.length!==d||
      p.codeLut.length%256||p.positionToCodebook.some(c=>c*256>=p.codeLut.length))throw new RangeError("Invalid calibration table shape");
  for(const values of [p.rotation,p.centroids,p.offsets,p.inverseScales])if(values.some(x=>!Number.isFinite(x)))throw new RangeError("Nonfinite calibration table");
  for(let i=0;i<d;i++)for(let bin=0;bin<256;bin++) {
    const code=p.codeLut[p.positionToCodebook[i]*256+bin];
    if(code>=p.maxCentroids||(i<p.nibbleDims&&code>15))throw new RangeError("Code LUT incompatible with packing");
  }
}
export function tables(p:BlockGTQParameters):{integer:Uint32Array;floating:Float32Array} {
  validateLayout(p);
  const config=[p.dim,p.groups,p.rowBytes,p.normStride,p.nibbleDims,p.maxCentroids,0,0,0,0];
  for(const [slot,values] of [[6,p.permutation],[7,p.groupOf],[8,p.positionToCodebook],[9,p.codeLut]] as const) {
    config[slot]=config.length;for(const v of values)config.push(v);
  }
  return {integer:new Uint32Array(config),floating:Float32Array.from([p.rotation,p.centroids,p.offsets,p.inverseScales].flatMap(a=>Array.from(a)))};
}
