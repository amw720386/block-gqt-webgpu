import {loadFixture} from "../fixture.js";
import {mixedPipelines,mixedBuffers,type Kernel} from "./mixed.js";

function exact(a:ArrayLike<number>,b:ArrayLike<number>,label:string) {
  if(a.length!==b.length) throw new Error(`${label} length ${a.length} != ${b.length}`);
  for(let i=0;i<a.length;i++) if(a[i]!==b[i]) throw new Error(`${label}[${i}]: ${a[i]} != ${b[i]}`);
}
function floats(a:Float32Array,b:Float32Array,atol:number,rtol:number,bound?:Float64Array) {
  let max=0, diff2=0, ref2=0, maxRef=0;
  for(let i=0;i<a.length;i++) {
    const e=Math.abs(a[i]-b[i]);
    if(!Number.isFinite(e)||e>atol+rtol*Math.abs(b[i])+(bound?.[i]??0)) throw new Error(`Float ${i}: ${a[i]} vs ${b[i]}, error ${e}`);
    max=Math.max(max,e);diff2+=e*e;ref2+=b[i]*b[i];maxRef=Math.max(maxRef,Math.abs(b[i]));
  }
  return {max_abs_error:max,relative_l2_error:Math.sqrt(diff2)/Math.max(Math.sqrt(ref2),1e-30),
    max_abs_over_reference_max:max/Math.max(maxRef,1e-30)};
}
function half(h:number) {return (h&32768?-1:1)*((h>>10&31)===0?(h&1023)*2**-24:(1+(h&1023)/1024)*2**((h>>10&31)-15));}
export function gpuMetadata(adapter:GPUAdapter,device:GPUDevice) {
  const limits:Record<string,number>={};
  for(const key in adapter.limits) {
    const value=(adapter.limits as unknown as Record<string,unknown>)[key];
    if(typeof value==="number") limits[key]=value;
  }
  const i=adapter.info;
  return {vendor:i.vendor,architecture:i.architecture,device:i.device,description:i.description,
    is_fallback_adapter:i.isFallbackAdapter,adapter_features:Array.from(adapter.features).sort(),
    adapter_limits:limits,device_features:Array.from(device.features).sort()};
}
export async function validateMixed(device:GPUDevice) {
  const compiled=await mixedPipelines(device);
  const index=await (await fetch("/fixtures/mixed/index.json")).json();
  const results=[];
  for(const c of index.cases) {
    const f=await loadFixture(`/fixtures/mixed/${c.name}.json`), m=f.manifest;
    const b=mixedBuffers(device,f,compiled);
    const run=(kernel:Kernel)=>{
      const encoder=device.createCommandEncoder();b.dispatch(encoder,kernel);device.queue.submit([encoder.finish()]);
    };
    try {
      b.golden();run("decode");const independent=await b.read();
      const decode=floats(independent.values,f.f32("reconstructed_f16"),m.tolerance.atol,m.tolerance.rtol);
      run("decode_f32");const f32=await b.read();
      const decodeF32=floats(f32.values,f.f32("reconstructed_f32"),m.tolerance.atol,m.tolerance.rtol);
      run("convert_norms");const converted=await b.read();
      exact(converted.norms.subarray(0,m.rows*m.norm_stride),f.u16("norm_bits"),`${c.name} identical-scale half conversion`);
      const probes=[0,-0,2**-24,2**-25,3*2**-25,1+2**-11,65504,65520];
      const expected=[0,32768,1,0,2,15360,31743,31744];
      b.writeScales(Float32Array.from({length:m.rows*m.norm_stride},(_,i)=>probes[i%probes.length]));
      run("convert_norms");const edges=await b.read();
      exact(edges.norms.subarray(0,m.rows*m.norm_stride),Uint16Array.from({length:m.rows*m.norm_stride},(_,i)=>expected[i%expected.length]),"half conversion edges");
      run("quantize");const encoded=await b.read();
      exact(encoded.codes,f.u8("codes"),`${c.name} codes`);
      exact(encoded.packed.subarray(0,m.capacity*m.row_bytes),f.u8("cache_packed"),`${c.name} packed`);
      const normReference=f.halfBits("cache_norms");let normMismatches=0;
      for(let i=0;i<normReference.length;i++) {
        if(encoded.norms[i]!==normReference[i]) normMismatches++;
        if(Math.abs(encoded.norms[i]-normReference[i])>1) throw new Error(`${c.name} norm difference exceeds one half ULP at ${i}`);
      }
      // Compare FP32 corrected scales before conversion; production stores half.
      const scale=floats(encoded.scales,f.f32("corrected_f32"),0.0005,0.00002);
      // Quantize -> decode in one command buffer, with no host transfer between.
      const encoder=device.createCommandEncoder();b.dispatch(encoder,"quantize");b.dispatch(encoder,"decode");
      device.queue.submit([encoder.finish()]);const roundtrip=await b.read();
      // Analytic bound for the *observed* half-scale change, holding codes fixed.
      const bound=new Float64Array(m.rows*m.dim), groups=f.u32("group_of"),perm=f.u32("head_perm"),rot=f.f32("rotation"),cents=f.f32("centroids");
      for(let r=0;r<m.rows;r++) for(let j=0;j<m.dim;j++) for(let i=0;i<m.dim;i++) {
        const h=r*m.norm_stride+groups[i], delta=half(encoded.norms[h])-half(normReference[h]);
        bound[r*m.dim+perm[j]]+=Math.abs(delta*cents[i*m.max_centroids+encoded.codes[r*m.dim+i]]*rot[i*m.dim+j]);
      }
      const rt=floats(roundtrip.values,f.f32("reconstructed_f16"),m.tolerance.atol,m.tolerance.rtol,bound);
      results.push({name:c.name,fixture_sha256:m.sha256,indices_exact:true,packed_bytes_exact:true,norm_bits_exact:normMismatches===0,
        norm_bit_mismatches:normMismatches,identical_scale_fp16_conversion_exact:true,half_conversion_edges_exact:true,
        norm_rounding_reconstruction_bound_max:Math.max(...bound),gpu_corrected_scales:Array.from(encoded.scales),
        gpu_norm_bits:Array.from(encoded.norms.subarray(0,m.rows*m.norm_stride)),
        independent_decode:decode,independent_decode_f32:decodeF32,corrected_scale:scale,roundtrip:rt,
        norm_conversion_max_abs:m.norm_conversion_max_abs,quantization_max_abs:m.quantization_max_abs,
        roundtrip_actual:Array.from(roundtrip.values),reference:Array.from(f.f32("reconstructed_f16")),
        allocated_buffers:b.ledger});
    } finally {b.destroy();}
  }
  return results;
}

export async function runCorrectness() {
  if(!navigator.gpu) throw new Error("WebGPU unavailable");
  const adapter=await navigator.gpu.requestAdapter();if(!adapter) throw new Error("No GPU adapter");
  const device=await adapter.requestDevice();const errors:string[]=[];
  device.addEventListener("uncapturederror",e=>errors.push(e.error.message));
  try {
    const cases=await validateMixed(device);await device.queue.onSubmittedWorkDone();
    if(errors.length) throw new Error(errors.join("\n"));
    return {passed:true,cuda_encoder_validated:false,environment:gpuMetadata(adapter,device),cases};
  } finally {device.destroy();}
}
