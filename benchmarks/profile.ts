import {AttentionEngine} from "../src/attention/attention.js";
import {CompressedKeys} from "../src/runtime/compressed-keys.js";
import {Values} from "../src/runtime/values.js";
import {storage,readBuffer} from "../src/webgpu/buffers.js";
import {attentionFixture} from "../tests/webgpu/attention-fixture.js";
import {gpuMetadata} from "../tests/webgpu/phase2/phase2-test.js";
import {compare} from "../tests/webgpu/compare.js";
import {timePasses} from "./timing.js";

export async function profile(){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter?.features.has("timestamp-query"))throw new Error("Timestamps required");
  const device=await adapter.requestDevice({requiredFeatures:["timestamp-query"]}),errors:string[]=[];
  device.addEventListener("uncapturederror",e=>errors.push(e.error.message));
  const {fixture:f,m,parameters}=await attentionFixture("t4096_d128");
  const keys=await CompressedKeys.create(device,parameters,m.context_length),values=new Values(device,m.dim,m.context_length);
  const engine=await AttentionEngine.create(device),session=engine.session(keys,values,3);
  const extra:GPUBuffer[]=[];
  try{
    await keys.append(f.halfBits("k"));values.append(f.f32("v"));session.setQueries(f.f32("q"),f.u32("positions"));
    const unchanged=await timePasses(device,e=>session.record(e),4);
    const baselineOutput=new Float32Array(await readBuffer(device,session.output));
    compare(baselineOutput,f.f32("compressed_output"),m.tolerances.port.output);
    // Test-only inspection of TS-private handles: no production API or source edits.
    const handles=keys as unknown as {integers:GPUBuffer;floats:GPUBuffer;packed:GPUBuffer;norms:GPUBuffer;decoded:GPUBuffer;dummy:GPUBuffer;decodeParams:GPUBuffer};
    const code=await (await fetch("/src/webgpu/mixed.wgsl")).text(),half=await (await fetch("/src/webgpu/half.wgsl")).text();
    const decode=code.slice(code.indexOf("@compute @workgroup_size(1)\nfn decode")).replaceAll("\r","");
    if(!decode.includes("fn decode"))throw new Error("Profile extraction failed");
    const split=decode.indexOf("  for(var j=0u;j<d;j++)");
    const unpack=decode.slice(0,split).replace("fn decode","fn unpack")+"for(var i=0u;i<d;i++){output[r*d+i]=values[i];}\n}";
    const inverse='@compute @workgroup_size(1)\nfn inverse(@builtin(global_invocation_id) id:vec3<u32>){let r=id.x;let d=config[0];var values:array<f32,128>;for(var i=0u;i<d;i++){values[i]=bitcast<f32>(input[r*d+i]);}\n'+decode.slice(split);
    const module=device.createShaderModule({code:half+code.slice(0,code.indexOf("@compute"))+unpack+inverse});
    const info=await module.getCompilationInfo();if(info.messages.some(x=>x.type==="error"))throw new Error(JSON.stringify(info.messages));
    const layout=device.createBindGroupLayout({entries:Array.from({length:7},(_,binding)=>({binding,visibility:4,buffer:{type:binding===6?"uniform" as const:binding<3?"read-only-storage" as const:"storage" as const}}))});
    const pl=device.createPipelineLayout({bindGroupLayouts:[layout]});
    const pipelines=await Promise.all(["unpack","inverse"].map(entryPoint=>device.createComputePipelineAsync({layout:pl,compute:{module,entryPoint}})));
    const temporary=storage(device,m.dim*m.context_length*4,"profile split centroids");extra.push(temporary);
    const bindings=(input:GPUBuffer,output:GPUBuffer)=>device.createBindGroup({layout,entries:[handles.integers,handles.floats,input,handles.packed,handles.norms,output,handles.decodeParams].map((buffer,binding)=>({binding,resource:{buffer}}))});
    const groups=[bindings(handles.dummy,temporary),bindings(temporary,handles.decoded)];
    const before=new Float32Array(await readBuffer(device,handles.decoded));
    const separated=await timePasses(device,e=>{for(let i=0;i<2;i++){const pass=e.beginComputePass();pass.setPipeline(pipelines[i]);pass.setBindGroup(0,groups[i]);pass.dispatchWorkgroups(m.context_length);pass.end();}},2);
    const decompositionParity=compare(new Float32Array(await readBuffer(device,handles.decoded)),before,[0,0]);
    const probe=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code:'@compute @workgroup_size(1) fn main(){}'}),entryPoint:"main"}});
    const empty=await timePasses(device,e=>{const p=e.beginComputePass();p.setPipeline(probe);p.dispatchWorkgroups(1);p.end();},1);
    // Bracket transfer probes with empty timestamped passes; not production work.
    const transfer=async(clear:boolean)=>timePasses(device,e=>{
      let p=e.beginComputePass();p.setPipeline(probe);p.dispatchWorkgroups(1);p.end();
      if(clear)e.clearBuffer(temporary);else e.copyBufferToBuffer(handles.decoded,0,temporary,0,temporary.size);
      p=e.beginComputePass();p.setPipeline(probe);p.dispatchWorkgroups(1);p.end();
    },2);
    const copy=await transfer(false),clear=await transfer(true);
    if(errors.length)throw new Error(errors.join("\n"));
    return {passed:true,environment:gpuMetadata(adapter,device),context_length:4096,head_dimension:128,queries:3,upstream_commit:m.upstream_commit,
      unchanged,separated,decompositionParity,empty,copy,clear,allocations:[...keys.allocations,...values.allocations,...session.allocations],
      notes:"Unchanged passes: decode including inverse rotation, QK, softmax, V accumulation. Separated diagnostic: unpack/scales then dense inverse rotation, adds 2 MiB write/read. Copy/clear are bracketed probes, not hot-path costs. GPU timestamp quantization may yield zero short-pass samples. Host record is command construction, not GPU dispatch overhead. Actual cache/DRAM traffic cannot be observed by WebGPU."};
  }finally{extra.forEach(b=>b.destroy());session.destroy();keys.destroy();values.destroy();device.destroy();}
}
