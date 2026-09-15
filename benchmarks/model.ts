import {Weights} from "../src/model/weights.js";
import {Calibration} from "../src/model/calibration.js";
import {Decoder} from "../src/model/decoder.js";
import {Tokenizer} from "../src/model/tokenizer.js";
import {generate} from "../src/model/generation.js";
import {compare} from "../tests/webgpu/compare.js";
import {capture} from "../tests/webgpu/capture.js";
import {gpuMetadata} from "../tests/webgpu/phase2/phase2-test.js";
function summary(samples:number[]){
  if(!samples.length)return null;const s=[...samples].sort((a,b)=>a-b),mean=s.reduce((a,b)=>a+b,0)/s.length;
  return {samples,count:s.length,mean,median:(s[Math.floor((s.length-1)/2)]+s[Math.floor(s.length/2)])/2,p50:s[Math.ceil(s.length*.5)-1],p95:s.length>=20?s[Math.ceil(s.length*.95)-1]:null,stddev:Math.sqrt(s.reduce((a,b)=>a+(b-mean)**2,0)/s.length),min:s[0],max:s[s.length-1],unit:'ms'};
}
function agreement(a:number[],b:number[]){const n=Math.min(a.length,b.length);let first=-1,same=0;for(let i=0;i<n;i++){if(a[i]===b[i])same++;else if(first<0)first=i;}if(first<0&&a.length!==b.length)first=n;return {matching_tokens:same,compared_tokens:n,first_divergence_index:first<0?null:first,exact_sequence:a.length===b.length&&same===n};}
export async function run(){
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error('No adapter');const device=await adapter.requestDevice(),errors:string[]=[];
  device.addEventListener('uncapturederror',e=>errors.push(e.error.message));const environment=gpuMetadata(adapter,device),weights=await Weights.load(device),tokenizer=await Tokenizer.load();
  const cases:unknown[]=[],quality:unknown[]=[],allocationProbes:unknown[]=[];
  async function execute(ids:Uint32Array,compressed:boolean,maxTokens=25){
    const calibration=compressed?await Calibration.load(device):undefined;let model:Decoder|undefined;
    try{
      model=await Decoder.create(weights,ids.length+32,32,calibration);const result=await generate(model,ids,maxTokens);
      const roles:Record<string,number>={};for(const a of model.allocations)roles[a.role]=(roles[a.role]??0)+a.bytes;
      const buffers=model.allocations.reduce((s,a)=>s+a.bytes,0);
      if(compressed&&model.encodedRows!==model.length*90)throw new Error('Historical re-encode count');
      return {...result,roles,allocations:model.allocations,capacity:model.capacity,active_length:model.length,encoded_key_rows:model.encodedRows,
        explicit_resident_bytes:buffers,max_transient_uniform_bytes:model.maxTransientUniformBytes,logit_readback_buffer_bytes:49152*4,
        maximum_live_explicit_buffer_bytes:buffers+Math.max(model.maxTransientUniformBytes,49152*4)};
    }finally{model?.destroy();calibration?.destroy();}
  }
  try{
    // One short warmup per path, then three repetitions per matched workload.
    for(const compressed of [false,true])await execute(tokenizer.encode('A clear explanation starts with'),compressed,3);
    const source=tokenizer.encode('A library contains books about science, history, art, and language. Readers learn by studying examples, asking questions, and testing their ideas. '),tail=tokenizer.encode('Write a detailed paragraph about why reading books is useful.');
    for(const length of [256,512,1024,2048,4096]){
      const ids=new Uint32Array(length);for(let i=0;i<length-tail.length;i++)ids[i]=source[i%source.length];ids.set(tail,length-tail.length);
      const byPath:{fp16:Awaited<ReturnType<typeof execute>>[];compressed:Awaited<ReturnType<typeof execute>>[]}={fp16:[],compressed:[]};
      for(let rep=0;rep<3;rep++)for(const compressed of (rep%2?[true,false]:[false,true])){
        console.log(`Final model suite: context ${length}, ${compressed?'Block-GTQ':'FP16'}, repetition ${rep+1}/3`);
        byPath[compressed?'compressed':'fp16'].push(await execute(ids,compressed));
      }
      const measurements=[];
      for(const path of ['fp16','compressed'] as const){
        const records=byPath[path],decode=records.flatMap(r=>r.decodeMs),prefill=records.map(r=>r.prefillMs);
        measurements.push({path,repetitions:3,warmup:'one short model warmup per path before the suite; no workload-specific warmup',
          prefill:summary(prefill),ttft:summary(records.map(r=>r.ttftMs)),decode:summary(decode),prefill_tokens_per_second:1000*length*3/prefill.reduce((a,b)=>a+b,0),
          decode_tokens_per_second:decode.length?1000*decode.length/decode.reduce((a,b)=>a+b,0):null,
          allocations:records[0].allocations,roles:records[0].roles,capacity:records[0].capacity,explicit_resident_bytes:records[0].explicit_resident_bytes,maximum_live_explicit_buffer_bytes:records[0].maximum_live_explicit_buffer_bytes,
          max_transient_uniform_bytes:records[0].max_transient_uniform_bytes,logit_readback_buffer_bytes:records[0].logit_readback_buffer_bytes,
          runs:records.map(({firstLogits,allocations,roles,...r})=>({...r,text:tokenizer.decode(r.tokens)}))});
      }
      const a=byPath.fp16[0],b=byPath.compressed[0];
      cases.push({name:`context_${length}`,context_length:length,prompt_ids:Array.from(ids),measurements,
        first_logit_error:compare(b.firstLogits,a.firstLogits),top10_overlap:a.firstTopK.filter(x=>b.firstTopK.some(y=>y.id===x.id)).length/10,
        greedy_agreement:agreement(a.tokens,b.tokens),persistent_k_ratio:a.roles.k/(b.roles.k+b.roles.tables),full_kv_ratio:(a.roles.k+a.roles.v)/(b.roles.k+b.roles.v+b.roles.tables),
        capture:capture({fp16_first_logits:a.firstLogits,compressed_first_logits:b.firstLogits})});
    }
    for(const prompt of ['Explain why plants need sunlight.','Give two tips for keeping a desk organized.','Name one planet in our solar system.']){
      console.log(`Final quality example: ${prompt}`);const ids=tokenizer.encode(tokenizer.chat(prompt)),a=await execute(ids,false,32),b=await execute(ids,true,32);
      quality.push({prompt,prompt_ids:Array.from(ids),fp16:{tokens:a.tokens,text:tokenizer.decode(a.tokens),top10:a.firstTopK},compressed:{tokens:b.tokens,text:tokenizer.decode(b.tokens),top10:b.firstTopK},
        first_logit_error:compare(b.firstLogits,a.firstLogits),top10_overlap:a.firstTopK.filter(x=>b.firstTopK.some(y=>y.id===x.id)).length/10,greedy_agreement:agreement(a.tokens,b.tokens)});
    }
    // Allocation pressure probe at the model's configured context ceiling.
    // Touch all runtime-owned buffers, but do not label this full-context inference.
    for(const compressed of [false,true]){
      let calibration:Calibration|undefined,model:Decoder|undefined;device.pushErrorScope('out-of-memory');
      try{
        if(compressed)calibration=await Calibration.load(device);model=await Decoder.create(weights,8192,32,calibration);
        const encoder=device.createCommandEncoder();for(const b of (model as unknown as {owned:GPUBuffer[]}).owned)encoder.clearBuffer(b);
        device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();const error=await device.popErrorScope();
        allocationProbes.push({path:compressed?'compressed':'fp16',capacity:8192,success:!error,error:error?.message??null,explicit_bytes:model.allocations.reduce((s,a)=>s+a.bytes,0),all_owned_buffers_touched:true,full_context_inference_tested:false});
      }catch(e){await device.popErrorScope();allocationProbes.push({path:compressed?'compressed':'fp16',capacity:8192,success:false,error:String(e),full_context_inference_tested:false});}
      finally{model?.destroy();calibration?.destroy();}
    }
    return {passed:errors.length===0,errors,environment,model:weights.manifest,cases,quality,allocationProbes,
      boundaries:'Host wall-clock after tokenization/model loading/pipeline creation; includes CPU command construction, uploads, all WebGPU computation, GPU waits and logits readback. TTFT includes first greedy selection. Decode latency includes each subsequent forward/readback and greedy selection; excludes UI rendering. Prefill chunks=32; logits are computed/read after every chunk. No CUDA comparison. Capacity=prompt tokens+32. EOS stops generation, so sample counts may differ. Single final suite; three repetitions, one short warmup per path.',
      memory_note:'Only one cache path is resident per run; weights are shared across runs. Calibration tables exist only during compressed runs. Roles report actual GPUBuffer sizes including padding. Maximum live explicit bytes adds the larger of transient uniform buffers and logits readback (non-overlapping lifetimes). Driver/pipeline/workgroup/allocator reservations are unobservable; this is not measured physical peak VRAM.'};
  }finally{weights.destroy();device.destroy();}
}
