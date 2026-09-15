import {Weights} from "../src/model/weights.js";
import {Calibration} from "../src/model/calibration.js";
import {Decoder} from "../src/model/decoder.js";
import {Tokenizer} from "../src/model/tokenizer.js";
import {greedy} from "../src/model/generation.js";
import {compare} from "../tests/webgpu/compare.js";
import {gpuMetadata} from "../tests/webgpu/phase2/phase2-test.js";

// Bounded repair of missing decode samples caused by immediate EOS in the
// final greedy suite. Identical supplied tokens keep both paths on one prefix.
const parentPath='benchmarks/raw/2026-09-15T10-43-22-413Z-model.json';
function summary(samples:number[]){
  const sorted=[...samples].sort((a,b)=>a-b),mean=samples.reduce((a,b)=>a+b,0)/samples.length;
  return {samples,count:samples.length,mean,median:(sorted[11]+sorted[12])/2,p50:sorted[11],p95:sorted[22],
    stddev:Math.sqrt(samples.reduce((s,x)=>s+(x-mean)**2,0)/samples.length),min:sorted[0],max:sorted[23],unit:'ms'};
}
export async function run(){
  const parentBytes=await (await fetch('/'+parentPath)).arrayBuffer(),parent=JSON.parse(new TextDecoder().decode(parentBytes));
  if(!parent.passed||parent.cases.length!==5)throw new Error('Invalid parent evaluation');
  const parentSha=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',parentBytes)),b=>b.toString(16).padStart(2,'0')).join('');
  const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw new Error('No adapter');
  const device=await adapter.requestDevice(),errors:string[]=[];device.addEventListener('uncapturederror',e=>errors.push(e.error.message));
  const environment=gpuMetadata(adapter,device),weights=await Weights.load(device),tokenizer=await Tokenizer.load(),cases=[];
  const continuation=tokenizer.encode('Reading books helps people learn about the world, understand different perspectives, develop their vocabulary, and explore ideas through stories and clear explanations.').slice(0,24);
  if(continuation.length!==24||continuation.includes(2))throw new Error('Invalid fixed continuation');
  async function execute(ids:Uint32Array,compressed:boolean,warmup=false){
    const calibration=compressed?await Calibration.load(device):undefined;let model:Decoder|undefined;
    try{
      model=await Decoder.create(weights,ids.length+32,32,calibration);await model.prefill(ids);
      const samples:number[]=[],selected:number[]=[],logits:Float32Array[]=[];
      for(const token of continuation.subarray(0,warmup?2:24)){
        const start=performance.now(),result=await model.decode(token),id=greedy(result.logits),elapsed=performance.now()-start;
        if(!Number.isFinite(elapsed)||elapsed<=0)throw new Error('Invalid timing');samples.push(elapsed);selected.push(id);logits.push(result.logits);
      }
      if(model.length!==ids.length+samples.length||(compressed&&model.encodedRows!==90*model.length))throw new Error('Append count mismatch');
      return {samples,selected,logits,capacity:model.capacity,active_length:model.length,encoded_key_rows:model.encodedRows};
    }finally{model?.destroy();calibration?.destroy();}
  }
  try{
    for(const compressed of [false,true])await execute(tokenizer.encode('A clear explanation starts with'),compressed,true);
    for(let index=0;index<parent.cases.length;index++){
      const original=parent.cases[index],ids=new Uint32Array(original.prompt_ids),outputs=new Map<string,Awaited<ReturnType<typeof execute>>>();
      for(const compressed of (index%2?[true,false]:[false,true])){
        const path=compressed?'compressed':'fp16';console.log(`Fixed-continuation decode: context ${ids.length}, ${path}`);
        outputs.set(path,await execute(ids,compressed));
      }
      const a=outputs.get('fp16')!,b=outputs.get('compressed')!;
      cases.push({context_length:ids.length,prompt_ids:Array.from(ids),continuation_ids:Array.from(continuation),
        measurements:Array.from(outputs,([path,r])=>({path,decode:summary(r.samples),decode_tokens_per_second:24000/r.samples.reduce((a,b)=>a+b,0),
          capacity:r.capacity,active_length:r.active_length,encoded_key_rows:r.encoded_key_rows,selected_ids:r.selected})),
        same_prefix_logit_errors:a.logits.map((x,i)=>compare(b.logits[i],x))});
    }
    return {passed:errors.length===0,errors,environment,model:weights.manifest,parent_record:parentPath,parent_sha256:parentSha,cases,
      trajectory_repetitions:1,decode_steps_per_path:24,warmup:'one short prefill plus two decode steps per path before the suite',
      boundaries:'Supplemental decode only; parent greedy run retained unchanged. One trajectory per path/context, 24 consecutive supplied tokens identical across paths, EOS does not stop supplied-token evaluation. Host wall-clock includes forward, live append, GPU waits, logits readback and greedy selection; excludes prefill, loading, compilation and logit comparison. This is teacher-forced decode timing, not autonomous-generation throughput; p95 describes 24 steps, not repeated-run variability.'};
  }finally{weights.destroy();device.destroy();}
}
