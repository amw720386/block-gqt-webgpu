import {Weights} from "../../src/model/weights.js";
import {Calibration} from "../../src/model/calibration.js";
import {Decoder} from "../../src/model/decoder.js";
import {Tokenizer} from "../../src/model/tokenizer.js";
import {loadFixture} from "./fixture.js";
import {compare,exact} from "./compare.js";
export async function modelIntegration(base:string,weights:Weights,calibration:Calibration,tokenizer:Tokenizer){
  const index=await (await fetch(base+'/integration-index.json')).json(),a=weights.adapter,cases=[];
  for(const [actual,expected] of [[a.layers,index.layers],[a.hiddenSize,index.hidden_size],[a.intermediateSize,index.intermediate_size],[a.queryHeads,index.query_heads],[a.kvHeads,index.kv_heads],[a.headDim,index.head_dim],[a.ropeTheta,index.rope_theta],[a.rmsNormEps,index.rms_norm_eps],[a.qkvBias,index.qkv_bias]])if(actual!==expected)throw new Error('Model adapter metadata mismatch');
  for(const t of index.tokenizer_tests)exact(tokenizer.encode(t.text),t.ids,"model tokenizer");
  if(index.reserved_token_id!==null&&tokenizer.decode([index.reserved_token_id])!=="")throw new Error('Reserved model token was not ignored');let outOfRangeRejected=false;try{tokenizer.decode([a.vocabSize]);}catch{outOfRangeRejected=true;}if(!outOfRangeRejected)throw new Error('Out-of-range model token was accepted');
  for(const workload of index.workloads)for(const compressed of [false,true]){
    const names:string[]=workload.paths[compressed?'compressed':'fp16'],model=await Decoder.create(weights,workload.context_length+4,32,compressed?calibration:undefined);
    try{
      let actual=await model.prefill(Uint32Array.from(workload.prompt_ids),true),expected=await loadFixture(`${base}/${names[Math.ceil(workload.context_length/32)-1]}.json`);
      const checks=[];for(const layer of (expected.manifest as any).layers){const error=compare(actual.layers[layer],expected.f32(`attention_${layer}`));if(!compressed||layer===0)compare(actual.layers[layer],expected.f32(`attention_${layer}`),compressed?[0.002,0.001]:[0.02,0.001]);checks.push({stage:'prefill',layer,error});}
      const logitErrors=[{stage:'prefill',error:compare(actual.logits,expected.f32('logits'))}];if(!compressed)compare(actual.logits,expected.f32('logits'),[0.01,0.001]);
      for(let i=Math.ceil(workload.context_length/32);i<names.length;i++){
        const token=Number((expected.manifest as any).selected_token);actual=await model.decode(token,true);expected=await loadFixture(`${base}/${names[i]}.json`);
        for(const layer of (expected.manifest as any).layers){const error=compare(actual.layers[layer],expected.f32(`attention_${layer}`));if(!compressed||layer===0)compare(actual.layers[layer],expected.f32(`attention_${layer}`),compressed?[0.002,0.001]:[0.02,0.001]);checks.push({stage:'decode',layer,error});}
        logitErrors.push({stage:'decode',error:compare(actual.logits,expected.f32('logits'))});if(!compressed)compare(actual.logits,expected.f32('logits'),[0.01,0.001]);
      }
      if(model.length!==workload.context_length+names.length-Math.ceil(workload.context_length/32))throw new Error('Model cache length mismatch');
      if(compressed&&model.encodedRows!==model.length*a.layers*a.kvHeads)throw new Error('Compressed keys were not encoded exactly once');
      if(compressed&&model.allocations.some(x=>x.role==='k'&&x.name.includes('FP16 K')))throw new Error('Compressed path allocated an FP16 K cache');
      cases.push({context_length:workload.context_length,path:compressed?'compressed':'fp16',length:model.length,encoded_rows:model.encodedRows,attention:checks,logits:logitErrors});
    }finally{model.destroy();}
  }
  return {cases,tokenizerTests:index.tokenizer_tests.length};
}
