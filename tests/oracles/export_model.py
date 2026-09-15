"""Trusted Transformers eager model with explicit FP16 persistent KV boundary."""
import json
import hashlib
import torch
import numpy as np
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer, DynamicCache
from safetensors.torch import load_file
from tests.oracles.oracle import ROOT
from tests.oracles.fixture import write_fixture

class HalfCache(DynamicCache):
    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        return super().update(key_states.half().float(),value_states.half().float(),layer_idx,cache_kwargs)

def load_model():
    directory=ROOT/'models/smollm2'; config=LlamaConfig.from_pretrained(directory);config._attn_implementation='eager'
    model=LlamaForCausalLM(config).eval();state=load_file(str(directory/'fp16.safetensors'))
    state['lm_head.weight']=state['model.embed_tokens.weight'];model.load_state_dict(state,strict=True)
    return model,AutoTokenizer.from_pretrained(directory,local_files_only=True)

def main():
    torch.set_num_threads(4);torch.manual_seed(0)
    model,tokenizer=load_model();prompt='The capital of France is';ids=tokenizer.encode(prompt,add_special_tokens=False)
    cache=HalfCache();records=[];current=torch.tensor([ids]);generated=[]
    with torch.no_grad():
        for step in range(4):
            captured=[];hooks=[layer.self_attn.o_proj.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone())) for layer in model.model.layers]
            result=model(current,past_key_values=cache,use_cache=True);[h.remove() for h in hooks]
            logits=result.logits[0,-1].float();token=int(logits.argmax());generated.append(token)
            arrays={'input_ids':('u32',current[0].numpy()),'logits':('f32',logits.numpy())}
            for i,value in enumerate(captured):arrays[f'attention_{i}']=('f32',value[0].numpy())
            meta={'version':2,'name':f'fp16_step{step}','model_revision':'12fd25f77366fa6b3b4b768ec3050bf629380bac','step':step,
                  'rows':current.shape[1],'length':cache.get_seq_length(),'selected_token':token,'reference':'Transformers 4.51.3 eager; FP16 exported weights promoted to FP32; post-RoPE K/V rounded to FP16 then promoted for CPU attention',
                  'tolerance':{'logits':[0.003,0.0003],'attention':[0.001,0.0003]}}
            write_fixture(ROOT/f'fixtures/model/fp16_step{step}.json',meta,arrays);records.append(meta['name']);current=torch.tensor([[token]])
    texts=[prompt,'Hello, world!','123 45.67',' café\n東京 😀',' spaces  \n\n','<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n',"I'm HERE; we're testing."]
    report={'cases':records,'prompt':prompt,'prompt_ids':ids,'greedy_ids':generated,'greedy_text':tokenizer.decode(generated),'tokenizer_tests':[{'text':t,'ids':tokenizer.encode(t,add_special_tokens=False)} for t in texts],
            'weights_sha256':json.loads((ROOT/'models/smollm2/manifest.json').read_text())['sha256']['fp16.safetensors']}
    (ROOT/'fixtures/model/index.json').write_text(json.dumps(report,indent=2)+'\n');print(report['greedy_text'])

if __name__=='__main__':main()
