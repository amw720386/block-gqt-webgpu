"""Generate ignored, deterministic real-model integration references."""
import json
import sys
import torch
from transformers import DynamicCache
from tests.oracles.export_model import load_model
from tests.oracles.fixture import write_fixture
from tests.oracles.model_codec import calibrations,compress
from tests.oracles.oracle import ROOT

class HalfCache(DynamicCache):
    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        return super().update(key_states.half().float(),value_states.half().float(),layer_idx,cache_kwargs)
class CompressedCache(DynamicCache):
    def __init__(self,calibration,kvheads):super().__init__();self.calibration=calibration;self.kvheads=kvheads
    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        keys=torch.stack([compress(key_states[0,h],self.calibration[layer_idx*self.kvheads+h]) for h in range(self.kvheads)])[None]
        return super().update(keys,value_states.half().float(),layer_idx,cache_kwargs)

def main():
    if '--model' not in sys.argv:raise SystemExit('Usage: python -m tests.oracles.export_model_integration --model MODEL')
    name=sys.argv[sys.argv.index('--model')+1];directory=ROOT/'models'/name;torch.set_num_threads(4);torch.manual_seed(0);model,tokenizer=load_model(name);cfg=model.config
    kvheads=cfg.num_key_value_heads;layers=cfg.num_hidden_layers;selected_layers=(0,layers-1);calibration=calibrations(directory/'calibration.json',layers*kvheads)
    source=tokenizer.encode('Clear explanations use concrete examples and verify each step. ',add_special_tokens=False);tail=tokenizer.encode('The capital of France is',add_special_tokens=False);workloads=[]
    tokenizer_texts=['Hello, world!','123 45.67',' café\n東京 😀',"I'm HERE; we're testing.",'<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n']
    for length in (16,48):
        ids=(source*((length-len(tail)+len(source)-1)//len(source))+tail)[-length:];paths={}
        for compressed in (False,True):
            cache=CompressedCache(calibration,kvheads) if compressed else HalfCache();prefix=f'c{length}_{"compressed" if compressed else "fp16"}';names=[];logits=None;invocations=[ids[i:i+32] for i in range(0,len(ids),32)]
            for step in range(len(invocations)+2):
                current=invocations[step] if step<len(invocations) else [int(logits.argmax())];captured={}
                hooks=[model.model.layers[i].self_attn.o_proj.register_forward_pre_hook(lambda m,a,i=i:captured.__setitem__(i,a[0].detach().clone())) for i in selected_layers]
                with torch.no_grad():result=model(torch.tensor([current]),past_key_values=cache,use_cache=True)
                for hook in hooks:hook.remove()
                logits=result.logits[0,-1].float();fixture=f'{prefix}_step{step}';arrays={'input_ids':('u32',current),'logits':('f32',logits.numpy())};arrays.update({f'attention_{i}':('f32',captured[i][0].numpy()) for i in selected_layers})
                write_fixture(directory/f'{fixture}.json',{'version':2,'name':fixture,'path':'compressed' if compressed else 'fp16','stage':'prefill' if step<len(invocations) else 'decode','rows':len(current),'length':cache.get_seq_length(),'selected_token':int(logits.argmax()),'layers':selected_layers,'reference':'Transformers 4.51.3 eager; exported FP16 weights promoted to FP32; FP16 V and FP16 or Block-GTQ post-RoPE K cache'},arrays);names.append(fixture)
            paths['compressed' if compressed else 'fp16']=names
        workloads.append({'context_length':length,'prompt_ids':ids,'paths':paths})
    manifest=json.loads((directory/'manifest.json').read_text());qkv_bias=any(name.endswith('self_attn.q_proj.bias') for name,_ in model.named_parameters());known_ids=set(tokenizer.get_vocab().values());reserved_token_id=next((i for i in range(cfg.vocab_size-1,-1,-1) if i not in known_ids),None)
    report={'model':manifest['repository'].split('/')[-1],'revision':manifest['revision'],'layers':layers,'hidden_size':cfg.hidden_size,'intermediate_size':cfg.intermediate_size,'query_heads':cfg.num_attention_heads,'kv_heads':kvheads,'head_dim':cfg.hidden_size//cfg.num_attention_heads,'rope_theta':cfg.rope_theta,'rms_norm_eps':cfg.rms_norm_eps,'qkv_bias':qkv_bias,'reserved_token_id':reserved_token_id,'workloads':workloads,'tokenizer_tests':[{'text':text,'ids':tokenizer.encode(text,add_special_tokens=False)} for text in tokenizer_texts]}
    (directory/'integration-index.json').write_text(json.dumps(report,indent=2)+'\n');print(f'Exported {report["model"]} FP16 and compressed integration references')

if __name__=='__main__':main()
