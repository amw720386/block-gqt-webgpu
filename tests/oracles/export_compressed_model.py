import json
import torch
from transformers import DynamicCache
from tests.oracles.export_model import load_model
from tests.oracles.model_codec import calibrations,compress
from tests.oracles.fixture import write_fixture
from tests.oracles.oracle import ROOT

class CompressedCache(DynamicCache):
    def __init__(self,calibration):super().__init__();self.calibration=calibration
    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        k=torch.stack([compress(key_states[0,h],self.calibration[layer_idx*3+h]) for h in range(3)])[None]
        return super().update(k,value_states.half().float(),layer_idx,cache_kwargs)

def main():
    torch.set_num_threads(4);model,tokenizer=load_model();cache=CompressedCache(calibrations());index=json.loads((ROOT/'fixtures/model/index.json').read_text());cases=[]
    # Teacher forcing keeps FP16 and compressed comparisons on identical contexts.
    inputs=[index['prompt_ids'],*[[i] for i in index['greedy_ids'][:-1]]]
    with torch.no_grad():
        for step,ids in enumerate(inputs):
            captured=[];hooks=[l.self_attn.o_proj.register_forward_pre_hook(lambda m,a:captured.append(a[0].detach().clone())) for l in model.model.layers]
            out=model(torch.tensor([ids]),past_key_values=cache,use_cache=True);[h.remove() for h in hooks]
            logits=out.logits[0,-1].float();name=f'compressed_step{step}'
            arrays={'input_ids':('u32',ids),'logits':('f32',logits.numpy())};arrays.update({f'attention_{i}':('f32',t[0].numpy()) for i,t in enumerate(captured)})
            write_fixture(ROOT/f'fixtures/model/{name}.json',{'version':2,'name':name,'rows':len(ids),'step':step,'selected_token':int(logits.argmax()),'reference':'Transformers eager plus pinned Block-GTQ CPU encoder emulator; not CUDA parity'},arrays);cases.append(name)
    (ROOT/'fixtures/model/compressed-index.json').write_text(json.dumps({'cases':cases,'calibration_sha256':json.loads((ROOT/'fixtures/model/calibration.json').read_text())['sha256']},indent=2)+'\n');print('Compressed CPU model reference exported')

if __name__=='__main__':main()
