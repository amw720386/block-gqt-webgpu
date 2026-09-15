import torch
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding,apply_rotary_pos_emb
from transformers import LlamaConfig
from tests.oracles.fixture import write_fixture
from tests.oracles.oracle import ROOT

def main():
    cfg=LlamaConfig.from_pretrained(ROOT/'models/smollm2');rope=LlamaRotaryEmbedding(cfg);arrays={};cases=[]
    for heads in [3,9]:
        for start in [0,1,1023,4095,8189]:
            name=f'h{heads}_p{start}';x=torch.sin(torch.arange(3*heads*64).float()*0.173).reshape(3,heads,64)
            positions=torch.arange(start,start+3)[None];cos,sin=rope(x,positions)
            y,_=apply_rotary_pos_emb(x.transpose(0,1)[None],x.transpose(0,1)[None],cos,sin)
            arrays[name+'_x']=('f32',x.numpy());arrays[name+'_y']=('f32',y[0].transpose(0,1).numpy());cases.append({'name':name,'start':start,'heads':heads,'rows':3})
    arrays['frequencies']=('f32',rope.inv_freq.numpy())
    write_fixture(ROOT/'fixtures/model/rope.json',{'version':2,'name':'model-rope','cases':cases,'convention':'Transformers split-half RoPE, theta=100000; no scaling'},arrays)

if __name__=='__main__':main()
