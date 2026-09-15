"""Keep strict local attention error separate from propagated full-model drift."""
import hashlib
import json
import numpy as np
import torch
import sys
from tests.oracles.oracle import ROOT
from tests.oracles.model_codec import calibrations,unpack
from reference.production_math import encode

def main():
    compressed='--compressed' in sys.argv;stem='model-compressed-test' if compressed else 'model-test';path=ROOT/f'results/model/{stem}.json';r=json.loads(path.read_text());assert r['passed'];torch.set_num_threads(4);checks=[];codec=[];calibration=calibrations() if compressed else None
    for case in r['cases']:
        cap=case['capture'];payload=(ROOT/cap['binary']).read_bytes();assert hashlib.sha256(payload).hexdigest()==cap['sha256']
        def get(key):
            a=cap['arrays'][key];return np.frombuffer(payload,dtype='<f4',count=a['length'],offset=a['offset']).copy()
        for layer in range(30):
            rows=case['rows'];length=case['length'];q=torch.from_numpy(get(f'q_{layer}').reshape(rows,9,64)).transpose(0,1)
            if compressed:k=torch.stack([unpack(get(f'codes_{layer*3+h}').astype(np.uint8),get(f'norms_{layer*3+h}').astype(np.uint16),length,calibration[layer*3+h]) for h in range(3)]).repeat_interleave(3,dim=0)
            else:k=torch.from_numpy(get(f'k_bits_{layer}').astype(np.uint16).view(np.float16).astype(np.float32).reshape(3,length,64)).repeat_interleave(3,dim=0)
            v=torch.from_numpy(get(f'v_bits_{layer}').astype(np.uint16).view(np.float16).astype(np.float32).reshape(3,length,64)).repeat_interleave(3,dim=0)
            logits=q@k.transpose(-1,-2)/8;mask=torch.arange(length)[None,:]>torch.arange(length-rows,length)[:,None]
            expected=(logits.masked_fill(mask[None],-torch.inf).softmax(-1)@v).transpose(0,1).reshape(rows,576).numpy()
            actual=get(f'attention_{layer}').reshape(rows,576)
            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
            checks.append({'step':case['name'],'layer':layer,'max_abs':float(np.max(np.abs(actual-expected)))})
            if compressed:
                new_k=torch.from_numpy(get(f'new_k_{layer}').reshape(rows,3,64)).half().float()
                for head in range(3):
                    a=calibration[layer*3+head];codes,_,scales,_=encode(new_k[:,head],a)
                    raw=get(f'codes_{layer*3+head}').astype(np.uint8)[:length*a['row_bytes']].reshape(length,a['row_bytes'])[length-rows:];ns=a['nibble_dims'];actual_codes=np.empty((rows,64),np.uint8)
                    for j in range(64):actual_codes[:,j]=(raw[:,j//2]>>((j%2)*4))&15 if j<ns else raw[:,ns//2+j-ns]
                    actual_norms=get(f'norms_{layer*3+head}').astype(np.uint16)[:length*a['norm_stride']].reshape(length,a['norm_stride'])[length-rows:]
                    codec.append({'step':case['name'],'layer':layer,'head':head,'index_differences_on_identical_input':int((actual_codes!=codes.numpy()).sum()),'fp16_scale_bit_differences':int((actual_norms!=scales.half().view(torch.uint16).numpy()).sum())})
    out={'passed':True,'gpu_report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checks':checks,'codec_comparison_on_identical_new_k':codec,
         'note':'Strict attention with identical GPU Q/K/V. Unchanged full-model logit tolerance and greedy-token gate pass separately; propagated layer differences and local-threshold exceedances remain in GPU report.'}
    target='model-compressed-cpu-gate' if compressed else 'model-cpu-gate';(ROOT/f'results/model/{target}.json').write_text(json.dumps(out,indent=2)+'\n');print(f'{len(checks)} strict per-layer checks passed; max error {max(c["max_abs"] for c in checks)}')

if __name__=='__main__':main()
