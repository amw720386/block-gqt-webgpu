"""Strict attention oracle with the exact effective K used by both GPU paths."""
import hashlib
import json
import sys
import numpy as np
import torch
from reference.attention import attention
from tests.oracles.fixture import read_fixture
from tests.oracles.oracle import ROOT

def main():
    path=ROOT/sys.argv[1]; report=json.loads(path.read_text()); assert report['passed']
    torch.set_num_threads(1); checks=[]
    for case in report['cases']:
        m,a=read_fixture(ROOT/f"fixtures/attention/{case['name']}.json")
        c=case['capture']; payload=(ROOT/c['binary']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==c['sha256']
        def get(name,shape):
            desc=c['arrays'][name]
            return np.frombuffer(payload,dtype='<f4',offset=desc['offset'],count=desc['length']).reshape(shape).copy()
        k=get('reconstructed_k',(m['context_length'],m['dim']))
        for causal in [False,True]:
            expected=attention(torch.from_numpy(a['q']),torch.from_numpy(k),torch.from_numpy(a['v']),torch.from_numpy(a['positions'].astype(np.int64)),causal)
            errors={}
            for stage,ref in zip(['logits','probabilities','output'],expected):
                actual=get(f'{str(causal).lower()}_{stage}',tuple(ref.shape)); atol,rtol=m['tolerances']['baseline'][stage]
                np.testing.assert_allclose(actual,ref.numpy(),atol=atol,rtol=rtol)
                finite=np.isfinite(ref.numpy()); delta=actual[finite]-ref.numpy()[finite]
                errors[stage]={'max_absolute_error':float(np.max(np.abs(delta))), 'relative_l2_error':float(np.linalg.norm(delta)/max(np.linalg.norm(ref.numpy()[finite]),1e-30))}
            checks.append({'name':case['name'],'causal':causal,'errors':errors})
    out=path.with_name(path.stem+'-cpu.json')
    out.write_text(json.dumps({'passed':True,'gpu_record':str(path.relative_to(ROOT)),'gpu_record_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checks':checks},indent=2)+'\n')
    print(f'{len(checks)} strict comparisons passed: {out}')

if __name__=='__main__':main()
