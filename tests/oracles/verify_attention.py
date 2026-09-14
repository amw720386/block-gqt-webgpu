"""Strict CPU attention check on GPU-reconstructed K; codec port error stays separate.

The runtime never reads compressed K to CPU. These test captures contain only
FP32 reconstruction/attention intermediates. Checking identical effective K
distinguishes discontinuous LUT code selection from a faulty attention kernel.
"""
import hashlib
import json
import numpy as np
import torch
from reference.attention import attention
from tests.oracles.fixture import read_fixture
from tests.oracles.oracle import ROOT


def main():
    torch.set_num_threads(1)
    report=json.loads((ROOT/'results/attention-compressed.json').read_text());assert report['passed']
    checks=[]
    for c in report['cases']:
        m,a=read_fixture(ROOT/f"fixtures/attention/{c['name']}.json")
        payload=(ROOT/'results'/c['capture']['binary']).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==c['capture']['sha256']
        def array(name,shape):
            desc=c['capture']['arrays'][name]
            return np.frombuffer(payload,dtype='<f4',count=desc['length'],offset=desc['offset']).reshape(shape).copy()
        decoded=array('reconstructed_k',(m['context_length'],m['dim']))
        q=torch.from_numpy(a['q']);v=torch.from_numpy(a['v']);positions=torch.from_numpy(a['positions'].astype(np.int64))
        for causal in (False,True):
            for label,k in (('baseline',a['k']),('compressed',decoded)):
                expected=attention(q,torch.from_numpy(k),v,positions,causal)
                errors={}
                for stage,ref in zip(('logits','probabilities','output'),expected):
                    actual=array(f'{label}_{str(causal).lower()}_{stage}',tuple(ref.shape))
                    atol,rtol=m['tolerances']['baseline'][stage]
                    np.testing.assert_allclose(actual,ref.numpy(),atol=atol,rtol=rtol)
                    finite=np.isfinite(ref.numpy());delta=actual[finite]-ref.numpy()[finite]
                    errors[stage]=dict(max_absolute_error=float(np.max(np.abs(delta))),
                                       relative_l2_error=float(np.linalg.norm(delta)/max(np.linalg.norm(ref.numpy()[finite]),1e-30)))
                checks.append(dict(name=c['name'],path=label,causal=causal,attention_error_given_identical_k=errors))
    out=dict(passed=True,checks=checks,codec_note='Full encoder port error remains in attention-compressed.json, independently of strict attention arithmetic checks.')
    (ROOT/'results/attention-cpu-verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print(f'Passed {len(checks)} strict CPU/GPU attention comparisons, logits/probabilities/output, holding effective K fixed.')


if __name__=='__main__':main()
