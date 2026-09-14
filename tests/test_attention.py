import json
import unittest
import numpy as np
import torch
import torch.nn.functional as F
from reference.attention import attention
from tests.oracles.fixture import read_fixture
from tests.oracles.oracle import ROOT


class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_twelve_workloads_against_goldens_and_pytorch_sdpa(self):
        index=json.loads((ROOT/"fixtures/attention/index.json").read_text())
        self.assertEqual(len(index["cases"]),12)
        for c in index["cases"]:
            m,a=read_fixture(ROOT/f"fixtures/attention/{c['name']}.json")
            q,k,v=(torch.from_numpy(a[x]) for x in ("q","k","v"))
            positions=torch.from_numpy(a["positions"].astype(np.int64))
            for causal in (False,True):
                for label,keys in (("baseline",k),("compressed",torch.from_numpy(a["reconstructed_k"])),("fp32_source",torch.from_numpy(a["k_source"]))):
                    with self.subTest(case=c["name"],causal=causal,path=label):
                        prefix=("causal_" if causal else "")+label+"_"
                        logits,probs,out=attention(q,keys,v,positions,causal)
                        for kind,result in (("logits",logits),("probabilities",probs),("output",out)):
                            np.testing.assert_array_equal(result,a[prefix+kind])
                        mask=(torch.arange(len(k))[None,:]<=positions[:,None]) if causal else None
                        expected=F.scaled_dot_product_attention(q[None,None],keys.float()[None,None],v[None,None],attn_mask=mask,dropout_p=0)[0,0]
                        torch.testing.assert_close(out,expected,atol=2e-6,rtol=2e-5)
                        torch.testing.assert_close(probs.sum(1),torch.ones(len(q)),atol=1e-6,rtol=0)
                        if causal:
                            self.assertEqual(torch.count_nonzero(probs[~mask]).item(),0)
                            torch.testing.assert_close(out[0],v[0],atol=0,rtol=0)

    def test_hand_computable_uniform_and_stable_softmax(self):
        q=torch.zeros(1,2);k=torch.tensor([[1.,2.],[3.,4.]]);v=torch.tensor([[2.,4.],[6.,8.]])
        logits,probs,out=attention(q,k,v,torch.tensor([1]))
        torch.testing.assert_close(probs,torch.tensor([[.5,.5]]),atol=0,rtol=0)
        torch.testing.assert_close(out,torch.tensor([[4.,6.]]),atol=0,rtol=0)
        _,probs,out=attention(torch.full((1,2),100.),k,v,torch.tensor([1]))
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertAlmostEqual(float(probs.sum()),1.)

    def test_invalid_shapes_positions_and_nonfinite_inputs(self):
        q=torch.zeros(1,64);k=torch.zeros(4,64);v=k.clone()
        for positions in (torch.tensor([-1]),torch.tensor([4]),torch.tensor([0,1]),torch.tensor([0.])):
            with self.assertRaises(ValueError):attention(q,k,v,positions)
        with self.assertRaises(ValueError):attention(q,k,v[:2],torch.tensor([0]))
        with self.assertRaises(ValueError):attention(q[:,None],k,v,torch.tensor([0]))
        q[0,0]=float('nan')
        with self.assertRaises(ValueError):attention(q,k,v,torch.tensor([0]))
