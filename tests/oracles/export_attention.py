"""Deterministic synthetic attention fixtures. No CUDA or model-quality oracle."""
import json
import torch
from reference.attention import attention
from reference.production_math import encode, reconstruct
from reference.mixed_layout import pack
from tests.oracles.fixture import read_fixture, write_fixture
from tests.oracles.oracle import ROOT, COMMIT


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases=[]
    for d in (64, 128):
        layout, data=read_fixture(ROOT/f"fixtures/mixed/mixed_d{d}.json")
        a={k:torch.from_numpy(v) for k,v in data.items()}
        for length in (128,256,512,1024,2048,4096):
            generator=torch.Generator().manual_seed(731+d+length)
            q=torch.randn(3,d,generator=generator)
            k_source=torch.randn(length,d,generator=generator)*0.7
            k=k_source.half()
            v=torch.randn(length,d,generator=generator)*0.5
            positions=torch.tensor([0,length//2,length-1])
            codes,_,scales,_=encode(k.float(),a)
            norm_stride=layout["norm_stride"]
            persistent=torch.zeros(length,norm_stride,dtype=torch.float16)
            persistent[:,:layout["n_groups"]]=scales.half()
            compressed=reconstruct(codes,persistent,a)
            packed=pack(codes.numpy(),layout["nopack_start"],layout["row_bytes"])
            arrays={"q":("f32",q.numpy()),"k":("f16",k.numpy()),"k_source":("f32",k_source.numpy()),
                    "v":("f32",v.numpy()),"positions":("u32",positions.numpy()),
                    "packed_k":("u8",packed),"norms":("f16",persistent.numpy()),
                    "reconstructed_k":("f32",compressed.numpy())}
            for causal in (False,True):
                prefix="causal_" if causal else ""
                for label,keys in (("baseline",k),("compressed",compressed),("fp32_source",k_source)):
                    result=attention(q,keys,v,positions,causal)
                    for kind,value in zip(("logits","probabilities","output"),result):
                        arrays[prefix+label+"_"+kind]=("f32",value.numpy())
            name=f"t{length}_d{d}"
            metadata=dict(version=2,name=name,upstream_commit=COMMIT,layout=f"mixed_d{d}",layout_sha256=layout["sha256"],
                          oracle="CPU PyTorch attention; production compression arithmetic emulator",
                          cuda_encoder_validated=False,coordinate_space="synthetic-rope-free",
                          context_length=length,dim=d,queries=3,k_storage="f16",q_storage="f32",v_storage="f32",
                          accumulation="f32",tolerances={
                              "baseline":{"logits":[0.0001,0.00002],"probabilities":[0.00001,0.0001],"output":[0.0001,0.0001]},
                              "port":{"logits":[0.005,0.002],"probabilities":[0.0005,0.002],"output":[0.001,0.002]}},
                          environment={"torch":torch.__version__,"device":"cpu","threads":1})
            write_fixture(ROOT/f"fixtures/attention/{name}.json",metadata,arrays)
            cases.append({"name":name,"dim":d,"context_length":length})
    (ROOT/"fixtures/attention/index.json").write_text(json.dumps({"version":2,"cases":cases},indent=2)+"\n")
    print(f"Exported {len(cases)} attention workloads; causal and noncausal references in each.")


if __name__=="__main__":main()
