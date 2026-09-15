"""Fetch pinned public weights/tokenizer and export explicitly rounded FP16 tensors."""
import hashlib
import json
from pathlib import Path
import urllib.request
import torch
from safetensors.torch import load_file, save_file

ROOT=Path(__file__).resolve().parents[1]
def main():
    spec=json.loads((ROOT/'models/model.json').read_text()); dest=ROOT/'models/smollm2';dest.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','generation_config.json','model.safetensors']:
        path=dest/name
        if not path.exists():
            url=f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{name}"
            print(f'Downloading {name}',flush=True); temporary=path.with_suffix(path.suffix+'.download')
            urllib.request.urlretrieve(url,temporary);temporary.replace(path)
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    state=load_file(str(dest/'model.safetensors')); count=sum(t.numel() for t in state.values())
    assert count==spec['parameters'],(count,spec['parameters'])
    rounded={k:v.to(torch.float16).contiguous() for k,v in state.items()}
    assert all(torch.isfinite(t).all() for t in rounded.values())
    save_file(rounded,str(dest/'fp16.safetensors'))
    hashes['fp16.safetensors']=hashlib.sha256((dest/'fp16.safetensors').read_bytes()).hexdigest()
    report={**spec,'sha256':hashes,'weight_bytes':sum(t.numel()*2 for t in rounded.values()),
            'conversion':'BF16 -> FP16, round-to-nearest-even via PyTorch; no weight quantization'}
    (dest/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(f'Ready: {dest}',flush=True)

if __name__=='__main__':main()
