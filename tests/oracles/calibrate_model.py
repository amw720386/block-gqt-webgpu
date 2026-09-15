"""Export pinned upstream energy allocation/transforms using real pre-RoPE Q/K."""
import hashlib
import importlib.util
import json
import torch
from tests.oracles.export_model import load_model
from tests.oracles.oracle import ROOT, COMMIT, load
from tests.oracles import production_oracle as oracle
from tests.oracles.fixture import write_fixture

CALIBRATION='''An algorithm is a sequence of instructions for solving a problem. Sorting places records in an order, and searching finds a matching record. A program should be tested on empty inputs, ordinary inputs, and boundary conditions. The oceans store heat and move it around the planet. Water evaporates, condenses into clouds, and returns as rain or snow. Rivers carry water from high ground toward lakes and the sea. A library organizes books so readers can discover information. Catalogs describe authors, titles, and subjects. Clear writing helps readers distinguish observations from explanations. In a workshop, measurements should be repeated and tools should be checked before use. A careful record describes the materials, procedure, and outcome so another person can repeat the work.'''
def vendor(name):
    path=ROOT/'vendor/blockgtq-calibration'/f'{name}.py';provenance=json.loads(path.with_name('provenance.json').read_text())
    assert provenance['commit']==COMMIT and hashlib.sha256(path.read_bytes()).hexdigest()==provenance['sha256'][path.name]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    torch.set_num_threads(4);model,tokenizer=load_model();ids=tokenizer.encode(CALIBRATION,add_special_tokens=False)
    qs={};ks={};hooks=[]
    for i,layer in enumerate(model.model.layers):
        hooks += [layer.self_attn.q_proj.register_forward_hook(lambda m,a,o,i=i:qs.__setitem__(i,o.detach())),layer.self_attn.k_proj.register_forward_hook(lambda m,a,o,i=i:ks.__setitem__(i,o.detach()))]
    with torch.no_grad():model(torch.tensor([ids]),use_cache=False)
    for h in hooks:h.remove()
    freq=vendor('freq_analysis');rope=vendor('rope_utils');arrays={};heads=[]
    for layer in range(30):
        for head in range(3):
            q=qs[layer].reshape(-1,9,64)[:,head*3:head*3+3].reshape(-1,64);k=ks[layer].reshape(-1,3,64)[:,head]
            scores=freq.compute_freq_importance_energy(rope.decompose_freq_blocks(q,64),rope.decompose_freq_blocks(k,64))
            widths=load('allocator').greedy_bit_allocation(scores,128,1,8);p=oracle.setup(widths);pm=oracle.pack_meta(widths,p._head_perm)
            assert torch.equal(pm['pack_perm'],torch.arange(64));lut,cb,offsets,inv=oracle.code_lut(p)
            ng=p._n_groups;row=pm['nopack_start']//2+pm['nopack_len'];ns=ng;maxc=p._pos_centroids.shape[1]
            config=[64,ng,row,ns,pm['nopack_start'],maxc,0,0,0,0]
            for slot,values in [(6,p._head_perm),(7,p._group_of),(8,cb),(9,lut.flatten())]:config[slot]=len(config);config.extend(values.tolist())
            rotation=p._block_rot_T.T.contiguous();table=torch.cat([rotation.flatten(),p._pos_centroids.flatten(),offsets,inv])
            ranges=[]
            for j in range(64):
                members=(p._group_of==p._group_of[j]).nonzero().flatten();ranges += [int(members[0]),int(members[-1])+1]
            prefix=f'h{layer*3+head}_';items={'config':('u32',config),'table':('f32',table.numpy()),'ranges':('u32',ranges),'scores':('f32',scores.numpy()),'widths':('u32',widths.numpy())}
            arrays.update({prefix+n:value for n,value in items.items()});heads.append({'layer':layer,'kv_head':head,'row_bytes':row,'norm_stride':ns,'groups':ng,'max_centroids':maxc,'average_bits':float(widths.float().mean())})
    meta={'version':2,'name':'smollm2-calibration','upstream_commit':COMMIT,'model_revision':'12fd25f77366fa6b3b4b768ec3050bf629380bac','heads':heads,'rotation_threshold':2,'average_bits':4,
          'coordinate_space':'post-RoPE cache; pre-RoPE pair-energy calibration','calibration_text':CALIBRATION,'calibration_ids':ids,
          'score':'(mean squared pair norm over all grouped Q heads + mean squared K pair norm)/2; pinned upstream function',
          'vendor_sha256':{f:hashlib.sha256((ROOT/'vendor/blockgtq-calibration'/f).read_bytes()).hexdigest() for f in ['freq_analysis.py','rope_utils.py']}}
    write_fixture(ROOT/'fixtures/model/calibration.json',meta,arrays);print('Exported 90 real-model calibrations',flush=True)

if __name__=='__main__':main()
