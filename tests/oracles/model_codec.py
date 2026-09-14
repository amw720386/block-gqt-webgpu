"""Model calibration adapter for the existing, explicitly labeled CPU emulator."""
import torch
from tests.oracles.fixture import read_fixture
from tests.oracles.oracle import ROOT
from reference.production_math import encode,reconstruct

def calibrations():
    m,arrays=read_fixture(ROOT/'fixtures/model/calibration.json');result=[]
    for i in range(90):
        c=torch.from_numpy(arrays[f'h{i}_config'].astype('int64'));t=torch.from_numpy(arrays[f'h{i}_table']);maxc=int(c[5]);offset=4096+64*maxc
        perm=c[int(c[6]):int(c[6])+64];inv=torch.argsort(perm)
        result.append({'rotation':t[:4096].reshape(64,64),'centroids':t[4096:offset].reshape(64,maxc),'lut_offsets':t[offset:offset+64],'lut_inv_scales':t[offset+64:offset+128],
                       'group_bits':torch.unique(torch.from_numpy(arrays[f'h{i}_widths'].astype('int64'))),'head_perm':perm,'inv_head_perm':inv,
                       'group_of':c[int(c[7]):int(c[7])+64],'code_lut':c[int(c[9]):].reshape(-1,256).byte(),'pos_to_cb':c[int(c[8]):int(c[8])+64],
                       'pack_perm':torch.arange(64),'inv_pack_perm':torch.arange(64),'row_bytes':int(c[2]),'norm_stride':int(c[3]),'nibble_dims':int(c[4])})
    return result

def compress(x,a):
    codes,raw,scales,_=encode(x.half().float(),a);return reconstruct(codes,scales.half(),a)

def unpack(codes_bytes,norm_bits,length,a):
    rows=torch.as_tensor(codes_bytes,dtype=torch.uint8).reshape(-1)[:length*a['row_bytes']].reshape(length,a['row_bytes']);ns=a['nibble_dims']
    codes=torch.empty(length,64,dtype=torch.uint8)
    for j in range(64):codes[:,j]=(rows[:,j//2]>>((j%2)*4))&15 if j<ns else rows[:,ns//2+j-ns]
    scales=torch.as_tensor(norm_bits,dtype=torch.uint16).reshape(-1)[:length*a['norm_stride']].view(torch.float16).reshape(length,a['norm_stride'])
    return reconstruct(codes,scales,a)
