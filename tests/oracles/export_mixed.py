"""Export upstream grouping/LUT/host packing plus labeled FP32 encoder emulation."""
import json
import platform

import numpy as np
import scipy
import torch

from tests.oracles.fixture import write_fixture
from tests.oracles.oracle import ROOT, COMMIT, load
from tests.oracles import production_oracle as oracle
from reference.production_math import encode, reconstruct


def export_case(name, desired):
    scores = torch.tensor([4.0 ** int(b) for b in desired], dtype=torch.float32)
    widths = load("allocator").greedy_bit_allocation(scores, sum(desired))
    assert widths.tolist() == desired
    p = oracle.setup(widths)
    pm = oracle.pack_meta(widths, p._head_perm)
    lut, cb, offsets, inverse = oracle.code_lut(p)
    d, ng = p.head_dim, p._n_groups
    group_bits = torch.unique(widths, sorted=True)
    group_offsets = torch.tensor([int((p._group_of < g).sum()) for g in range(ng + 1)])
    pair_groups = torch.empty(d // 2, dtype=torch.long)
    pair_groups[p._head_perm[::2]] = p._group_of[::2]
    ns, nl = pm["nopack_start"], pm["nopack_len"]
    logical_bytes = ns // 2 + nl
    # Deliberately reserve wider rows/groups, as a different head/layer would.
    row_bytes, norm_stride, rows, capacity = logical_bytes + 3, ng + 1, 5, 7
    index = torch.arange(d, dtype=torch.float32)
    x = torch.stack((torch.zeros(d), ((index * 7) % 23 - 11) / 7,
                     torch.cos(index * 0.37) * 3, torch.sin(index * 0.23) * 0.001,
                     ((index * 3) % 19 - 9) * 17)).half().float()
    a = dict(scores=scores, allocation=widths, pair_order=p._head_perm[::2], pair_groups=pair_groups,
             head_perm=p._head_perm, inv_head_perm=p._head_perm_inv,
             pack_perm=pm["pack_perm"], inv_pack_perm=pm["inv_pack_perm"],
             group_of=p._group_of, group_bits=group_bits, group_offsets=group_offsets,
             group_byte_offsets=torch.tensor([s // 2 if s < ns else ns // 2 + s - ns for s in group_offsets]),
             rotation=p._block_rot_T.T.contiguous(), centroids=p._pos_centroids,
             boundaries=p._pos_boundaries, code_lut=lut, pos_to_cb=cb,
             lut_offsets=offsets, lut_inv_scales=inverse)
    codes, raw, corrected, rotated = encode(x, a)
    raw_padded = torch.zeros(1, rows, norm_stride)
    raw_padded[0, :, :ng] = raw
    gmask = torch.stack([(p._group_of == g).float() for g in range(norm_stride)])
    args = dict(mixed_nopack_starts=[ns], mixed_nopack_lens=[nl], max_mixed_bytes=row_bytes,
                k_pos_cents_stack=p._pos_centroids[None], k_inv_pp_stack=pm["inv_pack_perm"][None],
                k_group_mask_stack=gmask[None], max_n_groups=norm_stride)
    packed, corrected_host = oracle.host_pack(codes[None], raw_padded, args)
    torch.testing.assert_close(corrected_host[0, :, :ng], corrected, atol=0, rtol=0)
    persistent = corrected_host[0].half()
    r32 = reconstruct(codes, corrected_host[0], a)
    r16 = reconstruct(codes, persistent, a)
    # Execute the unchanged upstream cache constructor on CPU; no append integration.
    cache = oracle.cache(2, 1, d, capacity, [logical_bytes, row_bytes], [[ng], [norm_stride]])
    cache.k_packed[0, 0, 0, :rows] = packed[0]
    cache.k_norms[0, 0, 0, :rows] = persistent
    a.update(original=x, codes=codes, raw_norms=raw_padded[0], corrected_f32=corrected_host[0],
             persistent_norms=persistent, norm_bits=persistent.view(torch.uint16),
             packed=packed[0], rotated=rotated, reconstructed_f32=r32, reconstructed_f16=r16,
             cache_packed=cache.k_packed[0, 0, 0], cache_norms=cache.k_norms[0, 0, 0])
    arrays = {}
    for key, value in a.items():
        dtype = "f16" if value.dtype == torch.float16 else "u16" if value.dtype == torch.uint16 else (
            "f32" if value.is_floating_point() else "u8" if value.dtype == torch.uint8 else "u32")
        arrays[key] = (dtype, value.numpy())
    meta = dict(version=2, name=name, upstream_commit=COMMIT,
                oracle="upstream CPU grouping/LUT/host packing; Triton encoder emulated in PyTorch FP32",
                cuda_encoder_validated=False, coordinate_space="synthetic-rope-free", rotation_threshold=2,
                input_dtype="FP16 values promoted to FP32", packing="production-k-nibble-byte-v1",
                rows=rows, dim=d, n_groups=ng, norm_stride=norm_stride, row_bytes=row_bytes,
                logical_row_bytes=logical_bytes, nopack_start=ns, capacity=capacity, n_bins=256,
                max_centroids=p._pos_centroids.shape[1], requested_budget=sum(desired),
                average_bits=float(widths.float().mean()),
                tolerance=dict(atol=0.001, rtol=0.00002),
                norm_conversion_max_abs=float((r16-r32).abs().max()),
                quantization_max_abs=float((r32-x).abs().max()),
                environment=dict(python=platform.python_version(), torch=torch.__version__,
                                 numpy=np.__version__, scipy=scipy.__version__, device="cpu", threads=1))
    write_fixture(ROOT / f"fixtures/mixed/{name}.json", meta, arrays)
    return {"name": name, "dim": d, "average_bits": meta["average_bits"]}


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases = []
    for d in (16, 32, 64, 128):
        for label, pattern in (("low", [1, 3, 2, 4]), ("mixed", [1, 5, 2, 3])):
            cases.append(export_case(f"{label}_d{d}", (pattern * d)[:d//2]))
    cases.append(export_case("awkward_all_widths", [8, 1, 5, 2, 7, 3, 6, 4]))
    cases.append(export_case("only_high", [5, 6, 5, 6, 5, 6, 5, 6]))
    path = ROOT / "fixtures/mixed/index.json"
    path.write_text(json.dumps(dict(version=2, cases=cases), indent=2) + "\n")


if __name__ == "__main__":
    main()
