"""Goldens are computed only with the pinned oracle, never local TQ/packing.

Run explicitly to regenerate; tests consume committed fixtures without writes.
"""

import platform

import numpy as np
import scipy
import torch

from tests.oracles.fixture import write_fixture
from tests.oracles.oracle import COMMIT, ROOT, load


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dim, bits = 10, 3
    tq = load("tq").TurboQuantMSE(dim, bits, device=torch.device("cpu"))
    # Seeds are used only inside the oracle's initial setup. Consumers get
    # the actual exported FP32 tensor bytes and never regenerate transforms.
    x = torch.tensor([
        [0] * dim,
        [1, -2, 3, -4, 5, -6, 7, -8, 9, -10],
        [0.25, -0.5, 0.125, 1, -1, 2, -2, 0.75, -0.25, 0.0625],
        [1e-12, -2e-12, 3e-12, 0, 1e-12, 0, -1e-12, 0, 0, 2e-12],
        [1000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    codes, norms = tq.quantize(x)
    reconstruction = tq.dequantize(codes, norms)
    full_roundtrip = tq.quantize_dequantize(x)
    tq.norm_correction = False
    uncorrected = tq.dequantize(codes, norms)
    tq.norm_correction = True
    oracle_packing = load("v_packing")
    packed = oracle_packing.pack_v_codes(codes.to(torch.uint8), bits)
    allocation = load("allocator").greedy_bit_allocation(torch.ones(dim // 2), 15)
    safe = torch.where(norms > 0, norms, torch.ones_like(norms))
    normalized = x / safe[:, None]
    rotated = normalized @ tq.rotation_T
    boundaries = tq.boundaries
    probes = torch.stack((torch.nextafter(boundaries, torch.full_like(boundaries, -float("inf"))),
                          boundaries,
                          torch.nextafter(boundaries, torch.full_like(boundaries, float("inf")))))
    metadata = dict(
        upstream_commit=COMMIT, oracle="tq.TurboQuantMSE.quantize + dequantize",
        packing_oracle="v_packing.pack_v_codes", packing="uniform-lsb-byte-v1",
        coordinate_space="synthetic-rope-free", norm_dtype="f32", norm_correction=True,
        rows=len(x), dim=dim, bits=bits, row_bytes=packed.shape[-1],
        group_order="one uniform contiguous group; no mixed-K permutation",
        group_dimensions=[list(range(dim))], requested_budget=15,
        row_cases=["zero", "signed_integers", "fractions", "tiny_norm", "large_norm"],
        tolerance=dict(atol=0.0005, rtol=0.000002),
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         numpy=np.__version__, scipy=scipy.__version__, device="cpu", threads=1),
    )
    arrays = {
        "original": ("f32", x.numpy()), "scores": ("f32", np.ones(dim // 2)),
        "allocation": ("u32", allocation.numpy()),
        "pair_groups": ("u32", np.zeros(dim // 2)),
        "dimension_groups": ("u32", np.zeros(dim)),
        "rotation": ("f32", tq.rotation.numpy()), "centroids": ("f32", tq.centroids.numpy()),
        "boundaries": ("f32", boundaries.numpy()), "normalized": ("f32", normalized.numpy()),
        "rotated": ("f32", rotated.numpy()), "indices": ("u32", codes.numpy()),
        "norms": ("f32", norms.numpy()), "packed": ("u8", packed.numpy()),
        "reconstructed": ("f32", reconstruction.numpy()),
        "roundtrip": ("f32", full_roundtrip.numpy()), "uncorrected": ("f32", uncorrected.numpy()),
        "boundary_probes": ("f32", probes.numpy()),
        "boundary_indices": ("u32", torch.searchsorted(boundaries, probes).numpy()),
    }
    write_fixture(ROOT / "fixtures/tq3.json", metadata, arrays)
    # Packing oracles exercise every width and incomplete rows, independently
    # of the quantizer's naturally occurring codebook indices.
    packing_arrays = {}
    cases = []
    for b in (1, 2, 3, 4):
        for d in (1, 2, 3, 7, 8, 9, 10, 13):
            name = f"b{b}_d{d}"
            c = (torch.arange(2 * d).reshape(2, d) % (1 << b)).to(torch.uint8)
            p = oracle_packing.pack_v_codes(c, b)
            packing_arrays[name + "_codes"] = ("u8", c.numpy())
            packing_arrays[name + "_packed"] = ("u8", p.numpy())
            cases.append(dict(name=name, bits=b, dim=d))
    write_fixture(ROOT / "fixtures/packing.json",
                  dict(upstream_commit=COMMIT, oracle="v_packing.pack_v_codes", cases=cases), packing_arrays)
    print(f"Exported TQ fixture: {len(x)}×{dim}, {bits}-bit, {packed.shape[-1]} bytes/row")
    print(f"Exported {len(cases)} direct upstream packing cases")


if __name__ == "__main__":
    main()
