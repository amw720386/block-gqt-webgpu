"""Deterministic synthetic attention fixtures. No CUDA or model-quality oracle.

Committed tests use lengths 128/256/512. Larger 1024/2048/4096 workloads are
generated locally for benchmarks: python -m tests.oracles.export_attention --bench
"""
import argparse
import json
import torch
from reference.attention import attention
from reference.production_math import encode, reconstruct
from reference.mixed_layout import pack
from tests.oracles.fixture import read_fixture, write_fixture
from tests.oracles.oracle import ROOT, COMMIT

TEST_LENGTHS = (128, 256, 512)
SCALE_LENGTHS = (1024, 2048, 4096)
DIMS = (64, 128)


def export_one(d, length, layout, arrays_in):
    generator = torch.Generator().manual_seed(731 + d + length)
    q = torch.randn(3, d, generator=generator)
    k_source = torch.randn(length, d, generator=generator) * 0.7
    k = k_source.half()
    v = torch.randn(length, d, generator=generator) * 0.5
    positions = torch.tensor([0, length // 2, length - 1])
    codes, _, scales, _ = encode(k.float(), arrays_in)
    persistent = torch.zeros(length, layout["norm_stride"], dtype=torch.float16)
    persistent[:, :layout["n_groups"]] = scales.half()
    compressed = reconstruct(codes, persistent, arrays_in)
    packed = pack(codes.numpy(), layout["nopack_start"], layout["row_bytes"])
    arrays = {"q": ("f32", q.numpy()), "k": ("f16", k.numpy()), "k_source": ("f32", k_source.numpy()),
              "v": ("f32", v.numpy()), "positions": ("u32", positions.numpy()),
              "packed_k": ("u8", packed), "norms": ("f16", persistent.numpy()),
              "reconstructed_k": ("f32", compressed.numpy())}
    for causal in (False, True):
        prefix = "causal_" if causal else ""
        for label, keys in (("baseline", k), ("compressed", compressed), ("fp32_source", k_source)):
            result = attention(q, keys, v, positions, causal)
            for kind, value in zip(("logits", "probabilities", "output"), result):
                arrays[prefix + label + "_" + kind] = ("f32", value.numpy())
    name = f"t{length}_d{d}"
    metadata = dict(version=2, name=name, upstream_commit=COMMIT, layout=f"mixed_d{d}", layout_sha256=layout["sha256"],
                    oracle="CPU PyTorch attention; production compression arithmetic emulator",
                    cuda_encoder_validated=False, coordinate_space="synthetic-rope-free",
                    context_length=length, dim=d, queries=3, k_storage="f16", q_storage="f32", v_storage="f32",
                    accumulation="f32", tolerances={
                        "baseline": {"logits": [0.0001, 0.00002], "probabilities": [0.00001, 0.0001], "output": [0.0001, 0.0001]},
                        "port": {"logits": [0.005, 0.002], "probabilities": [0.0005, 0.002], "output": [0.001, 0.002]}},
                    environment={"torch": torch.__version__, "device": "cpu", "threads": 1})
    write_fixture(ROOT / f"fixtures/attention/{name}.json", metadata, arrays)
    return {"name": name, "dim": d, "context_length": length}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true", help="also emit 1024/2048/4096 workloads for benchmarks")
    parser.add_argument("--force", action="store_true", help="rewrite existing files")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    lengths = TEST_LENGTHS + (SCALE_LENGTHS if args.bench else ())
    test_cases, bench_cases = [], []
    for d in DIMS:
        layout, data = read_fixture(ROOT / f"fixtures/mixed/mixed_d{d}.json")
        arrays_in = {k: torch.from_numpy(v) for k, v in data.items()}
        for length in lengths:
            path = ROOT / f"fixtures/attention/t{length}_d{d}.json"
            if path.exists() and not args.force:
                case = {"name": f"t{length}_d{d}", "dim": d, "context_length": length}
            else:
                case = export_one(d, length, layout, arrays_in)
            bench_cases.append(case)
            if length in TEST_LENGTHS:
                test_cases.append(case)
    (ROOT / "fixtures/attention/index.json").write_text(json.dumps({"version": 2, "cases": test_cases}, indent=2) + "\n")
    if args.bench:
        (ROOT / "fixtures/attention/bench-index.json").write_text(json.dumps({"version": 2, "cases": bench_cases}, indent=2) + "\n")
    print(f"Attention fixtures ready: {len(test_cases)} committed-test workloads"
          + (f", {len(bench_cases)} including scale benches" if args.bench else "") + ".")


if __name__ == "__main__":
    main()
