"""Execute hash-checked upstream CPU definitions and the actual host packing tail.

No CUDA kernel is replaced silently: encode arithmetic lives in production_math.py
and is explicitly an emulator. AST selection avoids importing Triton-only modules.
Selected function/class bodies are compiled unchanged; host_tail omits only the
CUDA launch/scratch extraction and accepts those codes and raw norms as inputs.
"""
import ast
import hashlib
import json
import math
from typing import Optional

import numpy as np
import torch

from tests.oracles.oracle import VENDOR, COMMIT, load


def source(name):
    p = VENDOR / f"{name}.py"
    manifest = json.loads((VENDOR / "provenance.json").read_text())
    assert manifest["commit"] == COMMIT
    assert hashlib.sha256(p.read_bytes()).hexdigest() == manifest["sha256"][p.name]
    return ast.parse(p.read_text(encoding="utf-8"))


def definitions(name, names):
    nodes = [n for n in source(name).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    env = dict(torch=torch, np=np, math=math, Optional=Optional,
               greedy_bit_allocation=load("allocator").greedy_bit_allocation)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(VENDOR / f"{name}.py"), "exec"), env)
    return env


def setup(widths):
    names = ["_optimal_centroids_torch", "_random_rotation_dense_torch", "BlockGTQPipeline"]
    cls = definitions("block_gtq_pipeline", names)["BlockGTQPipeline"]
    p = cls.__new__(cls)
    p.head_dim, p.n_freqs = 2 * len(widths), len(widths)
    p.device, p.rotation_threshold, p._perm_baked = torch.device("cpu"), 2, False
    p.bit_allocation = torch.as_tensor(widths, dtype=torch.long)
    p._post_allocation_setup()
    return p


def pack_meta(widths, permutation):
    return definitions("quantizer", ["build_pack_meta"])["build_pack_meta"](widths, permutation)


def code_lut(p):
    return definitions("quant_kernels", ["build_code_lut"])["build_code_lut"](
        p._pos_centroids, p._pos_boundaries, n_bins=256)


def host_pack(codes, raw_norms, args):
    node = next(n for n in source("quantizer").body if isinstance(n, ast.FunctionDef) and n.name == "compress_k_mixed_batched")
    start = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "nopack_starts" for t in n.targets))
    wrapper = ast.parse("def host_tail(codes, norms, args):\n n_kv, N, hd = codes.shape\n device = codes.device\n norm_correct = True\n").body[0]
    wrapper.body += node.body[start:]
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    env = {"torch": torch}
    exec(compile(module, str(VENDOR / "quantizer.py"), "exec"), env)
    return env["host_tail"](codes, raw_norms.clone(), args)


def cache(layers, heads, dim, capacity, row_bytes, groups):
    cls = load("production_cache").BlockGTQProductionCache
    from types import SimpleNamespace
    quantizers = [[SimpleNamespace(_n_groups=g) for g in layer] for layer in groups]
    args = [{"max_mixed_bytes": b} for b in row_bytes]
    return cls(quantizers, args, None, layers, heads, heads, dim, capacity, device="cpu")
