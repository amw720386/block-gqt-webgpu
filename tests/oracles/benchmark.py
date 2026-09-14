"""CPU wall-time diagnostics, not CUDA timing and not equivalent to GPU passes."""
import json
import platform
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

from tests.oracles.fixture import read_fixture
from tests.oracles.oracle import ROOT
from reference.production_math import encode, reconstruct
from reference.mixed_layout import pack


def measure(fn, warmup, repetitions):
    samples = []
    for rep in range(-warmup, repetitions):
        start = time.perf_counter_ns()
        result = fn()
        elapsed = (time.perf_counter_ns()-start)/1000
        if rep >= 0: samples.append(elapsed)
        del result
    return dict(samples=samples,statistics=dict(median=statistics.median(samples),mean=statistics.mean(samples),
                stddev=statistics.pstdev(samples),min=min(samples),max=max(samples)))


def main(path):
    torch.set_num_threads(1)
    source=json.loads(Path(path).read_text())
    records=[]
    for record in source["records"]:
        m,a=read_fixture(ROOT / f"fixtures/mixed/{record['fixture']}.json")
        t={k:torch.from_numpy(v) for k,v in a.items()}
        rows=record["workload"]["rows"]
        indices=torch.arange(rows)%m["rows"]
        x=t["original"][indices];codes=t["codes"][indices];norms=t["persistent_norms"][indices]
        def quantize():
            c,_,scale,_=encode(x,t)
            return pack(c.numpy(),m["nopack_start"],m["row_bytes"]),scale.half()
        def decode(): return reconstruct(codes,norms,t)
        timings=[]
        for name,fn in (("quantize",quantize),("decode",decode)):
            timings.append(dict(kernel=name,environment_kind="CPU PyTorch emulator wall time (non-equivalent)",
                unit="microseconds_per_call",warmup_count=10,repetition_count=30,
                measurement_boundary="perf_counter_ns around eager CPU operation including temporary tensor allocation; quantize includes NumPy packing; decode starts from unpacked codes",
                **measure(fn,10,30)))
        records.append(dict(fixture=record["fixture"],fixture_sha256=m["sha256"],upstream_commit=m["upstream_commit"],
            workload=record["workload"],
            memory=dict(explicit_input_tensor_bytes=x.numel()*x.element_size(),
                        code_tensor_bytes=codes.numel()*codes.element_size(),norm_tensor_bytes=norms.numel()*norms.element_size(),
                        note="Explicit tensor storage only; CPU allocator temporaries/peak not measured. Actual GPU allocation ledger is in companion GPU record."),
            timings=timings))
    report={k:v for k,v in source.items() if k!="records"}
    report.update(records=records,python_environment=dict(python=platform.python_version(),torch=torch.__version__,
                  numpy=np.__version__,threads=1),evidence_kind="CPU_reference_diagnostics_non_equivalent_to_GPU")
    target=Path(str(path).replace('-gpu.json','-cpu.json'))
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(f"Saved CPU reference diagnostics: {target.name}")


if __name__=="__main__": main(sys.argv[1])
