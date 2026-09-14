"""Report reproduction error separately from intentional quantization error."""

import json
import platform

import numpy as np
import scipy
import torch

from tests.oracles.fixture import read_fixture
from tests.oracles.oracle import ROOT
from reference.packing import pack
from reference.tq_mse import quantize, reconstruct


def main():
    torch.set_num_threads(1)
    meta, data = read_fixture(ROOT / "fixtures/tq3.json")
    a = {k: torch.from_numpy(v) for k, v in data.items()}
    codes, norms = quantize(a["original"], a["rotation"], a["boundaries"])
    actual = reconstruct(codes, norms, a["rotation"], a["centroids"])
    report = dict(upstream_commit=meta["upstream_commit"], fixture_sha256=meta["sha256"],
                  max_abs_reference_error=(actual - a["reconstructed"]).abs().max().item(),
                  max_abs_quantization_error=(actual - a["original"]).abs().max().item(),
                  packed_bytes_exact=bool(np.array_equal(pack(codes.numpy(), meta["bits"]), data["packed"])),
                  indices_exact=bool(np.array_equal(codes.numpy(), data["indices"])),
                  norms_exact=bool(np.array_equal(norms.numpy(), data["norms"])),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   numpy=np.__version__, scipy=scipy.__version__, device="cpu", threads=1))
    if not (report["max_abs_reference_error"] == 0 and report["packed_bytes_exact"]
            and report["indices_exact"] and report["norms_exact"]):
        raise AssertionError(report)
    target = ROOT / "results/python.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
