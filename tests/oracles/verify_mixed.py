"""Check saved GPU scales with PyTorch's FP16 conversion; report error components."""
import json
import numpy as np
import torch
from tests.oracles.fixture import read_fixture
from tests.oracles.oracle import ROOT, COMMIT
from reference.production_math import reconstruct


def main():
    torch.set_num_threads(1)
    gpu = json.loads((ROOT / "results/phase2-webgpu.json").read_text())
    assert gpu["passed"]
    reports = []
    for c in gpu["cases"]:
        m, a = read_fixture(ROOT / f"fixtures/mixed/{c['name']}.json")
        assert c["fixture_sha256"] == m["sha256"]
        scales = torch.tensor(c["gpu_corrected_scales"], dtype=torch.float32).reshape(m["rows"], m["norm_stride"])
        expected_bits = scales.half().view(torch.uint16).numpy().flatten()
        np.testing.assert_array_equal(expected_bits, c["gpu_norm_bits"])
        t = {k: torch.from_numpy(v) for k, v in a.items()}
        gpu_scale_ref = reconstruct(t["codes"], scales.half(), t).numpy()
        actual = np.array(c["roundtrip_actual"], dtype=np.float32).reshape(gpu_scale_ref.shape)
        err = float(np.max(np.abs(actual - gpu_scale_ref)))
        np.testing.assert_allclose(actual, gpu_scale_ref, atol=m["tolerance"]["atol"], rtol=m["tolerance"]["rtol"])
        reports.append(dict(name=c["name"], gpu_scale_half_bits_match_pytorch=True,
                            decoder_error_with_identical_gpu_scales=err,
                            port_max_abs=c["roundtrip"]["max_abs_error"],
                            port_relative_l2=c["roundtrip"]["relative_l2_error"],
                            norm_bit_mismatches_vs_cpu_encoder=c["norm_bit_mismatches"],
                            fp16_conversion_max_abs=m["norm_conversion_max_abs"],
                            quantization_max_abs=m["quantization_max_abs"]))
    result=dict(passed=True,upstream_commit=COMMIT,cuda_encoder_validated=False,cases=reports)
    (ROOT / "results/phase2-python.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
