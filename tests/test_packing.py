import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.oracles.fixture import read_fixture, write_fixture
from tests.oracles.oracle import ROOT, load
from reference.packing import pack, unpack


class PackingTests(unittest.TestCase):
    def test_all_widths_and_remainders_against_goldens_and_oracle(self):
        meta, arrays = read_fixture(ROOT / "fixtures/packing.json")
        oracle = load("v_packing")
        for case in meta["cases"]:
            with self.subTest(case=case["name"]):
                codes = arrays[case["name"] + "_codes"]
                expected = arrays[case["name"] + "_packed"]
                np.testing.assert_array_equal(pack(codes, case["bits"]), expected)
                np.testing.assert_array_equal(
                    oracle.pack_v_codes(torch.from_numpy(codes), case["bits"]).numpy(), expected)
                np.testing.assert_array_equal(unpack(expected, case["dim"], case["bits"]), codes)

    def test_tq_packed_bytes_and_codes(self):
        meta, a = read_fixture(ROOT / "fixtures/tq3.json")
        self.assertEqual(a["packed"].shape, (5, 6))
        np.testing.assert_array_equal(pack(a["indices"], meta["bits"]), a["packed"])
        np.testing.assert_array_equal(unpack(a["packed"], 10, 3), a["indices"])
        # Ten codes occupy 30 meaningful bits but upstream reserves 48 bits.
        np.testing.assert_array_equal(a["packed"][:, 4:], np.zeros((5, 2), dtype=np.uint8))
        self.assertTrue(np.all((a["packed"][:, 3] & 0xC0) == 0))

    def test_serialization_roundtrip_and_corruption_detection(self):
        _, arrays = read_fixture(ROOT / "fixtures/tq3.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            write_fixture(path, {}, {"x": ("f32", arrays["original"]), "packed": ("u8", arrays["packed"])})
            _, actual = read_fixture(path)
            np.testing.assert_array_equal(actual["x"], arrays["original"])
            np.testing.assert_array_equal(actual["packed"], arrays["packed"])
            payload = bytearray(path.with_suffix(".bin").read_bytes())
            payload[0] ^= 1
            path.with_suffix(".bin").write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_fixture(path)

