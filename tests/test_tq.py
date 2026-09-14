import unittest

import numpy as np
import torch

from tests.oracles.fixture import read_fixture
from tests.oracles.oracle import COMMIT, ROOT, load
from reference.tq_mse import quantize, reconstruct


class TQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.meta, arrays = read_fixture(ROOT / "fixtures/tq3.json")
        cls.a = {name: torch.from_numpy(value) for name, value in arrays.items()}

    def oracle(self):
        # Load actual exported parameters into the unmodified upstream class;
        # no regeneration, RNG seed comparison, or local implementation calls.
        cls = load("tq").TurboQuantMSE
        tq = cls.__new__(cls)
        tq.rotation = self.a["rotation"]
        tq.rotation_T = tq.rotation.T.contiguous()
        tq.centroids = self.a["centroids"]
        tq.boundaries = self.a["boundaries"]
        tq.norm_correction = True
        return tq

    def test_exported_parameters_and_direct_oracle(self):
        self.assertEqual(self.meta["upstream_commit"], COMMIT)
        self.assertEqual(self.meta["coordinate_space"], "synthetic-rope-free")
        tq = self.oracle()
        codes, norms = tq.quantize(self.a["original"])
        torch.testing.assert_close(codes, self.a["indices"].long(), atol=0, rtol=0)
        torch.testing.assert_close(norms, self.a["norms"], atol=0, rtol=0)
        torch.testing.assert_close(tq.dequantize(codes, norms), self.a["reconstructed"], atol=0, rtol=0)

    def test_local_quantize_exact_codes_and_norms(self):
        codes, norms = quantize(self.a["original"], self.a["rotation"], self.a["boundaries"])
        torch.testing.assert_close(codes, self.a["indices"].long(), atol=0, rtol=0)
        torch.testing.assert_close(norms, self.a["norms"], atol=0, rtol=0)

    def test_local_reconstruct_and_roundtrip(self):
        a = self.a
        result = reconstruct(a["indices"], a["norms"], a["rotation"], a["centroids"])
        torch.testing.assert_close(result, a["reconstructed"], atol=0, rtol=0)
        torch.testing.assert_close(result, a["roundtrip"], atol=0, rtol=0)

    def test_single_vector_and_zero_norm(self):
        a = self.a
        for i in range(len(a["original"])):
            with self.subTest(row=i):
                codes, norm = quantize(a["original"][i], a["rotation"], a["boundaries"])
                expected_codes, expected_norm = self.oracle().quantize(a["original"][i])
                torch.testing.assert_close(codes, expected_codes, atol=0, rtol=0)
                torch.testing.assert_close(norm, expected_norm, atol=0, rtol=0)
                result = reconstruct(codes, norm, a["rotation"], a["centroids"])
                expected = self.oracle().dequantize(expected_codes, expected_norm)
                torch.testing.assert_close(result, expected, atol=0, rtol=0)
        self.assertEqual(a["norms"][0].item(), 0)
        self.assertTrue(torch.equal(a["reconstructed"][0], torch.zeros(10)))

    def test_exact_boundaries_and_adjacent_floats(self):
        a = self.a
        actual = torch.searchsorted(a["boundaries"], a["boundary_probes"])
        torch.testing.assert_close(actual, a["boundary_indices"].long(), atol=0, rtol=0)
        torch.testing.assert_close(actual[1], torch.arange(7), atol=0, rtol=0)
        torch.testing.assert_close(actual[2], torch.arange(1, 8), atol=0, rtol=0)

    def test_correction_disabled_matches_oracle(self):
        a = self.a
        result = reconstruct(a["indices"], a["norms"], a["rotation"], a["centroids"], False)
        tq = self.oracle()
        tq.norm_correction = False
        expected = tq.dequantize(a["indices"].long(), a["norms"])
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(result, a["uncorrected"], atol=0, rtol=0)

    def test_correction_denominator_zero_and_small(self):
        a = self.a
        for value in (0, 1e-12, 1e-8):
            with self.subTest(centroid=value):
                tq = self.oracle()
                tq.centroids = torch.full_like(a["centroids"], value)
                expected = tq.dequantize(a["indices"].long(), a["norms"])
                actual = reconstruct(a["indices"], a["norms"], a["rotation"], tq.centroids)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_uniform_group_and_allocation_metadata(self):
        a = self.a
        np.testing.assert_array_equal(a["allocation"].numpy(), [3] * 5)
        np.testing.assert_array_equal(a["pair_groups"].numpy(), [0] * 5)
        np.testing.assert_array_equal(a["dimension_groups"].numpy(), [0] * 10)
        self.assertEqual(self.meta["group_dimensions"], [list(range(10))])
