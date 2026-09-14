import json
import unittest

import numpy as np
import torch

from tests.oracles.fixture import read_fixture
from tests.oracles.oracle import ROOT, load
from reference.mixed_layout import grouping, pack, unpack
from tests.oracles import production_oracle as oracle
from reference.production_math import encode, reconstruct


class MixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cases = json.loads((ROOT / "fixtures/mixed/index.json").read_text())["cases"]

    def fixtures(self):
        for c in self.cases:
            m, a = read_fixture(ROOT / f"fixtures/mixed/{c['name']}.json")
            yield m, a, {k: torch.from_numpy(v) for k, v in a.items()}

    def test_grouping_permutation_and_allocator(self):
        for m, a, t in self.fixtures():
            with self.subTest(m["name"]):
                out = grouping(a["allocation"])
                for key in ("pair_order", "head_perm", "group_bits", "group_offsets", "group_of", "pair_groups"):
                    np.testing.assert_array_equal(out[key], a[key])
                expected = load("allocator").greedy_bit_allocation(t["scores"], m["requested_budget"])
                np.testing.assert_array_equal(expected, a["allocation"])
                pm = oracle.pack_meta(t["allocation"].long(), t["head_perm"].long())
                np.testing.assert_array_equal(pm["pack_perm"], a["pack_perm"])
                self.assertEqual(pm["nopack_start"], m["nopack_start"])
                np.testing.assert_array_equal(a["head_perm"][a["inv_head_perm"]], np.arange(m["dim"]))

    def test_exact_production_bytes_all_widths_and_padding(self):
        for m, a, t in self.fixtures():
            with self.subTest(m["name"]):
                np.testing.assert_array_equal(pack(a["codes"], m["nopack_start"], m["row_bytes"]), a["packed"])
                np.testing.assert_array_equal(unpack(a["packed"], m["dim"], m["nopack_start"]), a["codes"])
                args = dict(mixed_nopack_starts=[m["nopack_start"]],
                            mixed_nopack_lens=[m["dim"]-m["nopack_start"]], max_mixed_bytes=m["row_bytes"])
                packed, _ = oracle.host_pack(t["codes"][None], t["raw_norms"][None], args)
                np.testing.assert_array_equal(packed[0], a["packed"])
                self.assertTrue(np.all(a["packed"][:, m["logical_row_bytes"]:] == 0))
                if m["name"] == "awkward_all_widths":
                    self.assertTrue(np.any(a["codes"][:, m["nopack_start"]:] > 15))

    def test_encoder_emulator_and_corrected_norms(self):
        for m, a, t in self.fixtures():
            with self.subTest(m["name"]):
                codes, raw, corr, _ = encode(t["original"], t)
                np.testing.assert_array_equal(codes, a["codes"])
                np.testing.assert_array_equal(raw, a["raw_norms"][:, :m["n_groups"]])
                np.testing.assert_array_equal(corr, a["corrected_f32"][:, :m["n_groups"]])
                ng = m["norm_stride"]
                args = dict(mixed_nopack_starts=[m["nopack_start"]], mixed_nopack_lens=[m["dim"]-m["nopack_start"]],
                            max_mixed_bytes=m["row_bytes"], max_n_groups=ng,
                            k_pos_cents_stack=t["centroids"][None], k_inv_pp_stack=t["inv_pack_perm"].long()[None],
                            k_group_mask_stack=torch.stack([(t["group_of"] == g).float() for g in range(ng)])[None])
                _, host_norms = oracle.host_pack(t["codes"][None], t["raw_norms"][None], args)
                np.testing.assert_array_equal(host_norms[0], a["corrected_f32"])

    def test_fp16_norm_bits_and_separate_conversion_error(self):
        for m, a, t in self.fixtures():
            with self.subTest(m["name"]):
                np.testing.assert_array_equal(t["corrected_f32"].half().view(torch.uint16), a["norm_bits"])
                for dtype in ("f32", "f16"):
                    scales = t["corrected_f32"] if dtype == "f32" else t["persistent_norms"]
                    np.testing.assert_array_equal(reconstruct(t["codes"], scales, t), a[f"reconstructed_{dtype}"])
                self.assertEqual(float(np.max(np.abs(a["reconstructed_f16"]-a["reconstructed_f32"]))), m["norm_conversion_max_abs"])
                self.assertFalse(m["cuda_encoder_validated"])

    def test_half_conversion_edges(self):
        values = np.array([0, -0.0, 2**-24, 2**-25, 3*2**-25, 1+2**-11, 65504, 65520], dtype=np.float32)
        with np.errstate(over="ignore"):
            np.testing.assert_array_equal(torch.from_numpy(values).half().view(torch.uint16), values.astype(np.float16).view(np.uint16))

    def test_reserved_cache_shape_dtype_and_zero_capacity(self):
        for m, a, _ in self.fixtures():
            cache = oracle.cache(2, 2, m["dim"], m["capacity"], [m["logical_row_bytes"], m["row_bytes"]],
                                 [[m["n_groups"], 1], [m["norm_stride"], 1]])
            self.assertEqual(tuple(cache.k_packed.shape), (2, 1, 2, m["capacity"], m["row_bytes"]))
            self.assertEqual(cache.k_norms.dtype, torch.float16)
            self.assertEqual(cache.k_packed.dtype, torch.uint8)
            self.assertEqual(torch.count_nonzero(cache.k_packed).item(), 0)
            self.assertTrue(np.all(a["cache_packed"][m["rows"]:] == 0))
            self.assertTrue(np.all(a["cache_norms"][m["rows"]:] == 0))
