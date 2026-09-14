import json
import unittest

import numpy as np
import torch

from reference.allocator import allocate_bits
from tests.oracles.oracle import COMMIT, ROOT, load


class AllocatorTests(unittest.TestCase):
    def test_golden_cases_and_direct_upstream(self):
        fixture = json.loads((ROOT / "fixtures/allocator.json").read_text())
        self.assertEqual(fixture["upstream_commit"], COMMIT)
        upstream = load("allocator").greedy_bit_allocation
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                args = (case["budget"], case["min_bits"], case["max_bits"])
                expected = np.array(case["expected"], dtype=np.int64)
                np.testing.assert_array_equal(allocate_bits(case["scores"], *args), expected)
                np.testing.assert_array_equal(
                    upstream(torch.tensor(case["scores"], dtype=torch.float64), *args).numpy(), expected)

    def test_exhaustive_small_score_budget_grid(self):
        import itertools
        upstream = load("allocator").greedy_bit_allocation
        for scores in itertools.product((0, 1, 4), repeat=3):
            for budget in range(0, 14):
                with self.subTest(scores=scores, budget=budget):
                    expected = upstream(torch.tensor(scores), budget, 1, 4).numpy()
                    np.testing.assert_array_equal(allocate_bits(scores, budget, 1, 4), expected)


if __name__ == "__main__":
    unittest.main()
