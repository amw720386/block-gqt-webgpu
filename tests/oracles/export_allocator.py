"""Generate allocation goldens by calling the pinned upstream function."""

import json

import torch

from tests.oracles.oracle import COMMIT, ROOT, load

CASES = [
    ("unequal", [1, 4, 16, 64], 12, 1, 8),
    ("equal_tie", [1, 1, 1, 1], 6, 1, 8),
    ("priority_tie", [4, 1, 1], 6, 1, 8),
    ("all_zero", [0, 0, 0], 15, 1, 8),
    ("mixed_zero", [0, 1, 0, 4], 20, 1, 4),
    ("below_minimum", [1, 2, 3], 0, 2, 4),
    ("at_minimum", [1, 2, 3], 6, 2, 4),
    ("above_maximum", [1, 2, 3], 99, 1, 3),
    ("fixed_bounds", [3, 2, 1], 99, 2, 2),
    ("float32_tie", [1.00000001, 1.00000002, 1], 4, 1, 8),
    ("single_pair", [2], 3, 1, 8),
    ("empty_zero_budget", [], 0, 1, 8),
    ("uniform_fixture", [1, 1, 1, 1, 1], 15, 1, 8),
]


def main():
    oracle = load("allocator").greedy_bit_allocation
    cases = []
    for name, scores, budget, low, high in CASES:
        result = oracle(torch.tensor(scores, dtype=torch.float64), budget, low, high)
        cases.append(dict(name=name, scores=scores, budget=budget,
                          min_bits=low, max_bits=high, expected=result.tolist()))
    target = ROOT / "fixtures" / "allocator.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(dict(upstream_commit=COMMIT, cases=cases), indent=2) + "\n")
    print(f"Exported {len(cases)} upstream allocation cases")


if __name__ == "__main__":
    main()

