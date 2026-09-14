import numpy as np


def allocate_bits(scores, budget: int, min_bits: int = 1, max_bits: int = 8):
    scores = np.asarray(scores, dtype=np.float32)
    widths = np.full(len(scores), min_bits, dtype=np.int64)
    remaining = budget - int(widths.sum())
    while remaining > 0:
        # Upstream casts scores to FP32 but evaluates powers in FP64.
        priorities = scores * np.power(4.0, -widths.astype(np.float64))
        priorities[widths >= max_bits] = -1.0
        index = int(np.argmax(priorities))  # First index wins ties.
        if priorities[index] <= 0:
            break
        widths[index] += 1
        remaining -= 1
    return widths

