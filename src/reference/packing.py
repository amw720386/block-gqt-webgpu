import numpy as np


def row_bytes(dim, bits):
    if bits not in (1, 2, 3, 4):
        raise ValueError("Uniform packing supports 1–4 bits")
    return ((dim + 7) // 8) * 3 if bits == 3 else (dim * bits + 7) // 8


def pack(codes, bits):
    values = np.asarray(codes)
    dim = values.shape[-1]
    rows = values.reshape(-1, dim)
    out = np.zeros((len(rows), row_bytes(dim, bits)), dtype=np.uint8)
    mask = (1 << bits) - 1
    for r, row in enumerate(rows):
        for d, code in enumerate(row):
            bit = d * bits
            value = (int(code) & mask) << (bit % 8)
            out[r, bit // 8] |= value & 255
            if bit % 8 + bits > 8:
                out[r, bit // 8 + 1] |= value >> 8
    return out.reshape(*values.shape[:-1], out.shape[-1])


def unpack(packed, dim, bits):
    values = np.asarray(packed, dtype=np.uint8)
    if values.shape[-1] != row_bytes(dim, bits):
        raise ValueError("Packed row stride mismatch")
    rows = values.reshape(-1, values.shape[-1])
    out = np.empty((len(rows), dim), dtype=np.uint8)
    for r, row in enumerate(rows):
        for d in range(dim):
            bit = d * bits
            word = int(row[bit // 8])
            if bit % 8 + bits > 8:
                word |= int(row[bit // 8 + 1]) << 8
            out[r, d] = (word >> (bit % 8)) & ((1 << bits) - 1)
    return out.reshape(*values.shape[:-1], dim)

