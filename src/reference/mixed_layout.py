import numpy as np
import torch


def grouping(widths):
    w = torch.as_tensor(widths, dtype=torch.long)
    # Match upstream's *unstable* torch argsort, not a stable NumPy surrogate.
    order = w.argsort()
    perm = torch.stack((order, order + len(w)), dim=1).flatten()
    bits, counts = torch.unique(w[order], sorted=True, return_counts=True)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), (counts * 2).cumsum(0)))
    groups = torch.repeat_interleave(torch.arange(len(bits)), counts * 2)
    pair_groups = torch.empty(len(w), dtype=torch.long)
    pair_groups[order] = groups[::2]
    ns = int((w <= 4).sum()) * 2
    return dict(pair_order=order.numpy(), head_perm=perm.numpy(), group_bits=bits.numpy(),
                group_offsets=offsets.numpy(), group_of=groups.numpy(), pair_groups=pair_groups.numpy(),
                nopack_start=ns, row_bytes=ns // 2 + len(w) * 2 - ns)


def pack(codes, nopack_start, row_bytes):
    c = np.asarray(codes, dtype=np.uint8)
    out = np.zeros((*c.shape[:-1], row_bytes), dtype=np.uint8)
    ns = nopack_start
    out[..., :ns // 2] = (c[..., :ns:2] & 15) | ((c[..., 1:ns:2] & 15) << 4)
    out[..., ns // 2:ns // 2 + c.shape[-1] - ns] = c[..., ns:]
    return out


def unpack(data, dim, nopack_start):
    p = np.asarray(data, dtype=np.uint8)
    c = np.zeros((*p.shape[:-1], dim), dtype=np.uint8)
    ns = nopack_start
    c[..., :ns:2] = p[..., :ns // 2] & 15
    c[..., 1:ns:2] = p[..., :ns // 2] >> 4
    c[..., ns:] = p[..., ns // 2:ns // 2 + dim - ns]
    return c
