import torch


def encode(x, a):
    dim, ng = a["rotation"].shape[0], len(a["group_bits"])
    xp = x.float()[:, a["head_perm"].long()]
    group = a["group_of"].long()
    raw = torch.zeros(len(x), ng)
    normalized = torch.zeros_like(xp)
    for g in range(ng):
        mask = group == g
        n = ((xp * xp) * mask).sum(1).add(1e-30).sqrt()
        # The multi-group Triton path averages repeated per-position norms.
        raw[:, g] = (n[:, None] * mask).sum(1) / mask.sum() if ng > 1 else n
        normalized[:, mask] = xp[:, mask] / torch.where(n > 1e-10, n, 1.0)[:, None]
    y = normalized @ a["rotation"].T.contiguous()
    bins = ((y - a["lut_offsets"]) * a["lut_inv_scales"]).clamp(0, 255).long()
    codes_hp = a["code_lut"][a["pos_to_cb"].long()[None, :], bins].to(torch.uint8)
    codes = codes_hp[:, a["pack_perm"].long()]
    cents = a["centroids"][torch.arange(dim)[None, :], codes_hp.long()]
    corrected = raw.clone()
    for g in range(ng):
        corrected[:, g] /= ((cents * cents) * (group == g)).sum(1).sqrt().clamp(min=1e-10)
    return codes, raw, corrected, y


def reconstruct(codes, scales, a):
    hp = codes[:, a["inv_pack_perm"].long()].long()
    c = a["centroids"][torch.arange(hp.shape[1])[None, :], hp]
    y = c * scales.float()[:, a["group_of"].long()]
    return (y @ a["rotation"])[:, a["inv_head_perm"].long()]
