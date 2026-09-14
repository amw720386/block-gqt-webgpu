import torch


def quantize(x, rotation, boundaries):
    single = x.ndim == 1
    rows = x.unsqueeze(0) if single else x
    norms = torch.norm(rows, dim=1)
    safe = torch.where(norms > 0, norms, torch.ones_like(norms))
    rotated = (rows / safe.unsqueeze(1)) @ rotation.T.contiguous()
    codes = torch.searchsorted(boundaries, rotated.contiguous())
    return (codes[0], norms[0]) if single else (codes, norms)


def reconstruct(codes, norms, rotation, centroids, correction=True):
    single = codes.ndim == 1
    rows = codes.unsqueeze(0) if single else codes
    scales = norms.unsqueeze(0) if single else norms
    decoded = centroids[rows.long()]
    if correction:
        lengths = torch.norm(decoded, dim=1, keepdim=True)
        lengths = torch.where(lengths > 1e-10, lengths, torch.ones_like(lengths))
        decoded = decoded / lengths
    result = (decoded @ rotation) * scales.unsqueeze(1)
    return result[0] if single else result

