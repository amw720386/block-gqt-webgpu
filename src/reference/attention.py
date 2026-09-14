import math
import torch


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              positions: torch.Tensor, causal: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("Q/K/V must be contiguous conceptual [rows, dimension] matrices")
    if k.shape != v.shape or q.shape[1] != k.shape[1] or not k.shape[0] or not q.shape[0]:
        raise ValueError("Mismatched or empty attention shapes")
    if positions.shape != (q.shape[0],) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("One integer position per query is required")
    if bool(((positions < 0) | (positions >= len(k))).any()):
        raise ValueError("Query positions must identify an active context position")
    if not all(bool(torch.isfinite(x).all()) for x in (q, k, v)):
        raise ValueError("Nonfinite attention inputs are unsupported")
    logits = (q.float() @ k.float().T) / math.sqrt(k.shape[1])
    if causal:
        logits = logits.masked_fill(torch.arange(len(k))[None, :] > positions[:, None], -torch.inf)
    probabilities = torch.softmax(logits, dim=-1)
    output = probabilities @ v.float()
    return logits, probabilities, output
