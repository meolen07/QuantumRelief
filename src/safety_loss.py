"""
Safety-aware auxiliary loss for next-hop routing.

Dijkstra remains the primary teacher (cross-entropy on the oracle next hop).
The safety term softly boosts probability mass on neighbors farther from the
epicenter / on lower-hazard edges so policies avoid diving into r_epi when
travel-time ties leave room.

Table I layout (size 36):
  [x_epi, y_epi, x_start, y_start, x_dest, y_dest,
   x_e1, y_e1, w1, e1, d1, c1, … ×5 zero-padded]

Per neighbor slot i:
  dist_i  = ||(x_ei, y_ei) − (x_epi, y_epi)||
  hazard_i = w_i   (dynamic travel weight; higher near epicenter)
  s_i     = normalize(dist_i) − γ · normalize(log(1 + w_i))   among valid slots

Loss:
  L = L_CE(y_Dijkstra) + λ_safe · L_safe
  L_safe = −∑_i p_i · s_i     (p = softmax(logits))

Optional soft-target mix (``mode="kl"``):
  t ∝ (1−α)·one_hot(y) + α·softmax(s / T)
  L_safe = CE(logits, t)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .utils import INPUT_DIM, MAX_DEGREE, N_OUTPUTS

# Defaults tuned so Dijkstra stays dominant; safety is a soft preference.
DEFAULT_LAMBDA_SAFE = 0.35
DEFAULT_HAZARD_GAMMA = 0.35
DEFAULT_SOFT_MIX = 0.25
DEFAULT_SOFT_TEMP = 0.5


def _as_feature_tensors(
    x: torch.Tensor,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> torch.Tensor:
    """Return (B, 36) features in approx. raw km / weight units when possible."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.size(-1) != INPUT_DIM:
        raise ValueError(f"Expected feature dim {INPUT_DIM}, got {tuple(x.shape)}")
    if mean is None or std is None:
        return x
    mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1)
    std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1)
    std_t = torch.clamp(std_t, min=1e-6)
    return x * std_t + mean_t


def neighbor_safety_scores(
    x: torch.Tensor,
    *,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std: Optional[Union[torch.Tensor, np.ndarray]] = None,
    hazard_gamma: float = DEFAULT_HAZARD_GAMMA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-neighbor safety scores and validity mask.

    Returns
    -------
    scores : (B, 5)  higher = safer (farther from epi / lower edge hazard)
    valid  : (B, 5)  True for real neighbor slots (not Table-I zero-pad)
    """
    raw = _as_feature_tensors(x, mean=mean, std=std)
    epi = raw[:, 0:2]
    dist_list = []
    haz_list = []
    valid_list = []
    for i in range(MAX_DEGREE):
        base = 6 + i * 6
        nb = raw[:, base : base + 2]
        w = raw[:, base + 2].clamp_min(0.0)
        # Raw pad is all-zeros; after denorm this is ~0. Without mean/std,
        # fall back to "near-zero weight + near-zero coords" on the tensor as-is.
        valid = (nb.abs().sum(dim=-1) + w.abs()) > 1e-5
        dist = torch.sqrt(((nb - epi) ** 2).sum(dim=-1) + 1e-8)
        dist_list.append(dist)
        haz_list.append(torch.log1p(w))
        valid_list.append(valid)

    dist = torch.stack(dist_list, dim=1)  # (B, 5)
    haz = torch.stack(haz_list, dim=1)
    valid = torch.stack(valid_list, dim=1)

    # Min-max normalize among valid neighbors per row (invalid → 0).
    pos_inf = torch.tensor(float("inf"), device=dist.device, dtype=dist.dtype)
    neg_inf = torch.tensor(float("-inf"), device=dist.device, dtype=dist.dtype)
    d_min = dist.masked_fill(~valid, pos_inf).min(dim=1, keepdim=True).values
    d_max = dist.masked_fill(~valid, neg_inf).max(dim=1, keepdim=True).values
    h_min = haz.masked_fill(~valid, pos_inf).min(dim=1, keepdim=True).values
    h_max = haz.masked_fill(~valid, neg_inf).max(dim=1, keepdim=True).values

    d_span = (d_max - d_min).clamp_min(1e-6)
    h_span = (h_max - h_min).clamp_min(1e-6)
    dist_n = ((dist - d_min) / d_span).clamp(0.0, 1.0)
    haz_n = ((haz - h_min) / h_span).clamp(0.0, 1.0)
    # Single-valid-neighbor rows: span≈0 → treat as neutral 0.5
    only_one = valid.sum(dim=1, keepdim=True) <= 1
    dist_n = torch.where(only_one & valid, torch.full_like(dist_n, 0.5), dist_n)
    haz_n = torch.where(only_one & valid, torch.full_like(haz_n, 0.5), haz_n)

    scores = dist_n - float(hazard_gamma) * haz_n
    scores = scores.masked_fill(~valid, 0.0)
    return scores, valid


def safety_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    *,
    y: Optional[torch.Tensor] = None,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std: Optional[Union[torch.Tensor, np.ndarray]] = None,
    hazard_gamma: float = DEFAULT_HAZARD_GAMMA,
    mode: str = "expect",
    soft_mix: float = DEFAULT_SOFT_MIX,
    soft_temp: float = DEFAULT_SOFT_TEMP,
) -> torch.Tensor:
    """
    Auxiliary safety term (scalar).

    mode="expect":  L_safe = -∑ p_i s_i
    mode="kl":      L_safe = CE(logits, (1-α) one_hot(y) + α softmax(s/T))
    """
    if logits.size(-1) != N_OUTPUTS:
        raise ValueError(f"Expected {N_OUTPUTS} logits, got {tuple(logits.shape)}")
    scores, valid = neighbor_safety_scores(
        x, mean=mean, std=std, hazard_gamma=hazard_gamma
    )
    # Mask padded logits so softmax mass stays on real neighbors.
    neg_large = torch.finfo(logits.dtype).min / 4
    masked_logits = logits.masked_fill(~valid, neg_large)

    if mode == "kl":
        if y is None:
            raise ValueError("mode='kl' requires Dijkstra labels y")
        alpha = float(np.clip(soft_mix, 0.0, 1.0))
        temp = max(float(soft_temp), 1e-3)
        safe_logits = (scores / temp).masked_fill(~valid, neg_large)
        safe_p = F.softmax(safe_logits, dim=-1)
        y_oh = F.one_hot(y.long(), num_classes=N_OUTPUTS).to(dtype=logits.dtype)
        # Zero pad mass on one-hot already (label is always a real neighbor).
        target = (1.0 - alpha) * y_oh + alpha * safe_p
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        log_p = F.log_softmax(masked_logits, dim=-1)
        return -(target * log_p).sum(dim=-1).mean()

    # Default: expected safety reward (maximize ∑ p·s ⇒ minimize −∑ p·s)
    p = F.softmax(masked_logits, dim=-1)
    return -(p * scores).sum(dim=-1).mean()


def total_routing_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    *,
    lambda_safe: float = DEFAULT_LAMBDA_SAFE,
    mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
    std: Optional[Union[torch.Tensor, np.ndarray]] = None,
    hazard_gamma: float = DEFAULT_HAZARD_GAMMA,
    mode: str = "expect",
    soft_mix: float = DEFAULT_SOFT_MIX,
    soft_temp: float = DEFAULT_SOFT_TEMP,
    sample_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L = L_CE + λ_safe · L_safe.

    Returns (total, ce, safe) for logging.
    """
    if sample_weight is None:
        ce = F.cross_entropy(logits, y)
    else:
        ce_vec = F.cross_entropy(logits, y, reduction="none")
        w = sample_weight.to(dtype=ce_vec.dtype)
        ce = (ce_vec * w).sum() / w.sum().clamp_min(1e-6)

    lam = float(lambda_safe)
    if lam <= 0.0:
        zero = ce.detach() * 0.0
        return ce, ce, zero

    safe = safety_loss(
        logits,
        x,
        y=y,
        mean=mean,
        std=std,
        hazard_gamma=hazard_gamma,
        mode=mode,
        soft_mix=soft_mix,
        soft_temp=soft_temp,
    )
    total = ce + lam * safe
    return total, ce, safe
