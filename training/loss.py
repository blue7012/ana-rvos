"""Multi-resolution logistic loss — Paper Sec 3.4, Eqs. 3-5."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def make_multiscale_targets(mask_512: Tensor) -> Dict[int, Tensor]:
    """
    Build {32, 128, 512} ground-truth masks from one 512×512 binary mask.

    Args:
        mask_512: (B, 1, 512, 512) values in {0, 1}
    Returns:
        dict of (B, 1, r, r) tensors in {-1, +1}
    """
    targets: Dict[int, Tensor] = {}
    for r in (32, 128, 512):
        m = F.interpolate(mask_512, size=(r, r), mode="nearest")
        targets[r] = m * 2.0 - 1.0
    return targets


def multiresolution_loss(
    responses: Dict[int, Tensor],
    targets: Dict[int, Tensor],
    alphas: Optional[Dict[int, float]] = None,
) -> Tensor:
    """
    L = sum_r alpha_r * mean_ij log(1 + exp(-S_r_ij * Y_r_ij)). Paper Sec 3.4.

    Args:
        responses: dict {32,128,512} of (B,1,r,r) logits
        targets:   dict {32,128,512} of (B,1,r,r) in {-1,+1}
        alphas:    per-resolution weights; paper uses 1.0 for all
    Returns:
        scalar loss
    """
    if alphas is None:
        alphas = {32: 1.0, 128: 1.0, 512: 1.0}

    total = torch.tensor(0.0, device=responses[512].device)
    for r in (32, 128, 512):
        S = responses[r]
        Y = targets[r]
        L_r = F.softplus(-S * Y).mean()
        total = total + alphas[r] * L_r
    return total
