"""I3D backbone loading utilities."""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pytorch_i3d import InceptionI3d

logger = logging.getLogger(__name__)


def _ensure_pytorch_i3d_on_path(pytorch_i3d_path: str) -> None:
    """Add pytorch-i3d repo to sys.path if not already importable."""
    try:
        import pytorch_i3d  # noqa: F401
        return
    except ImportError:
        pass

    abs_path = os.path.abspath(pytorch_i3d_path)
    if not os.path.isdir(abs_path):
        raise ImportError(
            f"pytorch-i3d not found. Clone it to '{abs_path}' or install it:\n"
            "  git clone https://github.com/piergiaj/pytorch-i3d.git external/pytorch-i3d"
        )
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)


def load_i3d(
    weights_path: str,
    pytorch_i3d_path: str = "external/pytorch-i3d",
) -> "InceptionI3d":
    """
    Load RGB I3D pre-trained on ImageNet + Kinetics. Paper Sec 3.2.

    Args:
        weights_path: path to rgb_imagenet.pt
        pytorch_i3d_path: path to cloned pytorch-i3d repo
    Returns:
        InceptionI3d in eval mode
    """
    _ensure_pytorch_i3d_on_path(pytorch_i3d_path)
    from pytorch_i3d import InceptionI3d

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"I3D weights not found: {weights_path}\n"
            "Download from pytorch-i3d/models/rgb_imagenet.pt"
        )

    size = os.path.getsize(weights_path)
    if size < 1_000_000:
        raise ValueError(
            f"Weight file too small ({size} bytes) — likely a Git LFS pointer. "
            "Run: git lfs pull"
        )

    i3d = InceptionI3d(num_classes=400, in_channels=3)
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")
    i3d.load_state_dict(state, strict=False)
    i3d.eval()

    total = sum(p.numel() for p in i3d.parameters())
    logger.info("I3D loaded from %s — %s params", weights_path, f"{total:,}")
    return i3d
