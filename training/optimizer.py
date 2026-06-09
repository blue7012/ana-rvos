"""Optimizer and LR scheduler factory — Paper §3.4."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from config import TrainConfig
from model.modules import VideoEncoder

logger = logging.getLogger(__name__)


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    train_cfg: TrainConfig,
) -> Tuple[optim.Adam, StepLR]:
    """
    Set up Adam + StepLR exactly as described in Paper §3.4.

    Fine-tuning strategy (Paper §3.4):
      - Freeze all I3D layers before Mixed_4f
      - Fine-tune Mixed_4f, the full TextualEncoder, and the full Decoder
    LR schedule: divide by 10 every 5 000 iterations (StepLR stepped per iter).

    Args:
        model:     ActorActionVideoSegmentation
        train_cfg: TrainConfig with lr, lr_step, lr_gamma
    Returns:
        (optimizer, scheduler)
    """
    # --- 1. Freeze entire I3D backbone first ---
    for param in model.video_encoder.i3d.parameters():
        param.requires_grad = False

    frozen = sum(p.numel() for p in model.video_encoder.i3d.parameters())
    logger.info("Frozen %s I3D params", f"{frozen:,}")

    # --- 2. Unfreeze Mixed_4f (last inception block) — Paper §3.4 ---
    mixed4f, name = VideoEncoder._find_mixed4f(model.video_encoder.i3d)
    for param in mixed4f.parameters():
        param.requires_grad = True

    unfrozen_i3d = sum(
        p.numel() for p in mixed4f.parameters() if p.requires_grad
    )
    logger.info("Unfrozen Mixed_4f ('%s') — %s params", name, f"{unfrozen_i3d:,}")

    # --- 3. Collect all trainable parameters ---
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %s / %s  (%.1f %%)",
        f"{n_trainable:,}",
        f"{n_total:,}",
        100.0 * n_trainable / n_total,
    )

    # --- 4. Adam with paper lr — Paper §3.4 ---
    optimizer = optim.Adam(trainable, lr=train_cfg.lr)

    # --- 5. StepLR stepped per ITERATION so lr divides at exactly step 5k/10k ---
    # Paper §3.4: "divide lr by 10 every 5 000 iterations"
    scheduler = StepLR(
        optimizer,
        step_size=train_cfg.lr_step,   # 5 000
        gamma=train_cfg.lr_gamma,      # 0.1
    )

    return optimizer, scheduler
