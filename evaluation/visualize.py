"""Visualization helpers for debugging and demos."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


def visualize_prediction(
    frame_paths: List[str],
    responses: Dict[int, Tensor],
    sentence: str,
    *,
    frame_size: int = 512,
    threshold: float = 0.5,
) -> None:
    """Show middle frame, sigmoid heatmap, and thresholded overlay."""
    mid_idx = len(frame_paths) // 2
    mid_frame = Image.open(frame_paths[mid_idx]).convert("RGB").resize(
        (frame_size, frame_size)
    )
    heat = torch.sigmoid(responses[512])[0, 0].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(mid_frame)
    ax[0].set_title("Middle frame")
    ax[0].axis("off")

    ax[1].imshow(heat, cmap="jet")
    ax[1].set_title("Response (sigmoid)")
    ax[1].axis("off")

    ax[2].imshow(mid_frame)
    ax[2].imshow(heat > threshold, cmap="Reds", alpha=0.5)
    ax[2].set_title(f"Mask overlay (>{threshold})")
    ax[2].axis("off")

    plt.suptitle(f'"{sentence}"', fontsize=12)
    plt.tight_layout()
    plt.show()


def visualize_video_encoder(
    frame_paths: List[str],
    feat_map: Tensor,
    video_dir: str,
    *,
    num_frames: int = 16,
    frame_size: int = 512,
) -> None:
    """Feature-map sanity-check plot from the original notebook."""
    feat_np = feat_map[0].cpu().numpy()
    H_prime = feat_np.shape[1]

    mid_img = Image.open(frame_paths[num_frames // 2]).convert("RGB")
    mid_img = mid_img.resize((frame_size, frame_size), Image.BILINEAR)
    mid_np = np.array(mid_img, dtype=np.float32) / 255.0

    mean_feat = feat_np[:832].mean(axis=0)
    mean_t = torch.tensor(mean_feat).unsqueeze(0).unsqueeze(0)
    mean_up = F.interpolate(mean_t, (frame_size, frame_size), mode="bilinear", align_corners=False)
    mean_up_np = mean_up[0, 0].numpy()
    mean_norm = (mean_up_np - mean_up_np.min()) / (mean_up_np.max() - mean_up_np.min() + 1e-8)
    heatmap_rgb = cm.jet(mean_norm)[:, :, :3]
    overlay = 0.55 * mid_np + 0.45 * heatmap_rgb

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 5, figure=fig, wspace=0.22, hspace=0.38)

    for col, frame_idx in enumerate([0, 5, 10, 15]):
        ax = fig.add_subplot(gs[0, col])
        img = Image.open(frame_paths[frame_idx]).convert("RGB").resize((224, 224))
        ax.imshow(img)
        ax.set_title(
            f"frame[{frame_idx}]  {os.path.basename(frame_paths[frame_idx])}",
            fontsize=8,
        )
        ax.axis("off")

    ax = fig.add_subplot(gs[0, 4])
    im = ax.imshow(feat_np[833], cmap="coolwarm", interpolation="nearest", vmin=-1, vmax=1)
    ax.set_title("y-coord channel [833]", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(mean_feat, cmap="viridis", interpolation="nearest")
    ax.set_title(f"Mean across 832 ch\n({H_prime}×{H_prime})", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    for col, ch in enumerate([0, 200, 600]):
        ax = fig.add_subplot(gs[1, col + 1])
        im = ax.imshow(feat_np[ch], cmap="plasma", interpolation="nearest")
        ax.set_title(f"Channel {ch}\n({H_prime}×{H_prime})", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 4])
    ax.imshow(overlay)
    ax.set_title(f"Heatmap overlay\n(frame[{num_frames // 2}])", fontsize=8)
    ax.axis("off")

    plt.suptitle(
        f"VideoEncoder — '{os.path.basename(video_dir)}'\n"
        f"Output: {tuple(feat_map.shape[1:])}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


def visualize_gt_vs_pred(
    frame_paths: List[str],
    gt_mask: Tensor,
    responses: Dict[int, Tensor],
    sentence: str,
    *,
    frame_size: int = 512,
) -> None:
    """Side-by-side: middle frame, ground truth, model response."""
    mid_idx = len(frame_paths) // 2
    mid_frame = Image.open(frame_paths[mid_idx]).convert("RGB").resize((frame_size, frame_size))
    gt = gt_mask[0, 0].cpu().numpy()
    pred = torch.sigmoid(responses[512])[0, 0].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(mid_frame)
    ax[0].set_title("Middle frame")
    ax[0].axis("off")

    ax[1].imshow(gt, cmap="gray")
    ax[1].set_title("Ground truth")
    ax[1].axis("off")

    ax[2].imshow(pred, cmap="jet")
    ax[2].set_title("Model response")
    ax[2].axis("off")

    plt.suptitle(f'"{sentence}"', fontsize=12)
    plt.tight_layout()
    plt.show()
