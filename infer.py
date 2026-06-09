"""Entry point: run inference on a video folder + sentence."""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image

from data.preprocessing import load_word2vec
from evaluation.inference import predict
from model.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Actor-Action Video Segmentation — Inference")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--video_dir", required=True, help="Path to folder of video frames (JPEGs)")
    p.add_argument("--sentence", required=True, help='Query sentence, e.g. "a man running"')
    p.add_argument("--output", default="mask.png", help="Where to save the binary mask PNG")
    p.add_argument("--visualize", action="store_true",
                   help="Also save a 3-panel visualization alongside the mask")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    return p.parse_args()


def _save_visualization(
    frame_path: str,
    logits: np.ndarray,
    sentence: str,
    output_path: str,
) -> None:
    """
    3-panel visualization saved next to the mask:
      [Original frame] | [Soft heatmap overlay] | [Binary mask overlay]
    """
    import matplotlib.pyplot as plt

    frame = np.array(Image.open(frame_path).resize((512, 512)).convert("RGB"))

    # sigmoid → soft probability in [0, 1]
    prob = 1.0 / (1.0 + np.exp(-logits))
    binary = logits > 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'"{sentence}"', fontsize=14, y=1.01)

    axes[0].imshow(frame)
    axes[0].set_title("Original frame (middle)")
    axes[0].axis("off")

    axes[1].imshow(frame)
    axes[1].imshow(prob, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title("Soft heatmap overlay")
    axes[1].axis("off")

    red = np.zeros_like(frame)
    red[..., 0] = 255
    alpha = binary[..., None].astype(np.float32) * 0.5
    overlay = (frame * (1 - alpha) + red * alpha).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title("Binary mask overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.getLogger(__name__).info("Visualization saved → %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    logging.info("Checkpoint from step %d  (val_loss=%.4f)", ckpt["step"], ckpt["val_loss"])

    model = build_model(cfg, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logging.info("Model weights loaded")

    word2vec = load_word2vec(cfg.data.word2vec_model)

    logging.info("Running inference — sentence: %r", args.sentence)
    responses, frame_paths = predict(
        model=model,
        video_dir=args.video_dir,
        sentence=args.sentence,
        word2vec_model=word2vec,
        num_frames=cfg.model.num_frames,
        frame_size=cfg.model.frame_size,
        max_len=cfg.model.max_sentence_len,
        device=device,
    )

    logits = responses[512].squeeze().cpu().numpy()   # (512, 512)

    # Always save binary mask
    mask = (logits > 0).astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(args.output)
    logging.info("Mask saved → %s", args.output)

    # Optionally save visualization
    if args.visualize:
        stem, ext = os.path.splitext(args.output)
        vis_path = f"{stem}_vis{ext}"
        middle_frame = frame_paths[len(frame_paths) // 2]
        _save_visualization(middle_frame, logits, args.sentence, vis_path)


if __name__ == "__main__":
    main()
