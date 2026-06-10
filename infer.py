"""Entry point: run inference on a video folder + sentence."""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image

from data.preprocessing import load_word2vec
from evaluation.inference import predict, predict_video
from model.model import build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Actor-Action Video Segmentation — Inference"
    )
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument(
        "--video_dir", required=True, help="Path to folder of video frames (JPEGs)"
    )
    p.add_argument(
        "--sentence", required=True, help='Query sentence, e.g. "a man running"'
    )
    p.add_argument(
        "--output",
        default="mask.png",
        help="Where to save the binary mask PNG (single-frame mode)",
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Directory for per-frame masks (sliding-window mode: one forward pass per frame).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="In sliding-window mode, segment every k-th frame (1 = every frame).",
    )
    p.add_argument(
        "--frame_idx",
        type=int,
        default=None,
        help="In single-frame mode, which frame index to segment (default: middle of video).",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Also save a 3-panel visualization alongside each mask",
    )
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    return p.parse_args()


def _save_visualization(
    frame_path: str,
    logits: np.ndarray,
    sentence: str,
    output_path: str,
    *,
    log: bool = True,
) -> None:
    """
    3-panel visualization: [Original frame] | [Soft heatmap overlay] | [Binary mask overlay]
    """
    import matplotlib.pyplot as plt

    frame_stem = os.path.splitext(os.path.basename(frame_path))[0]
    frame = np.array(Image.open(frame_path).resize((512, 512)).convert("RGB"))

    # sigmoid → soft probability in [0, 1]
    prob = 1.0 / (1.0 + np.exp(-logits))
    binary = logits > 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'"{sentence}"', fontsize=14, y=1.01)

    axes[0].imshow(frame)
    axes[0].set_title(f"Frame {frame_stem}")
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
    if log:
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
    logging.info(
        "Checkpoint from step %d  (val_loss=%.4f)", ckpt["step"], ckpt["val_loss"]
    )

    model = build_model(cfg, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logging.info("Model weights loaded")

    word2vec = load_word2vec(cfg.data.word2vec_model)

    logging.info("Running inference — sentence: %r", args.sentence)

    if args.output_dir is not None:
        # ── All-frames mode: sliding window, one mask PER frame (paper) ──────
        os.makedirs(args.output_dir, exist_ok=True)

        n_saved = 0
        for frame_path, logits_t in predict_video(
            model=model,
            video_dir=args.video_dir,
            sentence=args.sentence,
            word2vec_model=word2vec,
            stride=args.stride,
            num_frames=cfg.model.num_frames,
            frame_size=cfg.model.frame_size,
            max_len=cfg.model.max_sentence_len,
            device=device,
        ):
            logits = logits_t.numpy()  # (512, 512)
            mask = (logits > 0).astype(np.uint8) * 255
            stem = os.path.splitext(os.path.basename(frame_path))[0]
            Image.fromarray(mask, mode="L").save(
                os.path.join(args.output_dir, stem + ".png")
            )
            if args.visualize:
                _save_visualization(
                    frame_path,
                    logits,
                    args.sentence,
                    os.path.join(args.output_dir, stem + "_vis.png"),
                    log=False,
                )
            n_saved += 1

        if args.visualize:
            logging.info(
                "Saved %d masks + %d visualizations → %s",
                n_saved,
                n_saved,
                args.output_dir,
            )
        else:
            logging.info("Saved %d masks → %s", n_saved, args.output_dir)

    else:
        # ── Single-frame mode: dense clip centered on one frame ─────────────
        responses, all_paths, center_idx = predict(
            model=model,
            video_dir=args.video_dir,
            sentence=args.sentence,
            word2vec_model=word2vec,
            center_idx=args.frame_idx,  # None → middle of video
            num_frames=cfg.model.num_frames,
            frame_size=cfg.model.frame_size,
            max_len=cfg.model.max_sentence_len,
            device=device,
        )

        logits = responses[512].squeeze().cpu().numpy()  # (512, 512)
        mask = (logits > 0).astype(np.uint8) * 255

        Image.fromarray(mask, mode="L").save(args.output)
        logging.info("Mask for frame %d saved → %s", center_idx, args.output)

        if args.visualize:
            stem, ext = os.path.splitext(args.output)
            _save_visualization(
                all_paths[center_idx], logits, args.sentence, f"{stem}_vis{ext}"
            )


if __name__ == "__main__":
    main()
