"""Demo: end-to-end forward pass on one video + sentence."""

import argparse
import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from data.preprocessing import (
    load_annotation_mask,
    load_video_frames,
    load_word2vec,
    middle_frame_anno_path,
    sentence_to_embedding,
)
from evaluation.visualize import visualize_gt_vs_pred, visualize_prediction
from model.model import build_model
from training.loss import make_multiscale_targets, multiresolution_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full model demo")
    parser.add_argument("--video-dir", required=True, help="Path to JPEGImages/<id>/")
    parser.add_argument("--sentence", required=True, help="Natural language query")
    parser.add_argument(
        "--anno-dir",
        default=None,
        help="Path to Annotations/<id>/ (optional, for loss demo)",
    )
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    device = args.device or (cfg.device if torch.cuda.is_available() else "cpu")

    word2vec = load_word2vec(cfg.data.word2vec_model)
    model = build_model(cfg, device=device)
    model.eval()

    video_tensor, frame_paths = load_video_frames(
        args.video_dir, args.num_frames, args.frame_size
    )
    video_tensor = video_tensor.to(device)

    emb = sentence_to_embedding(args.sentence, word2vec, cfg.model.max_sentence_len)
    emb = emb.unsqueeze(0).to(device)

    logger.info("Video: %s | Sentence: '%s'", tuple(video_tensor.shape), args.sentence)

    with torch.no_grad():
        responses = model(video_tensor, emb)

    for r in (32, 128, 512):
        logger.info("Response[%d]: %s", r, tuple(responses[r].shape))

    visualize_prediction(
        frame_paths, responses, args.sentence, frame_size=args.frame_size
    )

    if args.anno_dir:
        anno_path = middle_frame_anno_path(args.anno_dir, frame_paths)
        gt_mask = load_annotation_mask(anno_path, size=args.frame_size).to(device)
        targets = make_multiscale_targets(gt_mask)
        loss = multiresolution_loss(responses, targets)
        logger.info("Real annotation loss: %.4f", loss.item())
        visualize_gt_vs_pred(
            frame_paths, gt_mask, responses, args.sentence, frame_size=args.frame_size
        )


if __name__ == "__main__":
    main()
