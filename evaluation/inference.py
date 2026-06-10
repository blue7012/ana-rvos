"""Single-sample inference helpers."""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from data.preprocessing import load_clip_around_frame, sentence_to_embedding

logger = logging.getLogger(__name__)


def encode_sentence(
    sentence: str,
    word2vec_model,
    encoder: nn.Module,
    max_len: int = 20,
) -> Tensor:
    """
    Raw sentence → 300-dim feature vector.

    Args:
        sentence: natural language query
        word2vec_model: gensim KeyedVectors
        encoder: TextualEncoder instance
        max_len: padding length
    Returns:
        (300,)
    """
    enc_device = next(encoder.parameters()).device
    emb = sentence_to_embedding(sentence, word2vec_model, max_len)
    emb = emb.unsqueeze(0).to(enc_device)

    encoder.eval()
    with torch.no_grad():
        feat = encoder(emb)

    return feat.squeeze(0)


def predict(
    model: nn.Module,
    video_dir: str,
    sentence: str,
    word2vec_model,
    *,
    center_idx: int | None = None,
    num_frames: int = 16,
    frame_size: int = 512,
    max_len: int = 20,
    device: str = "cpu",
) -> Tuple[Dict[int, Tensor], List[str], int]:
    """
    Segment ONE frame: forward pass on the dense clip around it.

    This mirrors training exactly — same tensor shape (1, 3, N, H, W), same
    dense consecutive window, mask supervised/predicted for the clip's middle
    frame (Paper §3.4, §5.2). The temporal avg-pool means one mask per clip,
    so segmenting a whole video = sliding this window (see predict_video).

    Args:
        model: ActorActionVideoSegmentation
        video_dir: path to JPEGImages/<video_id>/
        sentence: natural language query
        word2vec_model: gensim KeyedVectors
        center_idx: index of the frame to segment (default: middle of video)
    Returns:
        responses:  dict {32, 128, 512} of logit maps — each (1, 1, r, r),
                    valid for the frame at center_idx
        all_paths:  every .jpg path in video_dir (sorted)
        center_idx: the index actually used
    """
    model.eval()
    model.to(device)

    all_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    if not all_paths:
        raise FileNotFoundError(f"No .jpg files in: {video_dir}")
    if center_idx is None:
        center_idx = len(all_paths) // 2

    video_tensor, _ = load_clip_around_frame(
        video_dir, center_idx, num_frames, frame_size
    )
    video_tensor = video_tensor.to(device)

    emb = sentence_to_embedding(sentence, word2vec_model, max_len)
    emb = emb.unsqueeze(0).to(device)

    with torch.no_grad():
        responses = model(video_tensor, emb)

    logger.info(
        "Segmented frame %d/%d ('%s') with a dense %d-frame window",
        center_idx,
        len(all_paths),
        os.path.basename(all_paths[center_idx]),
        num_frames,
    )
    return responses, all_paths, center_idx


def predict_video(
    model: nn.Module,
    video_dir: str,
    sentence: str,
    word2vec_model,
    *,
    stride: int = 1,
    num_frames: int = 16,
    frame_size: int = 512,
    max_len: int = 20,
    device: str = "cpu",
) -> Iterator[Tuple[str, Tensor]]:
    """
    Segment a whole video with a sliding window — one forward pass per
    target frame, dense clip centered on it. Paper-faithful per-frame
    protocol (each frame gets ITS OWN mask; nothing is broadcast).

    The sentence is encoded once; only the video clip changes per step.

    Args:
        stride: segment every `stride`-th frame (1 = every frame; higher
                values trade temporal density for speed)
    Yields:
        (frame_path, logits_512) per target frame, logits_512: (512, 512)
    """
    model.eval()
    model.to(device)

    all_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    if not all_paths:
        raise FileNotFoundError(f"No .jpg files in: {video_dir}")

    emb = sentence_to_embedding(sentence, word2vec_model, max_len)
    emb = emb.unsqueeze(0).to(device)

    n = len(all_paths)
    logger.info(
        "Sliding-window inference: %d frames, stride %d → %d forward passes",
        n,
        stride,
        len(range(0, n, stride)),
    )

    for center_idx in range(0, n, stride):
        clip, _ = load_clip_around_frame(video_dir, center_idx, num_frames, frame_size)
        with torch.no_grad():
            responses = model(clip.to(device), emb)
        yield all_paths[center_idx], responses[512].squeeze(0).squeeze(0).cpu()
