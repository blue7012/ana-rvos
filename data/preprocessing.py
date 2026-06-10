"""Video frame, annotation, and Word2Vec preprocessing."""

from __future__ import annotations

import glob
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def load_word2vec(model_name: str = "word2vec-google-news-300"):
    """
    Download and return the Google News Word2Vec model (300-dim).

    Returns:
        gensim.models.KeyedVectors — vocab ~3M, dim=300
    Complexity: O(1) time after download | O(vocab * dim) memory
    """
    import gensim.downloader as api

    logger.info("Downloading Word2Vec (%s) — ~1.5 GB, runs once ...", model_name)
    model = api.load(model_name)
    logger.info(
        "Word2Vec loaded: vocab=%s dim=%s", f"{len(model):,}", model.vector_size
    )
    return model


def sentence_to_embedding(
    sentence: str,
    word2vec_model,
    max_len: int = 20,
) -> torch.Tensor:
    """
    Sentence string → padded word-embedding matrix. Paper Sec 3.1.

    Args:
        sentence: raw text, e.g. "a man in dark suit standing on the back"
        word2vec_model: gensim KeyedVectors (300-dim)
        max_len: pad or truncate to this many tokens
    Returns:
        (max_len, 300) — zero-padded; OOV words are zero-filled
    """
    tokens = sentence.lower().split()
    embed_dim = word2vec_model.vector_size

    matrix = np.zeros((max_len, embed_dim), dtype=np.float32)
    for i, token in enumerate(tokens[:max_len]):
        if token in word2vec_model:
            matrix[i] = word2vec_model[token]

    return torch.from_numpy(matrix)


def _frames_to_tensor(paths: List[str], frame_size: int) -> torch.Tensor:
    """
    Load JPEG paths → (1, 3, N, H, W) float tensor in [-1, 1] (I3D norm).
    """
    frames = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img = img.resize((frame_size, frame_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        frames.append(arr)

    video = np.stack(frames)  # (N, H, W, 3)
    video = video.transpose(3, 0, 1, 2)  # (3, N, H, W)
    return torch.from_numpy(video).unsqueeze(0)


def dense_window_indices(
    num_total: int, center_idx: int, num_frames: int
) -> np.ndarray:
    """
    Indices of `num_frames` CONSECUTIVE frames centered on `center_idx`,
    clamped at the video boundaries (edge frames repeat). Paper §3.4 / §5.2:
    "16 frames around the frame to be segmented".

    Args:
        num_total:  total frames available in the video
        center_idx: index of the frame to be segmented
        num_frames: window length N
    Returns:
        (num_frames,) int array; window[num_frames // 2] == center_idx
        whenever the window fits inside the video
    """
    half = num_frames // 2
    start = center_idx - half
    return np.clip(np.arange(start, start + num_frames), 0, num_total - 1)


def load_clip_around_frame(
    video_dir: str,
    center_idx: int,
    num_frames: int = 16,
    frame_size: int = 512,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Load a dense clip of N consecutive frames around one frame — the
    paper-faithful input for segmenting that frame.

    Args:
        video_dir:  path to folder with *.jpg frames
        center_idx: index (into the sorted frame list) of the target frame
        num_frames: window length N (paper: 16)
        frame_size: resize to (frame_size, frame_size)
    Returns:
        video: (1, 3, N, H, W) values in [-1, 1]
        paths: the selected file paths (center frame at position N // 2)
    """
    all_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    if not all_paths:
        raise FileNotFoundError(f"No .jpg files in: {video_dir}")
    if not 0 <= center_idx < len(all_paths):
        raise IndexError(
            f"center_idx {center_idx} out of range for {len(all_paths)} frames"
        )

    indices = dense_window_indices(len(all_paths), center_idx, num_frames)
    selected = [all_paths[i] for i in indices]
    return _frames_to_tensor(selected, frame_size), selected


def load_video_frames(
    video_dir: str,
    num_frames: int = 16,
    frame_size: int = 512,
) -> Tuple[torch.Tensor, List[str]]:
    """
    DEPRECATED for training/inference — kept only for old notebook cells.

    Loads N evenly-spaced (sparse) frames across the whole video. This does
    NOT match the paper, which uses dense consecutive clips around the frame
    to be segmented (see `load_clip_around_frame`). Sparse frames break the
    temporal continuity that I3D's 3D convolutions rely on.
    """
    all_paths = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))
    if not all_paths:
        raise FileNotFoundError(f"No .jpg files in: {video_dir}")

    indices = np.linspace(0, len(all_paths) - 1, num_frames, dtype=int)
    selected = [all_paths[i] for i in indices]
    tensor = _frames_to_tensor(selected, frame_size)

    logger.warning(
        "load_video_frames() uses sparse linspace sampling (non-paper). "
        "Prefer load_clip_around_frame(). Loaded %d/%d frames from '%s'",
        num_frames,
        len(all_paths),
        os.path.basename(video_dir),
    )
    return tensor, selected


def load_annotation_mask(anno_path: str, size: int = 512) -> torch.Tensor:
    """
    Load one YouTube-VOS palette-PNG annotation → binary mask.

    Args:
        anno_path: path to mask PNG (index 0=bg, >=1=object)
        size: resize to (size, size)
    Returns:
        (1, 1, size, size) float tensor in {0, 1}
    """
    m = Image.open(anno_path)
    idx = np.array(m)
    binary = (idx > 0).astype(np.float32)

    t = torch.from_numpy(binary)[None, None]
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t


def middle_frame_anno_path(anno_video_dir: str, frame_paths: List[str]) -> str:
    """
    Pick the annotation PNG matching the middle sampled video frame.

    Args:
        anno_video_dir: e.g. ".../Annotations/0a2f2bd294"
        frame_paths: sampled JPEG paths from load_video_frames()
    Returns:
        path to matching .png annotation
    """
    mid_jpg = frame_paths[len(frame_paths) // 2]
    stem = os.path.splitext(os.path.basename(mid_jpg))[0]
    return os.path.join(anno_video_dir, stem + ".png")
