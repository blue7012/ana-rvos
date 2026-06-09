"""YouTube-VOS 2019 dataset for referring video object segmentation."""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from data.preprocessing import sentence_to_embedding

logger = logging.getLogger(__name__)


class YouTubeVOSDataset(Dataset):
    """
    YouTube-VOS 2019 for referring video object segmentation.
    Paper: Gavrilyuk et al., CVPR 2018.

    One data point = one (expression, object, video clip) triplet.

    Train / val subsets come from the train/ directory.
    The split is done at VIDEO level (90/10) so no video's frames appear
    in both train and val.  A fixed seed makes it reproducible.

    In train/: each object has 2 synonymous expressions → each is a separate
    data point with the SAME binary mask (palette PNG, pixel == obj_id).

    In valid/: expressions describe different objects across the video →
    masks differ per expression. Pre-extracted binary PNGs live in
    Annotations/{vid}/{expr_id}/.

    The valid/ directory is reserved as a benchmark test set ('test' subset).

    Returns per sample:
        video:     (3, N, H, W)   float32 in [-1, 1]
        embedding: (max_len, 300) float32
        mask:      (1, 512, 512)  float32 in {0, 1}

    Args:
        root:            path to youtube-vos-2019/
        subset:          'train' | 'val' | 'test'
        word2vec_model:  gensim KeyedVectors; only used during __init__
        val_fraction:    share of train videos reserved for val  (default 0.10)
        seed:            controls the 90/10 shuffle  (default 42)
        num_frames:      frames per clip  (Paper §3.2: N=16)
        frame_size:      resize target in pixels  (Paper: 512)
        max_sentence_len: pad/truncate sentences  (Paper §3.1: 20)
    """

    def __init__(
        self,
        root: str,
        subset: str,
        word2vec_model,
        val_fraction: float = 0.10,
        seed: int = 42,
        num_frames: int = 16,
        frame_size: int = 512,
        max_sentence_len: int = 20,
    ) -> None:
        assert subset in ("train", "val", "test"), (
            f"subset must be 'train', 'val', or 'test'; got {subset!r}"
        )
        self.root = root
        self.subset = subset
        self.num_frames = num_frames
        self.frame_size = frame_size

        self.samples: List[Dict] = self._build_samples(val_fraction, seed)

        # Pre-compute all embeddings once at init so DataLoader workers don't
        # need a copy of the 1.5 GB word2vec model.
        # Memory: ~311 MB for 12 k train samples (12913 × 20 × 300 × 4 bytes).
        logger.info("Pre-computing %d sentence embeddings …", len(self.samples))
        self._embeddings: List[Tensor] = [
            sentence_to_embedding(s["expression"], word2vec_model, max_sentence_len)
            for s in self.samples
        ]
        logger.info("Dataset '%s' ready — %d samples", subset, len(self.samples))

    # ------------------------------------------------------------------
    # Build sample list
    # ------------------------------------------------------------------

    def _build_samples(self, val_fraction: float, seed: int) -> List[Dict]:
        if self.subset == "test":
            return self._samples_from_valid()
        return self._samples_from_train(val_fraction, seed)

    def _samples_from_train(
        self, val_fraction: float, seed: int
    ) -> List[Dict]:
        """
        Source: train/meta_expressions.json

        Structure per video:
          objects → {obj_id: {expressions: [str, str], frames: [stem, ...]}}

        Each expression is a separate data point sharing the same binary mask
        (palette PNG pixel == obj_id).  The split is at the video level.
        """
        meta_path = os.path.join(self.root, "train", "meta_expressions.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        all_vid_ids = sorted(meta["videos"].keys())   # sorted → deterministic base
        rng = random.Random(seed)
        rng.shuffle(all_vid_ids)

        n_val = max(1, int(len(all_vid_ids) * val_fraction))
        if self.subset == "val":
            keep = set(all_vid_ids[-n_val:])
        else:
            keep = set(all_vid_ids[:-n_val])

        samples: List[Dict] = []
        for vid_id in sorted(keep):          # sorted again for stable ordering
            for obj_id, odata in meta["videos"][vid_id]["objects"].items():
                for expr in odata.get("expressions", []):
                    samples.append(
                        {
                            "video_id": vid_id,
                            "obj_id": obj_id,    # str "1"/"2"/… used for palette mask
                            "expression": expr,
                            "frames": odata["frames"],
                            "expr_id": None,
                            "_anno_dir": "train",
                        }
                    )
        return samples

    def _samples_from_valid(self) -> List[Dict]:
        """
        Source: valid/meta_expressions.json

        Structure per video:
          frames: [stem, ...]
          expressions: {expr_id: {exp: str, obj_id: int}}

        Each expression may reference a different obj_id → different binary mask.
        Masks are pre-extracted as L-mode PNGs in
        valid/Annotations/{vid_id}/{expr_id}/{frame}.png  ({0, 255}).
        """
        meta_path = os.path.join(self.root, "valid", "meta_expressions.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        samples: List[Dict] = []
        for vid_id, vdata in meta["videos"].items():
            frames: List[str] = vdata["frames"]
            for expr_id, edata in vdata["expressions"].items():
                samples.append(
                    {
                        "video_id": vid_id,
                        "obj_id": str(edata["obj_id"]),
                        "expression": edata["exp"],
                        "frames": frames,
                        "expr_id": expr_id,   # matches annotation subdir name
                        "_anno_dir": "valid",
                    }
                )
        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            video:     (3, N, H, W)   float32 in [-1, 1]
            embedding: (max_len, 300) float32
            mask:      (1, 512, 512)  float32 in {0, 1}
        """
        s = self.samples[idx]

        # Sample N evenly-spaced frames; np.linspace repeats when clip is short.
        frames: List[str] = s["frames"]
        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        stems = [frames[i] for i in indices]

        video = self._load_frames(s["_anno_dir"], s["video_id"], stems)   # (3, N, H, W)
        mask = self._load_mask(s, stems[self.num_frames // 2])             # (1, 512, 512)
        embedding = self._embeddings[idx]                                   # (max_len, 300)

        return video, embedding, mask

    # ------------------------------------------------------------------
    # Frame and mask loaders
    # ------------------------------------------------------------------

    def _load_frames(
        self, split_dir: str, video_id: str, stems: List[str]
    ) -> Tensor:
        """
        Load N JPEG frames → (frame_size × frame_size) → normalise to [-1, 1].

        Returns: (3, N, H, W) float32
        """
        frame_dir = os.path.join(self.root, split_dir, "JPEGImages", video_id)
        frames = []
        for stem in stems:
            img = (
                Image.open(os.path.join(frame_dir, stem + ".jpg"))
                .convert("RGB")
                .resize((self.frame_size, self.frame_size), Image.BILINEAR)
            )
            arr = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0   # [-1, 1]
            frames.append(arr)   # (H, W, 3)

        clip = np.stack(frames)              # (N, H, W, 3)
        clip = clip.transpose(3, 0, 1, 2)   # (3, N, H, W)
        return torch.from_numpy(clip)

    def _load_mask(self, sample: Dict, stem: str) -> Tensor:
        """
        Load binary segmentation mask for one frame.

        train/val: palette-mode PNG → (arr == int(obj_id)) → {0, 1}
        test:      L-mode binary PNG in per-expression subdir → (arr > 0) → {0, 1}

        Returns: (1, 512, 512) float32
        """
        if sample["_anno_dir"] == "train":
            anno_path = os.path.join(
                self.root, "train", "Annotations",
                sample["video_id"], stem + ".png",
            )
            arr = np.array(Image.open(anno_path))          # (H, W) palette values
            binary = (arr == int(sample["obj_id"])).astype(np.float32)
        else:
            anno_path = os.path.join(
                self.root, "valid", "Annotations",
                sample["video_id"], sample["expr_id"], stem + ".png",
            )
            arr = np.array(Image.open(anno_path))          # (H, W) {0, 255}
            binary = (arr > 0).astype(np.float32)

        mask = torch.from_numpy(binary)[None, None]        # (1, 1, H, W)
        mask = F.interpolate(
            mask, size=(512, 512), mode="nearest"
        ).squeeze(0)                                       # (1, 512, 512)
        return mask
