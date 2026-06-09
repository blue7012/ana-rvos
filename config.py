"""Central configuration for Actor and Action Video Segmentation."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainConfig:
    """Training hyperparameters — Paper §3.4."""

    lr: float = 0.001  # Paper §3.4
    lr_step: int = 5000  # Paper §3.4 — divide lr by 10 every N iterations
    lr_gamma: float = 0.1  # Paper §3.4 ("divide by 10")
    total_iters: int = 15000  # Paper §3.4
    batch_size: int = 2  # ⚠️ not in paper — GPU memory constraint (🟡)
    num_workers: int = 2  # ⚠️ not in paper (🟢)
    save_every: int = 1000  # ⚠️ not in paper — checkpoint cadence (🟢)
    log_every: int = 50  # ⚠️ not in paper — log cadence (🟢)
    checkpoint_dir: str = "checkpoints"
    split: str = "train"  # which YouTube-VOS split to train on


@dataclass
class ModelConfig:
    """Architecture hyperparameters — Gavrilyuk et al. CVPR 2018."""

    embed_dim: int = 300
    text_kernel_size: int = 2
    max_sentence_len: int = 20

    i3d_feature_ch: int = 832
    video_ch: int = 834  # 832 visual + 2 spatial coord channels

    decoder_ch_32: int = 834
    decoder_ch_128: int = 256
    decoder_ch_512: int = 128

    num_frames: int = 16
    frame_size: int = 512
    output_size: int = 512


@dataclass
class DataConfig:
    """Paths and data-loading settings."""

    word2vec_model: str = "word2vec-google-news-300"
    i3d_weights_path: str = "external/pytorch-i3d/models/rgb_imagenet.pt"
    pytorch_i3d_path: str = "external/pytorch-i3d"

    youtube_vos_root: str = "data/youtube-vos-2019"


@dataclass
class LossConfig:
    """Multi-resolution logistic loss weights — Paper Sec 3.4."""

    alphas: Dict[int, float] = field(
        default_factory=lambda: {32: 1.0, 128: 1.0, 512: 1.0}
    )


@dataclass
class Config:
    """Top-level config grouping all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"
    seed: int = 42
