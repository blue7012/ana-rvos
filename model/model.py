"""Full end-to-end model assembly."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from config import Config, ModelConfig
from model.i3d_loader import load_i3d
from model.layers import TextualEncoder
from model.modules import Decoder, VideoEncoder


class ActorActionVideoSegmentation(nn.Module):
    """
    Full model: (video, sentence_embedding) → multi-resolution response maps.
    Paper: Gavrilyuk et al., CVPR 2018.

    Inputs:
        video:              (B, 3, N, H, W) RGB frames in [-1, 1]
        sentence_embedding: (B, max_len, 300) padded Word2Vec matrices
    Output:
        dict {32, 128, 512} of (B, 1, r, r) logits; use [512] as final mask
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        video_encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.video_encoder = video_encoder
        self.decoder = decoder

    def forward(self, video: Tensor, sentence_embedding: Tensor) -> dict[int, Tensor]:
        text_feat = self.text_encoder(sentence_embedding)
        video_feat = self.video_encoder(video)
        return self.decoder(text_feat, video_feat)


# Backward-compatible alias from the Colab notebook
ActorActionSegmenter = ActorActionVideoSegmentation


def build_model(cfg: Config, device: str | None = None) -> ActorActionVideoSegmentation:
    """
    Factory: load I3D weights and wire all three components.

    Args:
        cfg: top-level Config
        device: optional torch device string
    Returns:
        model on the requested device
    """
    mcfg = cfg.model
    dcfg = cfg.data

    i3d = load_i3d(dcfg.i3d_weights_path, dcfg.pytorch_i3d_path)
    text_encoder = TextualEncoder(
        embed_dim=mcfg.embed_dim,
        kernel_size=mcfg.text_kernel_size,
    )
    video_encoder = VideoEncoder(i3d)
    decoder = Decoder(text_dim=mcfg.embed_dim, video_ch=mcfg.video_ch)

    model = ActorActionVideoSegmentation(text_encoder, video_encoder, decoder)

    if device is not None:
        model = model.to(device)

    return model
