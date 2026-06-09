"""Atomic neural network blocks — Paper Sec 3.1 and 3.3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TextualEncoder(nn.Module):
    """
    1D-CNN sentence encoder. Paper Sec 3.1.

    Input:  (B, max_len, 300)
    Output: (B, 300)
    """

    def __init__(self, embed_dim: int = 300, kernel_size: int = 2):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=0,
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, max_len, 300)
        Returns:
            (B, 300)
        """
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = x.max(dim=-1).values
        return x


class DynamicFilterGenerator(nn.Module):
    """
    Sentence vector → three dynamic filters. Paper Sec 3.3, Eq. 1.

    Input:  (B, 300)
    Output: dict {32: (B, ch_32), 128: (B, ch_128), 512: (B, ch_512)}
    """

    def __init__(
        self,
        text_dim: int = 300,
        ch_32: int = 834,
        ch_128: int = 256,
        ch_512: int = 128,
    ):
        super().__init__()
        self.fc_32 = nn.Linear(text_dim, ch_32)
        self.fc_128 = nn.Linear(text_dim, ch_128)
        self.fc_512 = nn.Linear(text_dim, ch_512)

    def forward(self, T: Tensor) -> dict[int, Tensor]:
        return {
            32: torch.tanh(self.fc_32(T)),
            128: torch.tanh(self.fc_128(T)),
            512: torch.tanh(self.fc_512(T)),
        }


class DeconvBlock(nn.Module):
    """
    Upsample video feature map by 4x. Paper Sec 3.3.

    (B, in_ch, H, W) → (B, out_ch, 4H, 4W)
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=8, stride=4, padding=2
        )
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.deconv(x))
        x = F.relu(self.conv(x))
        return x


def dynamic_conv(feature: Tensor, filt: Tensor) -> Tensor:
    """
    Per-sample 1x1 dynamic convolution. Paper Sec 3.3, Eq. 2: S_r = f_r * V_r.

    Args:
        feature: (B, C, H, W)
        filt:    (B, C)
    Returns:
        (B, 1, H, W)
    """
    return (feature * filt[:, :, None, None]).sum(dim=1, keepdim=True)
