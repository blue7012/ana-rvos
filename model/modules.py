"""Functional encoder and decoder modules — Paper Sec 3.2 and 3.3."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.layers import DeconvBlock, DynamicFilterGenerator, dynamic_conv

logger = logging.getLogger(__name__)


class VideoEncoder(nn.Module):
    """
    I3D feature extractor with post-processing. Paper Sec 3.2.

    (B, 3, N, H, W) → (B, 834, H', W')
    """

    def __init__(self, i3d: nn.Module):
        super().__init__()
        self.i3d = i3d
        self._feat: dict[str, Tensor] = {}

        target_module, target_name = self._find_mixed4f(i3d)
        self._hook = target_module.register_forward_hook(
            lambda _m, _inp, out: self._feat.update({"mixed_4f": out})
        )
        logger.info(
            "VideoEncoder: hooked '%s' (type=%s)",
            target_name,
            type(target_module).__name__,
        )

    @staticmethod
    def _find_mixed4f(i3d: nn.Module) -> Tuple[nn.Module, str]:
        for attr_name in ("mixed_4f", "Mixed_4f"):
            if hasattr(i3d, attr_name):
                return getattr(i3d, attr_name), attr_name

        for name, mod in i3d.named_modules():
            normalised = name.lower().replace("/", "_").replace("-", "_")
            if "mixed_4f" in normalised:
                return mod, name

        inception_count = 0
        for name, mod in i3d.named_modules():
            if type(mod).__name__ == "InceptionModule":
                inception_count += 1
                if inception_count == 6:
                    return mod, f"{name} (strategy-3 fallback)"

        all_names = [n for n, _ in i3d.named_modules() if n]
        raise RuntimeError(
            "Could not locate Mixed_4f in the I3D model.\n"
            + "All module names:\n"
            + "\n".join(f"  {n}" for n in all_names)
        )

    @staticmethod
    def _append_spatial_coords(x: Tensor) -> Tensor:
        """Append y-grid and x-grid channels. Values in [-1, 1]. Paper Sec 3.2."""
        B, C, H, W = x.shape
        y_line = torch.linspace(-1, 1, H, device=x.device)
        x_line = torch.linspace(-1, 1, W, device=x.device)
        gy, gx = torch.meshgrid(y_line, x_line, indexing="ij")
        gy = gy.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        gx = gx.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([x, gy, gx], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, N, H, W) values in [-1, 1]
        Returns:
            (B, 834, H', W') — H'=W'=32 for 512×512 input
        """
        self._feat.clear()
        _ = self.i3d(x)

        feat = self._feat["mixed_4f"]
        feat = feat.mean(dim=2)
        feat = F.normalize(feat, p=2, dim=1)
        feat = self._append_spatial_coords(feat)
        return feat

    def __del__(self) -> None:
        if hasattr(self, "_hook"):
            self._hook.remove()


class Decoder(nn.Module):
    """
    Multi-resolution decoder with dynamic filters. Paper Sec 3.3.

    Inputs:  T (B, 300), video_feat (B, 834, 32, 32)
    Outputs: dict {32, 128, 512} of (B, 1, r, r) logits
    """

    def __init__(self, text_dim: int = 300, video_ch: int = 834):
        super().__init__()
        self.filters = DynamicFilterGenerator(
            text_dim=text_dim, ch_32=video_ch, ch_128=256, ch_512=128
        )
        self.deconv1 = DeconvBlock(in_ch=video_ch, out_ch=256)
        self.deconv2 = DeconvBlock(in_ch=256, out_ch=128)

    def forward(self, T: Tensor, video_feat: Tensor) -> dict[int, Tensor]:
        f = self.filters(T)

        v32 = video_feat
        s32 = dynamic_conv(v32, f[32])

        v128 = self.deconv1(v32)
        s128 = dynamic_conv(v128, f[128])

        v512 = self.deconv2(v128)
        s512 = dynamic_conv(v512, f[512])

        return {32: s32, 128: s128, 512: s512}
