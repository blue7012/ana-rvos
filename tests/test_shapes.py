"""Tensor shape unit tests — run before training."""

import torch

from model.layers import TextualEncoder
from model.modules import Decoder
from training.loss import make_multiscale_targets, multiresolution_loss


def test_textual_encoder_shape():
    B, max_len, D = 2, 20, 300
    encoder = TextualEncoder(embed_dim=D, kernel_size=2)
    x = torch.randn(B, max_len, D)
    out = encoder(x)
    assert out.shape == (B, D), f"got {out.shape}"


def test_decoder_shapes():
    B, text_dim, video_ch = 2, 300, 834
    decoder = Decoder(text_dim=text_dim, video_ch=video_ch)
    decoder.eval()

    dummy_text = torch.randn(B, text_dim)
    dummy_video = torch.randn(B, video_ch, 32, 32)

    with torch.no_grad():
        responses = decoder(dummy_text, dummy_video)

    expected = {
        32: (B, 1, 32, 32),
        128: (B, 1, 128, 128),
        512: (B, 1, 512, 512),
    }
    for r, shape in expected.items():
        assert tuple(responses[r].shape) == shape, f"res {r}: got {responses[r].shape}"


def test_multiresolution_loss():
    B = 2
    decoder = Decoder()
    dummy_text = torch.randn(B, 300)
    dummy_video = torch.randn(B, 834, 32, 32)

    with torch.no_grad():
        responses = decoder(dummy_text, dummy_video)

    fake_mask = (torch.rand(B, 1, 512, 512) > 0.5).float()
    targets = make_multiscale_targets(fake_mask)
    loss = multiresolution_loss(responses, targets)

    assert loss.ndim == 0
    assert loss.item() > 0
