"""Single-sample inference helpers."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from data.preprocessing import load_video_frames, sentence_to_embedding


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
    num_frames: int = 16,
    frame_size: int = 512,
    max_len: int = 20,
    device: str = "cpu",
) -> Dict[int, Tensor]:
    """
    End-to-end forward pass on one video folder + sentence.

    Args:
        model: ActorActionVideoSegmentation
        video_dir: path to JPEGImages/<video_id>/
        sentence: natural language query
        word2vec_model: gensim KeyedVectors
    Returns:
        dict {32, 128, 512} of response maps (logits)
    """
    model.eval()
    model.to(device)

    video_tensor, _ = load_video_frames(video_dir, num_frames, frame_size)
    video_tensor = video_tensor.to(device)

    emb = sentence_to_embedding(sentence, word2vec_model, max_len)
    emb = emb.unsqueeze(0).to(device)

    with torch.no_grad():
        responses = model(video_tensor, emb)

    return responses
