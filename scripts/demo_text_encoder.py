"""Demo: Textual Encoder sanity check."""

import logging
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from data.preprocessing import load_word2vec, sentence_to_embedding
from model.layers import TextualEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = Config()
    device = cfg.device if torch.cuda.is_available() else "cpu"

    word2vec = load_word2vec(cfg.data.word2vec_model)
    encoder = TextualEncoder(
        embed_dim=cfg.model.embed_dim,
        kernel_size=cfg.model.text_kernel_size,
    ).to(device)
    encoder.eval()

    test_sentences = [
        "a man in dark suit standing on the back",
        "car jumping into the water",
        "small white fluffy puppy biting the cat",
        "woman in green dress is walking on the street",
    ]

    logger.info("Textual Encoder — Sanity Check")
    for sent in test_sentences:
        emb = sentence_to_embedding(sent, word2vec, cfg.model.max_sentence_len)
        emb_batch = emb.unsqueeze(0).to(device)

        with torch.no_grad():
            feat = encoder(emb_batch)

        tokens = sent.lower().split()
        in_vocab = sum(1 for t in tokens if t in word2vec)
        logger.info(
            "'%s' — tokens=%d (%d in vocab) → %s",
            sent,
            len(tokens),
            in_vocab,
            tuple(feat.shape),
        )

    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info("Learnable parameters: %s", f"{n_params:,}")


if __name__ == "__main__":
    main()
