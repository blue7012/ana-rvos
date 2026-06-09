"""Entry point: train the Actor and Action Video Segmentation model."""

from __future__ import annotations

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, DataConfig, ModelConfig, TrainConfig
from data.dataset import YouTubeVOSDataset
from data.preprocessing import load_word2vec
from model.model import build_model
from training.optimizer import build_optimizer_and_scheduler
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Actor-Action Video Segmentation")
    p.add_argument("--data_root", default=None, help="Override data.youtube_vos_root")
    p.add_argument("--device", default=None, help="Override cfg.device (e.g. 'cuda:0')")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--total_iters", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg = Config()

    # Apply CLI overrides
    if args.data_root is not None:
        cfg.data.youtube_vos_root = args.data_root
    if args.device is not None:
        cfg.device = args.device
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.total_iters is not None:
        cfg.train.total_iters = args.total_iters
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.checkpoint_dir is not None:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers

    seed_everything(cfg.seed)
    logging.info("Config: %s", cfg)

    # ---- Word2Vec (loaded once, then word2vec model is discarded after embedding
    #      pre-computation inside the Dataset __init__) ----
    word2vec = load_word2vec(cfg.data.word2vec_model)

    root = cfg.data.youtube_vos_root
    logging.info("Building datasets from: %s", os.path.abspath(root))

    train_dataset = YouTubeVOSDataset(
        root=root,
        subset="train",
        word2vec_model=word2vec,
        num_frames=cfg.model.num_frames,
        frame_size=cfg.model.frame_size,
        max_sentence_len=cfg.model.max_sentence_len,
    )
    val_dataset = YouTubeVOSDataset(
        root=root,
        subset="val",
        word2vec_model=word2vec,
        num_frames=cfg.model.num_frames,
        frame_size=cfg.model.frame_size,
        max_sentence_len=cfg.model.max_sentence_len,
    )

    # word2vec no longer needed — let GC reclaim the 1.5 GB
    del word2vec

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logging.info(
        "Train: %d samples | Val: %d samples",
        len(train_dataset), len(val_dataset),
    )

    # ---- Model ----
    model = build_model(cfg, device=cfg.device)
    logging.info(
        "Total params: %s",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    # ---- Optimizer + scheduler ----
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg.train)

    # ---- Train ----
    Trainer().train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
