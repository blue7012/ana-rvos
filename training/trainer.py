"""Iteration-based training loop — Paper §3.4."""

from __future__ import annotations

import logging
import math
import os
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from training.loss import make_multiscale_targets, multiresolution_loss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Iteration-based trainer matching Paper §3.4 (15 000 total iterations).

    One 'iteration' = one gradient step on one mini-batch.
    The DataLoader is restarted when exhausted so training always reaches
    total_iters regardless of dataset size.

    Checkpoints are saved every save_every iterations and at the end.
    """

    def train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        cfg: Config,
    ) -> None:
        """
        Args:
            model:        ActorActionVideoSegmentation
            train_loader: DataLoader yielding (video, embedding, mask)
            val_loader:   DataLoader for validation loss tracking
            optimizer:    Adam (from optimizer.py)
            scheduler:    StepLR stepped per iteration (from optimizer.py)
            cfg:          top-level Config
        """
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

        tcfg = cfg.train

        # ---- Sanity check on first batch ----
        # For logistic loss log(1+exp(-S*Y)) at init, S≈0 → expected loss ≈ ln(2)≈0.693
        self._sanity_check(model, train_loader, device, cfg)

        model.train()
        step = 0
        running_loss = 0.0
        best_val_loss = math.inf

        pbar = tqdm(total=tcfg.total_iters, unit="iter", desc="Training")

        # Restart the dataloader whenever exhausted
        while step < tcfg.total_iters:
            for video, embedding, mask in train_loader:
                if step >= tcfg.total_iters:
                    break

                video = video.to(device)         # (B, 3, N, H, W)
                embedding = embedding.to(device)  # (B, max_len, 300)
                mask = mask.to(device)            # (B, 1, 512, 512) in {0,1}

                # L = sum_r alpha_r * mean_ij log(1 + exp(-S_r_ij * Y_r_ij))  §3.4
                responses: Dict[int, Tensor] = model(video, embedding)
                targets = make_multiscale_targets(mask)
                loss = multiresolution_loss(responses, targets, cfg.loss.alphas)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()   # per-iteration — divides lr at exactly step 5k/10k

                step += 1
                running_loss += loss.item()

                # ---- Logging ----
                if step % tcfg.log_every == 0:
                    avg = running_loss / tcfg.log_every
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.2e}")
                    logger.info("step %d  loss=%.4f  lr=%.2e", step, avg, lr)
                    running_loss = 0.0

                # ---- Checkpoint ----
                if step % tcfg.save_every == 0:
                    val_loss = self._eval_loss(model, val_loader, device, cfg)
                    self._save_checkpoint(model, optimizer, scheduler, step, val_loss, cfg)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self._save_checkpoint(
                            model, optimizer, scheduler, step, val_loss, cfg,
                            filename="best_model.pt",
                        )
                    model.train()

                pbar.update(1)

        pbar.close()

        # Final checkpoint
        val_loss = self._eval_loss(model, val_loader, device, cfg)
        self._save_checkpoint(
            model, optimizer, scheduler, step, val_loss, cfg,
            filename="final_model.pt",
        )
        logger.info("Training complete — final val_loss=%.4f", val_loss)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sanity_check(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        cfg: Config,
    ) -> None:
        """
        One-batch sanity check before training.
        At init, dynamic filters ≈ 0 → S≈0 → expected loss ≈ ln(2) ≈ 0.693.
        """
        model.eval()
        video, embedding, mask = next(iter(loader))
        video = video.to(device)
        embedding = embedding.to(device)
        mask = mask.to(device)

        responses = model(video, embedding)
        targets = make_multiscale_targets(mask)
        loss = multiresolution_loss(responses, targets, cfg.loss.alphas)

        expected = 0.693   # ln(2) for balanced random logits
        logger.info(
            "Sanity check — loss=%.4f  (expected ≈ %.3f for random init)",
            loss.item(),
            expected,
        )
        model.train()

    @torch.no_grad()
    def _eval_loss(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        cfg: Config,
    ) -> float:
        """
        Average validation loss over all val batches.
        """
        model.eval()
        total, count = 0.0, 0
        for video, embedding, mask in tqdm(loader, desc="Validating", leave=False):
            video = video.to(device)
            embedding = embedding.to(device)
            mask = mask.to(device)
            responses = model(video, embedding)
            targets = make_multiscale_targets(mask)
            loss = multiresolution_loss(responses, targets, cfg.loss.alphas)
            total += loss.item()
            count += 1
        return total / max(count, 1)

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        step: int,
        val_loss: float,
        cfg: Config,
        filename: str | None = None,
    ) -> None:
        fname = filename or f"checkpoint_step{step:06d}.pt"
        path = os.path.join(cfg.train.checkpoint_dir, fname)
        torch.save(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "cfg": cfg,
            },
            path,
        )
        logger.info("Saved checkpoint → %s  (val_loss=%.4f)", path, val_loss)
