"""ST-GCN+BC 训练器（PyTorch 标准训练循环）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1d
依据: dev-docs/stages/phase-3.md §3.1d

特性:
    1. 标准训练循环: EPOCHS × (train + validate)
    2. 混合精度训练: torch.cuda.amp（GPU 可用时自动启用）
    3. 学习率调度: CosineAnnealingLR + warmup
    4. 检查点保存: best_val_acc + last_epoch
    5. 早停: patience 轮无提升则停止
    6. 日志: JSON 格式训练历史（可读 + 可视化）

用法:
    from backend.ml.behavior.stgcn_bc import build_stgcn_bc
    from backend.ml.behavior.stgcn_bc.trainer import STGCNBCTrainer
    from backend.ml.behavior.stgcn_bc.dataset import STGCNBCDataset, collate_fn

    model = build_stgcn_bc(in_channels=3, num_classes=22)
    train_ds = STGCNBCDataset(pkl_path="train.pkl", augment=True)
    val_ds = STGCNBCDataset(pkl_path="val.pkl", augment=False)
    trainer = STGCNBCTrainer(model, train_ds, val_ds, output_dir="runs/stgcn_bc")
    trainer.fit(epochs=100)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from backend.ml.behavior.stgcn_bc.dataset import STGCNBCDataset, collate_fn
from backend.ml.behavior.stgcn_bc.model import STGCNBC

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """训练配置."""

    # 优化器
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)

    # 训练
    epochs: int = 100
    batch_size: int = 32
    num_workers: int = 0  # Windows 默认 0（避免多进程问题）
    val_interval: int = 1  # 每 N epoch 验证一次
    save_interval: int = 5  # 每 N epoch 保存检查点

    # 学习率调度
    lr_scheduler: str = "cosine"  # "cosine" | "none"
    warmup_epochs: int = 5

    # 早停
    early_stopping: bool = True
    patience: int = 20  # 无提升轮数

    # 混合精度
    use_amp: bool = True  # GPU 可用时自动启用

    # 设备
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # 梯度裁剪
    grad_clip: float = 1.0

    # 输出
    output_dir: str = "runs/stgcn_bc"


@dataclass
class EpochMetrics:
    """单轮训练指标."""

    epoch: int
    train_loss: float
    train_cls_loss: float
    train_boundary_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    duration_sec: float


class STGCNBCTrainer:
    """ST-GCN+BC 训练器.

    Args:
        model: STGCNBC 模型实例
        train_dataset: 训练集
        val_dataset: 验证集
        config: 训练配置
    """

    def __init__(
        self,
        model: STGCNBC,
        train_dataset: STGCNBCDataset,
        val_dataset: STGCNBCDataset,
        config: Optional[TrainConfig] = None,
    ):
        self.model = model
        self.config = config or TrainConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设备选择
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model = self.model.to(self.device)

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )

        # 学习率调度
        if self.config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs - self.config.warmup_epochs
            )
        else:
            self.scheduler = None

        # 混合精度
        self.use_amp = (
            self.config.use_amp
            and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=self.device.type == "cuda",
        )

        # 训练历史
        self.history: List[EpochMetrics] = []
        self.best_val_acc: float = 0.0
        self.best_epoch: int = -1
        self.no_improve_count: int = 0

    def fit(self, epochs: Optional[int] = None) -> Dict:
        """完整训练流程.

        Args:
            epochs: 覆盖 config.epochs（可选）

        Returns:
            训练摘要 dict
        """
        total_epochs = epochs or self.config.epochs
        logger.info(
            f"开始训练: {total_epochs} epochs, device={self.device}, "
            f"amp={self.use_amp}, train_size={len(self.train_loader.dataset)}, "
            f"val_size={len(self.val_loader.dataset)}"
        )

        for epoch in range(1, total_epochs + 1):
            # Warmup
            if epoch <= self.config.warmup_epochs:
                warmup_lr = self.config.lr * epoch / self.config.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            t_start = time.time()
            train_metrics = self._train_one_epoch(epoch)
            t_train = time.time() - t_start

            # 验证
            if epoch % self.config.val_interval == 0:
                t_val_start = time.time()
                val_metrics = self._validate(epoch)
                t_val = time.time() - t_val_start
            else:
                val_metrics = {"val_loss": 0.0, "val_acc": 0.0}
                t_val = 0.0

            # 学习率调度
            if self.scheduler and epoch > self.config.warmup_epochs:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_metric = EpochMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                train_cls_loss=train_metrics["cls_loss"],
                train_boundary_loss=train_metrics["boundary_loss"],
                train_acc=train_metrics["acc"],
                val_loss=val_metrics["val_loss"],
                val_acc=val_metrics["val_acc"],
                lr=current_lr,
                duration_sec=t_train + t_val,
            )
            self.history.append(epoch_metric)

            logger.info(
                f"Epoch {epoch}/{total_epochs} "
                f"train_loss={epoch_metric.train_loss:.4f} "
                f"train_acc={epoch_metric.train_acc:.4f} "
                f"val_loss={epoch_metric.val_loss:.4f} "
                f"val_acc={epoch_metric.val_acc:.4f} "
                f"lr={current_lr:.2e} "
                f"time={epoch_metric.duration_sec:.1f}s"
            )

            # 保存最佳
            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self.best_epoch = epoch
                self.no_improve_count = 0
                self._save_checkpoint(epoch, val_metrics["val_acc"], is_best=True)
            else:
                self.no_improve_count += 1

            # 定期保存
            if epoch % self.config.save_interval == 0:
                self._save_checkpoint(epoch, val_metrics["val_acc"])

            # 早停
            if (
                self.config.early_stopping
                and self.no_improve_count >= self.config.patience
            ):
                logger.info(
                    f"早停触发: {self.config.patience} 轮无提升 "
                    f"(best_val_acc={self.best_val_acc:.4f} @ epoch {self.best_epoch})"
                )
                break

        # 保存训练历史
        self._save_history()

        summary = {
            "total_epochs_trained": len(self.history),
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "final_train_acc": self.history[-1].train_acc if self.history else 0.0,
            "final_val_acc": self.history[-1].val_acc if self.history else 0.0,
            "device": str(self.device),
            "use_amp": self.use_amp,
        }
        logger.info(f"训练完成: {summary}")
        return summary

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """单轮训练."""
        self.model.train()
        total_loss = 0.0
        total_cls_loss = 0.0
        total_boundary_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.train_loader:
            keypoints = batch["keypoints"].to(self.device)  # (B, T, 24, 3)
            labels = batch["labels"].to(self.device)  # (B,)
            boundaries = batch["boundaries"].to(self.device)  # (B, T)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    cls_logits, boundary_logits = self.model(keypoints)
                    loss_dict = self.model.compute_loss(
                        cls_logits, boundary_logits, labels, boundaries
                    )
                loss = loss_dict["total"]
                self.scaler.scale(loss).backward()
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                cls_logits, boundary_logits = self.model(keypoints)
                loss_dict = self.model.compute_loss(
                    cls_logits, boundary_logits, labels, boundaries
                )
                loss = loss_dict["total"]
                loss.backward()
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.optimizer.step()

            total_loss += loss.item() * keypoints.size(0)
            total_cls_loss += loss_dict["cls"].item() * keypoints.size(0)
            total_boundary_loss += loss_dict["boundary"].item() * keypoints.size(0)
            preds = cls_logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += keypoints.size(0)

        return {
            "loss": total_loss / max(total_samples, 1),
            "cls_loss": total_cls_loss / max(total_samples, 1),
            "boundary_loss": total_boundary_loss / max(total_samples, 1),
            "acc": total_correct / max(total_samples, 1),
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """验证."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.val_loader:
            keypoints = batch["keypoints"].to(self.device)
            labels = batch["labels"].to(self.device)
            boundaries = batch["boundaries"].to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    cls_logits, boundary_logits = self.model(keypoints)
                    loss_dict = self.model.compute_loss(
                        cls_logits, boundary_logits, labels, boundaries
                    )
            else:
                cls_logits, boundary_logits = self.model(keypoints)
                loss_dict = self.model.compute_loss(
                    cls_logits, boundary_logits, labels, boundaries
                )

            total_loss += loss_dict["total"].item() * keypoints.size(0)
            preds = cls_logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += keypoints.size(0)

        return {
            "val_loss": total_loss / max(total_samples, 1),
            "val_acc": total_correct / max(total_samples, 1),
        }

    def _save_checkpoint(
        self, epoch: int, val_acc: float, is_best: bool = False
    ) -> None:
        """保存检查点."""
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_acc": val_acc,
            "best_val_acc": self.best_val_acc,
            "config": {
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
            },
        }
        if self.scheduler is not None:
            ckpt["scheduler_state_dict"] = self.scheduler.state_dict()

        # 保存 last
        last_path = self.output_dir / "last.pt"
        torch.save(ckpt, last_path)

        # 保存 best
        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(ckpt, best_path)

    def _save_history(self) -> None:
        """保存训练历史为 JSON."""
        history_path = self.output_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "epoch": m.epoch,
                        "train_loss": m.train_loss,
                        "train_cls_loss": m.train_cls_loss,
                        "train_boundary_loss": m.train_boundary_loss,
                        "train_acc": m.train_acc,
                        "val_loss": m.val_loss,
                        "val_acc": m.val_acc,
                        "lr": m.lr,
                        "duration_sec": m.duration_sec,
                    }
                    for m in self.history
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )

    def load_checkpoint(self, path: str) -> Dict:
        """加载检查点."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        return ckpt


__all__ = [
    "TrainConfig",
    "EpochMetrics",
    "STGCNBCTrainer",
]
