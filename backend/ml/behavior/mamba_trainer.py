"""Mamba 序列模型训练器（Phase 4.2）.

Owner: ML 开发
Phase: 4.2
依据: dev-docs/stages/phase-4-transformer-mamba.md

特性:
    1. 标准训练循环: EPOCHS × (train + validate)
    2. 混合精度训练: torch.cuda.amp（GPU 可用时自动启用）
    3. 学习率调度: CosineAnnealingLR + warmup
    4. 检查点保存: best_val_acc + last_epoch
    5. 早停: patience 轮无提升则停止
    6. 日志: JSON 格式训练历史

注意:
    - Mamba 模型仅输出分类 logits（无边界头），损失函数为 CrossEntropyLoss
    - 复用 stgcn_bc 的合成数据生成（make_synthetic_dataset）
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from backend.ml.behavior.mamba_sequence import MambaSequenceBaseline
from backend.ml.behavior.stgcn_bc.labels import NUM_BEHAVIORS

logger = logging.getLogger(__name__)


@dataclass
class MambaTrainConfig:
    """训练配置."""

    # 优化器
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)

    # 训练
    epochs: int = 50
    batch_size: int = 32
    num_workers: int = 0
    val_interval: int = 1
    save_interval: int = 5

    # 学习率调度
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 5

    # 早停
    early_stopping: bool = True
    patience: int = 15

    # 混合精度
    use_amp: bool = True

    # 设备
    device: str = "auto"

    # 梯度裁剪
    grad_clip: float = 1.0

    # 模型参数
    d_model: int = 128
    dropout: float = 0.1

    # 输出
    output_dir: str = "runs/mamba_synthetic"


@dataclass
class MambaEpochMetrics:
    """单轮训练指标."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    duration_sec: float


class MambaDataset(torch.utils.data.Dataset):
    """Mamba 训练数据集（复用合成数据格式）."""

    def __init__(
        self,
        samples: List[Dict],
        T: int = 30,
        augment: bool = False,
    ):
        self.samples = samples
        self.T = T
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        kpt = sample["keypoints"].copy()  # (T_orig, 24, 3) numpy
        label = sample["label"]

        T_orig = kpt.shape[0]

        # 时间裁剪/补齐到固定 T
        if T_orig >= self.T:
            if self.augment:
                start = np.random.randint(0, T_orig - self.T + 1)
            else:
                start = (T_orig - self.T) // 2
            kpt = kpt[start:start + self.T]
        else:
            pad_len = self.T - T_orig
            kpt = np.concatenate(
                [kpt, np.tile(kpt[-1:], (pad_len, 1, 1))], axis=0
            )

        # 数据增强
        if self.augment:
            if np.random.rand() < 0.5:
                kpt[..., 0] = -kpt[..., 0]  # 水平翻转 (x → -x)
            scale = 1.0 + np.random.uniform(-0.1, 0.1)
            kpt[..., :3] = kpt[..., :3] * scale

        # 归一化（numpy 版本）
        kpt = self._normalize(kpt)

        return {
            "keypoints": torch.from_numpy(kpt).float(),
            "label": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def _normalize(kpt: np.ndarray) -> np.ndarray:
        """按 withers 中心 + 体长尺度归一化（numpy 实现）."""
        center = kpt[:, 22:23, :].mean(axis=0, keepdims=True)  # (1, 1, 3)
        kpt = kpt - center
        bone_ref = kpt[:, 22, :2] - kpt[:, 12, :2]  # (T, 2)
        bone_len = np.linalg.norm(bone_ref, axis=-1).mean()
        if bone_len < 1e-6:
            bone_len = 1.0
        kpt[..., :2] = kpt[..., :2] / bone_len
        return kpt


def mamba_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """批处理 collate 函数."""
    return {
        "keypoints": torch.stack([b["keypoints"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
    }


class MambaTrainer:
    """Mamba 训练器.

    Args:
        model: MambaSequenceBaseline 模型实例
        train_dataset: 训练集
        val_dataset: 验证集
        config: 训练配置
    """

    def __init__(
        self,
        model: MambaSequenceBaseline,
        train_dataset: MambaDataset,
        val_dataset: MambaDataset,
        config: Optional[MambaTrainConfig] = None,
    ):
        self.model = model
        self.config = config or MambaTrainConfig()
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
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # 损失函数（仅分类，无边界头）
        self.criterion = nn.CrossEntropyLoss()

        # DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=mamba_collate_fn,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=mamba_collate_fn,
            pin_memory=self.device.type == "cuda",
        )

        # 训练历史
        self.history: List[MambaEpochMetrics] = []
        self.best_val_acc: float = 0.0
        self.best_epoch: int = -1
        self.no_improve_count: int = 0

    def fit(self, epochs: Optional[int] = None) -> Dict:
        """完整训练流程."""
        total_epochs = epochs or self.config.epochs
        logger.info(
            f"开始 Mamba 训练: {total_epochs} epochs, device={self.device}, "
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
            epoch_metric = MambaEpochMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
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
            if self.config.early_stopping and self.no_improve_count >= self.config.patience:
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
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "use_amp": self.use_amp,
        }
        logger.info(f"Mamba 训练完成: {summary}")
        return summary

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """单轮训练."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.train_loader:
            keypoints = batch["keypoints"].to(self.device)  # (B, T, 24, 3)
            labels = batch["labels"].to(self.device)  # (B,)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    cls_logits = self.model(keypoints)  # (B, num_classes)
                    loss = self.criterion(cls_logits, labels)
                self.scaler.scale(loss).backward()
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                cls_logits = self.model(keypoints)
                loss = self.criterion(cls_logits, labels)
                loss.backward()
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )
                self.optimizer.step()

            total_loss += loss.item() * keypoints.size(0)
            preds = cls_logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += keypoints.size(0)

        return {
            "loss": total_loss / max(total_samples, 1),
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

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    cls_logits = self.model(keypoints)
                    loss = self.criterion(cls_logits, labels)
            else:
                cls_logits = self.model(keypoints)
                loss = self.criterion(cls_logits, labels)

            total_loss += loss.item() * keypoints.size(0)
            preds = cls_logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += keypoints.size(0)

        return {
            "val_loss": total_loss / max(total_samples, 1),
            "val_acc": total_correct / max(total_samples, 1),
        }

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False) -> None:
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
                "d_model": self.config.d_model,
                "dropout": self.config.dropout,
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
    "MambaTrainConfig",
    "MambaDataset",
    "MambaTrainer",
]
