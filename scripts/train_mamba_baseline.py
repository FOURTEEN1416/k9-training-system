"""Mamba 基线模型训练脚本 - Phase 4.2

Owner: ML 开发
Phase: 4.2
依据: VideoMamba (ECCV 2024) 架构简化实现

功能:
- 使用与 ST-GCN+BC 完全相同的合成数据管线训练 Mamba 基线
- 输出: 训练历史 + 最佳模型权重
- 对比基线: ST-GCN+BC (synthetic baseline 46.97% / 边界 F1 58.45%)

用法:
    python scripts/train_mamba_baseline.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.mamba_sequence import MambaSequenceBaseline
from backend.ml.behavior.stgcn_bc.dataset import (
    make_synthetic_dataset,
    save_synthetic_dataset,
)


class MambaDataset(Dataset):
    """Mamba 模型数据集适配器.

    输入: keypoints (T, 24, 3) -> 转置为 (T, J, 3)
    """

    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        kpt = s["keypoints"]  # (T, 24, 3)
        label = s["label"]
        # 转 float32 tensor
        x = torch.tensor(kpt, dtype=torch.float32)  # (T, 24, 3)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_dir: Path,
) -> Dict:
    """训练循环."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": -1,
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += y.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        train_acc = train_correct / train_total if train_total else 0
        val_acc = val_correct / val_total if val_total else 0
        avg_train_loss = train_loss / train_total if train_total else 0
        avg_val_loss = val_loss / val_total if val_total else 0

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        # Save best model
        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), save_dir / "best.pt")

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # Save last model
    torch.save(model.state_dict(), save_dir / "last.pt")

    # Save history
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    return history


def main():
    parser = argparse.ArgumentParser(description="Mamba 基线训练")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--samples-per-class", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="runs/mamba_baseline")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    save_dir = PROJECT_ROOT / args.output
    save_dir.mkdir(parents=True, exist_ok=True)

    # 生成合成数据 (与 ST-GCN+BC 相同管线)
    print("生成合成数据...")
    all_samples = make_synthetic_dataset(
        samples_per_class=args.samples_per_class,
        T=30,
        noise_std=0.05,
        seed=args.seed,
    )
    print(f"总样本: {len(all_samples)}")

    # 划分 train/val (80/20)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(all_samples))
    n_val = int(len(all_samples) * 0.2)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    print(f"训练: {len(train_samples)}, 验证: {len(val_samples)}")

    # 数据集
    train_ds = MambaDataset(train_samples)
    val_ds = MambaDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 模型
    model = MambaSequenceBaseline(
        num_joints=24,
        num_classes=22,
        d_model=args.d_model,
        n_layers=8,
        d_state=16,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # 训练
    start = time.time()
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=save_dir,
    )
    elapsed = time.time() - start

    # 汇总
    summary = {
        "model": "MambaSequenceBaseline",
        "params": n_params,
        "epochs": args.epochs,
        "best_val_acc": history["best_val_acc"],
        "best_epoch": history["best_epoch"],
        "train_time_sec": round(elapsed, 1),
        "device": str(device),
        "d_model": args.d_model,
        "samples_per_class": args.samples_per_class,
    }
    print("\n=== 训练完成 ===")
    print(f"最佳验证准确率: {history['best_val_acc']:.4f} (epoch {history['best_epoch']+1})")
    print(f"参数量: {n_params:,}")
    print(f"训练耗时: {elapsed:.1f}s")

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"结果保存至: {save_dir}")


if __name__ == "__main__":
    main()
