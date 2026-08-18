"""Mamba+BC 训练入口脚本（Phase 4.2 MS-Temba 集成）.

Owner: ML 开发
Phase: 4.2

用法:
    python scripts/train_mamba_bc.py \
        --synthetic \
        --samples-per-class 20 \
        --output-dir runs/mamba_bc_synthetic \
        --epochs 50
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from backend.ml.behavior.mamba_bc import build_mamba_bc, MambaBC
from backend.ml.behavior.stgcn_bc.dataset import make_synthetic_dataset
from backend.ml.behavior.stgcn_bc.labels import NUM_BEHAVIORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mamba+BC 训练入口（Phase 4.2 MS-Temba）")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output-dir", type=str, default="runs/mamba_bc_synthetic")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def setup_logging(level: str):
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


class MambaBCDataset(Dataset):
    def __init__(self, samples, T=30, augment=False):
        self.samples = samples
        self.T = T
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        kpt = sample["keypoints"].copy()
        label = sample["label"]
        boundary = sample["boundary"].copy()
        T_orig = kpt.shape[0]

        if T_orig >= self.T:
            if self.augment:
                start = random.randint(0, T_orig - self.T)
            else:
                start = (T_orig - self.T) // 2
            kpt = kpt[start:start + self.T]
            boundary = boundary[start:start + self.T]
        else:
            pad_len = self.T - T_orig
            kpt = np.concatenate([kpt, np.tile(kpt[-1:], (pad_len, 1, 1))], axis=0)
            boundary = np.concatenate([boundary, np.zeros(pad_len)])

        if self.augment:
            if random.random() < 0.5:
                kpt[..., 0] = -kpt[..., 0]
            scale = 1.0 + random.uniform(-0.1, 0.1)
            kpt[..., :3] = kpt[..., :3] * scale

        kpt = self._normalize(kpt)
        return {
            "keypoints": torch.from_numpy(kpt).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "boundary": torch.from_numpy(boundary).float(),
        }

    @staticmethod
    def _normalize(kpt):
        center = kpt[:, 22:23, :].mean(axis=0, keepdims=True)
        kpt = kpt - center
        bone_ref = kpt[:, 22, :2] - kpt[:, 12, :2]
        bone_len = np.linalg.norm(bone_ref, axis=-1).mean()
        if bone_len < 1e-6:
            bone_len = 1.0
        kpt[..., :2] = kpt[..., :2] / bone_len
        return kpt


def collate_fn(batch):
    return {
        "keypoints": torch.stack([b["keypoints"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
        "boundaries": torch.stack([b["boundary"] for b in batch]),
    }


def main():
    import numpy as np
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("train_mamba_bc")

    logger.info("=" * 60)
    logger.info("Mamba+BC 训练 (Phase 4.2 MS-Temba 集成)")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    logger.info(f"设备: {device}, AMP: {use_amp}")

    # 1. 准备数据
    all_samples = make_synthetic_dataset(samples_per_class=args.samples_per_class, T=args.T, seed=42)
    random.seed(42)
    random.shuffle(all_samples)
    val_size = int(len(all_samples) * args.val_ratio)
    train_ds = MambaBCDataset(all_samples[val_size:], T=args.T, augment=True)
    val_ds = MambaBCDataset(all_samples[:val_size], T=args.T, augment=False)
    logger.info(f"训练集: {len(train_ds)}, 验证集: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. 构建模型
    model = build_mamba_bc(
        num_classes=NUM_BEHAVIORS,
        d_model=args.d_model,
        d_state=args.d_state,
        embed_dims=[args.d_model, args.d_model + 64, args.d_model + 128],
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数: {n_params:,}")

    # 3. 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # 4. 训练循环
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * epoch / args.warmup_epochs

        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in train_loader:
            kpts = batch["keypoints"].to(device)
            labels = batch["labels"].to(device)
            boundaries = batch["boundaries"].to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    cls_logits, bd_logits = model(kpts)
                    loss_dict = model.compute_loss(cls_logits, bd_logits, labels, boundaries)
                    loss = loss_dict["total"]
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                cls_logits, bd_logits = model(kpts)
                loss_dict = model.compute_loss(cls_logits, bd_logits, labels, boundaries)
                loss = loss_dict["total"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss += loss.item() * kpts.size(0)
            preds = cls_logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += kpts.size(0)

        train_acc = train_correct / max(train_total, 1)
        train_loss_avg = train_loss / max(train_total, 1)

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                kpts = batch["keypoints"].to(device)
                labels = batch["labels"].to(device)
                boundaries = batch["boundaries"].to(device)
                cls_logits, bd_logits = model(kpts)
                loss_dict = model.compute_loss(cls_logits, bd_logits, labels, boundaries)
                val_loss += loss_dict["total"].item() * kpts.size(0)
                preds = cls_logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += kpts.size(0)

        val_acc = val_correct / max(val_total, 1)
        val_loss_avg = val_loss / max(val_total, 1)

        if scheduler and epoch > args.warmup_epochs:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch, "train_loss": train_loss_avg, "train_acc": train_acc,
            "val_loss": val_loss_avg, "val_acc": val_acc, "lr": current_lr,
        })
        logger.info(f"Epoch {epoch}/{args.epochs} train_loss={train_loss_avg:.4f} train_acc={train_acc:.4f} val_loss={val_loss_avg:.4f} val_acc={val_acc:.4f} lr={current_lr:.2e}")

        # 保存最佳
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc, "best_val_acc": best_val_acc,
            }, output_dir / "best.pt")
        else:
            no_improve += 1

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, output_dir / "last.pt")

        if no_improve >= args.patience:
            logger.info(f"早停触发 (best_val_acc={best_val_acc:.4f} @ epoch {best_epoch})")
            break

    # 保存历史
    import json
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    print(f"  best_val_acc: {best_val_acc:.4f} @ epoch {best_epoch}")
    print(f"  模型参数: {n_params:,}")
    print(f"  ST-GCN+BC 对比: 42.27% / 1.43M params")
    print(f"  Mamba 对比: 63.18% / 21K params")
    print(f"  最佳检查点: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
