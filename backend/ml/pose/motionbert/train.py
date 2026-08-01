"""MotionBERT 17→24 关键点 InterPet4D 微调脚本.

Owner: ML 开发
Phase: 3.3c

设计:
    - 加载预训练 MB_lite 权重 → 24 关键点 DSTformer
    - 在 InterPet4D 合成 2D-3D 配对上微调
    - 损失: MPJPE + velocity + scale（num_joints 无关）
    - limb_var/limb_gt/angle/angle_velocity 损失禁用（H36M 17 关节硬编码，不适配 24）
    - 每 epoch 验证 + 保存最佳 checkpoint

用法:
    # 完整训练
    python -m backend.ml.pose.motionbert.train --epochs 30

    # 快速测试（少量数据）
    python -m backend.ml.pose.motionbert.train --max-clips 10 --epochs 2

    # 指定预训练权重
    python -m backend.ml.pose.motionbert.train \\
        --pretrained data/models/MB_lite_pose3d_ft.bin
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from backend.ml.pose.motionbert.config import MotionBERTConfig, load_config
from backend.ml.pose.motionbert.dataset import build_datasets
from backend.ml.pose.motionbert.model import (
    DSTformerWrapper,
    build_model_from_config,
    load_pretrained_weights,
)

# MotionBERT 损失函数（num_joints 无关的部分）
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "external" / "MotionBERT"))
from lib.model.loss import loss_mpjpe, loss_velocity, n_mpjpe  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: MotionBERTConfig,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """计算多任务损失.

    只使用 num_joints 无关的损失:
        - loss_mpjpe: 主损失，关节位置误差
        - loss_velocity: 时序一致性
        - n_mpjpe: 尺度归一化 MPJPE

    禁用（H36M 17 关节硬编码）:
        - loss_limb_var, loss_limb_gt, loss_angle, loss_angle_velocity
    """
    loss_3d_pos = loss_mpjpe(pred, target)
    loss_3d_scale = n_mpjpe(pred, target)
    loss_3d_velocity = loss_velocity(pred, target)

    loss_total = (
        config.lambda_3d_pos * loss_3d_pos
        + config.lambda_scale * loss_3d_scale
        + config.lambda_3d_velocity * loss_3d_velocity
    )

    loss_dict = {
        "3d_pos": float(loss_3d_pos.item()),
        "3d_scale": float(loss_3d_scale.item()),
        "3d_velocity": float(loss_3d_velocity.item()),
        "total": float(loss_total.item()),
    }
    return loss_total, loss_dict


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    config: MotionBERTConfig,
    device: str,
) -> tuple[float, Dict[str, float]]:
    """验证集评估，返回平均 MPJPE (mm，归一化空间) 和损失字典."""
    model.eval()
    total_mpjpe = 0.0
    total_loss = {k: 0.0 for k in ["3d_pos", "3d_scale", "3d_velocity", "total"]}
    num_batches = 0

    for batch_2d, batch_3d in loader:
        batch_2d = batch_2d.to(device)
        batch_3d = batch_3d.to(device)

        # 根关节中心化 GT
        if config.rootrel:
            batch_3d = batch_3d - batch_3d[..., 0:1, :]

        pred = model(batch_2d)
        loss, loss_dict = compute_loss(pred, batch_3d, config)

        # MPJPE (归一化空间)
        mpjpe = torch.mean(torch.norm(pred - batch_3d, dim=-1))
        total_mpjpe += float(mpjpe.item())
        for k, v in loss_dict.items():
            total_loss[k] += v
        num_batches += 1

    return total_mpjpe / max(num_batches, 1), {
        k: v / max(num_batches, 1) for k, v in total_loss.items()
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    config: MotionBERTConfig,
    device: str,
    epoch: int = 0,
) -> Dict[str, float]:
    """单 epoch 训练."""
    model.train()
    total_loss = {k: 0.0 for k in ["3d_pos", "3d_scale", "3d_velocity", "total"]}
    num_batches = 0
    log_interval = max(1, len(loader) // 10)  # 每 10% 打印一次

    for batch_idx, (batch_2d, batch_3d) in enumerate(loader):
        batch_2d = batch_2d.to(device)
        batch_3d = batch_3d.to(device)

        # 根关节中心化 GT
        if config.rootrel:
            batch_3d = batch_3d - batch_3d[..., 0:1, :]

        pred = model(batch_2d)
        loss, loss_dict = compute_loss(pred, batch_3d, config)

        # NaN 检查（防御性，理论上数据已过滤）
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(
                f"[Epoch {epoch}] Batch {batch_idx}: loss=nan/inf, skip "
                f"(pred range=[{pred.min().item():.3f}, {pred.max().item():.3f}])"
            )
            continue

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（防止梯度爆炸导致 NaN 传播）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in loss_dict.items():
            total_loss[k] += v
        num_batches += 1

        if batch_idx % log_interval == 0:
            logger.info(
                f"[Epoch {epoch}] Batch {batch_idx}/{len(loader)} "
                f"({batch_idx/len(loader)*100:.0f}%) "
                f"loss={loss_dict['total']:.4f}"
            )

    return {k: v / max(num_batches, 1) for k, v in total_loss.items()}


def train(
    config: MotionBERTConfig,
    max_clips: int | None = None,
    device: str | None = None,
) -> Dict:
    """完整训练流程.

    Returns:
        训练结果字典（含最佳 checkpoint 路径、最终 MPJPE 等）
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config.ensure_dirs()

    # 启用 cudnn benchmark 加速训练（固定输入 shape 时显著加速）
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    logger.info(f"[train] 设备: {device}, 关节数: {config.num_joints}")

    # 1. 构建数据
    datasets = build_datasets(config, max_clips=max_clips)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows 兼容
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    # 2. 构建模型 + 加载预训练权重
    model = build_model_from_config(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"[train] 模型参数量: {param_count:,}")

    if Path(config.pretrained_path).exists():
        matched, total, discarded = load_pretrained_weights(
            model, config.pretrained_path
        )
        logger.info(
            f"[train] 预训练权重迁移: {matched}/{total} 层 "
            f"(丢弃 {len(discarded)} 层, 预期 pos_embed 被丢弃)"
        )
    else:
        logger.warning(
            f"[train] 预训练权重不存在: {config.pretrained_path}, 从头训练"
        )

    # 3. 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # 4. 训练循环
    best_mpjpe = float("inf")
    best_epoch = -1
    best_path = Path(config.checkpoint_dir) / "best_epoch.bin"
    latest_path = Path(config.checkpoint_dir) / "latest_epoch.bin"
    history = []

    for epoch in range(config.epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, config, device, epoch + 1)
        val_mpjpe, val_loss = evaluate(model, val_loader, config, device)
        epoch_time = time.time() - start_time

        # 学习率衰减
        for pg in optimizer.param_groups:
            pg["lr"] *= config.lr_decay

        logger.info(
            f"[Epoch {epoch+1}/{config.epochs}] "
            f"train_loss={train_loss['total']:.4f} "
            f"val_mpjpe={val_mpjpe:.4f} "
            f"val_loss={val_loss['total']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"time={epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_mpjpe": val_mpjpe,
            "val_loss": val_loss,
            "time_sec": epoch_time,
        })

        # 保存最佳
        if val_mpjpe < best_mpjpe:
            best_mpjpe = val_mpjpe
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model_pos": model.state_dict(),
                "val_mpjpe": val_mpjpe,
                "config": config.to_dict(),
            }, best_path)
            logger.info(f"[train] 新最佳 MPJPE={val_mpjpe:.4f}, 保存到 {best_path}")

        # 保存最新
        torch.save({
            "epoch": epoch + 1,
            "model_pos": model.state_dict(),
            "val_mpjpe": val_mpjpe,
            "config": config.to_dict(),
        }, latest_path)

    logger.info(
        f"[train] 训练完成: 最佳 epoch={best_epoch}, MPJPE={best_mpjpe:.4f}"
    )

    return {
        "best_epoch": best_epoch,
        "best_mpjpe": best_mpjpe,
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
        "history": history,
        "param_count": param_count,
        "num_joints": config.num_joints,
    }


def main():
    parser = argparse.ArgumentParser(description="MotionBERT 17→24 InterPet4D 微调")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 epochs")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=None, help="最大 clip 数（测试用）")
    parser.add_argument("--pretrained", type=str, default=None, help="预训练权重路径")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    # 强制 flush（避免 Tee-Object 缓冲）
    for handler in logging.getLogger().handlers:
        handler.flush = lambda: sys.stdout.flush()

    config = load_config(args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.pretrained is not None:
        config.pretrained_path = args.pretrained
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir

    result = train(config, max_clips=args.max_clips)
    print(f"\n训练结果: {result['best_epoch']=}, {result['best_mpjpe']:.4f}")
    print(f"最佳 checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
