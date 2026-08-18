"""Mamba 序列模型训练入口脚本（Phase 4.2）.

Owner: ML 开发
Phase: 4.2
依据: dev-docs/stages/phase-4-transformer-mamba.md

用法:
    # 合成数据 baseline 训练
    python scripts/train_mamba.py \
        --synthetic \
        --samples-per-class 20 \
        --output-dir runs/mamba_synthetic \
        --epochs 50

    # 从检查点继续训练
    python scripts/train_mamba.py \
        --synthetic \
        --resume runs/mamba_synthetic/last.pt

    # 自定义模型参数
    python scripts/train_mamba.py \
        --synthetic \
        --d-model 256 \
        --dropout 0.2 \
        --epochs 100
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.mamba_sequence import get_model
from backend.ml.behavior.stgcn_bc.labels import NUM_BEHAVIORS
from backend.ml.behavior.mamba_trainer import (
    MambaTrainer,
    MambaTrainConfig,
    MambaDataset,
)
from backend.ml.behavior.stgcn_bc.dataset import make_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mamba 序列模型训练入口（Phase 4.2）"
    )

    # 数据源
    data_group = parser.add_argument_group("数据源")
    data_group.add_argument(
        "--synthetic", action="store_true", default=True,
        help="使用合成数据（默认）"
    )
    data_group.add_argument(
        "--samples-per-class", type=int, default=20,
        help="合成数据每类样本数"
    )
    data_group.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="合成数据验证集比例"
    )
    data_group.add_argument(
        "--T", type=int, default=30,
        help="每个样本帧数（时间裁剪/补齐）"
    )

    # 模型
    model_group = parser.add_argument_group("模型")
    model_group.add_argument(
        "--d-model", type=int, default=128,
        help="模型维度"
    )
    model_group.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout 率"
    )

    # 训练
    train_group = parser.add_argument_group("训练")
    train_group.add_argument(
        "--epochs", type=int, default=50,
        help="训练轮数"
    )
    train_group.add_argument(
        "--batch-size", type=int, default=32,
        help="批大小"
    )
    train_group.add_argument(
        "--lr", type=float, default=1e-3,
        help="学习率"
    )
    train_group.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="权重衰减"
    )
    train_group.add_argument(
        "--warmup-epochs", type=int, default=5,
        help="warmup 轮数"
    )
    train_group.add_argument(
        "--patience", type=int, default=15,
        help="早停耐心值"
    )

    # 输出
    out_group = parser.add_argument_group("输出")
    out_group.add_argument(
        "--output-dir", type=str, default="runs/mamba_synthetic",
        help="输出目录（检查点 + 历史）"
    )
    out_group.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="训练设备"
    )
    out_group.add_argument(
        "--no-amp", action="store_true",
        help="禁用混合精度训练"
    )
    out_group.add_argument(
        "--resume", type=str, default=None,
        help="从检查点恢复训练（last.pt 路径）"
    )

    # 日志
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def prepare_datasets(args: argparse.Namespace):
    """准备训练/验证数据集."""
    print(f"[合成数据] 生成 22 类 × {args.samples_per_class} 样本/类...")
    all_samples = make_synthetic_dataset(
        samples_per_class=args.samples_per_class,
        T=args.T,
        seed=42,
    )

    # 划分训练/验证
    random.seed(42)
    random.shuffle(all_samples)
    val_size = int(len(all_samples) * args.val_ratio)
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]

    print(
        f"[合成数据] 训练集 {len(train_samples)} 样本, "
        f"验证集 {len(val_samples)} 样本"
    )

    train_ds = MambaDataset(
        samples=train_samples,
        T=args.T,
        augment=True,
    )
    val_ds = MambaDataset(
        samples=val_samples,
        T=args.T,
        augment=False,
    )
    return train_ds, val_ds


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("train_mamba")

    logger.info("=" * 60)
    logger.info("Mamba 序列模型训练 (Phase 4.2)")
    logger.info("=" * 60)

    # 1. 准备数据
    train_ds, val_ds = prepare_datasets(args)
    logger.info(f"训练集: {len(train_ds)} 样本, 验证集: {len(val_ds)} 样本")

    # 2. 构建模型
    model = get_model(
        num_joints=24,
        num_classes=NUM_BEHAVIORS,
        d_model=args.d_model,
        n_layers=8,
        d_state=16,
        dropout=args.dropout,
    )

    # 参数量
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"模型参数: {n_params:,} 总数, {n_trainable:,} 可训练 "
        f"(d_model={args.d_model})"
    )

    # 3. 配置训练器
    config = MambaTrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=0,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        use_amp=not args.no_amp,
        device=args.device,
        d_model=args.d_model,
        n_layers=8,
        d_state=16,
        dropout=args.dropout,
        output_dir=args.output_dir,
    )

    trainer = MambaTrainer(model, train_ds, val_ds, config=config)

    # 4. 恢复训练
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        ckpt = trainer.load_checkpoint(args.resume)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"已加载 epoch {start_epoch - 1}，从 epoch {start_epoch} 继续")

    # 5. 训练
    summary = trainer.fit()

    # 6. 输出最终摘要
    print("\n" + "=" * 60)
    print("训练完成")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n最佳检查点: {Path(args.output_dir) / 'best.pt'}")
    print(f"训练历史: {Path(args.output_dir) / 'history.json'}")

    # 7. 对比 ST-GCN+BC 基线
    print("\n" + "-" * 60)
    print("基线对比")
    print("-" * 60)
    print(f"  ST-GCN+BC:  46.97% accuracy, 1.43M params")
    print(f"  Mamba:      {summary['best_val_acc']:.2f}% accuracy, {summary['model_params']:,} params")


if __name__ == "__main__":
    main()
