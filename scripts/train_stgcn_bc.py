"""ST-GCN+BC 训练入口脚本（Phase 3.1d）.

Owner: ML 开发
Phase: 3.1d
依据: dev-docs/stages/phase-3.md §3.1d

用法:
    # 真实数据训练
    python scripts/train_stgcn_bc.py \
        --train-pkl data/stgcn_bc/train.pkl \
        --val-pkl data/stgcn_bc/val.pkl \
        --output-dir runs/stgcn_bc \
        --epochs 100 --batch-size 32 --lr 1e-3

    # 合成数据 baseline 训练（无真实数据时）
    python scripts/train_stgcn_bc.py \
        --synthetic \
        --samples-per-class 20 \
        --output-dir runs/stgcn_bc_synthetic \
        --epochs 50

    # 从检查点继续训练
    python scripts/train_stgcn_bc.py \
        --train-pkl data/stgcn_bc/train.pkl \
        --val-pkl data/stgcn_bc/val.pkl \
        --resume runs/stgcn_bc/last.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.stgcn_bc import (
    STGCNBCDataset,
    STGCNBCTrainer,
    TrainConfig,
    build_stgcn_bc,
    make_synthetic_dataset,
    save_synthetic_dataset,
)
from backend.ml.behavior.stgcn_bc.labels import NUM_BEHAVIORS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ST-GCN+BC 训练入口（Phase 3.1d）"
    )

    # 数据源
    data_group = parser.add_argument_group("数据源")
    data_group.add_argument(
        "--train-pkl", type=str, default=None,
        help="训练集 pyskl pickle 路径"
    )
    data_group.add_argument(
        "--val-pkl", type=str, default=None,
        help="验证集 pyskl pickle 路径"
    )
    data_group.add_argument(
        "--synthetic", action="store_true",
        help="使用合成数据（无真实数据时，仅 baseline 验证）"
    )
    data_group.add_argument(
        "--samples-per-class", type=int, default=20,
        help="合成数据每类样本数（仅 --synthetic 时生效）"
    )
    data_group.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="合成数据验证集比例（仅 --synthetic 时生效）"
    )
    data_group.add_argument(
        "--save-synthetic", type=str, default=None,
        help="保存生成的合成数据到指定 pickle 路径"
    )

    # 模型
    model_group = parser.add_argument_group("模型")
    model_group.add_argument(
        "--in-channels", type=int, default=3,
        help="输入通道数（3D=3, 2D+conf=3）"
    )
    model_group.add_argument(
        "--base-channels", type=int, default=64,
        help="STGCN 基础通道数"
    )
    model_group.add_argument(
        "--num-stages", type=int, default=10,
        help="STGCN block 数量"
    )
    model_group.add_argument(
        "--boundary-weight", type=float, default=0.3,
        help="边界损失权重"
    )

    # 训练
    train_group = parser.add_argument_group("训练")
    train_group.add_argument(
        "--epochs", type=int, default=100,
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
        "--patience", type=int, default=20,
        help="早停耐心值"
    )
    train_group.add_argument(
        "--T", type=int, default=30,
        help="每个样本帧数（时间裁剪/补齐）"
    )
    train_group.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader 工作进程数（Windows 默认 0）"
    )

    # 输出
    out_group = parser.add_argument_group("输出")
    out_group.add_argument(
        "--output-dir", type=str, default="runs/stgcn_bc",
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
    if args.synthetic:
        print(f"[合成数据] 生成 22 类 × {args.samples_per_class} 样本/类...")
        all_samples = make_synthetic_dataset(
            samples_per_class=args.samples_per_class,
            T=args.T,
            seed=42,
        )
        # 划分训练/验证
        import random
        random.seed(42)
        random.shuffle(all_samples)
        val_size = int(len(all_samples) * args.val_ratio)
        val_samples = all_samples[:val_size]
        train_samples = all_samples[val_size:]

        if args.save_synthetic:
            save_synthetic_dataset(all_samples, args.save_synthetic)
            print(f"[合成数据] 已保存到 {args.save_synthetic}")

        print(
            f"[合成数据] 训练集 {len(train_samples)} 样本, "
            f"验证集 {len(val_samples)} 样本"
        )
    else:
        if not args.train_pkl or not args.val_pkl:
            raise ValueError(
                "非合成模式必须提供 --train-pkl 和 --val-pkl，"
                "或使用 --synthetic 启用合成数据"
            )
        print(f"[真实数据] 加载训练集 {args.train_pkl}...")
        train_samples = None  # STGCNBCDataset 内部加载
        val_samples = None

    train_ds = STGCNBCDataset(
        samples=train_samples,
        pkl_path=args.train_pkl if not args.synthetic else None,
        T=args.T,
        augment=True,
        normalize=True,
    )
    val_ds = STGCNBCDataset(
        samples=val_samples,
        pkl_path=args.val_pkl if not args.synthetic else None,
        T=args.T,
        augment=False,
        normalize=True,
    )
    return train_ds, val_ds


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("train_stgcn_bc")

    logger.info("=" * 60)
    logger.info("ST-GCN+BC 训练 (Phase 3.1d)")
    logger.info("=" * 60)

    # 1. 准备数据
    train_ds, val_ds = prepare_datasets(args)
    logger.info(f"训练集: {len(train_ds)} 样本, 验证集: {len(val_ds)} 样本")

    # 2. 构建模型
    model = build_stgcn_bc(
        in_channels=args.in_channels,
        num_classes=NUM_BEHAVIORS,  # 22
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        boundary_weight=args.boundary_weight,
    )

    # 参数量
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"模型参数: {n_params:,} 总数, {n_trainable:,} 可训练 "
        f"(base_channels={args.base_channels}, num_stages={args.num_stages})"
    )

    # 3. 配置训练器
    config = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        use_amp=not args.no_amp,
        device=args.device,
        output_dir=args.output_dir,
    )

    trainer = STGCNBCTrainer(model, train_ds, val_ds, config=config)

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


if __name__ == "__main__":
    main()
