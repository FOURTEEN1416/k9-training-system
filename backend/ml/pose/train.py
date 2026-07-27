"""YOLO26-pose 微调训练脚本.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.0
依据:
    - dev-docs/stages/phase-1.md §1.0c
    - dev-docs/decisions/0003-phase-0-to-phase-1.md §2.4
    - dev-docs/research/RESEARCH_YOLO26_DEPLOY_DEEP.md

任务: 在 Ultralytics 官方 Dog-Pose 数据集（6773 train / 1703 val，24 关键点）上
      微调 yolo26n-pose.pt，产出 runs/pose/train/weights/best.pt。

用法:
    # 默认参数（推荐）
    python -m backend.ml.pose.train

    # 自定义参数
    python -m backend.ml.pose.train --epochs 50 --batch 8 --imgsz 640

    # 仅验证已有模型（跳过训练）
    python -m backend.ml.pose.train --eval-only --model runs/pose/train/weights/best.pt

前置条件:
    1. 数据集已下载: python -m backend.ml.pose.download_dataset
    2. GPU 可用（RTX 5060 Laptop, sm_120, CUDA 12.8）
    3. ultralytics>=8.4.84（支持 YOLO26-pose 24 关键点）

输出:
    - runs/pose/train/weights/best.pt  ← 最佳模型
    - runs/pose/train/weights/last.pt  ← 最后一轮
    - runs/pose/train/results.csv      ← 训练曲线
    - runs/pose/train/confusion_matrix.png 等
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_YAML = PROJECT_ROOT / "data" / "dog-pose.yaml"
DEFAULT_MODEL = "yolo26n-pose.pt"  # Ultralytics 自动下载权重
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs"

# Phase 1.0 微调参数（来自 phase-1.md §1.0c + ADR 0003）
# 依据: RESEARCH_YOLO26_DEPLOY_DEEP.md §1.4
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 16
DEFAULT_LR0 = 0.001
DEFAULT_DEVICE = 0  # GPU 0

# 损失权重（YOLO26-pose 官方默认）
DEFAULT_BOX_LOSS = 7.5
DEFAULT_CLS_LOSS = 0.5
DEFAULT_POSE_LOSS = 12.0
DEFAULT_KOBJ_LOSS = 1.0

# 数据增强关闭项（dog-pose 微调推荐：避免破坏关键点几何结构）
# 依据: RESEARCH_YOLO26_DEPLOY_DEEP.md §1.4
DEFAULT_SHEAR = 0.0
DEFAULT_PERSPECTIVE = 0.0
DEFAULT_FLIPUD = 0.0
DEFAULT_MIXUP = 0.0
DEFAULT_COPY_PASTE = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26-pose 微调训练（Phase 1.0）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 数据与模型
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="初始权重（yolo26n-pose.pt 自动下载，或本地 .pt 路径）",
    )
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA_YAML),
        help="数据集 yaml 路径",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="输出根目录（runs/）",
    )
    # 训练超参
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="batch size")
    parser.add_argument("--lr0", type=float, default=DEFAULT_LR0, help="初始学习率")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="设备（0 / cpu / [0,1]）")
    parser.add_argument("--workers", type=int, default=4, help="数据加载 worker 数")
    # 损失权重
    parser.add_argument("--box", type=float, default=DEFAULT_BOX_LOSS, help="box loss 权重")
    parser.add_argument("--cls", type=float, default=DEFAULT_CLS_LOSS, help="cls loss 权重")
    parser.add_argument("--pose", type=float, default=DEFAULT_POSE_LOSS, help="pose loss 权重")
    parser.add_argument("--kobj", type=float, default=DEFAULT_KOBJ_LOSS, help="kobj loss 权重")
    # 数据增强（关闭项）
    parser.add_argument("--shear", type=float, default=DEFAULT_SHEAR, help="shear 增强")
    parser.add_argument("--perspective", type=float, default=DEFAULT_PERSPECTIVE, help="perspective 增强")
    parser.add_argument("--flipud", type=float, default=DEFAULT_FLIPUD, help="上下翻转概率")
    parser.add_argument("--mixup", type=float, default=DEFAULT_MIXUP, help="mixup 概率")
    parser.add_argument("--copy-paste", type=float, default=DEFAULT_COPY_PASTE, help="copy-paste 概率")
    # 验证与调试
    parser.add_argument("--eval-only", action="store_true", help="仅验证（跳过训练）")
    parser.add_argument("--resume", action="store_true", help="从 last.pt 恢复训练")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值（0=不早停）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    return parser.parse_args()


def verify_dataset(yaml_path: Path) -> None:
    """验证数据集 yaml 存在且数据集目录就绪。"""
    if not yaml_path.exists():
        print(f"❌ 数据集配置不存在: {yaml_path}", file=sys.stderr)
        print("   请先运行: python -m backend.ml.pose.download_dataset", file=sys.stderr)
        sys.exit(1)

    # 检查数据集目录
    dataset_dir = PROJECT_ROOT / "data" / "dog-pose"
    if not dataset_dir.exists():
        print(f"❌ 数据集目录不存在: {dataset_dir}", file=sys.stderr)
        print("   请先运行: python -m backend.ml.pose.download_dataset", file=sys.stderr)
        sys.exit(1)

    train_dir = dataset_dir / "images" / "train"
    val_dir = dataset_dir / "images" / "val"
    if not train_dir.exists() or not val_dir.exists():
        print(f"❌ 数据集子目录缺失: {train_dir} 或 {val_dir}", file=sys.stderr)
        print("   请重新运行: python -m backend.ml.pose.download_dataset --force", file=sys.stderr)
        sys.exit(1)

    train_count = sum(1 for _ in train_dir.glob("*.jpg"))
    val_count = sum(1 for _ in val_dir.glob("*.jpg"))
    print(f"[verify] 数据集: {dataset_dir}", flush=True)
    print(f"[verify] train images: {train_count}", flush=True)
    print(f"[verify] val   images: {val_count}", flush=True)
    if train_count == 0 or val_count == 0:
        print("❌ 数据集为空，请重新下载", file=sys.stderr)
        sys.exit(1)


def train_model(args: argparse.Namespace) -> None:
    """执行训练。"""
    yaml_path = Path(args.data)
    verify_dataset(yaml_path)

    print(f"\n[train] 加载模型: {args.model}", flush=True)
    model = YOLO(args.model)

    print(f"[train] 数据集: {yaml_path}", flush=True)
    print(f"[train] 参数: epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} "
          f"lr0={args.lr0} device={args.device}", flush=True)
    print(f"[train] 损失: box={args.box} cls={args.cls} pose={args.pose} kobj={args.kobj}", flush=True)
    print(f"[train] 增强(关闭): shear={args.shear} perspective={args.perspective} "
          f"flipud={args.flipud} mixup={args.mixup} copy_paste={args.copy_paste}", flush=True)
    print(f"[train] 输出: {Path(args.output_dir) / 'train'}\n", flush=True)

    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        cos_lr=True,
        device=args.device,
        workers=args.workers,
        project=args.output_dir,
        name="train",
        exist_ok=False,
        patience=args.patience,
        seed=args.seed,
        verbose=True,
        # 损失权重
        box=args.box,
        cls=args.cls,
        pose=args.pose,
        kobj=args.kobj,
        # 关闭破坏关键点几何的增强
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        resume=args.resume,
    )
    print(f"\n✅ 训练完成: {results.save_dir}", flush=True)
    print(f"   best.pt: {Path(results.save_dir) / 'weights' / 'best.pt'}", flush=True)
    print(f"   results.csv: {Path(results.save_dir) / 'results.csv'}", flush=True)


def evaluate_model(args: argparse.Namespace) -> None:
    """在 Dog-Pose 验证集上评估模型。"""
    yaml_path = Path(args.data)
    verify_dataset(yaml_path)

    model_path = args.model
    if model_path == DEFAULT_MODEL:
        # 自动查找最新的 runs/train*/weights/best.pt
        # ultralytics 会自动递增 train, train-2, train-3...
        runs_dir = Path(args.output_dir)
        candidates = sorted(
            runs_dir.glob("train*/weights/best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"❌ 未找到 best.pt: 在 {runs_dir}/train*/weights/", file=sys.stderr)
            print("   请先运行训练，或用 --model 指定路径", file=sys.stderr)
            sys.exit(1)
        model_path = str(candidates[0])
        print(f"[eval] 自动定位最新 best.pt: {model_path}", flush=True)

    print(f"\n[eval] 加载模型: {model_path}", flush=True)
    model = YOLO(model_path)

    print(f"[eval] 数据集: {yaml_path}", flush=True)
    metrics = model.val(
        data=str(yaml_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="val",
        verbose=True,
    )

    # YOLO-pose 评估指标
    print(f"\n[eval] === Dog-Pose 验证集指标 ===", flush=True)
    print(f"  mAP50:      {metrics.box.map50:.4f}", flush=True)
    print(f"  mAP50-95:   {metrics.box.map:.4f}", flush=True)
    print(f"  Pose mAP50: {metrics.pose.map50:.4f}", flush=True)
    print(f"  Pose mAP50-95: {metrics.pose.map:.4f}", flush=True)

    target_map = 0.70
    pose_map = metrics.pose.map
    if pose_map >= target_map:
        print(f"\n✅ Phase 1.0-e 验收通过: Pose mAP50-95 = {pose_map:.4f} ≥ {target_map}",
              flush=True)
    else:
        print(f"\n⚠️  Phase 1.0-e 验收未达标: Pose mAP50-95 = {pose_map:.4f} < {target_map}",
              file=sys.stderr)
        print(f"   建议: 增加 epochs / 检查数据集 / 评估是否需要工作犬数据采集",
              file=sys.stderr)


def main() -> int:
    args = parse_args()
    if args.eval_only:
        evaluate_model(args)
    else:
        train_model(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
