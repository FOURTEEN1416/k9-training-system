"""ST-GCN+BC 评估脚本（Phase 3.1d）.

Owner: ML 开发
Phase: 3.1d
依据: dev-docs/stages/phase-3.md §3.1d（指定路径 scripts/eval_stgcn_bc.py）

评估指标:
    1. 整体准确率 (Top-1 Accuracy)
    2. 22 类 precision / recall / F1
    3. 混淆矩阵 (22 × 22)
    4. 边界检测 F1（可选，需要 boundary 标签）
    5. P0/P1/P2 分层准确率
    6. FCI-IGP A/B/C 阶段分层准确率

用法:
    # 真实数据评估
    python scripts/eval_stgcn_bc.py \
        --checkpoint runs/stgcn_bc/best.pt \
        --test-pkl data/stgcn_bc/test.pkl \
        --output-report reports/phase-3.1d-eval.json

    # 合成数据 baseline 评估
    python scripts/eval_stgcn_bc.py \
        --checkpoint runs/stgcn_bc_synthetic/best.pt \
        --synthetic \
        --samples-per-class 10 \
        --output-report reports/phase-3.1d-synthetic-eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.stgcn_bc import (
    STGCNBCDataset,
    build_stgcn_bc,
    collate_fn,
    make_synthetic_dataset,
)
from backend.ml.behavior.stgcn_bc.dataset import load_pyskl_pickle
from backend.ml.behavior.stgcn_bc.labels import (
    ALL_BEHAVIORS_22,
    BEHAVIOR_TO_IDX,
    IDX_TO_BEHAVIOR,
    LAYER_LABELS,
    FCI_IGP_STAGE,
    NUM_BEHAVIORS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ST-GCN+BC 评估入口（Phase 3.1d）"
    )

    # 模型
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="模型检查点路径（best.pt）"
    )
    parser.add_argument(
        "--in-channels", type=int, default=3,
        help="输入通道数（需与训练时一致）"
    )
    parser.add_argument(
        "--base-channels", type=int, default=64,
        help="STGCN 基础通道数（需与训练时一致）"
    )
    parser.add_argument(
        "--num-stages", type=int, default=10,
        help="STGCN block 数量（需与训练时一致）"
    )

    # 数据
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--test-pkl", type=str,
        help="测试集 pyskl pickle 路径"
    )
    data_group.add_argument(
        "--synthetic", action="store_true",
        help="使用合成数据评估（仅 baseline）"
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=10,
        help="合成数据每类样本数（仅 --synthetic 时生效）"
    )
    parser.add_argument(
        "--T", type=int, default=30,
        help="每个样本帧数"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="批大小"
    )

    # 输出
    parser.add_argument(
        "--output-report", type=str, default=None,
        help="评估报告 JSON 输出路径（不指定则仅打印）"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="评估设备"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Dict:
    """完整评估流程."""
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_boundary_preds: List[np.ndarray] = []
    all_boundary_labels: List[np.ndarray] = []
    total_loss_sum = 0.0
    total_samples = 0

    import torch.nn.functional as F

    for batch in loader:
        keypoints = batch["keypoints"].to(device)
        labels = batch["labels"].to(device)
        boundaries = batch["boundaries"].to(device)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                cls_logits, boundary_logits = model(keypoints)
        else:
            cls_logits, boundary_logits = model(keypoints)

        preds = cls_logits.argmax(dim=-1).cpu().numpy().tolist()
        boundary_probs = torch.sigmoid(boundary_logits).cpu().numpy()  # (B, T')
        boundary_pred = (boundary_probs > 0.5).astype(np.int32)  # (B, T')

        # 对齐时间维度: boundary_logits 是 backbone 下采样后的 T'，需上采样到 T
        # 以与 boundary_labels (B, T) 对齐计算边界检测 F1
        T_pred = boundary_pred.shape[1]
        T_label = boundaries.shape[1]
        if T_pred != T_label:
            # 最近邻上采样（保持 0/1 二值语义，避免线性插值产生中间值）
            idx = np.linspace(0, T_pred - 1, T_label).astype(np.int32)
            boundary_pred = boundary_pred[:, idx]  # (B, T_label)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_boundary_preds.append(boundary_pred)
        all_boundary_labels.append(boundaries.cpu().numpy().astype(np.int32))
        total_samples += keypoints.size(0)

    # 聚合
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    boundary_preds_arr = np.concatenate(all_boundary_preds, axis=0) if all_boundary_preds else np.array([])
    boundary_labels_arr = np.concatenate(all_boundary_labels, axis=0) if all_boundary_labels else np.array([])

    # 指标计算
    metrics = compute_metrics(
        preds_arr, labels_arr, boundary_preds_arr, boundary_labels_arr
    )
    metrics["total_samples"] = total_samples
    return metrics


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    boundary_preds: np.ndarray,
    boundary_labels: np.ndarray,
) -> Dict:
    """计算完整评估指标."""
    num_classes = NUM_BEHAVIORS  # 22

    # 1. 整体准确率
    accuracy = float((preds == labels).mean())

    # 2. 每类 precision / recall / F1
    per_class = {}
    for cls_idx in range(num_classes):
        cls_name = IDX_TO_BEHAVIOR[cls_idx]
        tp = int(((preds == cls_idx) & (labels == cls_idx)).sum())
        fp = int(((preds == cls_idx) & (labels != cls_idx)).sum())
        fn = int(((preds != cls_idx) & (labels == cls_idx)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        per_class[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int((labels == cls_idx).sum()),
            "tp": tp, "fp": fp, "fn": fn,
        }

    # 3. 宏平均
    macro_precision = float(np.mean([c["precision"] for c in per_class.values()]))
    macro_recall = float(np.mean([c["recall"] for c in per_class.values()]))
    macro_f1 = float(np.mean([c["f1"] for c in per_class.values()]))

    # 4. 混淆矩阵
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    for p, l in zip(preds, labels):
        confusion[l, p] += 1

    # 5. P0/P1/P2 分层准确率
    layer_acc = {}
    for layer_name in ["P0", "P1", "P2"]:
        layer_indices = [
            i for i, lbl in enumerate(LAYER_LABELS) if lbl == layer_name
        ]
        mask = np.isin(labels, layer_indices)
        if mask.sum() > 0:
            layer_acc[layer_name] = float((preds[mask] == labels[mask]).mean())
        else:
            layer_acc[layer_name] = 0.0

    # 6. FCI-IGP A/B/C 分层准确率
    igp_acc = {}
    for stage_name in ["A", "B", "C"]:
        stage_behaviors = [
            b for b, s in FCI_IGP_STAGE.items() if s == stage_name
        ]
        stage_indices = [BEHAVIOR_TO_IDX[b] for b in stage_behaviors]
        mask = np.isin(labels, stage_indices)
        if mask.sum() > 0:
            igp_acc[stage_name] = float((preds[mask] == labels[mask]).mean())
        else:
            igp_acc[stage_name] = 0.0

    # 7. 边界检测 F1
    boundary_f1 = 0.0
    if boundary_preds.size > 0 and boundary_labels.size > 0:
        bp = boundary_preds.flatten()
        bl = boundary_labels.flatten()
        tp_b = int(((bp == 1) & (bl == 1)).sum())
        fp_b = int(((bp == 1) & (bl == 0)).sum())
        fn_b = int(((bp == 0) & (bl == 1)).sum())
        p_b = tp_b / max(tp_b + fp_b, 1)
        r_b = tp_b / max(tp_b + fn_b, 1)
        boundary_f1 = 2 * p_b * r_b / max(p_b + r_b, 1e-8)

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "layer_accuracy": {k: round(v, 4) for k, v in layer_acc.items()},
        "igp_stage_accuracy": {k: round(v, 4) for k, v in igp_acc.items()},
        "boundary_f1": round(boundary_f1, 4),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("eval_stgcn_bc")

    logger.info("=" * 60)
    logger.info("ST-GCN+BC 评估 (Phase 3.1d)")
    logger.info("=" * 60)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"评估设备: {device}")

    # 1. 加载模型
    logger.info(f"加载检查点: {args.checkpoint}")
    model = build_stgcn_bc(
        in_channels=args.in_channels,
        num_classes=NUM_BEHAVIORS,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    best_val_acc = ckpt.get("val_acc", 0.0)
    best_epoch = ckpt.get("epoch", -1)
    logger.info(
        f"检查点 epoch={best_epoch}, val_acc={best_val_acc:.4f}"
    )

    # 2. 准备测试集
    if args.synthetic:
        logger.info(
            f"[合成数据] 生成 22 类 × {args.samples_per_class} 样本/类 (seed=43)"
        )
        samples = make_synthetic_dataset(
            samples_per_class=args.samples_per_class,
            T=args.T,
            seed=43,  # 评估用不同 seed（避免与训练集重合）
        )
        test_ds = STGCNBCDataset(
            samples=samples, T=args.T, augment=False, normalize=True
        )
    else:
        logger.info(f"[真实数据] 加载测试集 {args.test_pkl}")
        test_ds = STGCNBCDataset(
            pkl_path=args.test_pkl, T=args.T, augment=False, normalize=True
        )

    logger.info(f"测试集: {len(test_ds)} 样本")

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 3. 评估
    use_amp = device.type == "cuda"
    metrics = evaluate_model(model, loader, device, use_amp=use_amp)

    # 4. 输出
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"测试样本数: {metrics['total_samples']}")
    print(f"整体准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"宏平均 Precision: {metrics['macro_precision']:.4f}")
    print(f"宏平均 Recall:    {metrics['macro_recall']:.4f}")
    print(f"宏平均 F1:        {metrics['macro_f1']:.4f}")
    print(f"边界检测 F1:      {metrics['boundary_f1']:.4f}")

    print("\n分层准确率:")
    print(f"  P0 (基础 8 类):   {metrics['layer_accuracy']['P0']:.4f}")
    print(f"  P1 (训练 8 类):   {metrics['layer_accuracy']['P1']:.4f}")
    print(f"  P2 (高级 6 类):   {metrics['layer_accuracy']['P2']:.4f}")
    print(f"  IGP-A (追踪):     {metrics['igp_stage_accuracy']['A']:.4f}")
    print(f"  IGP-B (服从):     {metrics['igp_stage_accuracy']['B']:.4f}")
    print(f"  IGP-C (护卫):     {metrics['igp_stage_accuracy']['C']:.4f}")

    # Top-5 最低 F1 类
    print("\nF1 最低 5 类（待改进）:")
    sorted_classes = sorted(
        metrics["per_class"].items(), key=lambda x: x[1]["f1"]
    )
    for cls_name, m in sorted_classes[:5]:
        print(
            f"  {cls_name:<14} P={m['precision']:.3f} "
            f"R={m['recall']:.3f} F1={m['f1']:.3f} "
            f"(support={m['support']})"
        )

    # 5. 保存报告
    if args.output_report:
        report = {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": best_epoch,
            "checkpoint_val_acc": best_val_acc,
            "test_samples": metrics["total_samples"],
            "metrics": metrics,
            "device": str(device),
        }
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"评估报告已保存: {report_path}")


if __name__ == "__main__":
    main()
