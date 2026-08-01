"""3D 姿态重建精度评估（Phase 3.3d）.

Owner: ML 开发
Phase: 3.3d

目标:
    - MPJPE ≤ 50mm（MotionBERT-Lite H36M finetune 基线 37.2mm）
    - 在 InterPet4D 验证集上评估微调后的 MotionBERT 24 关键点模型

方法:
    使用与训练完全相同的数据路径（build_datasets → DataLoader），
    确保预处理、归一化、验证集划分一致。
    MPJPE(mm) = MPJPE(norm) × mean_scale × 1000

用法:
    python scripts/eval_3d_pose.py
    python scripts/eval_3d_pose.py --max-clips 30  # 快速评估
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from backend.ml.pose.interpet4d_loader import load_all_clips
from backend.ml.pose.lifting_pairing import build_pairs_from_clip, normalize_3d_keypoints
from backend.ml.pose.motionbert.config import MotionBERTConfig
from backend.ml.pose.motionbert.dataset import build_datasets
from backend.ml.pose.motionbert.model import build_model_from_config

logger = logging.getLogger(__name__)

THRESHOLD_MM = 50.0
H36M_BASELINE_MM = 37.2


def evaluate(
    checkpoint_path: str = "data/models/motionbert_dog24/best_epoch.bin",
    max_clips: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """评估 MotionBERT 3D 姿态重建精度（与训练相同数据路径）."""
    t0 = time.time()

    # 1. 加载 checkpoint
    logger.info(f"[eval] 加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = MotionBERTConfig(**ckpt["config"]) if "config" in ckpt else MotionBERTConfig()
    best_epoch = ckpt.get("epoch", "?")
    train_mpjpe_norm = ckpt.get("val_mpjpe", None)
    logger.info(f"[eval] best_epoch={best_epoch}, train val_mpjpe(norm)={train_mpjpe_norm}")

    model = build_model_from_config(config).to(device)
    state_dict = ckpt.get("model_pos", ckpt)
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    # 2. 构建验证集（与训练完全相同的数据路径）
    logger.info(f"[eval] 构建数据集 (max_clips={max_clips})")
    datasets = build_datasets(config, max_clips=max_clips)
    val_ds = datasets["val"]
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    logger.info(f"[eval] 验证集 {len(val_ds)} 样本")

    # 3. 推理 + 计算 MPJPE（与 train.py evaluate 完全一致）
    total_mpjpe = 0.0
    total_p_mpjpe = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch_2d, batch_3d in val_loader:
            batch_2d = batch_2d.to(device)
            batch_3d = batch_3d.to(device)

            # rootrel（与 train.py 第 106-107 行一致）
            if config.rootrel:
                batch_3d = batch_3d - batch_3d[..., 0:1, :]

            pred = model(batch_2d)

            # MPJPE (normalized)
            mpjpe = torch.mean(torch.norm(pred - batch_3d, dim=-1))
            total_mpjpe += float(mpjpe.item()) * batch_2d.shape[0]

            # P-MPJPE (Procrustes aligned) — 逐样本逐帧
            pred_np = pred.cpu().numpy()
            target_np = batch_3d.cpu().numpy()
            for i in range(pred_np.shape[0]):
                for t in range(pred_np.shape[1]):
                    p = pred_np[i, t] - pred_np[i, t].mean(axis=0, keepdims=True)
                    g = target_np[i, t] - target_np[i, t].mean(axis=0, keepdims=True)
                    U, _, Vt = np.linalg.svd(p.T @ g)
                    R = U @ Vt
                    if np.linalg.det(R) < 0:
                        U[:, -1] *= -1
                        R = U @ Vt
                    aligned = p @ R.T
                    p_mpjpe = np.sqrt(((aligned - g) ** 2).sum(axis=-1)).mean()
                    total_p_mpjpe += float(p_mpjpe)
                    n_samples += 1

    # 4. 计算平均 scale（用于 mm 估算）
    clips = load_all_clips(config.smal_dir, min_frames=config.window_size)
    if max_clips:
        clips = clips[:max_clips]
    scales = []
    for clip in clips:
        _, scale = normalize_3d_keypoints(clip.kp_world)
        scales.append(scale)
    mean_scale = float(np.mean(scales))

    # 5. 结果
    mpjpe_norm = total_mpjpe / len(val_ds)
    p_mpjpe_norm = total_p_mpjpe / (len(val_ds) * config.window_size)  # 每帧每样本
    mpjpe_mm = mpjpe_norm * mean_scale * 1000
    p_mpjpe_mm = p_mpjpe_norm * mean_scale * 1000

    elapsed = time.time() - t0

    result = {
        "checkpoint": checkpoint_path,
        "best_epoch": best_epoch,
        "train_val_mpjpe_normalized": train_mpjpe_norm,
        "val_samples": len(val_ds),
        "mpjpe_normalized": round(mpjpe_norm, 6),
        "p_mpjpe_normalized": round(p_mpjpe_norm, 6),
        "mean_scale": round(mean_scale, 6),
        "mpjpe_mm": round(mpjpe_mm, 2),
        "p_mpjpe_mm": round(p_mpjpe_mm, 2),
        "threshold_mm": THRESHOLD_MM,
        "h36m_baseline_mm": H36M_BASELINE_MM,
        "passed": mpjpe_mm <= THRESHOLD_MM,
        "elapsed_seconds": round(elapsed, 1),
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="3D 姿态重建精度评估 (Phase 3.3d)")
    parser.add_argument("--checkpoint", default="data/models/motionbert_dog24/best_epoch.bin")
    parser.add_argument("--max-clips", type=int, default=None, help="限制 clip 数（快速评估）")
    parser.add_argument("--output", default="reports/phase-3.3d-3d-pose-eval.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = evaluate(
        checkpoint_path=args.checkpoint,
        max_clips=args.max_clips,
    )

    print("\n" + "=" * 60)
    print("3D 姿态重建精度评估 (Phase 3.3d)")
    print("=" * 60)
    print(f"  Checkpoint:       {result['checkpoint']}")
    print(f"  Best Epoch:       {result['best_epoch']}")
    print(f"  验证集样本:       {result['val_samples']}")
    print(f"  训练 MPJPE(norm): {result['train_val_mpjpe_normalized']}")
    print(f"  评估 MPJPE(norm): {result['mpjpe_normalized']}")
    print(f"  P-MPJPE(norm):    {result['p_mpjpe_normalized']}")
    print(f"  mean_scale:       {result['mean_scale']}")
    print(f"  MPJPE:            {result['mpjpe_mm']} mm (阈值 ≤ {result['threshold_mm']} mm)")
    print(f"  P-MPJPE:          {result['p_mpjpe_mm']} mm")
    print(f"  H36M 基线:        {result['h36m_baseline_mm']} mm")
    print(f"  结果:             {'✅ 通过' if result['passed'] else '❌ 未达标'}")
    print(f"  耗时:             {result['elapsed_seconds']}s")
    print("=" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {output_path}")


if __name__ == "__main__":
    main()
