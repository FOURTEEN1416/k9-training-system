"""Phase 3.3c E2E 验证脚本.

Owner: ML 开发
Phase: 3.3c

验证流程:
    1. 加载微调后的 MotionBERT checkpoint
    2. 导出 ONNX
    3. PyTorch vs ONNX 一致性验证
    4. MotionBERTLifter 滑动窗口推理验证
    5. 端到端: 合成 2D → 3D lifting → MPJPE 评估
    6. 输出 JSON 验证报告
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 添加项目本地 onnx 包路径
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_PYLIBS = _PROJECT_ROOT / "external" / "_pylibs"
if _LOCAL_PYLIBS.exists() and str(_LOCAL_PYLIBS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PYLIBS))

from backend.ml.pose.motionbert.config import MotionBERTConfig, load_config
from backend.ml.pose.motionbert.dataset import build_datasets
from backend.ml.pose.motionbert.export_onnx import export_onnx
from backend.ml.pose.motionbert.inference import MotionBERTLifter
from backend.ml.pose.motionbert.model import build_model_from_config, load_pretrained_weights

logger = logging.getLogger(__name__)


def evaluate_mpjpe(
    lifter: MotionBERTLifter,
    datasets: dict,
    max_samples: int = 200,
) -> dict:
    """在验证集上评估 MPJPE.

    Returns:
        {"mpjpe_mean", "mpjpe_std", "mpjpe_median", "num_samples"}
    """
    val_ds = datasets["val"]
    n = min(max_samples, len(val_ds))
    indices = np.linspace(0, len(val_ds) - 1, n, dtype=int)

    mpjpes = []
    for idx in indices:
        kp_2d, kp_3d_gt = val_ds[idx]
        # kp_2d: (T, 24, 3), kp_3d_gt: (T, 24, 3)
        pred_3d = lifter.lift(kp_2d.numpy())  # (T, 24, 3)

        # GT 根关节中心化
        gt_3d = kp_3d_gt.numpy()
        gt_3d = gt_3d - gt_3d[..., 0:1, :]

        # MPJPE
        mpjpe = np.mean(np.linalg.norm(pred_3d - gt_3d, axis=-1))
        mpjpes.append(float(mpjpe))

    mpjpes = np.array(mpjpes)
    return {
        "mpjpe_mean": round(float(mpjpes.mean()), 4),
        "mpjpe_std": round(float(mpjpes.std()), 4),
        "mpjpe_median": round(float(np.median(mpjpes)), 4),
        "num_samples": int(n),
    }


def verify_phase_3_3c(
    checkpoint_path: str,
    config_path: str | None = None,
    output_report: str = "reports/phase-3.3c-validation.json",
) -> dict:
    """Phase 3.3c 完整验证.

    Returns:
        验证结果字典
    """
    results = {
        "phase": "3.3c",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint_path,
    }

    # 1. 加载配置 + checkpoint
    config = load_config(config_path) if config_path else MotionBERTConfig()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        config = MotionBERTConfig(**checkpoint["config"])
    results["num_joints"] = config.num_joints
    results["model_params"] = sum(
        p.numel() for p in build_model_from_config(config).parameters()
    )

    # 2. 导出 ONNX
    onnx_path = Path(checkpoint_path).parent / "motionbert_dog24.onnx"
    logger.info(f"[verify] 导出 ONNX: {onnx_path}")
    export_result = export_onnx(
        checkpoint_path, onnx_path, config=config, verify=True
    )
    results["onnx_export"] = export_result

    # 3. PyTorch 后端推理
    logger.info("[verify] PyTorch 后端评估")
    lifter_torch = MotionBERTLifter(checkpoint_path=checkpoint_path, config=config)

    # 4. 构建验证数据集（小规模）
    logger.info("[verify] 加载验证数据集")
    datasets = build_datasets(config, max_clips=20)  # 20 clips 快速验证

    # 5. MPJPE 评估
    logger.info("[verify] MPJPE 评估 (PyTorch)")
    mpjpe_torch = evaluate_mpjpe(lifter_torch, datasets, max_samples=100)
    results["mpjpe_torch"] = mpjpe_torch
    logger.info(f"[verify] PyTorch MPJPE: {mpjpe_torch['mpjpe_mean']:.4f}")

    # 6. ONNX 后端推理 + 一致性
    logger.info("[verify] ONNX 后端评估")
    lifter_onnx = MotionBERTLifter(onnx_path=onnx_path, config=config)

    # 一致性检查：同一输入两个后端的输出差异
    val_ds = datasets["val"]
    sample_2d, _ = val_ds[0]
    out_torch = lifter_torch.lift(sample_2d.numpy())
    out_onnx = lifter_onnx.lift(sample_2d.numpy())
    max_diff = float(np.abs(out_torch - out_onnx).max())
    results["backend_consistency"] = {
        "max_diff": round(max_diff, 6),
        "passed": max_diff < 1e-3,
    }
    logger.info(f"[verify] 后端一致性 max_diff: {max_diff:.6e}")

    # ONNX MPJPE
    mpjpe_onnx = evaluate_mpjpe(lifter_onnx, datasets, max_samples=100)
    results["mpjpe_onnx"] = mpjpe_onnx
    logger.info(f"[verify] ONNX MPJPE: {mpjpe_onnx['mpjpe_mean']:.4f}")

    # 7. 滑动窗口推理测试（长序列）
    long_input = np.random.randn(200, 24, 3).astype(np.float32)
    long_out = lifter_torch.lift(long_input)
    results["long_sequence_inference"] = {
        "input_shape": list(long_input.shape),
        "output_shape": list(long_out.shape),
        "passed": long_out.shape == (200, 24, 3),
    }

    # 8. 总结
    results["overall_passed"] = (
        results["onnx_export"].get("verification", {}).get("passed", False)
        and results["backend_consistency"]["passed"]
        and results["long_sequence_inference"]["passed"]
    )

    # 保存报告
    report_path = Path(output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"[verify] 验证报告保存到: {report_path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3.3c E2E 验证")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default="reports/phase-3.3c-validation.json")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = verify_phase_3_3c(args.checkpoint, args.config, args.output)
    print("\n" + "=" * 60)
    print("Phase 3.3c 验证结果:")
    print("=" * 60)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
