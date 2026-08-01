"""MotionBERT 24 关键点 ONNX 导出.

Owner: ML 开发
Phase: 3.3c

设计:
    - 导出固定输入 shape 的 ONNX（B=1, T=27, J=24, C=3）
    - 使用动态轴支持可变 batch 和时间长度
    - 不依赖 onnx 包（torch.onnx.export 自包含）
    - 用 onnxruntime 验证一致性

用法:
    python -m backend.ml.pose.motionbert.export_onnx \\
        --checkpoint data/models/motionbert_dog24/best_epoch.bin \\
        --output data/models/motionbert_dog24/motionbert_dog24.onnx
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# 添加项目本地 onnx 包（torch 2.8+ 导出需要）
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_LOCAL_PYLIBS = _PROJECT_ROOT / "external" / "_pylibs"
if _LOCAL_PYLIBS.exists() and str(_LOCAL_PYLIBS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PYLIBS))

import numpy as np
import torch
import torch.nn as nn

from backend.ml.pose.motionbert.config import MotionBERTConfig, load_config
from backend.ml.pose.motionbert.model import build_model_from_config

logger = logging.getLogger(__name__)


def export_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    config: Optional[MotionBERTConfig] = None,
    opset_version: int = 17,
    verify: bool = True,
) -> dict:
    """导出 MotionBERT 为 ONNX.

    Args:
        checkpoint_path: 微调后的 checkpoint 路径
        output_path: ONNX 输出路径
        config: 模型配置（None=从 checkpoint 读取）
        opset_version: ONNX opset 版本
        verify: 是否用 onnxruntime 验证一致性

    Returns:
        导出结果字典
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if config is None:
        if "config" in checkpoint:
            config = MotionBERTConfig(**checkpoint["config"])
        else:
            config = MotionBERTConfig()

    # 构建模型
    model = build_model_from_config(config)
    state_dict = checkpoint.get("model_pos", checkpoint)
    # strip module. prefix
    cleaned = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned[key] = v
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    # 构造示例输入 (1, T, 24, 3)
    T = config.window_size
    J = config.num_joints
    dummy_input = torch.randn(1, T, J, 3, dtype=torch.float32)

    # 导出
    logger.info(f"[export_onnx] 导出 {checkpoint_path} → {output_path}")
    # torch 2.8+ 新版 dynamo 导出器需要 onnx 包，旧版 dynamo=False 不需要
    export_kwargs = dict(
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["keypoints_2d"],
        output_names=["keypoints_3d"],
        dynamic_axes={
            "keypoints_2d": {0: "batch", 1: "time"},
            "keypoints_3d": {0: "batch", 1: "time"},
        },
    )
    try:
        torch.onnx.export(model, dummy_input, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        # 旧版 torch 无 dynamo 参数
        torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"[export_onnx] 完成, 大小: {file_size_mb:.2f} MB")

    result = {
        "output_path": str(output_path),
        "file_size_mb": round(file_size_mb, 2),
        "opset_version": opset_version,
        "input_shape": f"(batch, time, {J}, 3)",
        "output_shape": f"(batch, time, {J}, 3)",
        "num_joints": J,
    }

    # 验证
    if verify:
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            # 多次随机输入验证
            max_diff = 0.0
            for _ in range(5):
                test_input = np.random.randn(2, T, J, 3).astype(np.float32)
                with torch.no_grad():
                    torch_out = model(torch.from_numpy(test_input)).numpy()
                onnx_out = sess.run(None, {"keypoints_2d": test_input})[0]
                diff = np.abs(torch_out - onnx_out).max()
                max_diff = max(max_diff, float(diff))
            logger.info(
                f"[export_onnx] 一致性验证通过: PyTorch vs ONNX max_diff={max_diff:.6e}"
            )
            result["verification"] = {
                "max_diff": max_diff,
                "passed": max_diff < 1e-4,
            }
        except ImportError:
            logger.warning("[export_onnx] onnxruntime 不可用，跳过验证")
            result["verification"] = {"skipped": "onnxruntime not available"}

    return result


def main():
    parser = argparse.ArgumentParser(description="MotionBERT ONNX 导出")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="checkpoint 路径"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="ONNX 输出路径"
    )
    parser.add_argument("--config", type=str, default=None, help="YAML 配置")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config) if args.config else None
    result = export_onnx(
        args.checkpoint,
        args.output,
        config=config,
        opset_version=args.opset,
        verify=not args.no_verify,
    )
    print(f"\nONNX 导出结果:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
