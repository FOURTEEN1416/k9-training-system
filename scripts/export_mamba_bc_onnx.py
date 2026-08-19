"""Mamba+BC ONNX 导出脚本（Phase 4.2）.

用法:
    python scripts/export_mamba_bc_onnx.py \
        --checkpoint runs/mamba_bc_synthetic/best.pt \
        --output data/models/mamba_bc/mamba_bc_dog24.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from backend.ml.behavior.mamba_bc import build_mamba_bc
from backend.ml.behavior.stgcn_bc.labels import NUM_BEHAVIORS


def parse_args():
    parser = argparse.ArgumentParser(description="Mamba+BC ONNX 导出")
    parser.add_argument("--checkpoint", type=str, default="runs/mamba_bc_synthetic/best.pt")
    parser.add_argument("--output", type=str, default="data/models/mamba_bc/mamba_bc_dog24.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-state", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 构建模型
    model = build_mamba_bc(
        num_classes=NUM_BEHAVIORS,
        d_model=args.d_model,
        d_state=args.d_state,
    )
    model.eval()

    # 2. 加载权重
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"加载检查点: {args.checkpoint}")
    print(f"  验证准确率: {ckpt.get('val_acc', 'N/A'):.4f}")
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 3. 导出 ONNX
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 动态 batch 和 time 轴
    dummy_input = torch.randn(1, 30, 24, 3)  # (B, T, 24, 3)

    # 使用传统 ONNX 导出（dynamo=False）
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["keypoints"],
            output_names=["cls_logits", "boundary_logits"],
            dynamic_axes={
                "keypoints": {0: "batch", 1: "time"},
                "cls_logits": {0: "batch"},
                "boundary_logits": {0: "batch", 1: "time"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,  # 使用传统导出器，避免 torch.export
        )
    print(f"ONNX 导出成功: {output_path}")

    # 4. 一致性验证
    print("\n验证 ONNX 一致性...")
    import onnxruntime as ort

    ort_session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )

    # 多组输入验证
    for B, T in [(1, 30), (2, 60), (1, 90)]:
        test_input = np.random.randn(B, T, 24, 3).astype(np.float32)

        # PyTorch 推理
        with torch.no_grad():
            torch_cls, torch_bd = model(torch.from_numpy(test_input))
            torch_cls = torch_cls.numpy()
            torch_bd = torch_bd.numpy()

        # ONNX 推理
        ort_cls, ort_bd = ort_session.run(
            None, {"keypoints": test_input.astype(np.float32)}
        )

        # 对比
        cls_diff = np.abs(torch_cls - ort_cls).max()
        bd_diff = np.abs(torch_bd - ort_bd).max()
        cls_match = cls_diff < 1e-3
        bd_match = bd_diff < 1e-3

        status = "✅" if cls_match and bd_match else "❌"
        print(f"  {status} B={B} T={T}: cls_diff={cls_diff:.6f} bd_diff={bd_diff:.6f}")

    print("\n导出完成!")


if __name__ == "__main__":
    main()