"""Mamba 长序列评估脚本（Phase 4.2）.

对比 Mamba T=90 与 ST-GCN+BC T=30 在合成数据上的表现.

用法:
    python scripts/eval_mamba_long.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from backend.ml.behavior.mamba_sequence import get_model
from backend.ml.behavior.mamba_inference import MambaInferer
from backend.ml.behavior.stgcn_bc.labels import IDX_TO_BEHAVIOR, NUM_BEHAVIORS
from backend.ml.behavior.stgcn_bc.inference import STGCNBCInferer
from backend.ml.behavior.stgcn_bc.dataset import make_synthetic_dataset


def evaluate_accuracy(model, samples, T: int, device: str) -> float:
    """评估模型在合成数据上的准确率."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for s in samples:
            kpt = s["keypoints"]  # (T_orig, 24, 3)
            label = s["label"]
            T_orig = kpt.shape[0]

            if T_orig >= T:
                start = (T_orig - T) // 2
                kpt = kpt[start:start + T]
            else:
                pad_len = T - T_orig
                kpt = np.concatenate([kpt, np.tile(kpt[-1:], (pad_len, 1, 1))], axis=0)

            # 归一化
            kpt = _normalize_numpy(kpt)

            x = torch.from_numpy(kpt).float().unsqueeze(0).to(device)  # (1, T, 24, 3)
            logits = model(x)  # (1, 22)
            pred = logits.argmax(dim=-1).item()
            correct += 1 if pred == label else 0
            total += 1

    return correct / max(total, 1)


def _normalize_numpy(kpt: np.ndarray) -> np.ndarray:
    """按 withers 中心 + 体长尺度归一化."""
    center = kpt[:, 22:23, :].mean(axis=0, keepdims=True)
    kpt = kpt - center
    bone_ref = kpt[:, 22, :2] - kpt[:, 12, :2]
    bone_len = np.linalg.norm(bone_ref, axis=-1).mean()
    if bone_len < 1e-6:
        bone_len = 1.0
    kpt[..., :2] = kpt[..., :2] / bone_len
    return kpt


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print("=" * 60)

    # 加载训练好的 Mamba 模型
    ckpt_path = Path("runs/mamba_synthetic/best.pt")
    if not ckpt_path.exists():
        print(f"错误: 未找到检查点 {ckpt_path}，请先运行训练脚本")
        return

    print(f"加载 Mamba 检查点: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = get_model(num_classes=NUM_BEHAVIORS)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Mamba 参数: {n_params:,}")

    # 加载 ST-GCN+BC 模型
    stgcn_ckpt_path = Path("runs/stgcn_bc_synthetic/best.pt")
    stgcn_available = stgcn_ckpt_path.exists()
    if stgcn_available:
        from backend.ml.behavior.stgcn_bc.model import build_stgcn_bc
        print(f"加载 ST-GCN+BC 检查点: {stgcn_ckpt_path}")
        stgcn_model = build_stgcn_bc(in_channels=3, num_classes=NUM_BEHAVIORS)
        stgcn_ckpt = torch.load(stgcn_ckpt_path, map_location="cpu", weights_only=False)
        stgcn_model.load_state_dict(stgcn_ckpt["model_state_dict"], strict=True)
        stgcn_model = stgcn_model.to(device)
        stgcn_model.eval()
        print(f"  ST-GCN+BC 参数: {sum(p.numel() for p in stgcn_model.parameters()):,}")

    # 生成合成测试数据
    print("\n生成合成测试数据...")
    test_samples = make_synthetic_dataset(samples_per_class=10, T=90, seed=99)
    print(f"  测试样本数: {len(test_samples)}")

    # ================================================================
    # 评估: Mamba T=30 vs Mamba T=90
    # ================================================================
    print("\n" + "=" * 60)
    print("Mamba 长序列评估: T=30 vs T=90")
    print("=" * 60)

    for T in [30, 60, 90]:
        acc = evaluate_accuracy(model, test_samples, T=T, device=device)
        print(f"  Mamba T={T:3d}: 准确率 {acc*100:.2f}%")

    # ================================================================
    # 对比: Mamba T=90 vs ST-GCN+BC T=30
    # ================================================================
    print("\n" + "=" * 60)
    print("基线对比: Mamba T=90 vs ST-GCN+BC T=30")
    print("=" * 60)

    mamba_acc_t90 = evaluate_accuracy(model, test_samples, T=90, device=device)
    print(f"  Mamba      T=90: 准确率 {mamba_acc_t90*100:.2f}%  | 参数 {n_params:,}")

    if stgcn_available:
        stgcn_acc_t30 = evaluate_accuracy_stgcn(stgcn_model, test_samples, T=30, device=device)
        stgcn_params = sum(p.numel() for p in stgcn_model.parameters())
        print(f"  ST-GCN+BC  T=30: 准确率 {stgcn_acc_t30*100:.2f}%  | 参数 {stgcn_params:,}")

        # 对比摘要
        print(f"\n对比摘要:")
        print(f"  Mamba 准确率提升: {mamba_acc_t90*100 - stgcn_acc_t30*100:+.2f}%")
        print(f"  Mamba 参数缩减:   {stgcn_params / n_params:.1f}x")
    else:
        print(f"  (ST-GCN+BC 检查点未找到，跳过对比)")
        print(f"  参考: Phase 3 报告 ST-GCN+BC 合成数据准确率 46.97%")

    # ================================================================
    # 推理延迟对比
    # ================================================================
    print("\n" + "=" * 60)
    print("推理延迟对比")
    print("=" * 60)

    import time

    # Mamba 延迟
    sample = test_samples[0]
    kpt = sample["keypoints"]
    T_orig = kpt.shape[0]
    if T_orig >= 90:
        start = (T_orig - 90) // 2
        kpt_90 = kpt[start:start + 90]
    else:
        kpt_90 = kpt
    kpt_norm = _normalize_numpy(kpt_90)

    model.eval()
    with torch.no_grad():
        # warmup
        x = torch.from_numpy(kpt_norm).float().unsqueeze(0).to(device)
        _ = model(x)

        # 计时
        n_runs = 50
        t0 = time.time()
        for _ in range(n_runs):
            _ = model(x)
        mamba_latency = (time.time() - t0) / n_runs * 1000
        print(f"  Mamba      推理延迟: {mamba_latency:.2f} ms/样本 (T=90)")

    if stgcn_available:
        stgcn_model.eval()
        with torch.no_grad():
            kpt_stgcn = _normalize_numpy(kpt[:30])
            x_stgcn = torch.from_numpy(kpt_stgcn).float().unsqueeze(0).to(device)
            _ = stgcn_model(x_stgcn)

            t0 = time.time()
            for _ in range(n_runs):
                _ = stgcn_model(x_stgcn)
            stgcn_latency = (time.time() - t0) / n_runs * 1000
            print(f"  ST-GCN+BC  推理延迟: {stgcn_latency:.2f} ms/样本 (T=30)")

    print("\n" + "=" * 60)
    print("评估完成")
    print("=" * 60)


def evaluate_accuracy_stgcn(model, samples, T: int, device: str) -> float:
    """评估 ST-GCN+BC 模型在合成数据上的准确率."""
    from backend.ml.behavior.stgcn_bc.dataset import STGCNBCDataset
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for s in samples:
            kpt = s["keypoints"].copy()
            label = s["label"]
            T_orig = kpt.shape[0]

            if T_orig >= T:
                start = (T_orig - T) // 2
                kpt = kpt[start:start + T]
            else:
                pad_len = T - T_orig
                kpt = np.concatenate([kpt, np.tile(kpt[-1:], (pad_len, 1, 1))], axis=0)

            kpt = STGCNBCDataset._normalize(kpt)

            x = torch.from_numpy(kpt).float().unsqueeze(0).to(device)
            cls_logits, _ = model(x)
            pred = cls_logits.argmax(dim=-1).item()
            correct += 1 if pred == label else 0
            total += 1

    return correct / max(total, 1)


if __name__ == "__main__":
    main()