"""生产就绪组件盘点（P0-2 health 聚合 / P0-4 权重盘点共用）.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: P0-2 / P0-4（dev-docs/production-readiness-gap-plan.md）

设计:
    - 只做「路径存在性 + 可选加载验证」，绝不在 API 请求路径内加载大模型
    - 路径候选与 backend/workers/tasks.py 的解析顺序保持一致（tasks 为运行时 owner）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from backend.app.core.config import PROJECT_ROOT

# === 模型路径候选（与 backend/workers/tasks.py 保持一致） ===
POSE_MODEL_CANDIDATES = [
    PROJECT_ROOT / "runs" / "train-2" / "weights" / "best.onnx",
    PROJECT_ROOT / "runs" / "train-2" / "weights" / "best.pt",
    PROJECT_ROOT / "yolo26n-pose.pt",
]
STGCN_BC_ONNX_CANDIDATES = [
    PROJECT_ROOT / "data" / "models" / "stgcn_bc" / "stgcn_bc_dog24.onnx",
]
STGCN_BC_CHECKPOINT_CANDIDATES = [
    PROJECT_ROOT / "runs" / "stgcn_bc_synthetic" / "best.pt",
]
MAMBA_ONNX_CANDIDATES = [
    PROJECT_ROOT / "data" / "models" / "mamba" / "mamba_dog24.onnx",
    PROJECT_ROOT / "runs" / "mamba_synthetic" / "best.pt",
]
MAMBA_BC_ONNX_CANDIDATES = [
    PROJECT_ROOT / "data" / "models" / "mamba_bc" / "mamba_bc_dog24.onnx",
    PROJECT_ROOT / "runs" / "mamba_bc_synthetic" / "best.pt",
]
MOTIONBERT_ONNX_CANDIDATES = [
    PROJECT_ROOT / "data" / "models" / "motionbert_dog24" / "motionbert_dog24.onnx",
]
MB_LITE_CANDIDATES = [
    PROJECT_ROOT / "data" / "models" / "MB_lite_pretrain.pth.tar",
    PROJECT_ROOT / "data" / "models" / "MB_lite_pretrain.bin",
]

# 分组展示名 → 候选列表（盘点用）
MODEL_GROUPS: dict[str, list[Path]] = {
    "pose_yolo": POSE_MODEL_CANDIDATES,
    "stgcn_bc": STGCN_BC_ONNX_CANDIDATES + STGCN_BC_CHECKPOINT_CANDIDATES,
    "mamba": MAMBA_ONNX_CANDIDATES,
    "mamba_bc": MAMBA_BC_ONNX_CANDIDATES,
    "motionbert": MOTIONBERT_ONNX_CANDIDATES + MB_LITE_CANDIDATES,
}


def resolve_first(candidates: list[Path]) -> Optional[Path]:
    """返回第一个存在的候选路径。"""
    for p in candidates:
        if p.exists():
            return p
    return None


def model_presence() -> dict[str, dict[str, Any]]:
    """各模型组路径存在性盘点（轻量，无加载）。

    Returns:
        {组名: {"present": bool, "path": str|None, "candidates": [str]}}
    """
    out: dict[str, dict[str, Any]] = {}
    for name, cands in MODEL_GROUPS.items():
        hit = resolve_first(cands)
        out[name] = {
            "present": hit is not None,
            "path": str(hit) if hit else None,
            "candidates": [str(c) for c in cands],
        }
    return out
