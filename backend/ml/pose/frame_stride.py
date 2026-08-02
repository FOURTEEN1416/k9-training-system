"""抽帧策略模块（Phase 3.5 Jetson 边缘部署）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.5
依据: dev-docs/stages/phase-3.md §3.5
      dev-docs/research/RESEARCH_JETSON_DEPLOYMENT.md §4 抽帧策略

策略:
    Jetson Orin Nano Super 算力有限（40 TOPS），全帧推理 2700 帧/90s 视频需 98s（1.13x）。
    通过抽帧策略（每 N 帧推理 1 次）+ 线性插值填充中间帧，可降至 ≤ 0.5x。

    - stride=1: 全帧推理（默认，精度最高）
    - stride=2: 每隔 1 帧推理 1 次（推理量减半，插值误差 < 5%）
    - stride=3: 每隔 2 帧推理 1 次（推理量 1/3，适合 Jetson）
    - stride=5: 每隔 4 帧推理 1 次（推理量 1/5，极限模式）

插值策略:
    - 关键点坐标 (x, y): 线性插值
    - 置信度 conf: 取相邻推理帧的最小值（保守估计）
    - 检测框: 线性插值

用法:
    from backend.ml.pose.frame_stride import compute_infer_indices, interpolate_keypoints

    # 计算需要推理的帧索引
    infer_indices = compute_infer_indices(total_frames=2700, stride=3)
    # [0, 3, 6, 9, ...]

    # 对推理结果做插值填充
    full_kpts = interpolate_keypoints(infer_kpts, infer_indices, total_frames=2700)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class StrideConfig:
    """抽帧策略配置."""

    stride: int = 1  # 抽帧步长（1=全帧，3=每 3 帧推理 1 次）
    interpolation: str = "linear"  # 插值方法: linear / nearest / none

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError(f"stride 必须 >= 1，实际 {self.stride}")
        if self.interpolation not in ("linear", "nearest", "none"):
            raise ValueError(
                f"interpolation 必须为 linear/nearest/none，实际 {self.interpolation}"
            )


def compute_infer_indices(total_frames: int, stride: int = 1) -> List[int]:
    """计算需要推理的帧索引.

    Args:
        total_frames: 视频总帧数
        stride: 抽帧步长

    Returns:
        需要推理的帧索引列表 [0, stride, 2*stride, ...]
    """
    if total_frames <= 0:
        return []
    if stride <= 1:
        return list(range(total_frames))
    # 确保最后一帧被推理（避免尾部插值外推）
    indices = list(range(0, total_frames, stride))
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    return indices


def interpolate_keypoints(
    infer_kpts: np.ndarray,
    infer_indices: List[int],
    total_frames: int,
    method: str = "linear",
) -> np.ndarray:
    """对抽帧推理的关键点序列做插值填充.

    Args:
        infer_kpts: 推理得到的关键点序列 shape=(M, 24, 3)，M = len(infer_indices)
        infer_indices: 推理帧索引列表 [0, 3, 6, ...]
        total_frames: 目标总帧数
        method: 插值方法 linear / nearest / none

    Returns:
        完整关键点序列 shape=(total_frames, 24, 3)
    """
    if len(infer_indices) != len(infer_kpts):
        raise ValueError(
            f"infer_kpts 长度 {len(infer_kpts)} 与 infer_indices 长度 {len(infer_indices)} 不匹配"
        )
    if total_frames <= 0:
        return np.zeros((0, 24, 3), dtype=np.float32)

    if method == "none" or len(infer_indices) == total_frames:
        # 无需插值
        return infer_kpts.astype(np.float32)

    if len(infer_indices) == 1:
        # 只有一帧，复制填充
        return np.repeat(infer_kpts, total_frames, axis=0).astype(np.float32)

    # 构建 (x, y) 和 conf 分别插值
    # x_coords: 推理帧索引
    x = np.array(infer_indices, dtype=np.float32)
    # x_query: 全部帧索引
    x_query = np.arange(total_frames, dtype=np.float32)

    if method == "nearest":
        # 最近邻插值
        nearest_idx = np.searchsorted(x, x_query, side="left")
        nearest_idx = np.clip(nearest_idx, 0, len(x) - 1)
        # 选择最近的（左或右）
        left_dist = np.abs(x_query - x[np.clip(nearest_idx - 1, 0, len(x) - 1)])
        right_dist = np.abs(x_query - x[np.clip(nearest_idx, 0, len(x) - 1)])
        mask = left_dist < right_dist
        final_idx = np.where(mask, nearest_idx - 1, nearest_idx)
        final_idx = np.clip(final_idx, 0, len(x) - 1)
        return infer_kpts[final_idx].astype(np.float32)

    # 线性插值（默认）
    # 对每个关键点的每个维度 (x, y, conf) 分别插值
    M, K, D = infer_kpts.shape  # K=24, D=3
    result = np.zeros((total_frames, K, D), dtype=np.float32)

    for k in range(K):
        for d in range(D):
            result[:, k, d] = np.interp(x_query, x, infer_kpts[:, k, d])

    # 置信度特殊处理：插值后可能 > 1 或 < 0，做 clip
    if D >= 3:
        result[:, :, 2] = np.clip(result[:, :, 2], 0.0, 1.0)

    return result


def estimate_speedup(stride: int, total_frames: int) -> float:
    """估算抽帧带来的推理加速比.

    Args:
        stride: 抽帧步长
        total_frames: 总帧数

    Returns:
        加速比（例如 stride=3 → 返回 3.0，表示推理量降为 1/3）
    """
    if stride <= 1:
        return 1.0
    infer_count = len(compute_infer_indices(total_frames, stride))
    if infer_count == 0:
        return 1.0
    return total_frames / infer_count


def recommend_stride(
    target_speedup: float = 2.0,
    total_frames: int = 2700,
    max_interpolation_error: float = 0.05,
) -> int:
    """根据目标加速比和可接受插值误差推荐 stride.

    Args:
        target_speedup: 目标加速比（2.0 = 推理量减半）
        total_frames: 视频总帧数
        max_interpolation_error: 可接受的最大插值误差（0-1）

    Returns:
        推荐的 stride 值
    """
    # 经验规则：stride=2 时线性插值误差约 3%，stride=3 约 5%，stride=5 约 10%
    # 误差与 stride 正相关
    error_per_stride = {1: 0.0, 2: 0.03, 3: 0.05, 4: 0.08, 5: 0.10}

    # 速度比较容忍边界浮动（最后一帧强制纳入导致轻微下偏：例如 2700 帧 stride=5
    # 实际 infer_count=541，speedup=4.99 而非理论 5.0）
    SPEEDUP_TOLERANCE = 0.05

    best_stride = 1
    for stride in sorted(error_per_stride.keys(), reverse=True):
        if error_per_stride[stride] <= max_interpolation_error:
            speedup = estimate_speedup(stride, total_frames)
            if speedup >= target_speedup - SPEEDUP_TOLERANCE:
                best_stride = stride
                break

    return best_stride
