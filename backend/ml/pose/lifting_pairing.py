"""2D-3D 配对构建 + 归一化（Phase 3.3b）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.3b
依据: dev-docs/research/RESEARCH_3D_POSE_RECONSTRUCTION.md §6.2 + dev-docs/stages/phase-3.md §3.3b

设计目标:
    将 InterPet4D kp_world 3D 坐标 + 合成相机投影的 2D 坐标，组织为
    MotionBERT-Lite 训练所需的 (2D_input, 3D_target) 配对。

MotionBERT 训练数据格式:
    - 2D input: (batch, T, 24, 2) — 归一化到 [-1, 1]
    - 3D target: (batch, T, 24, 3) — 根关节中心化 + 缩放归一化
    - confidence: (batch, T, 24) — 关键点置信度（可选，用于加权损失）

配对构建流程:
    1. 加载 InterPet4D kp_world (T, 24, 3) + kp_weight (T, 24)
    2. 生成合成相机阵列（N 个视角）
    3. 投影 kp_world → 2D (N, T, 24, 2)
    4. 归一化 3D（根关节中心化 + 尺度归一化）
    5. 滑动窗口切片（T → 多个 T_window 子序列）
    6. 输出 (2D, 3D, conf) 配对张量

归一化策略（与 MotionBERT 对齐）:
    - 2D: 归一化到 [-1, 1]（图像中心原点，短边为单位长度）
    - 3D: 根关节（withers, idx=22）中心化 + 骨骼长度归一化

不引入兜底层:
    - 不做缺失帧插补
    - 不混合正交/透视投影（每个 clip 固定一种模式）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from backend.ml.pose.interpet4d_loader import (
    InterPet4DClip,
    NUM_KEYPOINTS,
    load_all_clips,
    load_clip,
)
from backend.ml.pose.camera_projection import (
    SyntheticCamera,
    generate_synthetic_cameras,
    project_clip_to_2d,
)

logger = logging.getLogger(__name__)


# MotionBERT 默认窗口长度（27 帧，MotionBERT 原始设置）
DEFAULT_WINDOW_SIZE = 27
DEFAULT_WINDOW_STRIDE = 13  # 50% 重叠

# 项目 Dog-Pose 24 关键点中的根关节索引（与 K9Graph 一致，withers=22）
ROOT_KEYPOINT_IDX = 22


@dataclass
class LiftingSample:
    """单个 2D-to-3D lifting 训练样本.

    Attributes:
        keypoints_2d: (T, 24, 2) 归一化 2D 坐标 [-1, 1]
        keypoints_3d: (T, 24, 3) 归一化 3D 坐标（根关节中心化 + 尺度归一化）
        confidence: (T, 24) 关键点置信度 [0, 1]
        clip_id: 源 clip ID
        camera_idx: 相机索引
        frame_start: 起始帧索引（在原 clip 中的位置）
        window_size: 窗口长度
        scale: 3D 归一化缩放因子（推理时用于恢复真实尺度）
    """

    keypoints_2d: np.ndarray  # (T, 24, 2)
    keypoints_3d: np.ndarray  # (T, 24, 3)
    confidence: np.ndarray  # (T, 24)
    clip_id: str = ""
    camera_idx: int = 0
    frame_start: int = 0
    window_size: int = DEFAULT_WINDOW_SIZE
    scale: float = 1.0

    @property
    def num_frames(self) -> int:
        return int(self.keypoints_2d.shape[0])

    def summary(self) -> Dict:
        return {
            "clip_id": self.clip_id,
            "camera_idx": self.camera_idx,
            "frame_start": self.frame_start,
            "window_size": self.window_size,
            "num_frames": self.num_frames,
            "scale": round(float(self.scale), 4),
        }


def normalize_3d_keypoints(
    keypoints_3d: np.ndarray,
    root_idx: int = ROOT_KEYPOINT_IDX,
    scale_mode: str = "bone_length",
) -> Tuple[np.ndarray, float]:
    """3D 关键点归一化（根关节中心化 + 尺度归一化）.

    Args:
        keypoints_3d: (..., 24, 3) 3D 坐标
        root_idx: 根关节索引（默认 withers=22）
        scale_mode: 尺度归一化模式
            - "bone_length": 用平均骨骼长度归一化
            - "bbox": 用关键点包围盒对角线归一化
            - "none": 仅根关节中心化

    Returns:
        (normalized_3d, scale) — 归一化后的 3D 坐标 + 缩放因子
    """
    kp3d = np.asarray(keypoints_3d, dtype=np.float32)
    orig_shape = kp3d.shape
    kp3d_flat = kp3d.reshape(-1, NUM_KEYPOINTS, 3)

    # 根关节中心化
    root = kp3d_flat[:, root_idx:root_idx + 1, :]  # (N, 1, 3)
    kp3d_centered = kp3d_flat - root  # (N, 24, 3)

    # 尺度归一化
    scale = 1.0
    if scale_mode == "bone_length":
        # 平均骨骼长度（所有相邻关节对距离的均值）
        # 简化：用所有关键点到根关节的距离均值
        dists = np.linalg.norm(kp3d_centered, axis=-1)  # (N, 24)
        mean_dist = float(dists.mean())
        if mean_dist > 1e-6:
            scale = mean_dist
            kp3d_centered = kp3d_centered / scale
    elif scale_mode == "bbox":
        # 包围盒对角线
        bbox_min = kp3d_centered.min(axis=-2)  # (N, 3)
        bbox_max = kp3d_centered.max(axis=-2)  # (N, 3)
        diag = np.linalg.norm(bbox_max - bbox_min, axis=-1)  # (N,)
        diag_mean = float(diag.mean())
        if diag_mean > 1e-6:
            scale = diag_mean
            kp3d_centered = kp3d_centered / scale
    elif scale_mode == "none":
        pass
    else:
        raise ValueError(f"未知 scale_mode: {scale_mode}")

    return kp3d_centered.reshape(orig_shape), scale


def denormalize_3d_keypoints(
    keypoints_3d_normalized: np.ndarray,
    scale: float,
    root_position: Optional[np.ndarray] = None,
    root_idx: int = ROOT_KEYPOINT_IDX,
) -> np.ndarray:
    """3D 关键点反归一化（恢复真实尺度 + 根关节位置）.

    Args:
        keypoints_3d_normalized: (..., 24, 3) 归一化坐标
        scale: 缩放因子
        root_position: (3,) 根关节原始位置（None=原点）
        root_idx: 根关节索引

    Returns:
        np.ndarray (..., 24, 3) — 恢复后的 3D 坐标
    """
    kp3d = np.asarray(keypoints_3d_normalized, dtype=np.float32) * float(scale)
    if root_position is not None:
        kp3d = kp3d + np.asarray(root_position, dtype=np.float32).reshape(3)
    return kp3d


def slice_windows(
    sequence: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> List[np.ndarray]:
    """滑动窗口切片.

    Args:
        sequence: (T, ...) 时间序列
        window_size: 窗口长度
        stride: 步长

    Returns:
        List[np.ndarray] — 窗口列表，每个 (window_size, ...)
    """
    T = sequence.shape[0]
    if T < window_size:
        # 不足一个窗口，pad 到 window_size
        pad_len = window_size - T
        pad_shape = [(0, pad_len)] + [(0, 0)] * (sequence.ndim - 1)
        padded = np.pad(sequence, pad_shape, mode="edge")
        return [padded]

    windows = []
    for start in range(0, T - window_size + 1, stride):
        windows.append(sequence[start:start + window_size])
    return windows


def build_pairs_from_clip(
    clip: InterPet4DClip,
    num_cameras: int = 8,
    projection_mode: str = "orthographic",
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
    camera_distance: float = 3.0,
    random_seed: int = 42,
    min_kp_weight: float = 0.3,
) -> List[LiftingSample]:
    """从单个 InterPet4D clip 构建 2D-3D 配对样本.

    Args:
        clip: InterPet4DClip 数据
        num_cameras: 合成相机数量
        projection_mode: 投影模式（"orthographic" / "perspective"）
        window_size: 滑动窗口长度
        stride: 滑动窗口步长
        camera_distance: 相机距离（米）
        random_seed: 相机生成随机种子
        min_kp_weight: 最小关键点置信度（低于此值的点 confidence 置 0）

    Returns:
        List[LiftingSample] — 配对样本列表
    """
    T = clip.num_frames
    if T < 1:
        logger.warning(f"[build_pairs_from_clip] {clip.clip_id} 帧数不足: {T}")
        return []

    # 1. 生成合成相机阵列
    cameras = generate_synthetic_cameras(
        num_cameras=num_cameras,
        distance=camera_distance,
        random_seed=random_seed,
    )

    # 2. 投影 kp_world → 2D (num_cameras, T, 24, 2)
    projections_2d = project_clip_to_2d(
        clip.kp_world, cameras, mode=projection_mode, normalize=True
    )

    # 3. 3D 归一化（根关节中心化 + 尺度归一化）
    kp_3d_normalized, scale = normalize_3d_keypoints(
        clip.kp_world, root_idx=ROOT_KEYPOINT_IDX, scale_mode="bone_length"
    )

    # 4. 关键点置信度过滤
    confidence = clip.kp_weight.copy()
    confidence[confidence < min_kp_weight] = 0.0

    # 5. 滑动窗口切片
    samples: List[LiftingSample] = []
    for cam_idx in range(num_cameras):
        kp_2d_windows = slice_windows(projections_2d[cam_idx], window_size, stride)
        kp_3d_windows = slice_windows(kp_3d_normalized, window_size, stride)
        conf_windows = slice_windows(confidence, window_size, stride)
        frame_starts = list(range(0, T - window_size + 1, stride))
        if T < window_size:
            frame_starts = [0]

        for i, (w2d, w3d, wc, fstart) in enumerate(
            zip(kp_2d_windows, kp_3d_windows, conf_windows, frame_starts)
        ):
            samples.append(
                LiftingSample(
                    keypoints_2d=w2d.astype(np.float32),
                    keypoints_3d=w3d.astype(np.float32),
                    confidence=wc.astype(np.float32),
                    clip_id=clip.clip_id,
                    camera_idx=cam_idx,
                    frame_start=int(fstart),
                    window_size=window_size,
                    scale=float(scale),
                )
            )

    return samples


def build_dataset(
    smal_dir: Union[str, Path] = "data/interpet4d/smal_npy",
    num_cameras: int = 8,
    projection_mode: str = "orthographic",
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
    max_clips: Optional[int] = None,
    min_frames: int = 10,
    min_kp_weight: float = 0.3,
    min_valid_kp_ratio: float = 0.5,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """批量构建 2D-to-3D lifting 训练数据集.

    Args:
        smal_dir: InterPet4D smal_npy 目录
        num_cameras: 每个 clip 的合成相机数
        projection_mode: 投影模式
        window_size: 滑动窗口长度
        stride: 滑动窗口步长
        max_clips: 最大加载 clip 数（None=全部）
        min_frames: 最小帧数过滤
        min_kp_weight: 最小关键点置信度
        min_valid_kp_ratio: 最小有效关键点比例
        random_seed: 随机种子

    Returns:
        (keypoints_2d, keypoints_3d, confidence, metadata)
        - keypoints_2d: (N, T, 24, 2) 2D 输入
        - keypoints_3d: (N, T, 24, 3) 3D 真值
        - confidence: (N, T, 24) 置信度
        - metadata: List[Dict] 每个样本的元数据
    """
    clips = load_all_clips(
        smal_dir=smal_dir,
        load_optional=False,
        min_frames=min_frames,
        min_kp_weight=min_kp_weight,
        min_valid_kp_ratio=min_valid_kp_ratio,
    )

    if max_clips is not None:
        clips = clips[:max_clips]

    logger.info(f"[build_dataset] 加载 {len(clips)} clips, 每个生成 {num_cameras} 相机视角")

    all_samples: List[LiftingSample] = []
    for clip in clips:
        samples = build_pairs_from_clip(
            clip,
            num_cameras=num_cameras,
            projection_mode=projection_mode,
            window_size=window_size,
            stride=stride,
            random_seed=random_seed,
            min_kp_weight=min_kp_weight,
        )
        all_samples.extend(samples)

    if not all_samples:
        logger.warning("[build_dataset] 无有效样本")
        empty_2d = np.zeros((0, window_size, NUM_KEYPOINTS, 2), dtype=np.float32)
        empty_3d = np.zeros((0, window_size, NUM_KEYPOINTS, 3), dtype=np.float32)
        empty_conf = np.zeros((0, window_size, NUM_KEYPOINTS), dtype=np.float32)
        return empty_2d, empty_3d, empty_conf, []

    # 堆叠
    keypoints_2d = np.stack([s.keypoints_2d for s in all_samples], axis=0)
    keypoints_3d = np.stack([s.keypoints_3d for s in all_samples], axis=0)
    confidence = np.stack([s.confidence for s in all_samples], axis=0)
    metadata = [
        {
            "clip_id": s.clip_id,
            "camera_idx": s.camera_idx,
            "frame_start": s.frame_start,
            "window_size": s.window_size,
            "scale": s.scale,
        }
        for s in all_samples
    ]

    logger.info(
        f"[build_dataset] 构建完成: {len(all_samples)} 样本, "
        f"2D shape={keypoints_2d.shape}, 3D shape={keypoints_3d.shape}"
    )
    return keypoints_2d, keypoints_3d, confidence, metadata


def compute_dataset_statistics(
    keypoints_3d: np.ndarray,
    keypoints_2d: np.ndarray,
) -> Dict:
    """计算数据集统计信息（用于归一化/异常检测）.

    Args:
        keypoints_3d: (N, T, 24, 3) 3D 坐标
        keypoints_2d: (N, T, 24, 2) 2D 坐标

    Returns:
        Dict — 统计信息
    """
    if keypoints_3d.size == 0 or keypoints_2d.size == 0:
        return {"num_samples": 0}

    return {
        "num_samples": int(keypoints_3d.shape[0]),
        "window_size": int(keypoints_3d.shape[1]),
        "kp_3d_stats": {
            "min": round(float(keypoints_3d.min()), 4),
            "max": round(float(keypoints_3d.max()), 4),
            "mean": round(float(keypoints_3d.mean()), 4),
            "std": round(float(keypoints_3d.std()), 4),
        },
        "kp_2d_stats": {
            "min": round(float(keypoints_2d.min()), 4),
            "max": round(float(keypoints_2d.max()), 4),
            "mean": round(float(keypoints_2d.mean()), 4),
            "std": round(float(keypoints_2d.std()), 4),
        },
    }


def train_val_split(
    keypoints_2d: np.ndarray,
    keypoints_3d: np.ndarray,
    confidence: np.ndarray,
    metadata: List[Dict],
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """按 clip_id 划分训练/验证集（避免数据泄漏）.

    Args:
        keypoints_2d, keypoints_3d, confidence: 张量数据
        metadata: 元数据列表
        val_ratio: 验证集比例
        random_seed: 随机种子

    Returns:
        Dict — {
            "train_2d", "train_3d", "train_conf",
            "val_2d", "val_3d", "val_conf",
            "train_clips", "val_clips",
        }
    """
    rng = np.random.default_rng(random_seed)

    # 提取唯一 clip_id
    clip_ids = sorted(set(m["clip_id"] for m in metadata))
    num_val = max(1, int(len(clip_ids) * val_ratio))

    # 随机选 val clips
    val_clip_set = set(rng.choice(clip_ids, size=num_val, replace=False).tolist())
    train_clip_set = set(clip_ids) - val_clip_set

    # 划分
    train_idx = [i for i, m in enumerate(metadata) if m["clip_id"] in train_clip_set]
    val_idx = [i for i, m in enumerate(metadata) if m["clip_id"] in val_clip_set]

    return {
        "train_2d": keypoints_2d[train_idx],
        "train_3d": keypoints_3d[train_idx],
        "train_conf": confidence[train_idx],
        "val_2d": keypoints_2d[val_idx],
        "val_3d": keypoints_3d[val_idx],
        "val_conf": confidence[val_idx],
        "train_clips": sorted(train_clip_set),
        "val_clips": sorted(val_clip_set),
        "num_train": len(train_idx),
        "num_val": len(val_idx),
    }


__all__ = [
    "LiftingSample",
    "normalize_3d_keypoints",
    "denormalize_3d_keypoints",
    "slice_windows",
    "build_pairs_from_clip",
    "build_dataset",
    "compute_dataset_statistics",
    "train_val_split",
    "DEFAULT_WINDOW_SIZE",
    "DEFAULT_WINDOW_STRIDE",
    "ROOT_KEYPOINT_IDX",
]
