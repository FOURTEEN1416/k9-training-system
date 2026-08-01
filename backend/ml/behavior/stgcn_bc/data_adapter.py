"""数据格式适配器：YOLO26-pose ↔ pyskl 骨架格式.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1b
依据: dev-docs/research/RESEARCH_STGCN_BC.md §4.2 + §4.3

格式约定:
    YOLO26-pose 输出:
        keypoints_sequence: np.ndarray, shape=(T, 24, 3), 通道=[x, y, conf]

    pyskl 标准格式（dict）:
        {
            'split': 'train' | 'val',
            'frame_dir': str,         # 视频/片段 ID
            'total_frames': int,      # T
            'label': int,             # 类别索引（22 类）
            'keypoint': np.ndarray,   # shape=(M, T, V, C), M=犬数, V=24, C=2 或 3
            'keypoint_score': np.ndarray,  # shape=(M, T, V), 仅 2D 模式
            'ann_info': {
                'num_persons': M,     # 单犬 M=1
                'num_clases': 22,     # ST-GCN+BC 22 类
                'num_joints': 24,     # K9Graph 24 节点
            }
        }

转换策略:
    - 2D 模式（默认）: C=2 (x, y) + 独立 keypoint_score（与 pyskl NTU/HRNet 一致）
    - 3D 模式: C=3 (x, y, z)，无 keypoint_score（与 pyskl 3D 模式一致）
    - 单犬场景: M=1（多犬追踪 Phase 3.2 完成后扩展）

骨骼流计算:
    bone[v] = joint[v] - joint[parent[v]]（由 K9Graph.parent 定义父子关系）

归一化:
    按犬体 bbox 中心 + 尺度归一化（参考 pyskl preprocess）
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.ml.behavior.constants import NUM_KEYPOINTS
from backend.ml.behavior.stgcn_bc.k9_graph import K9Graph
from backend.ml.behavior.stgcn_bc.labels import NUM_BEHAVIORS


# 全局 K9Graph 实例（轻量，复用）
_K9_GRAPH = K9Graph()


def keypoints_to_pyskl(
    keypoints_sequence: np.ndarray,
    frame_dir: str = "k9_clip",
    label: Optional[int] = None,
    label_name: Optional[str] = None,
    split: str = "train",
    mode: str = "2d",
) -> Dict:
    """YOLO26-pose (T, 24, 3) → pyskl 标准格式 dict.

    Args:
        keypoints_sequence: shape=(T, 24, 3), 通道=[x, y, conf]
        frame_dir: 视频/片段标识符
        label: 类别索引（0-21）。若为 None 则不设标签（推理模式）
        label_name: 类别名称（可选，用于调试）
        split: 'train' | 'val' | 'test'
        mode: '2d'（默认，分离 score）或 '3d'（C=3 含 z，无独立 score）

    Returns:
        pyskl 格式 dict

    Raises:
        ValueError: shape 不匹配 / mode 非法
    """
    if keypoints_sequence.ndim != 3 or keypoints_sequence.shape[1:] != (NUM_KEYPOINTS, 3):
        raise ValueError(
            f"keypoints_sequence shape 应为 (T, {NUM_KEYPOINTS}, 3), "
            f"实际 {keypoints_sequence.shape}"
        )

    T, V, C = keypoints_sequence.shape  # T=帧数, V=24, C=3
    if mode not in ("2d", "3d"):
        raise ValueError(f"mode 应为 '2d' 或 '3d', 实际 {mode!r}")

    # 单犬场景 M=1
    M = 1

    # 构造 keypoint (M, T, V, C')
    if mode == "2d":
        # 2D: C'=2 (x, y)，score 单独存放
        # (T, V, 2) → (M=1, T, V, 2)
        kpt = keypoints_sequence[..., :2][np.newaxis, ...]
        # score: (M, T, V)
        score = keypoints_sequence[..., 2][np.newaxis, ...]  # (1, T, V)
    else:
        # 3D: C'=3 (x, y, z)，无独立 score（假设 z 来自 MotionBERT 或 InterPet4D kp_world）
        kpt = keypoints_sequence[np.newaxis, ...]  # (1, T, V, 3)
        score = None

    # 标签索引解析
    if label is None and label_name is not None:
        from backend.ml.behavior.stgcn_bc.labels import BEHAVIOR_TO_IDX
        label = BEHAVIOR_TO_IDX.get(label_name, -1)

    ann: Dict = {
        "split": split,
        "frame_dir": frame_dir,
        "total_frames": int(T),
        "label": int(label) if label is not None else -1,
        "keypoint": kpt.astype(np.float32),
        "ann_info": {
            "num_persons": M,
            "num_clases": NUM_BEHAVIORS,  # 22
            "num_joints": V,  # 24
        },
    }
    if score is not None:
        ann["keypoint_score"] = score.astype(np.float32)

    if label_name is not None:
        ann["label_name"] = label_name

    return ann


def pyskl_to_keypoints(ann: Dict, mode: str = "2d") -> np.ndarray:
    """pyskl 格式 → YOLO26-pose (T, 24, 3).

    Args:
        ann: pyskl 格式 dict
        mode: '2d'（合并 keypoint + keypoint_score）或 '3d'（直接取 keypoint）

    Returns:
        np.ndarray, shape=(T, 24, 3)

    Raises:
        ValueError: 格式不匹配
    """
    kpt = ann["keypoint"]  # (M, T, V, C)
    if kpt.ndim != 4:
        raise ValueError(f"keypoint 应为 4D (M, T, V, C), 实际 {kpt.shape}")

    M, T, V, C = kpt.shape
    if M != 1:
        raise ValueError(f"当前仅支持单犬 M=1, 实际 M={M}")
    if V != NUM_KEYPOINTS:
        raise ValueError(f"关键点数应为 {NUM_KEYPOINTS}, 实际 {V}")

    if mode == "2d":
        if C != 2:
            raise ValueError(f"2D 模式 keypoint C 应为 2, 实际 {C}")
        score = ann.get("keypoint_score")  # (M, T, V)
        if score is None:
            # 无 score 时填充 1.0
            score = np.ones((M, T, V), dtype=np.float32)
        # 合并: (T, V, 3) = [x, y, conf]
        kpt_2d = kpt[0]  # (T, V, 2)
        score_1d = score[0]  # (T, V)
        result = np.concatenate([kpt_2d, score_1d[..., np.newaxis]], axis=-1)  # (T, V, 3)
    elif mode == "3d":
        if C != 3:
            raise ValueError(f"3D 模式 keypoint C 应为 3, 实际 {C}")
        result = kpt[0]  # (T, V, 3)
    else:
        raise ValueError(f"mode 应为 '2d' 或 '3d', 实际 {mode!r}")

    return result.astype(np.float32)


def compute_bone_flow(keypoints_sequence: np.ndarray) -> np.ndarray:
    """计算骨骼流: bone[v] = joint[v] - joint[parent[v]].

    用于 ST-GCN++ 多流融合（joint + bone + motion 三流）。

    Args:
        keypoints_sequence: shape=(T, 24, 3)

    Returns:
        np.ndarray, shape=(T, 24, 3) — 骨骼向量（根节点 bone=0）
    """
    if keypoints_sequence.ndim != 3:
        raise ValueError(f"shape 应为 (T, 24, 3), 实际 {keypoints_sequence.shape}")

    parent = _K9_GRAPH.parent  # (24,) int64
    bone = np.zeros_like(keypoints_sequence, dtype=np.float32)
    for v in range(NUM_KEYPOINTS):
        p = parent[v]
        if p >= 0:  # 根节点 parent=-1，bone=0
            bone[:, v, :] = keypoints_sequence[:, v, :] - keypoints_sequence[:, p, :]
    return bone


def compute_motion_flow(keypoints_sequence: np.ndarray) -> np.ndarray:
    """计算运动流: motion[t] = joint[t] - joint[t-1].

    用于 ST-GCN++ 多流融合（joint + bone + motion 三流）。

    Args:
        keypoints_sequence: shape=(T, 24, 3)

    Returns:
        np.ndarray, shape=(T, 24, 3) — 第 0 帧运动为 0
    """
    if keypoints_sequence.ndim != 3:
        raise ValueError(f"shape 应为 (T, 24, 3), 实际 {keypoints_sequence.shape}")

    motion = np.zeros_like(keypoints_sequence, dtype=np.float32)
    if keypoints_sequence.shape[0] > 1:
        motion[1:, :, :] = keypoints_sequence[1:, :, :] - keypoints_sequence[:-1, :, :]
    return motion


def normalize_keypoints(
    keypoints_sequence: np.ndarray,
    center_idx: int = 22,  # WITHERS
    scale_idx: Tuple[int, int] = (22, 12),  # (withers, tail_start) 作为参考尺度
) -> np.ndarray:
    """归一化关键点序列（按犬体中心 + 尺度）.

    参考 pyskl preprocess 函数:
        1. 平移: 减去中心关键点坐标
        2. 缩放: 除以参考骨骼长度（如 withers→tail_start）

    Args:
        keypoints_sequence: shape=(T, 24, 3)
        center_idx: 中心关键点索引（默认 WITHERS=22）
        scale_idx: (a, b) 参考骨骼两端索引（默认 withers→tail_start）

    Returns:
        np.ndarray, shape=(T, 24, 3) — 归一化后的序列
    """
    if keypoints_sequence.ndim != 3:
        raise ValueError(f"shape 应为 (T, 24, 3), 实际 {keypoints_sequence.shape}")

    kpt = keypoints_sequence.astype(np.float32).copy()

    # 1. 平移: 每帧减去中心点
    center = kpt[:, center_idx:center_idx + 1, :]  # (T, 1, 3)
    kpt = kpt - center

    # 2. 缩放: 每帧除以参考骨骼长度
    a, b = scale_idx
    bone_ref = kpt[:, a, :2] - kpt[:, b, :2]  # (T, 2) — 仅 xy 计算长度
    bone_len = np.linalg.norm(bone_ref, axis=-1)  # (T,)
    bone_len = np.where(bone_len < 1e-6, 1.0, bone_len)  # 避免 0 除
    kpt[..., :2] = kpt[..., :2] / bone_len[:, np.newaxis, np.newaxis]
    # conf 通道不缩放
    if kpt.shape[-1] >= 3:
        kpt[..., 2] = keypoints_sequence[..., 2]

    return kpt


def build_pyskl_annotation(
    clips: List[Tuple[str, np.ndarray, Optional[int]]],
    split: str = "train",
    mode: str = "2d",
) -> List[Dict]:
    """批量构建 pyskl 标注列表.

    Args:
        clips: [(frame_dir, keypoints_sequence, label), ...]
            - frame_dir: str, 片段 ID
            - keypoints_sequence: np.ndarray, (T, 24, 3)
            - label: int 或 None
        split: 'train' | 'val' | 'test'
        mode: '2d' 或 '3d'

    Returns:
        List[Dict] — pyskl 标注列表（可 pickle 保存为 .pkl）
    """
    annotations: List[Dict] = []
    for frame_dir, kpt_seq, label in clips:
        ann = keypoints_to_pyskl(
            keypoints_sequence=kpt_seq,
            frame_dir=frame_dir,
            label=label,
            split=split,
            mode=mode,
        )
        annotations.append(ann)
    return annotations


def save_pyskl_pickle(annotations: List[Dict], output_path: str) -> None:
    """保存 pyskl 标注列表为 .pkl 文件.

    Args:
        annotations: pyskl 标注列表
        output_path: 输出路径（.pkl）
    """
    import pickle
    from pathlib import Path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(annotations, f)


__all__ = [
    "keypoints_to_pyskl",
    "pyskl_to_keypoints",
    "compute_bone_flow",
    "compute_motion_flow",
    "normalize_keypoints",
    "build_pyskl_annotation",
    "save_pyskl_pickle",
]
