"""ST-GCN+BC 训练数据集 + 合成数据生成.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1d
依据: dev-docs/stages/phase-3.md §3.1d + RESEARCH_STGCN_BC.md §4

数据源优先级:
    1. 真实数据（pyskl pickle，来自 YOLO26-pose + 人工标注）— 主路径
    2. InterPet4D kp_world 3D（无行为标签，仅用于无监督预训练）— 辅助
    3. 合成数据（基于行为模板 + 时序噪声）— baseline 验证

合成数据策略（无真实标注时的 baseline）:
    - 22 类行为各有 1 个"姿态模板"（关键点偏置）
    - 时序扰动：sin 波 + 高斯噪声模拟动作
    - 边界标签：每段动作前后 2 帧标记为边界
    - 用途：验证训练管线正确性 + 单元测试，非生产精度评估
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from backend.ml.behavior.constants import NUM_KEYPOINTS
from backend.ml.behavior.stgcn_bc.k9_graph import K9Graph
from backend.ml.behavior.stgcn_bc.labels import (
    ALL_BEHAVIORS_22,
    BEHAVIOR_TO_IDX,
    NUM_BEHAVIORS,
)


# 全局 K9Graph（轻量复用）
_K9_GRAPH = K9Graph()


# ============================================================================
# 22 类行为的"姿态模板"（用于合成数据生成）
# ============================================================================
# 每个模板是 (24, 3) 的关键点偏置，模拟该行为的典型姿态
# 基础姿态 = 站立（所有点在标准位置），各行为在基础上叠加偏置
# 偏置单位：相对犬体尺度的百分比（-0.3 ~ +0.3）

# 标准站立姿态（中型犬 ~0.6m 高，坐标系: x 左右 / y 前后 / z 上下）
# 索引顺序与 K9Graph.NODE_NAMES 一致（24 节点）
_BASE_POSE_3D: np.ndarray = np.array([
    # 前左肢 (0-2): paw → knee → elbow
    [-0.10,  0.10, 0.00],   # 0  front_left_paw
    [-0.10,  0.10, 0.30],   # 1  front_left_knee
    [-0.10,  0.10, 0.50],   # 2  front_left_elbow
    # 后左肢 (3-5): paw → knee → elbow
    [-0.10, -0.20, 0.00],   # 3  rear_left_paw
    [-0.10, -0.20, 0.30],   # 4  rear_left_knee
    [-0.10, -0.20, 0.50],   # 5  rear_left_elbow
    # 前右肢 (6-8): paw → knee → elbow
    [ 0.10,  0.10, 0.00],   # 6  front_right_paw
    [ 0.10,  0.10, 0.30],   # 7  front_right_knee
    [ 0.10,  0.10, 0.50],   # 8  front_right_elbow
    # 后右肢 (9-11): paw → knee → elbow
    [ 0.10, -0.20, 0.00],   # 9  rear_right_paw
    [ 0.10, -0.20, 0.30],   # 10 rear_right_knee
    [ 0.10, -0.20, 0.50],   # 11 rear_right_elbow
    # 尾部 (12-13)
    [ 0.00, -0.30, 0.55],   # 12 tail_start
    [ 0.00, -0.50, 0.50],   # 13 tail_end
    # 耳基 (14-15)
    [-0.05,  0.08, 0.78],   # 14 left_ear_base
    [ 0.05,  0.08, 0.78],   # 15 right_ear_base
    # 头部 (16-17)
    [ 0.00,  0.15, 0.80],   # 16 nose
    [ 0.00,  0.12, 0.75],   # 17 chin
    # 耳尖 (18-19)
    [-0.05,  0.10, 0.82],   # 18 left_ear_tip
    [ 0.05,  0.10, 0.82],   # 19 right_ear_tip
    # 眼睛 (20-21)
    [-0.03,  0.10, 0.80],   # 20 left_eye
    [ 0.03,  0.10, 0.80],   # 21 right_eye
    # 躯干根 (22-23)
    [ 0.00,  0.00, 0.60],   # 22 withers (根)
    [ 0.00,  0.05, 0.70],   # 23 throat
], dtype=np.float32)


def _behavior_pose_template(behavior: str) -> np.ndarray:
    """返回指定行为的姿态模板 (24, 3).

    基于行为语义构造关键点偏置：
        - sit: 后肢弯曲，臀部降低
        - down: 全身贴地，z 坐标降低
        - stand: 标准站立（无偏置）
        - bark: 头部上扬
        - bite: 头部前伸 + 张嘴
        - ...
    """
    pose = _BASE_POSE_3D.copy()
    # 24 节点索引（K9Graph）:
    # 0-4: 头部（nose, l_eye, r_eye, l_ear, r_ear）
    # 5-7: 颈部→肩部（throat, withers, chest）
    # 8-11: 前肢（l_shoulder, l_elbow, l_paw, r_shoulder, r_elbow, r_paw）
    # 12-15: 背部→尾部（tail_start, tail_mid, tail_end, hip）
    # 16-19: 后肢（l_hip, l_knee, l_foot, r_hip, r_knee, r_foot）
    # 20-23: 躯干补充

    behavior_lower = behavior.lower()
    if behavior_lower == "sit":
        # 后肢弯曲，臀部降低，前肢伸直
        pose[12:24, 2] -= 0.2  # 后半身 z 降低
        pose[16:24, 2] -= 0.3  # 后肢下沉
    elif behavior_lower == "down":
        # 全身贴地
        pose[:, 2] -= 0.5
    elif behavior_lower == "stand":
        pass  # 标准站立
    elif behavior_lower == "heel":
        # 站立 + 头部侧偏（跟随训导员）
        pose[0:5, 0] += 0.1
    elif behavior_lower == "sit_up":
        # 坐立：前肢抬起
        pose[8:14, 2] += 0.2
        pose[12:24, 2] -= 0.2
    elif behavior_lower == "stay":
        pass  # 保持当前位置（同 stand）
    elif behavior_lower == "bark":
        # 头部上扬 + 张嘴
        pose[0:5, 2] += 0.15
    elif behavior_lower == "bite":
        # 头部前伸 + 张嘴
        pose[0:5, 0] += 0.2
        pose[0:5, 2] += 0.05
    elif behavior_lower == "track":
        # 头部低伸（嗅闻地面）
        pose[0:5, 2] -= 0.2
        pose[0:5, 0] += 0.1
    elif behavior_lower in ("alert_sit", "alert_down"):
        # 物品指示：头部偏向一侧
        if "down" in behavior_lower:
            pose[:, 2] -= 0.5
        pose[0:5, 0] += 0.15
    elif behavior_lower == "search_blind":
        # 搜索：头部左右摆动（在模板中加偏置）
        pose[0:5, 0] += 0.1
    elif behavior_lower == "apprehend":
        # 扑咬：全身前冲
        pose[:, 0] += 0.15
        pose[0:5, 0] += 0.25
    elif behavior_lower == "escort":
        # 押解：站立 + 头部侧偏
        pose[0:5, 0] += 0.1
    elif behavior_lower == "obstacle":
        # 跨栏：前肢抬起
        pose[8:14, 2] += 0.3
    elif behavior_lower == "recall":
        # 召回：向训导员方向移动
        pose[:, 0] += 0.2
    elif behavior_lower == "watch":
        # 看守：头部凝视 + 警戒姿态
        pose[0:5, 2] += 0.1
    elif behavior_lower == "guard":
        # 警戒护卫：站立 + 全身紧绷
        pose[0:5, 2] += 0.05
    elif behavior_lower == "release":
        # 放口：头部后仰
        pose[0:5, 2] += 0.1
        pose[0:5, 0] -= 0.05
    elif behavior_lower == "retrieve":
        # 衔取：头部低伸含物
        pose[0:5, 2] -= 0.15
    elif behavior_lower == "jump":
        # 跳跃：全身腾空
        pose[:, 2] += 0.3
    elif behavior_lower == "scale":
        # 攀墙：前肢大幅抬起
        pose[8:14, 2] += 0.4
        pose[:, 2] += 0.1
    # 默认无偏置

    return pose.astype(np.float32)


def _generate_clip(
    behavior: str,
    T: int = 30,
    noise_std: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成单个合成 clip.

    Args:
        behavior: 行为名称（22 类之一）
        T: 帧数
        noise_std: 噪声标准差（相对犬体尺度）
        seed: 随机种子

    Returns:
        keypoints: (T, 24, 3) — 合成 3D 关键点
        boundary_labels: (T,) — 边界标签（前 2 帧 + 后 2 帧 = 1，中间 = 0）
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    template = _behavior_pose_template(behavior)  # (24, 3)

    # 时序扰动：sin 波 + 高斯噪声
    t = np.arange(T, dtype=np.float32) / max(T - 1, 1)  # [0, 1]
    # 每帧姿态 = template + sin 波扰动 + 高斯噪声
    sin_phase = rng.uniform(0, 2 * np.pi)
    sin_amp = 0.05  # 5% 体长的正弦扰动
    sin_wave = sin_amp * np.sin(2 * np.pi * t + sin_phase)  # (T,)

    # 每帧关键点 = template + sin 扰动（沿 x 方向）+ 高斯噪声
    keypoints = np.zeros((T, NUM_KEYPOINTS, 3), dtype=np.float32)
    for f in range(T):
        keypoints[f] = template + sin_wave[f] * 0.1  # 全身轻微摆动
        keypoints[f, :, 0] += rng.normal(0, noise_std, NUM_KEYPOINTS)  # x 噪声
        keypoints[f, :, 1] += rng.normal(0, noise_std, NUM_KEYPOINTS)  # y 噪声
        keypoints[f, :, 2] += rng.normal(0, noise_std, NUM_KEYPOINTS)  # z 噪声

    # 边界标签：前 2 帧 + 后 2 帧 = 1（动作边界），中间 = 0
    boundary_labels = np.zeros(T, dtype=np.float32)
    boundary_labels[:2] = 1.0
    boundary_labels[-2:] = 1.0

    return keypoints, boundary_labels


def make_synthetic_dataset(
    samples_per_class: int = 10,
    T: int = 30,
    noise_std: float = 0.05,
    seed: int = 42,
) -> List[Dict]:
    """生成合成数据集（22 类 × N 样本/类）.

    Args:
        samples_per_class: 每类样本数
        T: 每个样本帧数
        noise_std: 噪声标准差
        seed: 随机种子（可复现）

    Returns:
        List[Dict]，每个 Dict 含:
            - "keypoints": (T, 24, 3) np.ndarray
            - "label": int（行为索引 0-21）
            - "label_name": str（行为名称）
            - "boundary": (T,) np.ndarray
            - "frame_dir": str（clip ID）
    """
    rng = np.random.default_rng(seed)
    samples: List[Dict] = []

    for behavior in ALL_BEHAVIORS_22:
        label_idx = BEHAVIOR_TO_IDX[behavior]
        for i in range(samples_per_class):
            clip_seed = int(rng.integers(0, 2**31 - 1))
            kpt, boundary = _generate_clip(
                behavior, T=T, noise_std=noise_std, seed=clip_seed
            )
            samples.append({
                "keypoints": kpt,
                "label": label_idx,
                "label_name": behavior,
                "boundary": boundary,
                "frame_dir": f"syn_{behavior}_{i:03d}",
            })

    return samples


def save_synthetic_dataset(samples: List[Dict], output_path: str) -> None:
    """保存合成数据集为 pickle."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)


def load_pyskl_pickle(pkl_path: str) -> List[Dict]:
    """加载 pyskl 格式 pickle.

    预期格式: List[Dict]，每个 Dict 含:
        - "keypoint": (M, T, V, C) np.ndarray（M=1 单犬）
        - "label": int
        - "frame_dir": str
        - 可选 "keypoint_score": (M, T, V)
        - 可选 "boundary": (T,) — 边界标签

    Returns:
        标准化后的 List[Dict]，keypoint 字段为 (T, 24, 3)
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    samples: List[Dict] = []
    for ann in raw:
        kpt = ann["keypoint"]  # (M, T, V, C)
        if kpt.ndim != 4:
            raise ValueError(f"keypoint 应为 4D (M,T,V,C), 实际 {kpt.shape}")
        M, T, V, C = kpt.shape
        if M != 1:
            raise ValueError(f"当前仅支持单犬 M=1, 实际 M={M}")
        if V != NUM_KEYPOINTS:
            raise ValueError(f"关键点数应为 {NUM_KEYPOINTS}, 实际 {V}")

        # 单犬: 取 M=0
        kpt_single = kpt[0]  # (T, V, C)

        # 2D 模式: 合并 score 到 C=3
        if C == 2:
            score = ann.get("keypoint_score")
            if score is None:
                score = np.ones((T, V), dtype=np.float32)
            else:
                score = score[0]  # (T, V)
            kpt_3c = np.concatenate(
                [kpt_single, score[..., np.newaxis]], axis=-1
            )  # (T, V, 3)
        else:
            # 3D 模式: 直接使用
            kpt_3c = kpt_single

        samples.append({
            "keypoints": kpt_3c.astype(np.float32),
            "label": int(ann.get("label", -1)),
            "label_name": ann.get("label_name", ""),
            "boundary": ann.get("boundary", np.zeros(T, dtype=np.float32)),
            "frame_dir": ann.get("frame_dir", "unknown"),
        })

    return samples


# ============================================================================
# PyTorch Dataset
# ============================================================================


class STGCNBCDataset(Dataset):
    """ST-GCN+BC 训练数据集.

    支持两种加载方式:
        1. 从内存 List[Dict]（合成数据或已加载的真实数据）
        2. 从 pyskl pickle 文件

    数据增强:
        - 随机时间裁剪（保持 T 一致）
        - 随机水平翻转（x → -x）
        - 随机尺度抖动
    """

    def __init__(
        self,
        samples: Optional[List[Dict]] = None,
        pkl_path: Optional[str] = None,
        T: int = 30,
        augment: bool = False,
        normalize: bool = True,
    ):
        if samples is not None:
            self.samples = samples
        elif pkl_path is not None:
            self.samples = load_pyskl_pickle(pkl_path)
        else:
            raise ValueError("必须提供 samples 或 pkl_path")

        self.T = T
        self.augment = augment
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        kpt = sample["keypoints"].copy()  # (T_orig, 24, 3)
        label = sample["label"]
        boundary = sample["boundary"].copy()  # (T_orig,)

        T_orig = kpt.shape[0]

        # 时间裁剪/补齐到固定 T
        if T_orig >= self.T:
            # 随机裁剪
            if self.augment:
                start = np.random.randint(0, T_orig - self.T + 1)
            else:
                start = (T_orig - self.T) // 2  # 居中裁剪
            kpt = kpt[start:start + self.T]
            boundary = boundary[start:start + self.T]
        else:
            # 循环补齐
            pad_len = self.T - T_orig
            kpt = np.concatenate(
                [kpt, np.tile(kpt[-1:], (pad_len, 1, 1))], axis=0
            )
            boundary = np.concatenate(
                [boundary, np.zeros(pad_len, dtype=np.float32)]
            )

        # 数据增强
        if self.augment:
            # 随机水平翻转 (x → -x)
            if np.random.rand() < 0.5:
                kpt[..., 0] = -kpt[..., 0]
            # 随机尺度抖动 (0.9 ~ 1.1)
            scale = 1.0 + np.random.uniform(-0.1, 0.1)
            kpt[..., :3] = kpt[..., :3] * scale

        # 归一化（按 withers 中心 + 体长尺度）
        if self.normalize:
            kpt = self._normalize(kpt)

        return {
            "keypoints": torch.from_numpy(kpt).float(),  # (T, 24, 3)
            "label": torch.tensor(label, dtype=torch.long),
            "boundary": torch.from_numpy(boundary).float(),  # (T,)
            "frame_dir": sample.get("frame_dir", ""),
        }

    @staticmethod
    def _normalize(kpt: np.ndarray) -> np.ndarray:
        """按 withers 中心 + 体长尺度归一化."""
        # withers 索引 = 22（K9Graph）
        center = kpt[:, 22:23, :].mean(axis=0, keepdims=True)  # (1, 1, 3)
        kpt = kpt - center
        # 参考骨骼: withers(22) → tail_start(12)
        bone_ref = kpt[:, 22, :2] - kpt[:, 12, :2]  # (T, 2)
        bone_len = np.linalg.norm(bone_ref, axis=-1).mean()  # 标量
        if bone_len < 1e-6:
            bone_len = 1.0
        kpt[..., :2] = kpt[..., :2] / bone_len
        return kpt.astype(np.float32)


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """批处理 collate 函数.

    将 list of dict 转为 dict of batched tensor.
    输出:
        - keypoints: (B, T, V=24, C=3)
        - labels: (B,)
        - boundaries: (B, T)
        - frame_dirs: List[str]
    """
    keypoints = torch.stack([b["keypoints"] for b in batch])  # (B, T, 24, 3)
    labels = torch.stack([b["label"] for b in batch])  # (B,)
    boundaries = torch.stack([b["boundary"] for b in batch])  # (B, T)
    frame_dirs = [b["frame_dir"] for b in batch]
    return {
        "keypoints": keypoints,
        "labels": labels,
        "boundaries": boundaries,
        "frame_dirs": frame_dirs,
    }


__all__ = [
    "STGCNBCDataset",
    "make_synthetic_dataset",
    "save_synthetic_dataset",
    "load_pyskl_pickle",
    "collate_fn",
]
