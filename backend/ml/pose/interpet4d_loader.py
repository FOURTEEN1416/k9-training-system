"""InterPet4D SMAL 数据加载器（Phase 3.3b）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.3b
依据: dev-docs/research/RESEARCH_3D_POSE_RECONSTRUCTION.md §6 + dev-docs/stages/phase-3.md §3.3b

数据源:
    - `data/interpet4d/smal_npy/*.npz` — SMAL 宠物身体拟合
    - 每个 .npz 包含:
        * kp_world: (T, 24, 3) — 24 关键点世界坐标（米）✅ 与项目 Dog-Pose 24 点对齐
        * kp_weight: (T, 24) — 关键点置信度
        * frame_idx: (T,) int32 — 原始帧索引（稀疏，非连续）
        * R_world: (T, 3, 3) — 全局旋转
        * t_world: (T, 3) — 全局平移
        * s_world: (T,) — 全局缩放
        * pose_rotmat: (T, 35, 3, 3) — SMAL 关节旋转
        * betas: (T, 30) — 形状系数
        * betas_limbs: (T, 7) — 肢体形状系数

数据集规模:
    - 227 clips 总数，226 clips 有 SMAL fits（1 clip 缺失）
    - 13 只犬（dog00-dog12），~23 个参与者（p01-p23）
    - 每个 clip ~17-20 秒，帧数 ~150-400

命名约定:
    interpet_dog{DD}_p{PP}_take{TT}_ego_{NNN}
    clip_id = 文件名（去扩展名），跨目录联接键

用途:
    1. 加载 kp_world (T, 24, 3) 作为 2D-to-3D lifting 监督真值
    2. 加载 kp_weight 作为关键点置信度过滤
    3. 配合 camera_projection.py 合成 2D 投影 → (2D, 3D) 配对

不引入兜底层:
    - 不对缺失 clip 做插补
    - 不混合 pet_npy (20 点) 与 smal_npy (24 点)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# InterPet4D SMAL 关键字段
SMAL_KP_WORLD_KEY = "kp_world"        # (T, 24, 3)
SMAL_KP_WEIGHT_KEY = "kp_weight"      # (T, 24)
SMAL_FRAME_IDX_KEY = "frame_idx"      # (T,) int32
SMAL_R_WORLD_KEY = "R_world"          # (T, 3, 3)
SMAL_T_WORLD_KEY = "t_world"          # (T, 3)
SMAL_S_WORLD_KEY = "s_world"          # (T,)
SMAL_POSE_ROTMAT_KEY = "pose_rotmat"  # (T, 35, 3, 3)
SMAL_BETAS_KEY = "betas"              # (T, 30)
SMAL_BETAS_LIMBS_KEY = "betas_limbs"  # (T, 7)

# 项目 Dog-Pose 24 关键点（与 InterPet4D SMAL kp_world 对齐）
NUM_KEYPOINTS = 24

# 默认数据目录
DEFAULT_INTERPET4D_SMAL_DIR = "data/interpet4d/smal_npy"


@dataclass
class InterPet4DClip:
    """单个 InterPet4D clip 的 SMAL 数据.

    Attributes:
        clip_id: clip 标识符（文件名去扩展名）
        kp_world: (T, 24, 3) 世界坐标（米）
        kp_weight: (T, 24) 关键点置信度 [0, 1]
        frame_idx: (T,) 原始帧索引
        R_world: (T, 3, 3) 全局旋转矩阵
        t_world: (T, 3) 全局平移
        s_world: (T,) 全局缩放
        dog_id: 犬只 ID（dog00-dog12）
        participant_id: 参与者 ID（p01-p23）
        take_id: take 编号
        clip_idx: clip 序号（同一 take 内）
    """

    clip_id: str
    kp_world: np.ndarray  # (T, 24, 3)
    kp_weight: np.ndarray  # (T, 24)
    frame_idx: np.ndarray  # (T,)
    R_world: Optional[np.ndarray] = None  # (T, 3, 3)
    t_world: Optional[np.ndarray] = None  # (T, 3)
    s_world: Optional[np.ndarray] = None  # (T,)
    pose_rotmat: Optional[np.ndarray] = None  # (T, 35, 3, 3)
    betas: Optional[np.ndarray] = None  # (T, 30)
    betas_limbs: Optional[np.ndarray] = None  # (T, 7)
    dog_id: str = ""
    participant_id: str = ""
    take_id: str = ""
    clip_idx: str = ""

    @property
    def num_frames(self) -> int:
        """帧数."""
        return int(self.kp_world.shape[0])

    @property
    def duration_seconds(self) -> float:
        """估计时长（假设 30fps，InterPet4D 原始帧率）."""
        # frame_idx 是原始帧索引，取最大值估算
        if len(self.frame_idx) > 0:
            return float(self.frame_idx[-1]) / 30.0
        return float(self.num_frames) / 30.0

    def summary(self) -> Dict:
        """返回 clip 摘要."""
        return {
            "clip_id": self.clip_id,
            "num_frames": self.num_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "kp_world_shape": list(self.kp_world.shape),
            "kp_weight_mean": round(float(self.kp_weight.mean()), 4) if self.kp_weight.size > 0 else 0.0,
            "dog_id": self.dog_id,
            "participant_id": self.participant_id,
        }


def parse_clip_id(clip_id: str) -> Tuple[str, str, str, str]:
    """解析 clip_id 为 (dog_id, participant_id, take_id, clip_idx).

    Args:
        clip_id: e.g. "interpet_dog01_p01_take01_ego_001"

    Returns:
        (dog_id, participant_id, take_id, clip_idx) e.g. ("dog01", "p01", "take01", "001")
    """
    parts = clip_id.split("_")
    # 期望格式: interpet_dog{DD}_p{PP}_take{TT}_ego_{NNN}
    if len(parts) != 6 or parts[0] != "interpet" or parts[4] != "ego":
        logger.warning(f"[parse_clip_id] 异常 clip_id 格式: {clip_id}")
        return ("", "", "", "")

    return (parts[1], parts[2], parts[3], parts[5])


def load_clip(
    clip_id: str,
    smal_dir: Union[str, Path] = DEFAULT_INTERPET4D_SMAL_DIR,
    load_optional: bool = False,
) -> Optional[InterPet4DClip]:
    """加载单个 InterPet4D clip 的 SMAL 数据.

    Args:
        clip_id: clip 标识符（文件名去扩展名）
        smal_dir: smal_npy 目录路径
        load_optional: 是否加载可选字段（R/t/s_world, pose_rotmat, betas）

    Returns:
        InterPet4DClip 或 None（文件不存在时）
    """
    smal_dir = Path(smal_dir)
    npz_path = smal_dir / f"{clip_id}.npz"
    if not npz_path.exists():
        logger.debug(f"[load_clip] 文件不存在: {npz_path}")
        return None

    try:
        data = np.load(npz_path)
    except Exception as e:
        logger.warning(f"[load_clip] 加载失败 {npz_path}: {e}")
        return None

    # 必需字段
    if SMAL_KP_WORLD_KEY not in data:
        logger.warning(f"[load_clip] 缺少 {SMAL_KP_WORLD_KEY}: {npz_path}")
        return None

    kp_world = np.asarray(data[SMAL_KP_WORLD_KEY], dtype=np.float32)
    if kp_world.shape[1] != NUM_KEYPOINTS or kp_world.shape[2] != 3:
        logger.warning(
            f"[load_clip] kp_world 形状异常 {kp_world.shape}, 期望 (T, {NUM_KEYPOINTS}, 3): {npz_path}"
        )
        return None

    kp_weight = (
        np.asarray(data[SMAL_KP_WEIGHT_KEY], dtype=np.float32)
        if SMAL_KP_WEIGHT_KEY in data
        else np.ones((kp_world.shape[0], NUM_KEYPOINTS), dtype=np.float32)
    )
    frame_idx = (
        np.asarray(data[SMAL_FRAME_IDX_KEY], dtype=np.int32)
        if SMAL_FRAME_IDX_KEY in data
        else np.arange(kp_world.shape[0], dtype=np.int32)
    )

    # 可选字段
    R_world = t_world = s_world = pose_rotmat = betas = betas_limbs = None
    if load_optional:
        R_world = (
            np.asarray(data[SMAL_R_WORLD_KEY], dtype=np.float32)
            if SMAL_R_WORLD_KEY in data else None
        )
        t_world = (
            np.asarray(data[SMAL_T_WORLD_KEY], dtype=np.float32)
            if SMAL_T_WORLD_KEY in data else None
        )
        s_world = (
            np.asarray(data[SMAL_S_WORLD_KEY], dtype=np.float32)
            if SMAL_S_WORLD_KEY in data else None
        )
        pose_rotmat = (
            np.asarray(data[SMAL_POSE_ROTMAT_KEY], dtype=np.float32)
            if SMAL_POSE_ROTMAT_KEY in data else None
        )
        betas = (
            np.asarray(data[SMAL_BETAS_KEY], dtype=np.float32)
            if SMAL_BETAS_KEY in data else None
        )
        betas_limbs = (
            np.asarray(data[SMAL_BETAS_LIMBS_KEY], dtype=np.float32)
            if SMAL_BETAS_LIMBS_KEY in data else None
        )

    dog_id, participant_id, take_id, clip_idx = parse_clip_id(clip_id)

    return InterPet4DClip(
        clip_id=clip_id,
        kp_world=kp_world,
        kp_weight=kp_weight,
        frame_idx=frame_idx,
        R_world=R_world,
        t_world=t_world,
        s_world=s_world,
        pose_rotmat=pose_rotmat,
        betas=betas,
        betas_limbs=betas_limbs,
        dog_id=dog_id,
        participant_id=participant_id,
        take_id=take_id,
        clip_idx=clip_idx,
    )


def list_clips(
    smal_dir: Union[str, Path] = DEFAULT_INTERPET4D_SMAL_DIR,
) -> List[str]:
    """列出 smal_npy 目录下所有 clip_id.

    Args:
        smal_dir: smal_npy 目录路径

    Returns:
        List[str] — clip_id 列表（排序）
    """
    smal_dir = Path(smal_dir)
    if not smal_dir.exists():
        logger.warning(f"[list_clips] 目录不存在: {smal_dir}")
        return []
    clip_ids = sorted([f.stem for f in smal_dir.glob("*.npz")])
    return clip_ids


def load_all_clips(
    smal_dir: Union[str, Path] = DEFAULT_INTERPET4D_SMAL_DIR,
    load_optional: bool = False,
    min_frames: int = 10,
    min_kp_weight: float = 0.3,
    min_valid_kp_ratio: float = 0.5,
) -> List[InterPet4DClip]:
    """批量加载所有 clip，过滤低质量数据.

    Args:
        smal_dir: smal_npy 目录路径
        load_optional: 是否加载可选字段
        min_frames: 最小帧数（过滤过短 clip）
        min_kp_weight: 关键点置信度阈值（低于此值的点视为无效）
        min_valid_kp_ratio: 每帧有效关键点比例阈值（低于此值的帧过滤）

    Returns:
        List[InterPet4DClip] — 通过过滤的 clip 列表
    """
    clip_ids = list_clips(smal_dir)
    logger.info(f"[load_all_clips] 发现 {len(clip_ids)} clips in {smal_dir}")

    valid_clips: List[InterPet4DClip] = []
    skipped_count = 0

    for clip_id in clip_ids:
        clip = load_clip(clip_id, smal_dir, load_optional=load_optional)
        if clip is None:
            skipped_count += 1
            continue

        # 过滤：帧数不足
        if clip.num_frames < min_frames:
            logger.debug(f"[load_all_clips] 跳过 {clip_id}: 帧数 {clip.num_frames} < {min_frames}")
            skipped_count += 1
            continue

        # 过滤：kp_world 含 NaN/Inf（Phase 3.3c 训练 NaN bug 修复）
        # 个别 clip（如 interpet_dog06_p14_take01_ego_001）全部 kp_world 为 NaN，
        # 不过滤会导致训练 loss=nan
        if np.isnan(clip.kp_world).any() or np.isinf(clip.kp_world).any():
            nan_count = int(np.isnan(clip.kp_world).sum())
            logger.warning(
                f"[load_all_clips] 跳过 {clip_id}: kp_world 含 NaN/Inf "
                f"({nan_count}/{clip.kp_world.size} 个值)"
            )
            skipped_count += 1
            continue

        # 过滤：关键点置信度过低
        valid_mask = clip.kp_weight >= min_kp_weight  # (T, 24)
        valid_ratio = valid_mask.mean(axis=1)  # (T,)
        if float(valid_ratio.mean()) < min_valid_kp_ratio:
            logger.debug(
                f"[load_all_clips] 跳过 {clip_id}: 平均有效关键点比例 "
                f"{float(valid_ratio.mean()):.2f} < {min_valid_kp_ratio}"
            )
            skipped_count += 1
            continue

        valid_clips.append(clip)

    logger.info(
        f"[load_all_clips] 加载完成: {len(valid_clips)} 有效 / {len(clip_ids)} 总计 "
        f"({skipped_count} 跳过)"
    )
    return valid_clips


def get_dataset_statistics(
    smal_dir: Union[str, Path] = DEFAULT_INTERPET4D_SMAL_DIR,
    max_clips: Optional[int] = None,
) -> Dict:
    """获取 InterPet4D 数据集统计信息.

    Args:
        smal_dir: smal_npy 目录路径
        max_clips: 最大加载 clip 数（None=全部，用于快速采样）

    Returns:
        Dict — 统计信息
    """
    clip_ids = list_clips(smal_dir)
    if max_clips is not None:
        clip_ids = clip_ids[:max_clips]

    total_frames = 0
    dog_ids = set()
    participant_ids = set()
    frame_counts = []
    kp_weight_means = []

    for clip_id in clip_ids:
        clip = load_clip(clip_id, smal_dir, load_optional=False)
        if clip is None:
            continue
        total_frames += clip.num_frames
        frame_counts.append(clip.num_frames)
        dog_ids.add(clip.dog_id)
        participant_ids.add(clip.participant_id)
        kp_weight_means.append(float(clip.kp_weight.mean()))

    return {
        "total_clips": len(clip_ids),
        "total_frames": total_frames,
        "num_dogs": len(dog_ids),
        "num_participants": len(participant_ids),
        "dog_ids": sorted(dog_ids),
        "frame_count_stats": {
            "min": min(frame_counts) if frame_counts else 0,
            "max": max(frame_counts) if frame_counts else 0,
            "mean": round(sum(frame_counts) / len(frame_counts), 1) if frame_counts else 0,
        },
        "kp_weight_stats": {
            "min": round(min(kp_weight_means), 4) if kp_weight_means else 0,
            "max": round(max(kp_weight_means), 4) if kp_weight_means else 0,
            "mean": round(sum(kp_weight_means) / len(kp_weight_means), 4) if kp_weight_means else 0,
        },
    }


__all__ = [
    "InterPet4DClip",
    "parse_clip_id",
    "load_clip",
    "list_clips",
    "load_all_clips",
    "get_dataset_statistics",
    "NUM_KEYPOINTS",
    "DEFAULT_INTERPET4D_SMAL_DIR",
    "SMAL_KP_WORLD_KEY",
    "SMAL_KP_WEIGHT_KEY",
    "SMAL_FRAME_IDX_KEY",
]
