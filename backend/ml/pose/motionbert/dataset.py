"""MotionBERT 微调数据集（包装 lifting_pairing.py）.

Owner: ML 开发
Phase: 3.3c

设计:
    - 复用 backend.ml.pose.lifting_pairing.build_dataset 构建 (2D, 3D) 配对
    - 转换为 PyTorch Dataset，支持 DataLoader
    - 不缓存到磁盘（数据量小，内存足够）
    - confidence 作为额外 channel 拼接到 2D 输入（dim_in=3: x,y,conf）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from backend.ml.pose.lifting_pairing import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STRIDE,
    ROOT_KEYPOINT_IDX,
    build_dataset,
    train_val_split,
)
from backend.ml.pose.motionbert.config import MotionBERTConfig

logger = logging.getLogger(__name__)


class InterPet4DDataset(Dataset):
    """InterPet4D 2D-to-3D lifting 数据集.

    输出:
        keypoints_2d: (T, 24, 3) — (x, y, conf)，归一化到 [-1, 1]
        keypoints_3d: (T, 24, 3) — 根关节中心化 + 尺度归一化
    """

    def __init__(
        self,
        keypoints_2d: np.ndarray,
        keypoints_3d: np.ndarray,
        confidence: np.ndarray,
    ):
        self.kp_2d = keypoints_2d.astype(np.float32)  # (N, T, 24, 2)
        self.kp_3d = keypoints_3d.astype(np.float32)  # (N, T, 24, 3)
        self.conf = confidence.astype(np.float32)  # (N, T, 24)
        assert self.kp_2d.shape[0] == self.kp_3d.shape[0] == self.conf.shape[0]

    def __len__(self) -> int:
        return self.kp_2d.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        kp_2d = self.kp_2d[idx]  # (T, 24, 2)
        conf = self.conf[idx]  # (T, 24)
        # 拼接 (x, y, conf) → (T, 24, 3) — MotionBERT dim_in=3
        kp_2d_with_conf = np.concatenate(
            [kp_2d, conf[..., np.newaxis]], axis=-1
        ).astype(np.float32)
        kp_3d = self.kp_3d[idx]  # (T, 24, 3)
        return torch.from_numpy(kp_2d_with_conf), torch.from_numpy(kp_3d)


def build_datasets(
    config: MotionBERTConfig,
    max_clips: Optional[int] = None,
) -> Dict[str, InterPet4DDataset]:
    """构建训练/验证数据集.

    Args:
        config: MotionBERTConfig
        max_clips: 最大 clip 数（None=全部，用于快速测试）

    Returns:
        {"train": InterPet4DDataset, "val": InterPet4DDataset}
    """
    logger.info(f"[build_datasets] 加载 InterPet4D 数据: {config.smal_dir}")

    kp_2d, kp_3d, conf, metadata = build_dataset(
        smal_dir=config.smal_dir,
        num_cameras=config.num_cameras,
        projection_mode=config.projection_mode,
        window_size=config.window_size,
        stride=config.window_stride,
        max_clips=max_clips,
        min_kp_weight=config.min_kp_weight,
        min_valid_kp_ratio=config.min_valid_kp_ratio,
        random_seed=config.random_seed,
    )

    if kp_2d.shape[0] == 0:
        raise RuntimeError("无有效样本，请检查 InterPet4D 数据")

    logger.info(
        f"[build_datasets] 总样本: {kp_2d.shape[0]}, "
        f"窗口: {config.window_size} 帧"
    )

    split = train_val_split(
        kp_2d, kp_3d, conf, metadata,
        val_ratio=config.val_ratio,
        random_seed=config.random_seed,
    )

    train_ds = InterPet4DDataset(
        split["train_2d"], split["train_3d"], split["train_conf"]
    )
    val_ds = InterPet4DDataset(
        split["val_2d"], split["val_3d"], split["val_conf"]
    )

    logger.info(
        f"[build_datasets] 训练集: {len(train_ds)} 样本, "
        f"验证集: {len(val_ds)} 样本"
    )
    return {"train": train_ds, "val": val_ds}
