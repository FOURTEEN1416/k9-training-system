"""MotionBERT 配置加载器（不依赖 easydict）.

Owner: ML 开发
Phase: 3.3c

设计:
    - 用 dataclass 替代 easydict，保持类型安全
    - 支持 YAML 加载 + 默认值
    - 与 external/MotionBERT/configs/*.yaml 兼容
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class MotionBERTConfig:
    """MotionBERT 配置（MB_lite 架构 + 24 关键点适配）."""

    # === 模型架构 ===
    backbone: str = "DSTformer"
    dim_in: int = 3
    dim_out: int = 3
    dim_feat: int = 256
    dim_rep: int = 512
    depth: int = 5
    num_heads: int = 8
    mlp_ratio: int = 4
    num_joints: int = 24  # ← 17→24 适配核心
    maxlen: int = 243
    att_fuse: bool = True
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    # === 训练 ===
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    lr_decay: float = 0.99
    checkpoint_frequency: int = 10

    # === 数据 ===
    window_size: int = 27  # 与 lifting_pairing.DEFAULT_WINDOW_SIZE 一致
    window_stride: int = 13
    num_cameras: int = 8
    projection_mode: str = "orthographic"
    camera_distance: float = 3.0
    min_kp_weight: float = 0.3
    min_valid_kp_ratio: float = 0.5
    val_ratio: float = 0.2
    random_seed: int = 42

    # === 损失权重（与 MotionBERT 原始一致）===
    lambda_3d_pos: float = 1.0
    lambda_3d_velocity: float = 20.0
    lambda_scale: float = 0.5
    lambda_lv: float = 0.0
    lambda_lg: float = 0.0
    lambda_a: float = 0.0
    lambda_av: float = 0.0

    # === 推理 ===
    flip: bool = True
    rootrel: bool = True  # 根关节中心化输出
    no_conf: bool = False

    # === 路径 ===
    smal_dir: str = "data/interpet4d/smal_npy"
    pretrained_path: str = "data/models/MB_lite_pose3d_ft.bin"
    checkpoint_dir: str = "data/models/motionbert_dog24"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MotionBERTConfig":
        """从 YAML 文件加载配置."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}
        # 过滤未知字段
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def ensure_dirs(self) -> None:
        """创建必要的输出目录."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def load_config(path: Optional[str | Path] = None) -> MotionBERTConfig:
    """加载配置: 优先 YAML 文件，否则返回默认 24 关键点配置."""
    if path is not None:
        return MotionBERTConfig.from_yaml(path)
    return MotionBERTConfig()
