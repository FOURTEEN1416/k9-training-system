"""DSTformer 模型包装器 + 17→24 关键点权重迁移.

Owner: ML 开发
Phase: 3.3c

设计:
    - 通过 sys.path 注入引用 external/MotionBERT/lib/model/DSTformer.py
    - DSTformer 原生支持 num_joints 参数，仅需 pos_embed 适配
    - 加载预训练权重时，pos_embed (1,17,256) 与新模型 (1,24,256) 不匹配，
      会被自动丢弃；其他所有层（joints_embed/blocks/head/temp_embed）尺寸无关，
      全部迁移。新 pos_embed 用 trunc_normal_ 重新初始化。

权重迁移策略（无兜底层）:
    1. 加载预训练 checkpoint（MB_lite_pretrain.bin 或 MB_lite_pose3d_ft.bin）
    2. strip "module." prefix（DataParallel 保存格式）
    3. 逐 key 比对，shape 匹配则迁移，不匹配则跳过
    4. pos_embed 被跳过 → 保持模型构建时的 trunc_normal_ 初始化
"""
from __future__ import annotations

import collections
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from functools import partial

# 注入 external/MotionBERT 到 sys.path（仅 lib 目录的父目录）
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_MOTIONBERT_ROOT = _PROJECT_ROOT / "external" / "MotionBERT"
if str(_MOTIONBERT_ROOT) not in sys.path:
    sys.path.insert(0, str(_MOTIONBERT_ROOT))

# 现在可以导入 DSTformer
from lib.model.DSTformer import DSTformer  # type: ignore[import-not-found]
from lib.model.drop import DropPath  # noqa: F401  # 确保 drop 模块可用

from backend.ml.pose.motionbert.config import MotionBERTConfig

logger = logging.getLogger(__name__)


def build_model_from_config(config: MotionBERTConfig) -> DSTformer:
    """从配置构建 DSTformer 模型（24 关键点）."""
    model = DSTformer(
        dim_in=config.dim_in,
        dim_out=config.dim_out,
        dim_feat=config.dim_feat,
        dim_rep=config.dim_rep,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        num_joints=config.num_joints,
        maxlen=config.maxlen,
        qkv_bias=config.qkv_bias,
        qk_scale=config.qk_scale,
        drop_rate=config.drop_rate,
        attn_drop_rate=config.attn_drop_rate,
        drop_path_rate=config.drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        att_fuse=config.att_fuse,
    )
    return model


def load_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
    log_mismatches: bool = True,
) -> Tuple[int, int, List[str]]:
    """加载预训练权重到模型，自动跳过 shape 不匹配的层.

    Args:
        model: 目标模型（24 关键点 DSTformer）
        checkpoint_path: 预训练 checkpoint 路径
        strict: 是否严格加载（False=允许部分匹配）
        log_mismatches: 是否记录不匹配的层

    Returns:
        (matched_count, total_count, discarded_keys)

    17→24 适配说明:
        - pos_embed (1,17,256) → (1,24,256): 不匹配，被丢弃，保持模型初始化
        - joints_embed (256,3): 匹配，迁移
        - blocks_st/blocks_ts: 匹配，迁移
        - head (3,512): 匹配，迁移
        - temp_embed (1,243,1,256): 匹配，迁移
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"预训练权重不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_pos" in checkpoint:
        state_dict = checkpoint["model_pos"]
    else:
        state_dict = checkpoint

    # strip "module." prefix (DataParallel 保存格式)
    cleaned = collections.OrderedDict()
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        cleaned[key] = v

    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers: List[str] = []
    discarded_layers: List[str] = []

    for k, v in cleaned.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=strict)

    total = len(cleaned)
    matched = len(matched_layers)
    logger.info(
        f"[load_pretrained_weights] 迁移 {matched}/{total} 层 "
        f"({matched / total * 100:.1f}%), 丢弃 {len(discarded_layers)} 层"
    )
    if log_mismatches and discarded_layers:
        logger.info(
            f"[load_pretrained_weights] 丢弃的层: {discarded_layers[:5]}"
            + ("..." if len(discarded_layers) > 5 else "")
        )

    return matched, total, discarded_layers


class DSTformerWrapper(nn.Module):
    """DSTformer 推理/训练包装器.

    封装:
        - 模型构建
        - 预训练权重加载
        - forward / get_representation
        - 推理时的 flip 增强 + rootrel 后处理
    """

    def __init__(
        self,
        config: Optional[MotionBERTConfig] = None,
        pretrained_path: Optional[str | Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.config = config or MotionBERTConfig()
        self.device = device
        self.model = build_model_from_config(self.config).to(device)

        if pretrained_path is not None:
            load_pretrained_weights(self.model, pretrained_path)

    def forward(self, x: torch.Tensor, return_rep: bool = False) -> torch.Tensor:
        """前向推理.

        Args:
            x: (B, T, J, C) 2D 关键点序列
                - J=24, C=2 (xy) 或 C=3 (xyc, confidence)
                - 归一化到 [-1, 1]
            return_rep: 是否返回中间表示 (B, T, J, dim_rep)

        Returns:
            (B, T, 24, 3) 3D 关键点（根关节中心化）
        """
        return self.model(x, return_rep=return_rep)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """获取中间表示 (B, T, 24, dim_rep) — 用于下游任务."""
        return self.model.get_representation(x)

    @torch.no_grad()
    def infer(
        self,
        keypoints_2d: torch.Tensor,
        flip: Optional[bool] = None,
        rootrel: Optional[bool] = None,
    ) -> torch.Tensor:
        """推理接口（含 flip 增强和 rootrel 后处理）.

        Args:
            keypoints_2d: (B, T, 24, C) 2D 输入
            flip: 是否启用 flip 增强（默认用 config.flip）
            rootrel: 是否根关节中心化（默认用 config.rootrel）

        Returns:
            (B, T, 24, 3) 3D 关键点
        """
        self.model.eval()
        flip = self.config.flip if flip is None else flip
        rootrel = self.config.rootrel if rootrel is None else rootrel

        x = keypoints_2d.to(self.device)
        if self.config.no_conf and x.shape[-1] == 3:
            x = x[..., :2]
            # 需要补回 channel 维度给 joints_embed (dim_in=3)
            # MotionBERT 默认 dim_in=3，2D 输入 (x,y) 需补 0 confidence channel
            conf_pad = torch.zeros_like(x[..., :1])
            x = torch.cat([x, conf_pad], dim=-1)

        pred = self.model(x)

        if flip:
            pred_flip = self.model(self._flip_data(x))
            pred = (pred + self._flip_data(pred_flip)) / 2.0

        if rootrel:
            pred = pred - pred[..., 0:1, :]

        return pred

    @staticmethod
    def _flip_data(x: torch.Tensor) -> torch.Tensor:
        """左右翻转关键点序列.

        Dog-Pose 24 关键点左右对称对（与 K9Graph 一致）:
            left ↔ right 索引互换
        """
        # Dog-Pose 24 关键点对称映射（左↔右）
        # 0-2: front_left  ↔ 6-8: front_right
        # 3-5: rear_left   ↔ 9-11: rear_right
        # 14: left_ear_base ↔ 15: right_ear_base
        # 18: left_ear_tip  ↔ 19: right_ear_tip
        # 20: left_eye      ↔ 21: right_eye
        # 其他（12,13,16,17,22,23）无对称
        flip_pairs = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
                      (14, 15), (18, 19), (20, 21)]
        flipped = x.clone()
        for (i, j) in flip_pairs:
            flipped[..., i, :], flipped[..., j, :] = x[..., j, :], x[..., i, :]
        # x 坐标取反
        flipped[..., 0] = -flipped[..., 0]
        return flipped
