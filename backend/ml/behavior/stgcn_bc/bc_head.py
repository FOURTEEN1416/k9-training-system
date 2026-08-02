"""BC 头（Boundary-Classification Head）— 自研组件.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1c
依据: dev-docs/research/RESEARCH_STGCN_BC.md §3.2 自研触发 + phase-3.md §3.1c

设计动机:
    传统 ST-GCN 只做片段级分类（一个 clip 一个标签），存在两个问题:
    1. 边界模糊: 行为起止帧附近分类置信度低，影响评分卡延迟维度计算
    2. 段级表示: 无法捕捉行为内部时序动态，不利于 FCI-IGP 评分（保持/延迟）

    BC 头联合优化:
    - 分类头（cls_head）: 全局池化 → FC → 22 类 logits
    - 边界头（bc_head）: 1D Conv + Sigmoid → 每帧边界概率（行为起止帧概率高）

    联合损失: L = L_cls + 0.3 · L_boundary
    L_cls: CrossEntropyLoss（22 类）
    L_boundary: BCEWithLogitsLoss（边界二分类）

BC 头架构（自研，简单 1D Conv 路线）:
    Input: (B, C, T', V) — STGCN backbone 输出
    ├─ Global Average Pool over (T', V) → (B, C)
    │   └─ FC → (B, num_classes=22)  ← cls logits
    ├─ Mean over V → (B, C, T')
    │   └─ Conv1d(C, 1, kernel=5, pad=2) → (B, 1, T')
    │       └─ Sigmoid → (B, T')  ← boundary probability
    └─ Output: (cls_logits, boundary_probs)

边界标签生成（训练时）:
    给定 clip 的行为起止帧 [start, end]（segment-level label 扩展为 frame-level）:
    - boundary[t] = 1 if t ∈ {start, end} else 0
    - 实际中 start/end 通常不可精确标注，采用软标签:
      boundary[t] = exp(-((t - start)^2 + (t - end)^2) / (2 * sigma^2))
      sigma = max(1, T * 0.05)  # 5% 时间长度
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCHead(nn.Module):
    """Boundary-Classification Head（自研）.

    输入: STGCN backbone 输出 (B, M, C, T', V)
    输出: (cls_logits (B, num_classes), boundary_logits (B, M, T'))
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 22,
        boundary_kernel: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.boundary_kernel = boundary_kernel

        # 分类头: global pool + FC
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc_cls = nn.Linear(in_channels, num_classes)

        # 边界头: 1D Conv on temporal dim
        pad = (boundary_kernel - 1) // 2
        self.conv_boundary = nn.Conv1d(
            in_channels, 1,
            kernel_size=boundary_kernel,
            padding=pad,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            x: (B, M, C, T', V) — STGCN backbone 输出

        Returns:
            cls_logits: (B, num_classes) — 分类 logits
            boundary_logits: (B, M, T') — 边界 logits（未 Sigmoid，供 BCEWithLogitsLoss）
        """
        if x.ndim != 5:
            raise ValueError(f"输入应为 (B, M, C, T', V), 实际 {x.shape}")

        B, M, C, T_p, V = x.shape

        # 分类: global average pool over (T', V) + mean over M
        feat = x.mean(dim=(1, 3, 4))  # (B, C)
        feat = self.dropout(feat)
        cls_logits = self.fc_cls(feat)  # (B, num_classes)

        # 边界: pool over V → (B, C, T')，对 M 取均值
        feat_t = x.mean(dim=4).mean(dim=1)  # (B, C, T')
        boundary_logits = self.conv_boundary(feat_t).squeeze(1)  # (B, T')

        return cls_logits, boundary_logits


def generate_boundary_labels(
    total_frames: int,
    segment_starts: list,
    segment_ends: list,
    sigma_ratio: float = 0.05,
) -> torch.Tensor:
    """生成帧级边界软标签（高斯环绕 start/end）.

    Args:
        total_frames: T
        segment_starts: [start_1, start_2, ...] 行为起始帧索引列表
        segment_ends: [end_1, end_2, ...] 行为结束帧索引列表
        sigma_ratio: 高斯标准差 = max(1, T * sigma_ratio)

    Returns:
        torch.Tensor, shape=(T,), dtype=float32, 值域 [0, 1]
    """
    if len(segment_starts) != len(segment_ends):
        raise ValueError("segment_starts 与 segment_ends 长度不一致")

    sigma = max(1.0, total_frames * sigma_ratio)
    t = np.arange(total_frames, dtype=np.float32)
    labels = np.zeros(total_frames, dtype=np.float32)

    for s, e in zip(segment_starts, segment_ends):
        if s >= 0 and s < total_frames:
            labels = np.maximum(labels, np.exp(-((t - s) ** 2) / (2 * sigma ** 2)))
        if e >= 0 and e < total_frames:
            labels = np.maximum(labels, np.exp(-((t - e) ** 2) / (2 * sigma ** 2)))

    return torch.from_numpy(labels)


__all__ = [
    "BCHead",
    "generate_boundary_labels",
]
