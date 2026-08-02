"""ST-GCN+BC 联合损失: L = L_cls + 0.3 · L_boundary.

Owner: ML 开发
Phase: 3.1c
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class STGCNBCLoss(nn.Module):
    """ST-GCN+BC 联合损失.

    L = L_cls + boundary_weight · L_boundary
    L_cls: CrossEntropyLoss（22 类分类）
    L_boundary: BCEWithLogitsLoss（边界检测，支持软标签）

    边界对齐:
        cls_logits: (B, num_classes) — 段级
        boundary_logits: (B, T') — 帧级
        boundary_labels: (B, T') — 软标签 [0, 1]，无标注时填 0
    """

    def __init__(
        self,
        boundary_weight: float = 0.3,
        label_smoothing: float = 0.0,
        boundary_pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if boundary_pos_weight is not None:
            pos_weight = torch.tensor([boundary_pos_weight])
            self.boundary_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.boundary_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        cls_labels: torch.Tensor,
        boundary_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算联合损失.

        Args:
            cls_logits: (B, num_classes)
            boundary_logits: (B, T')
            cls_labels: (B,) long
            boundary_labels: (B, T') float, None 时跳过边界损失

        Returns:
            {"total": ..., "cls": ..., "boundary": ...}
        """
        l_cls = self.cls_loss(cls_logits, cls_labels)

        if boundary_labels is None:
            # 无边界标签时跳过边界损失
            return {
                "total": l_cls,
                "cls": l_cls,
                "boundary": torch.tensor(0.0, device=l_cls.device),
            }

        # 对齐时间维度: boundary_logits 可能比 boundary_labels 短（因 backbone 下采样）
        T_pred = boundary_logits.size(-1)
        T_label = boundary_labels.size(-1)
        if T_pred != T_label:
            # 双线性插值到相同长度
            boundary_labels = F.interpolate(
                boundary_labels.unsqueeze(1),  # (B, 1, T_label)
                size=T_pred,
                mode="linear",
                align_corners=False,
            ).squeeze(1)  # (B, T_pred)

        l_boundary = self.boundary_loss(boundary_logits, boundary_labels)
        total = l_cls + self.boundary_weight * l_boundary

        return {
            "total": total,
            "cls": l_cls,
            "boundary": l_boundary,
        }


__all__ = ["STGCNBCLoss"]
