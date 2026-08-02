"""STGCNBC 整体模型: ST-GCN backbone + BC Head + 联合损失.

Owner: ML 开发
Phase: 3.1c

集成:
    - backbone: STGCN（自研 PyTorch 原生，复用 K9Graph）
    - head: BCHead（自研边界分类联合头）
    - loss: STGCNBCLoss（L = L_cls + 0.3 · L_boundary）

输入: (B, T, V=24, C=3) — 来自 YOLO26-pose 2D 或 MotionBERT 3D
输出:
    - 推理: (cls_logits (B, 22), boundary_probs (B, T'))
    - 训练: 联合损失字典
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from backend.ml.behavior.stgcn_bc.bc_head import BCHead
from backend.ml.behavior.stgcn_bc.k9_graph import K9Graph
from backend.ml.behavior.stgcn_bc.labels import NUM_BEHAVIORS
from backend.ml.behavior.stgcn_bc.loss import STGCNBCLoss
from backend.ml.behavior.stgcn_bc.stgcn import STGCN


class STGCNBC(nn.Module):
    """ST-GCN+BC 整体模型.

    Args:
        in_channels: 输入通道数（3D=3, 2D+conf=3）
        num_classes: 类别数（默认 22）
        base_channels: STGCN 基础通道（默认 64）
        num_stages: STGCN block 数量（默认 10）
        tcn_type: 'mstcn' (ST-GCN++) 或 'unit_tcn' (原始 ST-GCN)
        adaptive: 邻接矩阵自适应模式
        boundary_weight: 边界损失权重（默认 0.3）
        dropout: 分类头 dropout
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_BEHAVIORS,  # 22
        base_channels: int = 64,
        num_stages: int = 10,
        tcn_type: str = "mstcn",
        adaptive: str = "importance",
        boundary_weight: float = 0.3,
        dropout: float = 0.0,
        graph: Optional[K9Graph] = None,
    ):
        super().__init__()
        self.backbone = STGCN(
            graph=graph,
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            tcn_type=tcn_type,
            adaptive=adaptive,
        )
        self.head = BCHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.loss_fn = STGCNBCLoss(boundary_weight=boundary_weight)
        self.num_classes = num_classes

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播.

        Args:
            x: (B, T, V=24, C=3) 或 (B, M, T, V, C)

        Returns:
            cls_logits: (B, num_classes)
            boundary_logits: (B, T') — 未 Sigmoid
        """
        feat = self.backbone(x)  # (B, M, C_out, T', V)
        cls_logits, boundary_logits = self.head(feat)
        return cls_logits, boundary_logits

    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        cls_labels: torch.Tensor,
        boundary_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算联合损失. 详见 STGCNBCLoss."""
        return self.loss_fn(cls_logits, boundary_logits, cls_labels, boundary_labels)

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """推理接口: 返回分类概率 + 边界概率.

        Returns:
            {
                "cls_probs": (B, num_classes),
                "cls_pred": (B,) int64,
                "boundary_probs": (B, T'),
                "boundary_pred": (B, T') bool — 阈值 0.5
            }
        """
        cls_logits, boundary_logits = self.forward(x)
        cls_probs = torch.softmax(cls_logits, dim=-1)
        cls_pred = cls_probs.argmax(dim=-1)
        boundary_probs = torch.sigmoid(boundary_logits)
        boundary_pred = boundary_probs > 0.5
        return {
            "cls_probs": cls_probs,
            "cls_pred": cls_pred,
            "boundary_probs": boundary_probs,
            "boundary_pred": boundary_pred,
        }


def build_stgcn_bc(
    in_channels: int = 3,
    num_classes: int = NUM_BEHAVIORS,
    base_channels: int = 64,
    num_stages: int = 10,
    **kwargs,
) -> STGCNBC:
    """便捷构建函数."""
    return STGCNBC(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        num_stages=num_stages,
        **kwargs,
    )


__all__ = [
    "STGCNBC",
    "build_stgcn_bc",
]
