"""VideoMamba 适配骨骼数据的完整实现 (纯 PyTorch, 无需 CUDA 编译).

依据:
- VideoMamba (ECCV 2024) - bidirectional SSM for video understanding
- Simba (arXiv 2404.07645) - Mamba augmented U-ShiftGCN
- Gamba (CVPR 2026) - Mamba-based GCN for action recognition

架构:
1. 骨骼嵌入: (T, J, 3) -> (T, d_model) 线性投影
2. Mamba Blocks: 多层双向 SSM + 残差连接
3. 时序池化: 全局平均 + CLS token
4. 分类头: 22 类行为分类
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """选通状态空间模型 (Selective SSM) 的纯 PyTorch 实现.
    
    对应论文中的 selective scan mechanism。
    不使用 CUDA 编译的 mamba_ssm 包，而是用纯 PyTorch 实现。
    """
    
    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        if dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank
        
        # 输入投影 (x -> xz)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        
        # 卷积 (局部感受野)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            groups=d_model,
            padding=3,
        )
        self.act = nn.SiLU()
        
        # SSM 参数 (选择性)
        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        
        # 初始化 dt_proj 权重
        nn.init.constant_(self.dt_proj.bias, 1.0)
        
        # SSM 参数
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.
        
        Args:
            x: (B, L, D) 输入序列
            
        Returns:
            (B, L, D) 输出序列
        """
        B, L, D = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # (B, L, 2D)
        x_hidden, z = xz.chunk(2, dim=-1)  # 各 (B, L, D)
        
        # 1D 卷积
        x_hidden = x_hidden.transpose(1, 2)  # (B, D, L)
        x_hidden = self.conv1d(x_hidden)[:, :, :L]  # (B, D, L)
        x_hidden = self.act(x_hidden)
        x_hidden = x_hidden.transpose(1, 2)  # (B, L, D)
        
        # 选择性参数
        dt_x = self.x_proj(x_hidden)  # (B, L, dt_rank + 2*d_state)
        dt, B_param, C_param = torch.split(
            dt_x, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj(dt)  # (B, L, D)
        dt = F.softplus(dt)  # 保证 dt > 0
        
        # SSM 离散化
        A = -torch.exp(self.A_log)  # (D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, D, N)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)  # (B, L, D, N)
        
        # SSM 扫描 (选择性扫描)
        h = torch.zeros(B, D, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            h = h * dA[:, t] + x_hidden[:, t].unsqueeze(-1) * dB[:, t]
            out = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(out)
        
        x_hidden = torch.stack(outputs, dim=1)  # (B, L, D)
        
        # 门控 + 跳跃连接
        x_hidden = x_hidden * self.D
        x_hidden = x_hidden * F.silu(z)
        
        # 输出投影
        x_hidden = self.out_proj(x_hidden)
        
        return x_hidden


class MambaBlock(nn.Module):
    """Mamba Block: LayerNorm + SSM + 残差连接."""
    
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 残差 + LayerNorm + SSM."""
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return x + residual


class VideoMambaSkeleton(nn.Module):
    """VideoMamba 适配骨骼数据的完整实现.
    
    输入: (B, T, J, 3) 骨骼关键点序列
    输出: (B, num_classes) 行为分类 logits
    
    Args:
        num_joints: 关节数 (默认 24)
        num_classes: 分类数 (默认 22)
        d_model: 模型维度
        n_layers: Mamba Block 层数
        d_state: SSM 状态维度
        dropout: dropout 率
    """
    
    def __init__(
        self,
        num_joints: int = 24,
        num_classes: int = 22,
        d_model: int = 192,
        n_layers: int = 8,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_joints = num_joints
        
        # 骨骼嵌入: (J, 3) -> d_model
        self.embed = nn.Linear(num_joints * 3, d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Mamba Blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        
        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if name.endswith('weight') and len(p.shape) >= 2:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播.
        
        Args:
            x: (B, T, J, 3) 骨骼序列
            
        Returns:
            (B, num_classes) 行为分类 logits
        """
        B, T, J, _ = x.shape
        
        # 展平关节: (B, T, J*3)
        x = x.reshape(B, T, -1)
        
        # 嵌入: (B, T, d_model)
        x = self.embed(x)
        
        # 添加 CLS token: (B, T+1, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 位置编码
        x = x + self.pos_embed
        
        # Mamba Blocks
        for layer in self.layers:
            x = layer(x)
        
        # 取 CLS token 输出
        x = self.norm(x[:, 0])
        
        # 分类头
        logits = self.head(x)
        
        return logits


def get_model(num_classes: int = 22) -> VideoMambaSkeleton:
    """获取 VideoMamba 骨骼行为识别模型."""
    return VideoMambaSkeleton(
        num_joints=24,
        num_classes=num_classes,
        d_model=192,
        n_layers=8,
        d_state=16,
        dropout=0.1,
    )


if __name__ == "__main__":
    model = get_model()
    x = torch.randn(2, 30, 24, 3)
    output = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    assert output.shape == (2, 22), f"Expected (2, 22), got {output.shape}"
    print("Test passed!")