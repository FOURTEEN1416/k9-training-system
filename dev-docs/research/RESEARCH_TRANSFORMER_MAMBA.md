# Transformer-Mamba 基线实验研究文档

## 状态: ✅ 调研完成，基线训练完成，待集成 (2026-08-17)

## 问题
GitHub + HuggingFace 均无法访问（网络代理问题）。

## 解决方案

### 当前采用: 本地实现标准基线
基于论文描述与现有仓库，优先使用 `mamba_ssm` 标准路径，当前 Windows 环境回退到项目内 VideoMamba 骨架。

## 论文关键信息

### VideoMamba (ECCV 2024, OpenGVLab)
- **架构**: Vision Mamba (Vim) + 视频理解
- **核心创新**: 双向状态空间模型 (Bi-directional SSM) 用于长序列建模
- **复杂度**: O(N) 线性 vs Transformer O(N^2)
- **参数量**: ~25M (base), ~5M (tiny)
- **关键模块**:
  - Patch Embedding
  - Mamba Blocks (SSM + selective scan)
  - Temporal Aggregation

### MS-Temba (CVPR 2026)
- **架构**: Multi-Scale Temporal Mamba
- **核心创新**: 多尺度膨胀 SSM + 辅助边界损失
- **参数量**: 17M
- **适用**: 动作边界检测 + 分类

## 集成计划

### 步骤 1: 环境准备
```bash
pip install mamba-ssm flash-attn timm
```

### 步骤 2: 创建基线模型
文件: `backend/ml/behavior/mamba_sequence.py`
- 实现简化版 VideoMamba
- 输入: 骨骼关键点序列 (T x 24 x 3)
- 输出: 22 类行为概率

### 步骤 3: 训练评估
- 使用现有 synthetic dataset (ST-GCN+BC 已训练)
- 对比基线: accuracy, F1, 推理延迟

## 阻塞项
- [✓] GitHub 访问恢复后克隆完整仓库 (VideoMamba + MS-Temba 均已克隆至 external/)
- [ ] APT-36K 数据集申请 (发送邮件至 viptae@gmail.com)
- [ ] HuggingFace Hub 模型下载测试

## 相关文件
- `dev-docs/stages/phase-4-transformer-mamba.md`
- `decisions/0011-phase-4-llm-explainer-adr.md`
