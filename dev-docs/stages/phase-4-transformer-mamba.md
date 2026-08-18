# Phase 4.2 Transformer-Mamba 长序列行为分析

## 状态: ✅ 调研完成，基线训练完成，待集成

## 依据
- GitHub 调研结果 (2026-08-17)
- `dev-docs/decisions/0011-phase-4-llm-explainer-adr.md`

## 调研结论

### 推荐架构: VideoMamba + MS-Temba 混合方案

| 组件 | 来源 | 角色 |
|------|------|------|
| **VideoMamba** | OpenGVLab (ECCV 2024) | 主干网络：线性复杂度长序列建模 |
| **MS-Temba** | arkaprava (CVPR 2026) | 多尺度膨胀 SSM + 动作边界定位 |
| **ST-GCN+BC** | 现有系统 | 时序骨骼特征提取基线 |

### 关键仓库

| 仓库 | Stars | 论文 | 适配度 | 关键文件 |
|------|-------|------|--------|----------|
| [VideoMamba](https://github.com/OpenGVLab/VideoMamba) | 1,124 | ECCV 2024 | ⭐⭐⭐⭐⭐ | `models/video_mamba.py`, `run_class_finetuning.py` |
| [MS-Temba](https://github.com/thearkaprava/MS-Temba) | 47 | CVPR 2026 | ⭐⭐⭐⭐⭐ | `vim/models/ms_temba.py`, `scripts/run_MSTemba_TSU.sh` |
| [Vamba](https://github.com/TIGER-AI-Lab/Vamba) | 108 | ICCV 2025 | ⭐⭐⭐⭐ | `models/vamba_mamba2/` |
| [MambaVision](https://github.com/NVlabs/MambaVision) | 2,222 | CVPR 2025 | ⭐⭐⭐ | `mambavision/models/` |

## 实施方案

### 阶段 1: 环境搭建 (1周)
```
[✓] 克隆 VideoMamba + MS-Temba 仓库至 external/
[✓] 安装依赖: mamba_ssm, flash-attn, timm
[✓] 验证 GPU 推理 (RTX 5060)
[✓] 准备数据加载器 (Kinetics-400 格式适配 22类)
```

### 阶段 2: 基线实验 (1周)
```
[✓] 复现 VideoMamba on Kinetics-400 (验证 pipeline)
[✓] 替换头层为 22 类工作犬行为分类
[✓] 训练评估: accuracy / macro-F1 / 推理延迟
[✓] 与 ST-GCN+BC 基线对比
```

### 阶段 3: 改进实验 (1-2周)
```
[ ] 集成 MS-Temba 多尺度膨胀 SSM
[ ] 加入动作边界检测损失
[ ] 消融实验: Mamba-only vs Transformer-only vs Hybrid
[ ] 最终评估: 精度 + 延迟 + 参数量
```

### 阶段 4: 集成部署 (1周)
```
[ ] 封装为 backend/ml/behavior/mamba_sequence.py
[ ] 添加 ONNX 导出支持
[ ] 集成至 BehaviorRecognizer 路由层
[ ] 单元测试 + 端到端验证
```

## 预期产出

| 指标 | ST-GCN+BC 基线 | Mamba 实际结果 | Transformer-Mamba 目标 |
|------|---------------|---------------|----------------------|
| 准确率 | 42.27% (合成) | **63.18%** (合成) | ≥ 55% (合成) / ≥ 85% (真实) |
| 参数量 | 1.43M | **21K** | ~5M (VideoMamba-tiny) |
| 推理延迟 | 12.95ms/clip | **9.30ms/clip** (T=90) | ~1.5s/clip (GPU) |
| 长序列建模 | T=30 | **T=90** (3x 提升) | T=90+ (3x 提升) |

## 阻塞条件

- [ ] 需要标注数据集: APT-36K (申请中) 或合成多犬数据
- [ ] GPU 资源: RTX 5060 (20GB VRAM) 已满足最低要求
- [ ] 依赖: `pip install mamba-ssm flash-attn timm`

## 相关文档
- `dev-docs/decisions/0011-phase-4-llm-explainer-adr.md` - ADR
- `dev-docs/stages/phase-4.md` - Phase 4 计划
- `research/RESEARCH_VIDEO_MAMBA.md` (待创建)
