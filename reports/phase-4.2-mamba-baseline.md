# Transformer-Mamba 基线实验 - 训练结果

## 状态: ✅ 训练完成 (2026-08-17)

## 实验设置

| 项目 | 值 |
|------|-----|
| 模型 | MambaSequenceBaseline (简化 VideoMamba) |
| 数据 | 合成数据 (22 类 × 60 样本/类) |
| 训练/验证 | 1056 / 264 |
| Epochs | 30 |
| 优化器 | AdamW (lr=3e-4, cosine) |
| 设备 | CUDA (RTX 5060) |
| 训练耗时 | 20.9s |

## 结果

| 指标 | ST-GCN+BC 基线 | **Mamba 基线** | 提升 |
|------|---------------|---------------|------|
| 最佳验证准确率 | 46.97% | **85.61%** | **+38.6pp** |
| 参数量 | 1.43M | **21,078** | **-98.5%** |
| 最佳 epoch | 21 | 12 | - |

## 训练曲线

```
Epoch  1: val_acc=18.2%
Epoch  5: val_acc=59.5%
Epoch 12: val_acc=85.6% ← 最佳
Epoch 30: val_acc=84.1%
```

## 分析

1. **大幅超越 ST-GCN+BC**: Mamba 简化版在合成数据上准确率 85.6% vs 46.97%
2. **参数量极小**: 21K 参数 (仅 ST-GCN 的 1.5%)，适合边缘部署
3. **快速收敛**: 12 epochs 即达最佳，训练仅 21 秒
4. **SSM 优势**: 线性复杂度时序建模比 GCN 更有效捕捉行为时序依赖

## 待验证项

- [ ] 真实数据验证 (APTv2 下载完成后)
- [ ] 与完整 VideoMamba (Mamba-2/SSM 原版) 对比
- [ ] ONNX 导出 + 部署延迟测试
- [ ] 长序列 (T>30) 性能

## 文件

- 模型: `backend/ml/behavior/mamba_sequence.py`
- 训练: `scripts/train_mamba_baseline.py`
- 权重: `runs/mamba_baseline/best.pt` (epoch 12)
- 历史: `runs/mamba_baseline/history.json`
- 总结: `runs/mamba_baseline/summary.json`