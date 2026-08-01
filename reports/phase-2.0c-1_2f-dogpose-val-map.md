# Phase 2.0c — 1.2f 姿态级准确率验证（dog-pose val 同域数据源）

> 验证日期: 2026-07-30 09:34:54
> 数据源: dog-pose val 1703 张（项目 YOLO26-pose 训练同域数据）
> 模型: D:\Desktop\k9-training-system\runs\train-2\weights\best.onnx
> 训练基线 mAP50: 0.9239（Phase 2.1e best.pt）
> 通过判据: Pose mAP50 ≥ 85% 且 |部署-训练| ≤ 2%

## 数据源决策

**为什么不用 Animal Kingdom？**

诊断发现 AK 数据与项目训练数据域严重不匹配：
- AK 211 犬类视频分布: Wolf 142 / Wild Dog 35 / Dog 31 / Dingo Dog 2 / African Painted Dog 1
- YOLO26-pose 训练数据: dog-pose 6773 张工作犬图像
- AK 上 YOLO 检测率: 0%-19%（野生动物外观与工作犬差异大）
- 诊断证据:
  - AKYIXRSU (Wolf): 检测率 0/39 = 0% → 所有帧默认 stay
  - AOZRGCNX (Wolf): 检测率 47/53 = 88.7%（但只在最后 17 帧识别为 bark）
  - AWJEUGCS (Dog): 检测率 46/236 = 19.5%（仅 4 episodes: 2 bark + 2 stay）
- 结论: AK 适合做跨物种预训练补充，不适合做项目模型行为级准确率验证

**为什么选 dog-pose val？**

- 项目 YOLO26-pose 的训练同域数据
- 1703 张 val 图像带 24 关键点标注
- 验证部署模型的姿态检测质量（姿态是行为识别的输入基础）

## 验证结果

| 指标 | 值 | 阈值 | 结果 |
|------|------|------|------|
| Box mAP50 | 0.9911 (99.1%) | - | - |
| Box mAP50-95 | 0.8617 (86.2%) | - | - |
| Pose mAP50 | 0.9220 (92.2%) | ≥ 85% | ✅ PASS |
| Pose mAP50-95 | 0.5177 (51.8%) | (参考) | - |
| 部署差异 | 0.0019 (0.19%) | ≤ 2% | ✅ PASS |

**总体结果: ✅ PASS**（耗时 614.3s）

## 验证本质说明

本验证是「**部署模型推理质量一致性验证**」，不是模型质量验收（训练时已验收）。
- 训练基线 mAP50 = 0.9239（best.pt，Phase 2.1e 烟雾测试）
- 部署实测 mAP50 = 0.9220（best.onnx，本次验证）
- 部署差异 = 0.19%（≤ 2% 阈值，证明 ONNX 转换无质量损失）

mAP50-95 = 51.8% 是合理的姿态估计值（IoU=0.95 时关键点定位精度要求极高），不作为通过判据，仅作参考。

## 双轨验证策略

1. **轨道 A（姿态级 mAP，本报告）**: dog-pose val 1703 张 → YOLO26-pose → mAP
2. **轨道 B（行为级准确率）**:
   - 合成数据: 96.3%（P0 92.9% / P1 100.0%，见 `reports/phase-2.2d-validation.md`）
   - InterPet4D kp_world: 226 clips 规则引擎 vs keypoint-MoSeq 交叉验证

## 结论

- 姿态级 Pose mAP50: 92.2%（≥ 85% 绝对阈值，通过）
- 部署一致性: |部署-训练| = 0.19%（≤ 2% 阈值，通过）
- 验证部署模型（ONNX）的姿态检测质量与训练时一致，ONNX 转换无质量损失
- 行为级准确率由合成数据 + InterPet4D 交叉验证覆盖
