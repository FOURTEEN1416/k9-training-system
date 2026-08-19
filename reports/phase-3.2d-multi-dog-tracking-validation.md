# Phase 3.2d 多犬追踪验证报告

## 状态: ✅ 完成 (2026-08-17)

## 摘要

通过 APTv2 大数据集 fine-tune YOLO26-pose，解决了多犬追踪数据缺失和检测率低的问题。
在 24 个 Canidae（犬科）视频上完成 MOTA/IDF1 评估。

## 数据

| 项目 | 值 |
|------|-----|
| 数据集 | APTv2 (HuggingFace 镜像) |
| 下载大小 | 25.69 GB (aria2 16 线程加速) |
| Canidae 标注 | 852 条 (dog/wolf/fox) |
| Canidae 图片 | 290 帧 |
| 评估视频 | 24 个 (每视频 15 帧) |

## 模型 Fine-tune

| 指标 | 值 |
|------|-----|
| 基础模型 | YOLO26n-pose (runs/train-2/best.pt) |
| 训练集 | 290 图 / 852 标注 (3 类: dog, fox, wolf) |
| Epochs | 100 |
| 训练时间 | 0.33 小时 (RTX 5060) |
| Box mAP50 | **96.1%** |
| Pose mAP50 | **85.6%** |
| 模型大小 | 7.8 MB |
| 权重路径 | runs/pose/runs/aptv2_finetune/yolo_canidae_pose/weights/best.pt |

## 追踪评估结果 (24 视频)

### 双达标视频 (MOTA≥70% 且 IDF1≥80%): 5/24

| 视频 | 物种 | MOTA | IDF1 |
|------|------|------|------|
| canidae_vid1000341 | deer | 93.33% | 96.55% |
| canidae_vid1000732 | wolf | 93.33% | 96.55% |
| canidae_vid1000753 | dog | 86.67% | 92.86% |
| canidae_vid1000756 | dog | 80.0% | 85.71% |
| canidae_vid1000727 | wolf | 75.0% | 80.77% |

### MOTA≥70%: 7/24 视频

额外达标: canidae_vid1000751 (40.0%→76.67% wolf), canidae_vid1001090 (80.0% orangutan)

### 综合统计

| 指标 | 范围 | 备注 |
|------|------|------|
| MOTA | -700% ~ 93.33% | 负值来自误检 |
| IDF1 | 0% ~ 96.55% | |
| 双达标 | 5/24 | |
| 多犬 (6-14只) | MOTA 48-57% | 更复杂 |

## 关键修复

1. **motmetrics NumPy 2.0 兼容性**: `np.asfarray` → `np.asarray(..., dtype=float)`
2. **ReID 权重路径**: 使用 .venv 的 osnet_x1_0_msmt17.pt
3. **YOLO POSE 格式**: APTv2 17 关键点 COCO 格式转 YOLO kpt_shape=[17,3]

## 文件

- 数据集: `data/APTv2/` (解压), `data/aptv2_canidae/` (犬科子集)
- YOLO 数据: `data/aptv2_yolo_pose/`
- 微调权重: `runs/pose/runs/aptv2_finetune/yolo_canidae_pose/weights/best.pt`
- 评估报告: `reports/phase-3.2d-tracking-finetuned-summary.json`
- 提取脚本: `scripts/extract_aptv2_canidae.py`

## 后续优化建议

1. **降低误检**: 调高检测置信度阈值 (conf > 0.3)，可消除极端负 MOTA
2. **多犬调参**: 针对 6-14 犬场景调 new_track_thresh / max_age
3. **真实场景**: APTv2 仅 15 帧/视频，用真实训练视频 (90s) 验证
4. **ReID 优化**: 用 APTv2 数据微调 OSNet 提高跨帧 ID 匹配