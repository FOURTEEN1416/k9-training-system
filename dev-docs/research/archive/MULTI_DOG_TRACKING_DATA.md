# Multi-Dog Tracking 测试数据调研

## 状态: ✅ 调研完成

## 问题现状
Phase 3.2d 阻塞原因: 缺少多犬视频 + GT 标注

## 调研结论

### 方案 A: APT-36K 数据集 (推荐)

| 属性 | 内容 |
|------|------|
| 链接 | https://github.com/ViTAE-Transformer/APTv2 |
| 规模 | 30 种动物物种，84,611 实例标注 |
| 犬科子集 | Canidae 类别，含 Wolf, Wild Dog, Dog 等 |
| 格式 | 视频 + 关键点 + 追踪 GT (MOT 格式) |
| 下载 | OneDrive (需邮件申请) |
| 许可证 | Apache-2.0 |
| 推荐理由 | 最大规模的动物多目标追踪数据集，含犬科 |

**申请方式**: 发送邮件至 viptae@gmail.com，说明研究用途

### 方案 B: 合成多犬数据 (短期可行)

利用现有 YouTube 犬类训练视频 + 人工标注 GT:
- 优点: 可控、可自定义、无许可问题
- 缺点: 规模有限 (10-20 段视频)

**可用视频源**:
- YouTube: "dog training obedience", "dog agility course"
- 已有: `data/uploads/warmup.mp4` (单犬)
- 目标: 2-3 犬同框训练场景

### 方案 C: MOT Challenge 基准 (基线测试)

| 数据集 | 链接 | 适用性 |
|--------|------|--------|
| MOT17/20 | motchallenge.net | ⚠️ 行人为主，IDF1~84% |
| SoccerNet | github.com/SoccerNet/sn-tracking | ⚠️ 足球运动员，IDF1=89% |

用途: 追踪器基线验证，非最终评测

## 推荐行动计划

### 短期 (1周内)
1. 申请 APT-36K 访问权限
2. 从 YouTube 收集 5-10 段多犬视频
3. 使用 Label Studio 人工标注 GT (预计 20h)

### 中期 (2-4周)
1. 获得 APT-36K 犬科子集
2. 运行现有评估脚本 `scripts/eval_multi_dog_tracking.py`
3. 报告 MOTA/IDF1/ID-switch 指标

## 现有代码就绪情况

```
backend/tests/ml/test_multi_dog_tracking.py  # 26/26 通过 ✅
scripts/eval_multi_dog_tracking.py           # 评估脚本就绪 ✅
backend/ml/tracking/multi_dog_tracker.py     # 追踪器实现 ✅
```

## 阻塞解除后

取消 `⏸️ DEFERRED` 标记，将 3.2d 状态更新为 `🔄 进行中`