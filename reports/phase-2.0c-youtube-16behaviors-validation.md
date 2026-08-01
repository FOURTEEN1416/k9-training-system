# Phase 2.0c — 1.2f YouTube 16 行为定性验证 报告

> 验证日期: 2026-07-30 02:16:28
> 视频数: 5
> 模型: D:\Desktop\k9-training-system\runs\train-2\weights\best.onnx
> 验证类型: 定性（无 ground truth，定量准确率阻塞于 2.0b 人工标注）

## 汇总

- 总 episodes: 538
- 行为种类: 4
- 行为分布: {
  "bark": 298,
  "bite": 66,
  "stay": 173,
  "down": 1
}

## 逐视频详情

| 视频 | 时长(s) | 帧 | 检测率 | episodes | 行为 | 耗时(s) |
|------|---------|----|--------|---------:|------|--------:|
| c9zbP90BMfc.mp4 | 250.0 | 5994 | 38.3% | 134 | bark:70, bite:24, stay:40 | 149.3 |
| n0LoPl6KzlA.mp4 | 159.3 | 3820 | 58.5% | 88 | bark:52, bite:11, down:1, stay:24 | 82.9 |
| obEZTF_Rr7E.mp4 | 32.1 | 961 | 68.5% | 18 | bark:15, bite:1, stay:2 | 20.2 |
| Obue6tS-AT8.mp4 | 175.7 | 5271 | 79.4% | 154 | bark:89, bite:7, stay:58 | 109.6 |
| uBmVYflCzSI.mp4 | 179.9 | 4497 | 80.5% | 144 | bark:72, bite:23, stay:49 | 91.3 |

## 结论

- 本验证为定性验证（YouTube 视频无 ground truth 标签）
- 定量准确率（≥80%）阻塞于 2.0b Label Studio 人工标注
- kp_world 替代验证已通过（10/10 clips，10 种行为，见 reports/phase-2-prereq-1.2f-validation.md）
- 合成数据基线 92.9%（reports/phase-1.2f-validation.md）
