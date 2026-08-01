# Phase 2 启动前置 — 1.2f 真实数据复核 报告

> 任务: 1.2f
> 状态: 通过（kp_world 模式）
> 验证日期: 2026-07-30 01:43:22
> 数据目录: `D:\Desktop\k9-training-system\data\interpet4d`
> 模型: `D:\Desktop\k9-training-system\runs\train-2\weights\best.onnx`
> 耗时: 1.9 秒
> 依据: ADR 0005 + ADR 0006

## 1. 摘要

- **总 clips**: 10
- **已处理**: 10
- **跳过**: 0
- **决策**: 通过（kp_world 模式）: 引擎 10/10 clips 无异常 + 10 种行为 + 9 种行为组合. 合成基线 92.9% 保留，确认跳过 Phase 1.3 PoseC3D

## 2. 详细统计

```json
{
  "total_clips": 10,
  "processed_clips": 10,
  "skipped_clips": 0,
  "clips_with_behaviors": 10,
  "clips_unknown_only": 0,
  "clips_no_episodes": 0,
  "behavior_distribution": {
    "down": 6,
    "obstacle": 9,
    "sit": 5,
    "alert_down": 2,
    "track": 4,
    "recall": 8,
    "stay": 7,
    "heel": 2,
    "sit_up": 1,
    "stand": 5
  },
  "distinct_behavior_types": 10,
  "distinct_behavior_combos": 9,
  "synthetic_baseline_accuracy": 0.929,
  "mode": "kp_world_fallback",
  "note": "InterPet4D v1 无视频文件和行为标签，无法执行原设计的 视频→YOLO26-pose→准确率 验证。改用 kp_world→y翻转→规则引擎→行为分布合理性 替代验证。合成数据 92.9% 基线已达标（reports/phase-1.2f-validation.md）。"
}
```

## 3. 前 20 条 clip 详情

| clip_id | label/valid | frames | detected/信号 |
|---------|-------------|--------|---------------|
| interpet_dog01_p01_take01_ego_001 | ep=2 | 326 | down |
| interpet_dog01_p01_take01_ego_003 | ep=9 | 543 | alert_down, down, obstacle, sit, track |
| interpet_dog01_p01_take02_ego_001 | ep=7 | 537 | down, obstacle, recall, stay |
| interpet_dog01_p01_take02_ego_002 | ep=16 | 1041 | heel, obstacle, recall, sit, sit_up, stand, stay |
| interpet_dog01_p01_take03_ego_001 | ep=4 | 559 | obstacle, recall, stay |
| interpet_dog01_p01_take03_ego_002 | ep=16 | 915 | obstacle, recall, stand, stay |
| interpet_dog01_p01_take04_ego_001 | ep=16 | 747 | obstacle, recall, stay |
| interpet_dog01_p01_take04_ego_002 | ep=59 | 629 | alert_down, down, heel, obstacle, recall, sit, stand, stay, track |
| interpet_dog01_p01_take05_ego_001 | ep=15 | 413 | down, obstacle, recall, sit, stand, track |
| interpet_dog01_p01_take05_ego_002 | ep=30 | 412 | down, obstacle, recall, sit, stand, stay, track |

## 4. 结论与下一步

- 通过（kp_world 模式）: 引擎 10/10 clips 无异常 + 10 种行为 + 9 种行为组合. 合成基线 92.9% 保留，确认跳过 Phase 1.3 PoseC3D
- 确认 Phase 1.3 PoseC3D 跳过决策
- 满足 Phase 2 启动前置条件之一
- 注: InterPet4D v1 无视频/标签，采用 kp_world 替代验证 + 合成基线 92.9%
- 待 1.6d + 1.2f 均通过后，用户决策是否升级 Phase 2（ADR 0005）

## 5. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-30 | 初始版本 |
