# Phase 2 启动前置 — 1.6d 真实序列验证 报告

> 任务: 1.6d
> 状态: 通过
> 验证日期: 2026-07-28 03:41:52
> 数据目录: `D:\Desktop\k9-training-system\data\interpet4d`
> 耗时: 1.8 秒
> 依据: ADR 0005 + ADR 0006

## 1. 摘要

- **总 clips**: 226
- **已处理**: 226
- **跳过**: 0
- **决策**: 通过: 管线 226/226 clips 无异常 + 9/9 姿态代理指标跨 clips 有变异（≥6）

## 2. 详细统计

```json
{
  "total_clips": 226,
  "processed_clips": 226,
  "pipeline_ok_clips": 226,
  "skipped_clips": 0,
  "pipeline_pass": true,
  "pose_metrics_pass_count": 9,
  "pose_metrics_total_count": 9,
  "pose_metrics_pass": true,
  "pose_metric_stats": {
    "nose_motion_energy": {
      "mean": 16.6008944711854,
      "std": 14.949076150927477,
      "min": 0.0,
      "max": 126.46788024902344,
      "nonzero_ratio": 0.995575221238938
    },
    "withers_motion_energy": {
      "mean": 10.461189493141344,
      "std": 8.58627633453975,
      "min": 0.0,
      "max": 72.91535949707031,
      "nonzero_ratio": 0.995575221238938
    },
    "nose_y_range": {
      "mean": 1.5276353736366846,
      "std": 0.6781822697111057,
      "min": 0.0,
      "max": 3.4294681549072266,
      "nonzero_ratio": 0.995575221238938
    },
    "withers_speed_mean": {
      "mean": 0.5153006112153551,
      "std": 0.3760690865624019,
      "min": 0.0,
      "max": 2.193324327468872,
      "nonzero_ratio": 0.995575221238938
    },
    "withers_speed_std": {
      "mean": 1.3321528858711233,
      "std": 1.0435474867017225,
      "min": 0.0,
      "max": 5.467984199523926,
      "nonzero_ratio": 0.995575221238938
    },
    "nose_speed_mean": {
      "mean": 0.8162777025944892,
      "std": 0.6753536016649435,
      "min": 0.0,
      "max": 4.3312177658081055,
      "nonzero_ratio": 0.995575221238938
    },
    "withers_x_range": {
      "mean": 0.8678762459781317,
      "std": 0.40818958839727576,
      "min": 0.0,
      "max": 1.992163062095642,
      "nonzero_ratio": 0.995575221238938
    },
    "motion_freeze_ratio": {
      "mean": 0.6475171862426389,
      "std": 0.18423827515878555,
      "min": 0.0,
      "max": 1.0,
      "nonzero_ratio": 0.995575221238938
    },
    "motion_recovery_time": {
      "mean": 0.1317109144542773,
      "std": 0.23554018128517282,
      "min": 0.0,
      "max": 1.2,
      "nonzero_ratio": 0.9823008849557522
    }
  },
  "note": "InterPet4D 无物体检测标注，puppy_signals 9 信号全降级为 penalty。姿态代理指标验证真实狗运动数据的姿态处理逻辑。物体检测场景的 9 信号验证需 YouTube 玩球视频（ADR 0006 §2.3）。"
}
```

## 3. 前 20 条 clip 详情

| clip_id | label/valid | frames | detected/信号 |
|---------|-------------|--------|---------------|
| interpet_dog01_p01_take01_ego_001 | 管线✓ | 326 | nose_motion_energy=3.849, withers_speed_mean=0.319, motion_freeze_ratio=0.735 |
| interpet_dog01_p01_take01_ego_003 | 管线✓ | 543 | nose_motion_energy=14.778, withers_speed_mean=0.602, motion_freeze_ratio=0.478 |
| interpet_dog01_p01_take02_ego_001 | 管线✓ | 537 | nose_motion_energy=9.579, withers_speed_mean=0.389, motion_freeze_ratio=0.750 |
| interpet_dog01_p01_take02_ego_002 | 管线✓ | 1041 | nose_motion_energy=17.370, withers_speed_mean=0.366, motion_freeze_ratio=0.709 |
| interpet_dog01_p01_take03_ego_001 | 管线✓ | 559 | nose_motion_energy=7.144, withers_speed_mean=0.300, motion_freeze_ratio=0.772 |
| interpet_dog01_p01_take03_ego_002 | 管线✓ | 915 | nose_motion_energy=31.981, withers_speed_mean=0.661, motion_freeze_ratio=0.575 |
| interpet_dog01_p01_take04_ego_001 | 管线✓ | 747 | nose_motion_energy=23.274, withers_speed_mean=0.692, motion_freeze_ratio=0.524 |
| interpet_dog01_p01_take04_ego_002 | 管线✓ | 629 | nose_motion_energy=25.193, withers_speed_mean=0.734, motion_freeze_ratio=0.591 |
| interpet_dog01_p01_take05_ego_001 | 管线✓ | 413 | nose_motion_energy=10.165, withers_speed_mean=0.617, motion_freeze_ratio=0.478 |
| interpet_dog01_p01_take05_ego_002 | 管线✓ | 412 | nose_motion_energy=9.823, withers_speed_mean=0.450, motion_freeze_ratio=0.723 |
| interpet_dog01_p01_take06_ego_001 | 管线✓ | 308 | nose_motion_energy=9.438, withers_speed_mean=0.819, motion_freeze_ratio=0.300 |
| interpet_dog01_p01_take06_ego_002 | 管线✓ | 1081 | nose_motion_energy=25.689, withers_speed_mean=0.432, motion_freeze_ratio=0.681 |
| interpet_dog01_p01_take07_ego_001 | 管线✓ | 723 | nose_motion_energy=27.480, withers_speed_mean=0.741, motion_freeze_ratio=0.540 |
| interpet_dog01_p01_take07_ego_002 | 管线✓ | 566 | nose_motion_energy=14.033, withers_speed_mean=0.443, motion_freeze_ratio=0.704 |
| interpet_dog01_p01_take08_ego_001 | 管线✓ | 640 | nose_motion_energy=16.912, withers_speed_mean=0.610, motion_freeze_ratio=0.510 |
| interpet_dog01_p01_take08_ego_002 | 管线✓ | 429 | nose_motion_energy=13.084, withers_speed_mean=0.542, motion_freeze_ratio=0.530 |
| interpet_dog01_p01_take09_ego_001 | 管线✓ | 309 | nose_motion_energy=9.768, withers_speed_mean=0.819, motion_freeze_ratio=0.289 |
| interpet_dog01_p01_take09_ego_002 | 管线✓ | 474 | nose_motion_energy=10.794, withers_speed_mean=0.468, motion_freeze_ratio=0.643 |
| interpet_dog01_p01_take09_ego_003 | 管线✓ | 318 | nose_motion_energy=13.039, withers_speed_mean=0.901, motion_freeze_ratio=0.454 |
| interpet_dog01_p01_take10_ego_001 | 管线✓ | 446 | nose_motion_energy=4.429, withers_speed_mean=0.237, motion_freeze_ratio=0.807 |

## 4. 结论与下一步

- 通过: 管线 226/226 clips 无异常 + 9/9 姿态代理指标跨 clips 有变异（≥6）
- 满足 Phase 2 启动前置条件之一
- 待 1.6d + 1.2f 均通过后，用户决策是否升级 Phase 2（ADR 0005）

## 5. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-28 | 初始版本 |
