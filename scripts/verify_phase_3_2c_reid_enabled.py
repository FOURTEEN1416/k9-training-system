"""Phase 3.2c 端到端验证 — ReID 启用前后追踪效果对比.

Owner: ML 开发
Phase: 3.2c

目标:
    1. 验证 MultiDogTracker 显式启用 with_reid=True 后能正常初始化
    2. 在 warmup.mp4 上对比 ReID 启用前后的追踪指标
    3. 运行 IDSwitchMonitor 评估 ID switch 事件
    4. 输出对比报告到 reports/phase-3.2c-reid-enabled-warmup.json

注意:
    - 本脚本为验证脚本，不属于 production 代码
    - 验证完成后保留作为 3.2c 验证证据
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.tracking import (
    MultiDogTracker,
    TrackerBackend,
    IDSwitchMonitor,
)
from backend.ml.tracking.multi_dog_tracker import _parse_yolo_detections


def run_tracking_with_reid_setting(
    video_path: str,
    model_path: str,
    with_reid: bool,
    device: int = 0,
):
    """运行追踪并返回结果 + 初始化日志.

    Args:
        video_path: 视频路径
        model_path: YOLO26-pose 模型路径
        with_reid: 是否启用 ReID
        device: 推理设备

    Returns:
        (result, init_log)
    """
    tracker = MultiDogTracker(
        pose_model_path=model_path,
        tracker_backend=TrackerBackend.OCCLUBOOST,
        reid_model="osnet_x1_0_msmt17.pt",
        device=device,
        new_track_thresh=0.3,
        min_hits=1,
        tracker_kwargs={"with_reid": with_reid},
    )

    # 验证 with_reid 设置
    tracker._init_tracker()
    actual_with_reid = tracker._tracker.with_reid

    init_log = {
        "requested_with_reid": with_reid,
        "actual_with_reid": actual_with_reid,
        "tracker_class": type(tracker._tracker).__name__,
    }

    # 重置后跑视频
    tracker.reset()
    t0 = time.time()
    result = tracker.track_video(video_path)
    elapsed = time.time() - t0

    init_log["elapsed_seconds"] = round(elapsed, 2)
    return result, init_log


def compute_metrics(result) -> dict:
    """计算定性指标."""
    if result.num_dogs == 0:
        return {
            "num_dogs": 0,
            "total_frames": result.total_frames,
            "track_ids": [],
            "avg_coverage": 0.0,
            "avg_track_length": 0.0,
        }

    coverages = []
    track_lengths = []
    per_track = []
    for tid in result.track_ids:
        track = result.get_track(tid)
        cov = track.num_frames / max(result.total_frames, 1)
        coverages.append(cov)
        track_lengths.append(track.num_frames)
        per_track.append({
            "track_id": tid,
            "num_frames": track.num_frames,
            "coverage_pct": round(cov * 100, 1),
            "frame_range": [track.frame_indices[0], track.frame_indices[-1]] if track.frame_indices else [0, 0],
        })

    return {
        "num_dogs": result.num_dogs,
        "total_frames": result.total_frames,
        "track_ids": result.track_ids,
        "avg_coverage_pct": round(float(np.mean(coverages)) * 100, 1),
        "avg_track_length": round(float(np.mean(track_lengths)), 1),
        "per_track": per_track,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3.2c ReID 启用对比验证")
    parser.add_argument(
        "--video", default="data/uploads/_warmup.mp4",
        help="视频路径"
    )
    parser.add_argument(
        "--model", default="runs/train-2/weights/best.pt",
        help="YOLO26-pose 模型路径"
    )
    parser.add_argument(
        "--output", default="reports/phase-3.2c-reid-enabled-warmup.json",
        help="输出 JSON 报告路径"
    )
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    video_path = str(PROJECT_ROOT / args.video)
    model_path = str(PROJECT_ROOT / args.model)
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Phase 3.2c ReID 启用对比验证")
    print(f"{'=' * 70}")
    print(f"视频: {video_path}")
    print(f"模型: {model_path}")
    print(f"输出: {output_path}\n")

    # === 阶段 1: with_reid=False (3.2b 状态) ===
    print(f"[1/2] 运行 with_reid=False（3.2b 基线）...")
    try:
        result_no_reid, log_no_reid = run_tracking_with_reid_setting(
            video_path, model_path, with_reid=False, device=args.device
        )
        metrics_no_reid = compute_metrics(result_no_reid)
        # ID switch 监测
        monitor = IDSwitchMonitor()
        report_no_reid = monitor.evaluate(result_no_reid)
        id_switch_no_reid = report_no_reid.summary()
    except Exception as e:
        print(f"  ❌ with_reid=False 失败: {e}")
        log_no_reid = {"error": str(e)}
        metrics_no_reid = {}
        id_switch_no_reid = {}

    print(f"  ✓ with_reid=False 完成: {log_no_reid.get('actual_with_reid', 'N/A')}")

    # === 阶段 2: with_reid=True (3.2c 修复) ===
    print(f"\n[2/2] 运行 with_reid=True（3.2c 修复后）...")
    try:
        result_with_reid, log_with_reid = run_tracking_with_reid_setting(
            video_path, model_path, with_reid=True, device=args.device
        )
        metrics_with_reid = compute_metrics(result_with_reid)
        monitor = IDSwitchMonitor()
        report_with_reid = monitor.evaluate(result_with_reid)
        id_switch_with_reid = report_with_reid.summary()
    except Exception as e:
        print(f"  ❌ with_reid=True 失败: {e}")
        log_with_reid = {"error": str(e)}
        metrics_with_reid = {}
        id_switch_with_reid = {}

    print(f"  ✓ with_reid=True 完成: {log_with_reid.get('actual_with_reid', 'N/A')}")

    # === 汇总对比 ===
    report = {
        "phase": "3.2c",
        "video_path": args.video,
        "model_path": args.model,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_no_reid": {
            "init_log": log_no_reid,
            "metrics": metrics_no_reid,
            "id_switch": id_switch_no_reid,
        },
        "with_reid_enabled": {
            "init_log": log_with_reid,
            "metrics": metrics_with_reid,
            "id_switch": id_switch_with_reid,
        },
    }

    # 控制台摘要
    print(f"\n{'=' * 70}")
    print(f"对比摘要")
    print(f"{'=' * 70}")
    print(f"{'指标':<30} {'with_reid=False':<20} {'with_reid=True':<20}")
    print(f"{'-' * 70}")

    if metrics_no_reid and metrics_with_reid:
        for key in ["num_dogs", "avg_coverage_pct", "avg_track_length"]:
            v1 = metrics_no_reid.get(key, "N/A")
            v2 = metrics_with_reid.get(key, "N/A")
            print(f"{key:<30} {str(v1):<20} {str(v2):<20}")

    if id_switch_no_reid and id_switch_with_reid:
        for key in ["total_events", "switch_rate", "confirmed_events"]:
            v1 = id_switch_no_reid.get(key, "N/A")
            v2 = id_switch_with_reid.get(key, "N/A")
            print(f"{'id_switch.' + key:<30} {str(v1):<20} {str(v2):<20}")

    # 保存报告
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 报告已保存: {output_path}")


if __name__ == "__main__":
    main()
