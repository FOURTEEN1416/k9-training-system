"""多犬追踪评估脚本（Phase 3.2d）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.2d
依据: dev-docs/stages/phase-3.md §3.2d + dev-docs/research/RESEARCH_MULTI_DOG_TRACKING.md

功能:
    1. 无 ground truth 模式: 定性指标（轨迹数 / 覆盖率 / ID 稳定性 / 帧分布）
    2. 有 ground truth 模式: 定量指标（MOTA / IDF1 / ID switch / 精确率 / 召回率）
    3. 输出 JSON 报告 + 控制台摘要

用法:
    # 无 GT（定性评估）
    python scripts/eval_multi_dog_tracking.py --video data/uploads/test.mp4

    # 有 GT（定量评估，MOT 标准格式）
    python scripts/eval_multi_dog_tracking.py --video data/uploads/test.mp4 --gt data/gt/gt.txt

    # 指定输出
    python scripts/eval_multi_dog_tracking.py --video test.mp4 --output reports/tracking_eval.json

GT 格式（MOT Challenge 标准）:
    每行: frame_idx, track_id, x, y, w, h, conf, class, visibility
    帧索引从 1 开始
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 项目根目录加入 sys.path（脚本从 scripts/ 目录运行）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_tracker(
    video_path: str,
    model_path: str = "runs/train-2/weights/best.pt",
    reid_model: str = "osnet_x1_0",
    device: int = 0,
    new_track_thresh: float = 0.3,
    min_hits: int = 1,
):
    """运行 MultiDogTracker 并返回结果."""
    from backend.ml.tracking import MultiDogTracker, TrackerBackend

    tracker = MultiDogTracker(
        pose_model_path=model_path,
        tracker_backend=TrackerBackend.OCCLUBOOST,
        reid_model=reid_model,
        device=device,
        new_track_thresh=new_track_thresh,
        min_hits=min_hits,
    )
    return tracker.track_video(video_path)


def compute_qualitative_metrics(result) -> Dict:
    """计算定性指标（无 ground truth）.

    指标:
        - num_dogs: 追踪到的不同犬只数量
        - total_frames: 视频总帧数
        - avg_coverage: 平均覆盖率（每犬出现帧数 / 总帧数）
        - avg_track_length: 平均轨迹长度
        - id_switch_estimate: ID switch 估计（基于轨迹断裂分析）
        - per_track: 每犬详细统计
    """
    metrics = {
        "num_dogs": result.num_dogs,
        "total_frames": result.total_frames,
        "track_ids": result.track_ids,
    }

    if result.num_dogs == 0:
        metrics["avg_coverage"] = 0.0
        metrics["avg_track_length"] = 0.0
        metrics["id_switch_estimate"] = 0
        metrics["per_track"] = []
        return metrics

    per_track = []
    coverages = []
    track_lengths = []
    total_gaps = 0

    for tid in result.track_ids:
        track = result.get_track(tid)
        indices = track.frame_indices
        coverage = track.num_frames / max(result.total_frames, 1)
        coverages.append(coverage)
        track_lengths.append(track.num_frames)

        # 计算轨迹断裂（gaps = ID switch 估计）
        gaps = 0
        for i in range(1, len(indices)):
            if indices[i] - indices[i - 1] > 1:
                gaps += 1
        total_gaps += gaps

        per_track.append({
            "track_id": tid,
            "num_frames": track.num_frames,
            "coverage": round(coverage * 100, 1),
            "frame_range": [indices[0], indices[-1]] if indices else [0, 0],
            "gaps": gaps,
        })

    metrics["avg_coverage"] = round(np.mean(coverages) * 100, 1)
    metrics["avg_track_length"] = round(np.mean(track_lengths), 1)
    metrics["id_switch_estimate"] = total_gaps
    metrics["per_track"] = per_track
    return metrics


def load_mot_gt(gt_path: str) -> Dict[int, List[Tuple]]:
    """加载 MOT Challenge 格式 ground truth.

    格式: frame_idx, track_id, x, y, w, h, conf, class, visibility
    帧索引从 1 开始

    Returns:
        {frame_idx (0-based): [(track_id, x, y, w, h), ...]}
    """
    gt = defaultdict(list)
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_idx = int(parts[0]) - 1  # 转为 0-based
            track_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            gt[frame_idx].append((track_id, x, y, w, h))
    return dict(gt)


def compute_quantitative_metrics(result, gt_data: Dict) -> Dict:
    """计算定量指标（有 ground truth，使用 motmetrics）.

    指标:
        - MOTA: 多目标追踪准确率
        - IDF1: ID F1 分数
        - ID_switch: ID 切换次数
        - precision: 精确率
        - recall: 召回率
        - mostly_tracked: 大部分时间被正确追踪的轨迹数
        - mostly_lost: 大部分时间丢失的轨迹数
    """
    import motmetrics as mm

    # 构建 motmetrics 格式数据
    acc = mm.MOTAccumulator(auto_id=True)

    # 预测结果: {frame_idx: [(track_id, x, y, w, h), ...]}
    pred_by_frame = defaultdict(list)
    for tid in result.track_ids:
        track = result.get_track(tid)
        for f in track.frames:
            x1, y1, x2, y2 = f.bbox
            pred_by_frame[f.frame_idx].append((tid, x1, y1, x2 - x1, y2 - y1))

    # 逐帧对比
    all_frames = sorted(set(list(gt_data.keys()) + list(pred_by_frame.keys())))
    for frame_idx in all_frames:
        gt_objs = gt_data.get(frame_idx, [])
        pred_objs = pred_by_frame.get(frame_idx, [])

        gt_ids = [o[0] for o in gt_objs]
        pred_ids = [o[0] for o in pred_objs]

        # 计算距离（IoU 距离）
        if gt_objs and pred_objs:
            gt_boxes = np.array([[o[1], o[2], o[3], o[4]] for o in gt_objs])
            pred_boxes = np.array([[o[1], o[2], o[3], o[4]] for o in pred_objs])
            distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        else:
            distances = np.zeros((len(gt_objs), len(pred_objs)))

        acc.update(gt_ids, pred_ids, distances)

    # 计算指标
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "idf1", "num_switches", "precision", "recall",
                 "mostly_tracked", "mostly_lost", "num_matches", "num_false_positives",
                 "num_misses"],
        name="multi_dog_tracking",
    )

    return {
        "MOTA": round(float(summary["mota"].iloc[0]) * 100, 2),
        "IDF1": round(float(summary["idf1"].iloc[0]) * 100, 2),
        "ID_switch": int(summary["num_switches"].iloc[0]),
        "precision": round(float(summary["precision"].iloc[0]) * 100, 2),
        "recall": round(float(summary["recall"].iloc[0]) * 100, 2),
        "mostly_tracked": int(summary["mostly_tracked"].iloc[0]),
        "mostly_lost": int(summary["mostly_lost"].iloc[0]),
        "num_matches": int(summary["num_matches"].iloc[0]),
        "num_false_positives": int(summary["num_false_positives"].iloc[0]),
        "num_misses": int(summary["num_misses"].iloc[0]),
    }


def evaluate(
    video_path: str,
    model_path: str = "runs/train-2/weights/best.pt",
    gt_path: Optional[str] = None,
    reid_model: str = "osnet_x1_0",
    device: int = 0,
    output_path: Optional[str] = None,
) -> Dict:
    """评估多犬追踪.

    Args:
        video_path: 视频路径
        model_path: YOLO26-pose 模型路径
        gt_path: ground truth 路径（MOT 格式，可选）
        reid_model: ReID 模型名
        device: 推理设备
        output_path: JSON 报告输出路径

    Returns:
        评估结果字典
    """
    print(f"[eval] 视频: {video_path}", flush=True)
    print(f"[eval] 模型: {model_path}", flush=True)
    print(f"[eval] ReID: {reid_model}", flush=True)

    # 运行追踪
    result = run_tracker(
        video_path=video_path,
        model_path=model_path,
        reid_model=reid_model,
        device=device,
    )
    print(f"[eval] 追踪完成: {result.num_dogs} 犬, {result.total_frames} 帧", flush=True)

    # 定性指标
    report = {
        "video_path": video_path,
        "model_path": model_path,
        "reid_model": reid_model,
        "tracker_backend": "OccluBoost",
        "total_frames": result.total_frames,
        "fps": result.fps,
    }
    report["qualitative"] = compute_qualitative_metrics(result)

    # 定量指标（有 GT 时）
    if gt_path and Path(gt_path).exists():
        print(f"[eval] 加载 GT: {gt_path}", flush=True)
        gt_data = load_mot_gt(gt_path)
        print(f"[eval] GT: {len(gt_data)} 帧标注", flush=True)
        report["quantitative"] = compute_quantitative_metrics(result, gt_data)
    else:
        report["quantitative"] = None

    # 输出摘要
    print("\n" + "=" * 60, flush=True)
    print("多犬追踪评估结果", flush=True)
    print("=" * 60, flush=True)
    q = report["qualitative"]
    print(f"  追踪犬数: {q['num_dogs']}", flush=True)
    print(f"  总帧数: {q['total_frames']}", flush=True)
    print(f"  平均覆盖率: {q['avg_coverage']}%", flush=True)
    print(f"  平均轨迹长度: {q['avg_track_length']} 帧", flush=True)
    print(f"  ID switch 估计: {q['id_switch_estimate']}", flush=True)
    for t in q["per_track"]:
        print(f"    track {t['track_id']}: {t['num_frames']} 帧, "
              f"覆盖率 {t['coverage']}%, range={t['frame_range']}, gaps={t['gaps']}", flush=True)

    if report["quantitative"]:
        m = report["quantitative"]
        print(f"\n  --- 定量指标 (vs GT) ---", flush=True)
        print(f"  MOTA: {m['MOTA']}%", flush=True)
        print(f"  IDF1: {m['IDF1']}%", flush=True)
        print(f"  ID switch: {m['ID_switch']}", flush=True)
        print(f"  精确率: {m['precision']}%", flush=True)
        print(f"  召回率: {m['recall']}%", flush=True)
        print(f"  mostly_tracked: {m['mostly_tracked']}", flush=True)
        print(f"  mostly_lost: {m['mostly_lost']}", flush=True)

    # 验收阈值检查
    if report["quantitative"]:
        thresholds = {"MOTA": 70.0, "IDF1": 80.0}
        print(f"\n  --- 验收阈值 ---", flush=True)
        for metric, threshold in thresholds.items():
            val = report["quantitative"][metric]
            status = "✅" if val >= threshold else "❌"
            print(f"  {status} {metric}: {val}% (阈值 ≥ {threshold}%)", flush=True)

    # 保存报告
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[eval] 报告已保存: {output_path}", flush=True)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="多犬追踪评估（Phase 3.2d）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--model", default="runs/train-2/weights/best.pt", help="YOLO26-pose 模型路径")
    parser.add_argument("--gt", default=None, help="Ground truth 路径（MOT 格式，可选）")
    parser.add_argument("--reid", default="osnet_x1_0", help="ReID 模型名")
    parser.add_argument("--device", type=int, default=0, help="推理设备")
    parser.add_argument("--output", default=None, help="JSON 报告输出路径")
    args = parser.parse_args()

    evaluate(
        video_path=args.video,
        model_path=args.model,
        gt_path=args.gt,
        reid_model=args.reid,
        device=args.device,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
