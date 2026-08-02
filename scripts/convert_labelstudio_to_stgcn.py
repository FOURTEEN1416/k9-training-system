"""Label Studio 标注 → ST-GCN+BC 训练格式转换器（Phase 3 真实数据标注辅线）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1d 辅线（真实数据标注支持）
依据: dev-docs/stages/phase-3.md §3.1d 数据源优先级

功能:
    将 Label Studio 导出的 JSON 标注转换为 ST-GCN+BC 训练所需的 pyskl pickle 格式。
    支持 Label Studio v1.23+ 的视频关键点标注导出格式。

输入: Label Studio JSON 导出（export_type=JSON）
    [
      {
        "id": 1,
        "data": {"video": "/upload/1/video.mp4"},
        "annotations": [
          {
            "result": [
              {
                "type": "videobbox" | "keypoint" | "videorectangle",
                "value": {...},
                "frame": int  # 帧索引
              }
            ]
          }
        ]
      }
    ]

输出: pyskl pickle 格式
    List[Dict]，每个 Dict 含:
        - keypoint: (1, T, 24, 3) np.ndarray
        - label: int (BEHAVIOR_TO_IDX)
        - label_name: str
        - frame_dir: str
        - boundary: (T,) np.ndarray

用法:
    python scripts/convert_labelstudio_to_stgcn.py \\
        --input data/labelstudio/export.json \\
        --keypoints-dir data/labelstudio/keypoints \\
        --output data/stgcn_bc/train_real.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.behavior.constants import NUM_KEYPOINTS
from backend.ml.behavior.stgcn_bc.labels import BEHAVIOR_TO_IDX


def parse_labelstudio_keypoints(
    annotation_result: List[Dict],
    total_frames: int,
    num_keypoints: int = NUM_KEYPOINTS,
) -> tuple[np.ndarray, Dict[int, str]]:
    """解析 Label Studio 关键点标注 → (T, 24, 3) 序列.

    Args:
        annotation_result: Label Studio annotation result 列表
        total_frames: 视频总帧数
        num_keypoints: 关键点数（24）

    Returns:
        (keypoints_sequence, frame_labels)
        - keypoints_sequence: shape=(T, 24, 3) [x, y, conf]
        - frame_labels: {frame_idx: behavior_name}
    """
    # 初始化空序列（conf=0 表示未标注）
    kpts_seq = np.zeros((total_frames, num_keypoints, 3), dtype=np.float32)

    # 帧级行为标签（keypointlabels 可能包含行为类别）
    frame_labels: Dict[int, str] = {}

    for item in annotation_result:
        item_type = item.get("type", "")
        value = item.get("value", {})
        frame = item.get("frame", 0)

        if frame < 0 or frame >= total_frames:
            continue

        if item_type == "keypoint":
            # 单点标注: value = {"x": 0.5, "y": 0.3, "keypointlabels": ["nose"]}
            x_norm = value.get("x", 0.0) / 100.0  # Label Studio x 是百分比
            y_norm = value.get("y", 0.0) / 100.0
            labels = value.get("keypointlabels", [])

            # 简化：将关键点按顺序映射到索引（实际需按 label 名映射）
            # 这里仅做位置记录，具体索引映射需结合项目 24 关键点定义
            if labels:
                kp_name = labels[0]
                kp_idx = _map_keypoint_name_to_idx(kp_name)
                if kp_idx is not None:
                    kpts_seq[frame, kp_idx, 0] = x_norm
                    kpts_seq[frame, kp_idx, 1] = y_norm
                    kpts_seq[frame, kp_idx, 2] = 1.0  # 置信度

        elif item_type == "videobbox":
            # 视频检测框 + 标签
            labels = value.get("labels", value.get("videolabels", []))
            if labels:
                frame_labels[frame] = labels[0]

        elif item_type == "videorectangle":
            # 视频矩形标注（行为段落）
            labels = value.get("labels", value.get("rectanglelabels", []))
            if labels:
                frame_labels[frame] = labels[0]

    return kpts_seq, frame_labels


def _map_keypoint_name_to_idx(name: str) -> Optional[int]:
    """Label Studio 关键点名 → 项目 24 索引映射.

    项目 24 关键点定义（K9Graph.NODE_NAMES）:
        0-2: front_left_paw/knee/elbow
        3-5: rear_left_paw/knee/elbow
        6-8: front_right_paw/knee/elbow
        9-11: rear_right_paw/knee/elbow
        12-13: tail_start/end
        14-15: left/right_ear_base
        16-17: nose/chin
        18-19: left/right_ear_tip
        20-21: left/right_eye
        22: withers
        23: throat
    """
    mapping = {
        "front_left_paw": 0, "front_left_knee": 1, "front_left_elbow": 2,
        "rear_left_paw": 3, "rear_left_knee": 4, "rear_left_elbow": 5,
        "front_right_paw": 6, "front_right_knee": 7, "front_right_elbow": 8,
        "rear_right_paw": 9, "rear_right_knee": 10, "rear_right_elbow": 11,
        "tail_start": 12, "tail_end": 13,
        "left_ear_base": 14, "right_ear_base": 15,
        "nose": 16, "chin": 17,
        "left_ear_tip": 18, "right_ear_tip": 19,
        "left_eye": 20, "right_eye": 21,
        "withers": 22, "throat": 23,
    }
    return mapping.get(name.lower().replace(" ", "_"))


def generate_boundary_labels(
    frame_labels: Dict[int, str],
    total_frames: int,
    boundary_window: int = 2,
) -> np.ndarray:
    """根据帧级行为标签生成边界标签.

    行为切换点前后 boundary_window 帧标记为 1（边界）。

    Args:
        frame_labels: {frame_idx: behavior_name}
        total_frames: 总帧数
        boundary_window: 边界窗口大小

    Returns:
        (T,) np.ndarray，0/1 边界标签
    """
    boundary = np.zeros(total_frames, dtype=np.float32)
    if not frame_labels:
        return boundary

    # 按帧排序
    sorted_frames = sorted(frame_labels.keys())
    prev_behavior = None

    for f in sorted_frames:
        curr_behavior = frame_labels[f]
        if prev_behavior is not None and curr_behavior != prev_behavior:
            # 行为切换点
            for offset in range(-boundary_window, boundary_window + 1):
                idx = f + offset
                if 0 <= idx < total_frames:
                    boundary[idx] = 1.0
        prev_behavior = curr_behavior

    return boundary


def convert_labelstudio_to_pyskl(
    json_path: str | Path,
    keypoints_dir: str | Path,
    output_path: str | Path,
    default_total_frames: int = 2700,
) -> dict:
    """转换 Label Studio JSON → pyskl pickle.

    Args:
        json_path: Label Studio JSON 导出路径
        keypoints_dir: YOLO26-pose 推理生成的 keypoints pkl 目录
        output_path: 输出 pyskl pickle 路径
        default_total_frames: 默认总帧数（无法读取视频时）

    Returns:
        转换统计
    """
    json_path = Path(json_path)
    keypoints_dir = Path(keypoints_dir)
    output_path = Path(output_path)

    with open(json_path, encoding="utf-8") as f:
        tasks = json.load(f)

    samples: List[Dict] = []
    stats = {
        "total_tasks": len(tasks),
        "converted": 0,
        "skipped_no_annotation": 0,
        "skipped_no_keypoints": 0,
        "skipped_unknown_behavior": 0,
        "behaviors": {},
    }

    for task in tasks:
        annotations = task.get("annotations", [])
        if not annotations:
            stats["skipped_no_annotation"] += 1
            continue

        result = annotations[0].get("result", [])
        if not result:
            stats["skipped_no_annotation"] += 1
            continue

        # 获取视频信息
        video_data = task.get("data", {}).get("video", "")
        video_id = task.get("id", 0)

        # 尝试加载对应的 YOLO26-pose keypoints pkl
        kpts_pkl_path = keypoints_dir / f"{video_id}.pkl"
        if kpts_pkl_path.exists():
            with open(kpts_pkl_path, "rb") as f:
                kpts_data = pickle.load(f)
            kpts_seq = kpts_data.get("keypoints_sequence", np.zeros((0, 24, 3)))
            total_frames = len(kpts_seq)
        else:
            # 无 keypoints pkl，使用空序列
            total_frames = default_total_frames
            kpts_seq = np.zeros((total_frames, 24, 3), dtype=np.float32)
            stats["skipped_no_keypoints"] += 1
            # 仍尝试解析行为标签

        # 解析 Label Studio 标注
        _, frame_labels = parse_labelstudio_keypoints(
            result, total_frames
        )

        # 如果有 keypoints pkl，用其关键点；否则用 Label Studio 标注的关键点
        if kpts_pkl_path.exists():
            # 用 YOLO26-pose 的关键点（更准确）
            final_kpts = kpts_seq
        else:
            # 用 Label Studio 标注的关键点（人工标注）
            final_kpts, _ = parse_labelstudio_keypoints(result, total_frames)

        if not frame_labels:
            # 无行为标签，跳过
            continue

        # 提取主导行为（出现次数最多的行为）
        behavior_counts: Dict[str, int] = {}
        for b in frame_labels.values():
            behavior_counts[b] = behavior_counts.get(b, 0) + 1

        dominant_behavior = max(behavior_counts, key=behavior_counts.get)

        # 映射到行为索引
        if dominant_behavior not in BEHAVIOR_TO_IDX:
            stats["skipped_unknown_behavior"] += 1
            continue

        label_idx = BEHAVIOR_TO_IDX[dominant_behavior]

        # 生成边界标签
        boundary = generate_boundary_labels(frame_labels, total_frames)

        # 构造 pyskl 格式样本
        sample = {
            "keypoint": final_kpts[np.newaxis, ...].astype(np.float32),  # (1, T, 24, 3)
            "label": label_idx,
            "label_name": dominant_behavior,
            "frame_dir": f"video_{video_id}",
            "boundary": boundary,
        }
        samples.append(sample)
        stats["converted"] += 1
        stats["behaviors"][dominant_behavior] = (
            stats["behaviors"].get(dominant_behavior, 0) + 1
        )

    # 保存 pyskl pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)

    stats["output_path"] = str(output_path)
    stats["output_samples"] = len(samples)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Studio JSON → ST-GCN+BC pyskl pickle 转换"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Label Studio JSON 导出路径",
    )
    parser.add_argument(
        "--keypoints-dir", type=str, required=True,
        help="YOLO26-pose 推理生成的 keypoints pkl 目录",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="输出 pyskl pickle 路径",
    )
    parser.add_argument(
        "--default-frames", type=int, default=2700,
        help="默认总帧数（无法读取视频时）",
    )
    args = parser.parse_args()

    stats = convert_labelstudio_to_pyskl(
        json_path=args.input,
        keypoints_dir=args.keypoints_dir,
        output_path=args.output,
        default_total_frames=args.default_frames,
    )

    print(f"\n转换完成:")
    print(f"  总任务: {stats['total_tasks']}")
    print(f"  成功转换: {stats['converted']}")
    print(f"  跳过（无标注）: {stats['skipped_no_annotation']}")
    print(f"  跳过（无 keypoints）: {stats['skipped_no_keypoints']}")
    print(f"  跳过（未知行为）: {stats['skipped_unknown_behavior']}")
    print(f"  输出: {stats['output_path']}")
    print(f"  行为分布: {stats['behaviors']}")

    sys.exit(0 if stats["converted"] > 0 else 1)


if __name__ == "__main__":
    main()
