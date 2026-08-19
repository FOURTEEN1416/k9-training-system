"""合成多犬视频 + 自动 GT 生成（Phase 3.2d 轨道 A）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.2d
依据: dev-docs/stages/phase-3.md §3.2d

策略:
    1. 选取两个不同的单犬视频（不同犬只/姿态/场景）
    2. 用 YOLO26-pose 跑两个源视频，得到单犬场景的 bbox 序列
       （单犬场景下 YOLO 检测 recall 高，可作为伪 GT 用于多犬 ID 关联评估）
    3. 水平拼接两个视频帧 → 合成多犬视频（左半 + 右半）
    4. 右犬 bbox 横坐标 +左视频宽度 偏移
    5. 生成 MOT Challenge 格式 GT: frame_idx, track_id, x, y, w, h, conf, class, visibility
       （帧索引从 1 开始；左犬 track_id=1，右犬 track_id=2）

用法:
    .venv/Scripts/python.exe scripts/create_synthetic_multi_dog_video.py \
        --left data/uploads/_phase1_7_puppy.mp4 \
        --right data/uploads/_phase1_7_obedience.mp4 \
        --output-video data/synthetic/multi_dog_synthetic.mp4 \
        --output-gt data/synthetic/gt.txt

设计权衡:
    - 合成场景简单（左右分界，犬只不越界），无法验证复杂遮挡下 ID switch
    - 但能直接验证 ReID 在不同犬只间的辨别能力 + tracker 多犬 ID 关联核心逻辑
    - 真实多犬场景作为轨道 B 补充（GitHub-First 调研公开数据集）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def detect_dog_bbox_per_frame(
    video_path: Path,
    model_path: str,
    device: int = 0,
    conf: float = 0.25,
    imgsz: int = 640,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
) -> Tuple[List[Optional[np.ndarray]], int, int, int, float]:
    """跑 YOLO26-pose 单犬检测，返回每帧主犬 bbox.

    使用 ultralytics 推荐的 stream=True 模式（与 backend/ml/pose/inference.py 生产代码一致）
    避免 source=ndarray 单帧模式的检测差异。

    Args:
        start_frame: 起始帧索引（跳过前 start_frame 帧）
        max_frames: 最大处理帧数（None 表示处理到视频末尾）

    Returns:
        bboxes: List[Optional[np.ndarray]]，每帧 (4,) [x1,y1,x2,y2] 或 None
        frame_count, width, height, fps
    """
    from ultralytics import YOLO

    print(f"[detect] 加载模型: {model_path}", flush=True)
    model = YOLO(model_path, task="pose")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    print(f"[detect] 视频: {video_path.name} ({frame_count} frames, {width}x{height} @ {fps:.1f}fps)", flush=True)
    print(f"[detect] start_frame={start_frame}, max_frames={max_frames}", flush=True)

    bboxes: List[Optional[np.ndarray]] = []
    frame_idx = 0
    skipped = 0
    processed = 0
    # stream=True 模式：ultralytics 内部用 cv2.VideoCapture 流式处理
    results_stream = model.predict(
        source=str(video_path),
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        device=device,
        verbose=False,
        save=False,
        stream=True,
    )

    for results in results_stream:
        if skipped < start_frame:
            skipped += 1
            continue

        if max_frames is not None and processed >= max_frames:
            break

        if results.boxes is not None and len(results.boxes.xyxy) > 0:
            # 单犬场景，取置信度最高的检测作为该帧主犬
            xyxy = results.boxes.xyxy.cpu().numpy()  # (N, 4)
            confs = results.boxes.conf.cpu().numpy()
            best = int(np.argmax(confs))
            bboxes.append(xyxy[best].astype(np.float32))
        else:
            bboxes.append(None)

        frame_idx += 1
        processed += 1
        if processed % 50 == 0:
            print(f"  [detect] frame {processed} (orig {start_frame + processed}/{frame_count})", flush=True)

    print(f"[detect] 完成: {sum(1 for b in bboxes if b is not None)}/{len(bboxes)} 帧检测到犬", flush=True)
    return bboxes, len(bboxes), width, height, fps


def synthesize_multi_dog_video(
    left_video: Path,
    right_video: Path,
    model_path: str,
    output_video: Path,
    output_gt: Path,
    device: int = 0,
    conf: float = 0.25,
    separator_width: int = 4,
    separator_color: Tuple[int, int, int] = (0, 0, 255),  # BGR 红色
    left_start: int = 0,
    right_start: int = 0,
    max_frames: Optional[int] = None,
) -> dict:
    """合成多犬视频并生成 GT.

    Args:
        left_video: 左侧源视频路径
        right_video: 右侧源视频路径
        model_path: YOLO26-pose 模型路径
        output_video: 合成视频输出路径
        output_gt: GT 文件输出路径（MOT 格式）
        device: 推理设备
        conf: YOLO 检测置信度
        separator_width: 中间分隔条宽度（像素）
        separator_color: 分隔条颜色 BGR
        left_start: 左视频起始帧（跳过前 left_start 帧）
        right_start: 右视频起始帧
        max_frames: 最大合成帧数（None 表示取 min(左, 右)）

    Returns:
        统计字典
    """
    print("=" * 70, flush=True)
    print("合成多犬视频 + GT 生成", flush=True)
    print("=" * 70, flush=True)

    # 1. 分别检测两个源视频
    print(f"\n[1/4] 检测左视频犬只 bbox (start={left_start})...", flush=True)
    left_bboxes, left_n, lw, lh, lfps = detect_dog_bbox_per_frame(
        left_video, model_path, device=device, conf=conf,
        start_frame=left_start, max_frames=max_frames,
    )

    print(f"\n[2/4] 检测右视频犬只 bbox (start={right_start})...", flush=True)
    right_bboxes, right_n, rw, rh, rfps = detect_dog_bbox_per_frame(
        right_video, model_path, device=device, conf=conf,
        start_frame=right_start, max_frames=max_frames,
    )

    # 2. 尺寸对齐：高度取 min，宽度按比例缩放后拼接
    target_h = min(lh, rh)
    # 重采样到目标高度（按高度等比缩放宽度）
    left_scale = target_h / lh
    right_scale = target_h / rh
    target_lw = int(round(lw * left_scale))
    target_rw = int(round(rw * right_scale))

    common_n = min(left_n, right_n)
    fps = min(lfps, rfps)
    total_w = target_lw + separator_width + target_rw

    print(f"\n[3/4] 合成视频: {total_w}x{target_h} @ {fps:.1f}fps, {common_n} frames", flush=True)
    print(f"  左: {target_lw}x{target_h} (scale={left_scale:.3f})", flush=True)
    print(f"  右: {target_rw}x{target_h} (scale={right_scale:.3f})", flush=True)
    print(f"  分隔条: {separator_width}px", flush=True)

    # 3. 写合成视频 + GT
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_gt.parent.mkdir(parents=True, exist_ok=True)

    # 编码选择：优先 H264（ultralytics stream 兼容），回退 mp4v
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H264 in mp4
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (total_w, target_h))
    if not writer.isOpened():
        # 回退 mp4v
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (total_w, target_h))
        print(f"  [warn] avc1 不可用，回退 mp4v 编码（可能影响 ultralytics stream 模式）", flush=True)
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {output_video}")

    cap_l = cv2.VideoCapture(str(left_video))
    cap_r = cv2.VideoCapture(str(right_video))

    # 跳过起始帧（与 detect 阶段一致）
    if left_start > 0:
        cap_l.set(cv2.CAP_PROP_POS_FRAMES, left_start)
    if right_start > 0:
        cap_r.set(cv2.CAP_PROP_POS_FRAMES, right_start)

    gt_lines: List[str] = []
    left_x_offset = 0
    right_x_offset = target_lw + separator_width

    written = 0
    for i in range(common_n):
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        # 尺寸重采样
        frame_l = cv2.resize(frame_l, (target_lw, target_h))
        frame_r = cv2.resize(frame_r, (target_rw, target_h))

        # 拼接：左 + 分隔条 + 右
        sep = np.full((target_h, separator_width, 3), separator_color, dtype=np.uint8)
        combined = np.concatenate([frame_l, sep, frame_r], axis=1)
        writer.write(combined)

        # GT：MOT 格式 frame_idx(1-based), track_id, x, y, w, h, conf, class, visibility
        frame_idx_1 = i + 1

        # 左犬 GT
        if left_bboxes[i] is not None:
            bx = left_bboxes[i].copy()
            # 应用缩放
            bx[0] *= left_scale
            bx[2] *= left_scale
            bx[1] *= left_scale
            bx[3] *= left_scale
            # 左犬无额外 x 偏移
            x, y = float(bx[0]), float(bx[1])
            w, h = float(bx[2] - bx[0]), float(bx[3] - bx[1])
            # track_id=1, conf=1.0 (GT), class=1 (dog), visibility=1.0
            gt_lines.append(f"{frame_idx_1},1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,1,1.0")

        # 右犬 GT（x 偏移）
        if right_bboxes[i] is not None:
            bx = right_bboxes[i].copy()
            bx[0] = bx[0] * right_scale + right_x_offset
            bx[2] = bx[2] * right_scale + right_x_offset
            bx[1] *= right_scale
            bx[3] *= right_scale
            x, y = float(bx[0]), float(bx[1])
            w, h = float(bx[2] - bx[0]), float(bx[3] - bx[1])
            gt_lines.append(f"{frame_idx_1},2,{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,1,1.0")

        written += 1
        if written % 50 == 0:
            print(f"  [write] frame {written}/{common_n}", flush=True)

    cap_l.release()
    cap_r.release()
    writer.release()

    # 写 GT 文件
    with open(output_gt, "w", encoding="utf-8") as f:
        f.write("\n".join(gt_lines) + "\n")

    left_detected = sum(1 for i in range(min(common_n, len(left_bboxes))) if left_bboxes[i] is not None)
    right_detected = sum(1 for i in range(min(common_n, len(right_bboxes))) if right_bboxes[i] is not None)

    stats = {
        "left_video": str(left_video),
        "right_video": str(right_video),
        "output_video": str(output_video),
        "output_gt": str(output_gt),
        "total_frames": written,
        "fps": float(fps),
        "width": total_w,
        "height": target_h,
        "left_track_id": 1,
        "right_track_id": 2,
        "left_detected_frames": left_detected,
        "right_detected_frames": right_detected,
        "gt_lines": len(gt_lines),
        "left_scale": float(left_scale),
        "right_scale": float(right_scale),
        "right_x_offset": int(right_x_offset),
    }

    print(f"\n[4/4] 完成:", flush=True)
    print(f"  合成视频: {output_video}", flush=True)
    print(f"  GT 文件: {output_gt}", flush=True)
    print(f"  总帧数: {written}", flush=True)
    print(f"  左犬 GT 行: {left_detected}（track_id=1）", flush=True)
    print(f"  右犬 GT 行: {right_detected}（track_id=2）", flush=True)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="合成多犬视频 + GT 生成（Phase 3.2d 轨道 A）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--left", required=True, help="左侧源视频路径")
    parser.add_argument("--right", required=True, help="右侧源视频路径")
    parser.add_argument("--model", default="runs/train-2/weights/best.pt", help="YOLO26-pose 模型路径")
    parser.add_argument("--output-video", default="data/synthetic/multi_dog_synthetic.mp4", help="合成视频输出路径")
    parser.add_argument("--output-gt", default="data/synthetic/gt.txt", help="GT 文件输出路径（MOT 格式）")
    parser.add_argument("--device", type=int, default=0, help="推理设备")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 检测置信度阈值")
    parser.add_argument("--left-start", type=int, default=0, help="左视频起始帧（跳过前 N 帧，截取检测率高的片段）")
    parser.add_argument("--right-start", type=int, default=0, help="右视频起始帧")
    parser.add_argument("--max-frames", type=int, default=None, help="最大合成帧数（None=取 min(左,右)）")
    args = parser.parse_args()

    synthesize_multi_dog_video(
        left_video=Path(args.left),
        right_video=Path(args.right),
        model_path=args.model,
        output_video=Path(args.output_video),
        output_gt=Path(args.output_gt),
        device=args.device,
        conf=args.conf,
        left_start=args.left_start,
        right_start=args.right_start,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
