"""YOLO26-pose pre-annotation for Label Studio import (Path B).

Runs YOLO26-pose on downloaded YouTube videos, generates:
  1. Keypoint pkl files (for project's rule engine + 1.2f validation)
  2. Label Studio pre-annotation JSON (for human correction)

Usage:
    python scripts/preannotate_youtube.py
    python scripts/preannotate_youtube.py --video data/youtube_self_label/raw/Obue6tS-AT8.mp4
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.pose.inference import PoseInferenceEngine, KPT_NAMES, NUM_KEYPOINTS

ROOT = Path("data/youtube_self_label")
RAW = ROOT / "raw"
PKL_DIR = ROOT / "pkl"
LS_DIR = ROOT / "label_studio"

# Default model: use fine-tuned best.pt if available, else yolo26n-pose.pt
DEFAULT_MODEL = "runs/train-2/weights/best.pt"
FALLBACK_MODEL = "yolo26n-pose.pt"

# Label Studio label names for keypoints (Dog-Pose 24 keypoints grouped)
KPT_LABEL_GROUPS = {
    "head": ["nose", "chin", "left_eye", "right_eye",
             "left_ear_base", "right_ear_base", "left_ear_tip", "right_ear_tip",
             "withers", "throat"],
    "front_legs": ["front_left_paw", "front_left_knee", "front_left_elbow",
                   "front_right_paw", "front_right_knee", "front_right_elbow"],
    "rear_legs": ["rear_left_paw", "rear_left_knee", "rear_left_elbow",
                  "rear_right_paw", "rear_right_knee", "rear_right_elbow"],
    "tail": ["tail_start", "tail_end"],
}

# Project behaviors for annotation
PROJECT_BEHAVIORS = [
    "sit", "down", "stand", "come", "heel", "sit_up", "stay", "bark",
    "fetch", "forward", "backward", "turn", "jump", "search", "bite", "release",
]


def select_model() -> str:
    """Select best available model."""
    for m in [DEFAULT_MODEL, FALLBACK_MODEL]:
        if Path(m).exists():
            return m
    return FALLBACK_MODEL  # will auto-download


def video_to_ls_tasks(video_path: Path, engine: PoseInferenceEngine) -> dict:
    """Run inference on one video, return Label Studio task dict + pkl result."""
    video_path = Path(video_path)
    video_id = video_path.stem
    print(f"  [infer] {video_id}", flush=True)

    # Run YOLO26-pose inference
    result = engine.infer_video(str(video_path))
    print(f"    frames={len(result.frames)}, shape={result.shape}", flush=True)

    # Save pkl for project's rule engine / 1.2f validation
    pkl_path = PKL_DIR / f"{video_id}.pkl"
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    pkl_data = {
        "meta": result.meta,
        "frames": [
            {
                "frame_idx": f.frame_idx,
                "frame_time_sec": f.frame_time_sec,
                "keypoints": f.keypoints,
                "box": f.box,
                "box_conf": f.box_conf,
            }
            for f in result.frames
        ],
        "keypoints_sequence": result.keypoints_sequence,
    }
    with open(pkl_path, "wb") as fh:
        pickle.dump(pkl_data, fh)
    print(f"    pkl: {pkl_path}", flush=True)

    # Build Label Studio pre-annotations
    # For video annotation, Label Studio uses Video annotation with timeline regions
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    # Sample frames for pre-annotation (every 30 frames = ~1 second)
    # Label Studio video annotation works with frame numbers
    regions = []
    sampled_frames = result.frames[::30]  # sample every ~1s
    for f in sampled_frames:
        if f.box is None or f.box_conf < 0.3:
            continue
        x1, y1, x2, y2 = f.box.tolist()
        # Convert to Label Studio format (percentages of video dimensions)
        regions.append({
            "original_width": width,
            "original_height": height,
            "frame": f.frame_idx,
            "frame_count": total_frames,
            "fps": fps,
            "box": {
                "x": x1 / width * 100,
                "y": y1 / height * 100,
                "width": (x2 - x1) / width * 100,
                "height": (y2 - y1) / height * 100,
            },
            "box_conf": float(f.box_conf),
            "keypoints": [
                {
                    "name": KPT_NAMES[i],
                    "x": float(f.keypoints[i, 0]),
                    "y": float(f.keypoints[i, 1]),
                    "conf": float(f.keypoints[i, 2]),
                    "visible": bool(f.keypoints[i, 2] > 0.3),
                }
                for i in range(NUM_KEYPOINTS)
            ],
        })

    ls_task = {
        "data": {
            "video": f"/data/upload/{video_id}.mp4",
            "video_id": video_id,
            "title": video_id,
        },
        "predictions": [
            {
                "model_version": "yolo26-pose-v1",
                "result": [
                    {
                        "from_name": "box",
                        "to_name": "video",
                        "type": "videorectangle",
                        "value": {
                            "frame": r["frame"],
                            "frameCount": r["frame_count"],
                            "fps": r["fps"],
                            "x": r["box"]["x"],
                            "y": r["box"]["y"],
                            "width": r["box"]["width"],
                            "height": r["box"]["height"],
                        },
                        "score": r["box_conf"],
                    }
                    for r in regions
                ],
            }
        ],
        "meta": {
            "video_id": video_id,
            "duration_sec": round(duration, 2),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "sampled_regions": len(regions),
            "pkl_path": str(pkl_path),
        },
    }
    return ls_task


def build_label_config() -> str:
    """Build Label Studio labeling config XML for dog behavior annotation."""
    # Video annotation with bounding box + behavior classification
    behaviors_choices = "\n".join(f'        <Choice value="{b}" />' for b in PROJECT_BEHAVIORS)
    return f"""<View>
  <Header value="犬类行为标注 — YOLO26-pose 预标注已生成，请修正关键点 + 标注行为" />
  <Video name="video" value="$video" framerate="$fps" />
  <VideoRectangle name="box" toName="video" />
  <Choices name="behavior" toName="video" choice="single" required="true">
    <Header value="当前帧行为分类" />
{behaviors_choices}
  </Choices>
  <TextArea name="comment" toName="video" rows="2" editable="true"
            placeholder="标注备注（可选）：如关键点偏差、多犬、遮挡等" />
</View>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, help="single video file (else process all raw/*.mp4)")
    parser.add_argument("--model", default=None, help="model path override")
    args = parser.parse_args()

    model = args.model or select_model()
    print(f"=== YOLO26-pose pre-annotation ===", flush=True)
    print(f"model: {model}", flush=True)

    engine = PoseInferenceEngine(model, verbose=False)

    if args.video:
        videos = [args.video]
    else:
        videos = sorted(RAW.glob("*.mp4"))

    print(f"videos: {len(videos)}", flush=True)
    if not videos:
        print("No videos found in data/youtube_self_label/raw/", flush=True)
        return

    PKL_DIR.mkdir(parents=True, exist_ok=True)
    LS_DIR.mkdir(parents=True, exist_ok=True)

    all_tasks = []
    for v in videos:
        try:
            task = video_to_ls_tasks(v, engine)
            all_tasks.append(task)
            # Save individual task JSON
            task_path = LS_DIR / f"{v.stem}.json"
            task_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    ls: {task_path}", flush=True)
        except Exception as e:
            print(f"  [ERROR] {v.name}: {e}", flush=True)

    # Save combined tasks for bulk import
    combined_path = LS_DIR / "tasks_all.json"
    combined_path.write_text(json.dumps(all_tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== Combined tasks: {combined_path} ({len(all_tasks)} tasks) ===", flush=True)

    # Save labeling config
    config_path = LS_DIR / "label_config.xml"
    config_path.write_text(build_label_config(), encoding="utf-8")
    print(f"=== Label config: {config_path} ===", flush=True)

    # Summary
    print(f"\n=== Summary ===", flush=True)
    print(f"  videos processed: {len(all_tasks)}/{len(videos)}", flush=True)
    print(f"  pkl files: {PKL_DIR}", flush=True)
    print(f"  LS tasks: {LS_DIR}", flush=True)
    print(f"  Next: import tasks_all.json + label_config.xml into Label Studio", flush=True)


if __name__ == "__main__":
    main()
