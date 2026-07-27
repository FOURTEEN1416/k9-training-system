"""Phase 1.0-e 推理 API shape 验证（CPU 模式，避免与训练抢 GPU）。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from backend.ml.pose.inference import PoseInferenceEngine

# 1. 生成 30 帧合成视频（放到项目根 data/ 目录）
video_path = PROJECT_ROOT / "data" / "_test_synthetic.mp4"
video_path.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(video_path), fourcc, 30, (640, 640))
for _ in range(30):
    writer.write(np.full((640, 640, 3), 128, dtype=np.uint8))
writer.release()
assert video_path.exists() and video_path.stat().st_size > 0, f"合成视频生成失败: {video_path}"
print(f"[synthetic] 30 帧合成视频已生成: {video_path} ({video_path.stat().st_size} bytes)")

# 2. CPU 推理（避免与训练抢 GPU）
engine = PoseInferenceEngine("yolo26n-pose.pt", device="cpu", verbose=False)
result = engine.infer_video(video_path, save_output=False)

# 3. shape 校验
print(f"[inference] shape={result.shape}")
assert result.shape[1] == 24, f"期望 24 关键点，实际 {result.shape[1]}"
assert result.shape[2] == 3, f"期望 3 维 (x,y,conf)，实际 {result.shape[2]}"
assert 25 <= result.shape[0] <= 35, f"期望约 30 帧，实际 {result.shape[0]}"

# 4. meta 校验
meta = result.meta
print(f"[inference] meta fps={meta['fps']} frames={meta['frame_count']} "
      f"{meta['width']}x{meta['height']} duration={meta['duration_sec']:.2f}s")
print(f"[inference] kpt_names[:5]={meta['kpt_names'][:5]}")
assert meta["num_keypoints"] == 24
assert len(meta["kpt_names"]) == 24

# 5. 清理合成视频
video_path.unlink()
print(f"[synthetic] 已清理合成视频")

print("[inference] ✅ Phase 1.0-e 推理 API shape 验证通过: (T, 24, 3)")
