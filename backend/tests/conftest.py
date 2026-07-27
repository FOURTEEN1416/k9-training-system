"""pytest 配置：合成视频 fixture 等共享资源.

Owner: ML 开发
Phase: 1.0
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# 项目根目录加入 sys.path（便于 backend.* 导入）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    """生成 30 帧合成视频（640x640 纯色 + 噪声），用于推理 shape 验证。

    合成视频不包含真实狗，推理时可能返回全 0 关键点（未检测到），
    但 shape 必须为 (T, 24, 3)。
    """
    tmp_dir = tmp_path_factory.mktemp("videos")
    video_path = tmp_dir / "synthetic.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    w, h = 640, 640
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        pytest.skip("无法创建 VideoWriter（编码器不可用）")

    rng = np.random.default_rng(42)
    for _ in range(30):
        # 灰色背景 + 随机噪声
        frame = np.full((h, w, 3), 128, dtype=np.uint8)
        noise = rng.integers(0, 30, size=(h, w, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        writer.write(frame)
    writer.release()

    assert video_path.exists() and video_path.stat().st_size > 0
    return video_path


@pytest.fixture(scope="session")
def yolo_pose_model() -> str:
    """YOLO26-pose 模型名（首次使用自动下载约 5MB）。"""
    return "yolo26n-pose.pt"


def pytest_collection_modifyitems(config, items):
    """自动标记 slow 测试（需要 GPU 或模型下载）。"""
    for item in items:
        if "slow" in item.keywords:
            continue
        # 默认 GPU 相关测试标记为 slow
        if item.module.__name__.startswith("backend.tests.ml"):
            item.add_marker(pytest.mark.slow)
