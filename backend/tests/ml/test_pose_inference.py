"""backend.ml.pose.inference 推理 API 单元测试.

Owner: ML 开发
Phase: 1.0
依据: dev-docs/stages/phase-1.md §1.0d/e/h

验收条件:
    - results[0].keypoints.xy.shape == (1, 24, 2)  [单犬场景]
    - results[0].keypoints.data.shape == (1, 24, 3)
    - 视频推理输出 shape == (T, 24, 3)
    - pkl 保存/加载一致
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from backend.ml.pose.inference import (
    KPT_NAMES,
    NUM_KEYPOINTS,
    PoseInferenceEngine,
)


@pytest.mark.slow
class TestPoseInferenceEngine:
    """推理引擎测试（需要 GPU + 模型下载）。"""

    def test_engine_init(self, yolo_pose_model: str) -> None:
        """引擎初始化（首次自动下载 yolo26n-pose.pt）。"""
        engine = PoseInferenceEngine(yolo_pose_model, device=0, verbose=False)
        assert engine.model is not None
        assert engine.model_path == yolo_pose_model

    def test_infer_image_shape(self, yolo_pose_model: str, tmp_path: Path) -> None:
        """单图推理 shape = (24, 3)。"""
        # 生成一张合成图像（不含狗，返回全 0 关键点，但 shape 必须对）
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        img_path = tmp_path / "test.jpg"
        import cv2
        cv2.imwrite(str(img_path), img)

        engine = PoseInferenceEngine(yolo_pose_model, device=0, verbose=False)
        kpts = engine.infer_image(img_path)
        assert kpts.shape == (NUM_KEYPOINTS, 3), f"期望 (24, 3)，实际 {kpts.shape}"
        assert kpts.dtype == np.float32

    def test_infer_video_shape(
        self, yolo_pose_model: str, synthetic_video: Path
    ) -> None:
        """视频推理 shape = (T, 24, 3)，T=帧数。"""
        engine = PoseInferenceEngine(yolo_pose_model, device=0, verbose=False)
        result = engine.infer_video(
            synthetic_video,
            save_output=False,
        )
        # shape 校验
        assert result.shape[1] == NUM_KEYPOINTS, f"期望 24 关键点，实际 {result.shape[1]}"
        assert result.shape[2] == 3, f"期望 3 维 (x,y,conf)，实际 {result.shape[2]}"
        # 帧数校验（合成视频 30 帧，可能因 VideoWriter 末尾丢帧略少）
        assert 25 <= result.shape[0] <= 35, f"期望约 30 帧，实际 {result.shape[0]}"

        # meta 校验
        assert result.meta["num_keypoints"] == NUM_KEYPOINTS
        assert len(result.meta["kpt_names"]) == NUM_KEYPOINTS
        assert result.meta["kpt_names"] == KPT_NAMES

    def test_pkl_save_load(
        self, yolo_pose_model: str, synthetic_video: Path, tmp_path: Path
    ) -> None:
        """pkl 保存与加载一致。"""
        engine = PoseInferenceEngine(yolo_pose_model, device=0, verbose=False)
        pkl_path = tmp_path / "result.pkl"
        result = engine.infer_video(
            synthetic_video,
            save_output=True,
            output_path=pkl_path,
        )

        assert pkl_path.exists()
        assert pkl_path.stat().st_size > 0

        # 加载验证
        loaded = PoseInferenceEngine.load_pkl(pkl_path)
        assert "meta" in loaded
        assert "frames" in loaded
        assert "keypoints_sequence" in loaded
        assert len(loaded["frames"]) == len(result.frames)
        assert loaded["keypoints_sequence"].shape == result.shape

    def test_keypoints_data_type(
        self, yolo_pose_model: str, synthetic_video: Path
    ) -> None:
        """关键点数据类型为 float32。"""
        engine = PoseInferenceEngine(yolo_pose_model, device=0, verbose=False)
        result = engine.infer_video(synthetic_video, save_output=False)
        for frame in result.frames:
            assert frame.keypoints.dtype == np.float32
            assert frame.keypoints.shape == (NUM_KEYPOINTS, 3)


@pytest.mark.fast
class TestConstants:
    """常量校验（无需 GPU，快速）。"""

    def test_num_keypoints(self) -> None:
        assert NUM_KEYPOINTS == 24

    def test_kpt_names_count(self) -> None:
        assert len(KPT_NAMES) == 24

    def test_kpt_names_unique(self) -> None:
        assert len(set(KPT_NAMES)) == 24, "24 关键点名称应唯一"

    def test_kpt_names_match_yaml(self) -> None:
        """关键点名称与 data/dog-pose.yaml 一致。"""
        import yaml
        yaml_path = Path(__file__).resolve().parents[3] / "data" / "dog-pose.yaml"
        if not yaml_path.exists():
            pytest.skip(f"yaml 不存在: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        yaml_names = cfg["kpt_names"][0]
        assert yaml_names == KPT_NAMES, "关键点名称与 yaml 不一致"
