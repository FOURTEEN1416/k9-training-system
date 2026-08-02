"""抽帧策略模块单元测试（Phase 3.5）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.5
依据: backend/ml/pose/frame_stride.py
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.ml.pose.frame_stride import (
    StrideConfig,
    compute_infer_indices,
    estimate_speedup,
    interpolate_keypoints,
    recommend_stride,
)


class TestStrideConfig:
    def test_default_config(self):
        cfg = StrideConfig()
        assert cfg.stride == 1
        assert cfg.interpolation == "linear"

    def test_invalid_stride(self):
        with pytest.raises(ValueError, match="stride"):
            StrideConfig(stride=0)

    def test_invalid_interpolation(self):
        with pytest.raises(ValueError, match="interpolation"):
            StrideConfig(interpolation="cubic")


class TestComputeInferIndices:
    def test_stride_1_returns_all_frames(self):
        indices = compute_infer_indices(total_frames=10, stride=1)
        assert indices == list(range(10))

    def test_stride_3(self):
        indices = compute_infer_indices(total_frames=10, stride=3)
        # 0, 3, 6, 9（最后一帧确保被包含）
        assert indices == [0, 3, 6, 9]

    def test_stride_3_exact_multiple(self):
        indices = compute_infer_indices(total_frames=9, stride=3)
        # 0, 3, 6（9 是最后一个，但 range(0, 9, 3) = [0,3,6]，9-1=8 不是 3 的倍数）
        # 实际 total_frames=9，最后一帧索引 8
        assert indices == [0, 3, 6, 8]

    def test_zero_frames(self):
        indices = compute_infer_indices(total_frames=0, stride=3)
        assert indices == []

    def test_single_frame(self):
        indices = compute_infer_indices(total_frames=1, stride=3)
        assert indices == [0]

    def test_last_frame_always_included(self):
        indices = compute_infer_indices(total_frames=100, stride=7)
        assert indices[-1] == 99  # 最后一帧


class TestInterpolateKeypoints:
    def test_no_interpolation_needed(self):
        """stride=1 时无需插值."""
        kpts = np.random.rand(10, 24, 3).astype(np.float32)
        indices = list(range(10))
        result = interpolate_keypoints(kpts, indices, total_frames=10, method="linear")
        assert result.shape == (10, 24, 3)
        np.testing.assert_array_almost_equal(result, kpts)

    def test_linear_interpolation(self):
        """线性插值精度验证."""
        # 2 帧推理，10 帧目标
        kpts = np.zeros((2, 24, 3), dtype=np.float32)
        kpts[0, 0, 0] = 0.0  # 第 0 帧 x=0
        kpts[1, 0, 0] = 1.0  # 第 9 帧 x=1
        indices = [0, 9]
        result = interpolate_keypoints(kpts, indices, total_frames=10, method="linear")
        assert result.shape == (10, 24, 3)
        # 第 5 帧在 [0,9] 区间内插值 = 5/9 * 1.0 ≈ 0.5556
        assert abs(result[5, 0, 0] - (5.0 / 9.0)) < 0.01

    def test_nearest_interpolation(self):
        """最近邻插值."""
        kpts = np.zeros((2, 24, 3), dtype=np.float32)
        kpts[0, 0, 0] = 0.0
        kpts[1, 0, 0] = 1.0
        indices = [0, 5]
        result = interpolate_keypoints(kpts, indices, total_frames=10, method="nearest")
        # 第 2 帧应该用第 0 帧的值（距离 0 更近）
        assert result[2, 0, 0] == 0.0
        # 第 7 帧应该用第 5 帧的值
        assert result[7, 0, 0] == 1.0

    def test_single_frame_repeat(self):
        """只有一帧推理时，复制填充."""
        kpts = np.ones((1, 24, 3), dtype=np.float32) * 0.5
        indices = [0]
        result = interpolate_keypoints(kpts, indices, total_frames=5, method="linear")
        assert result.shape == (5, 24, 3)
        np.testing.assert_array_almost_equal(result, np.ones((5, 24, 3)) * 0.5)

    def test_confidence_clip(self):
        """置信度插值后 clip 到 [0, 1]."""
        kpts = np.zeros((2, 1, 3), dtype=np.float32)
        kpts[0, 0, 2] = 0.0
        kpts[1, 0, 2] = 2.0  # 超过 1.0
        indices = [0, 1]
        result = interpolate_keypoints(kpts, indices, total_frames=3, method="linear")
        # 插值后第 1 帧置信度应为 1.0（clip）
        assert result[1, 0, 2] <= 1.0
        assert result[1, 0, 2] >= 0.0

    def test_mismatched_lengths(self):
        """infer_kpts 与 infer_indices 长度不匹配应报错."""
        kpts = np.zeros((3, 24, 3), dtype=np.float32)
        indices = [0, 5]  # 长度 2 != 3
        with pytest.raises(ValueError, match="不匹配"):
            interpolate_keypoints(kpts, indices, total_frames=10)

    def test_none_method(self):
        """method=none 时直接返回原始数据."""
        kpts = np.random.rand(2, 24, 3).astype(np.float32)
        indices = [0, 5]
        result = interpolate_keypoints(kpts, indices, total_frames=2, method="none")
        # total_frames=2 == len(indices)，直接返回
        np.testing.assert_array_almost_equal(result, kpts)


class TestEstimateSpeedup:
    def test_stride_1_no_speedup(self):
        assert estimate_speedup(stride=1, total_frames=100) == 1.0

    def test_stride_3_speedup(self):
        # 100 帧 / stride=3 → 34 帧推理 → 加速比 100/34 ≈ 2.94
        speedup = estimate_speedup(stride=3, total_frames=100)
        assert 2.5 < speedup < 3.5

    def test_stride_5_speedup(self):
        speedup = estimate_speedup(stride=5, total_frames=100)
        assert 4.0 < speedup < 6.0


class TestRecommendStride:
    def test_default_recommendation(self):
        """默认目标加速 2x + 5% 误差容忍."""
        stride = recommend_stride(target_speedup=2.0, total_frames=2700)
        assert stride >= 2  # 至少 stride=2 才能达到 2x 加速

    def test_strict_error_tolerance(self):
        """严格误差容忍（1%）→ stride=2."""
        stride = recommend_stride(
            target_speedup=2.0, total_frames=2700, max_interpolation_error=0.01
        )
        # 1% 误差容忍下，stride=2（误差 3%）也不行，应该返回 1
        assert stride == 1

    def test_relaxed_error_tolerance(self):
        """宽松误差容忍（10%）→ stride=5."""
        stride = recommend_stride(
            target_speedup=4.0, total_frames=2700, max_interpolation_error=0.10
        )
        assert stride >= 4

    def test_high_target_speedup(self):
        """高目标加速比（5x）→ stride=5."""
        stride = recommend_stride(
            target_speedup=5.0, total_frames=2700, max_interpolation_error=0.10
        )
        assert stride == 5
