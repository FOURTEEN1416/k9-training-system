"""ReID 犬只身份关联 + ID switch 监测单元测试（Phase 3.2c）.

Owner: ML 开发
Phase: 3.2c

测试策略:
    1. ReIDExtractor 单元测试（mock BoxMOT ReID，不加载真实模型）
    2. DogIdentityGallery 单元测试（注册/匹配/导出）
    3. cosine_similarity 单元测试
    4. IDSwitchMonitor 单元测试（合成多犬场景 + ID switch 注入）
    5. ReID 启用验证（构造 OccluBoost 实例，检查 with_reid=True）

注意:
    - 不加载真实 ReID 模型（重量级，留给端到端测试）
    - 不依赖真实视频（合成帧 + 合成轨迹）
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.ml.tracking.reid_extractor import (
    ReIDExtractor,
    DogIdentityGallery,
    cosine_similarity,
    DEFAULT_SIM_THRESHOLD,
)
from backend.ml.tracking.id_switch_monitor import (
    IDSwitchEvent,
    IDSwitchReport,
    IDSwitchMonitor,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_OVERLAP_FRAMES,
    DEFAULT_SWITCH_RATE_TRIGGER,
)
from backend.ml.tracking.types import (
    DogTrack,
    DogTrackFrame,
    MultiDogTrackingResult,
)


# ===== 辅助函数 =====

def make_synthetic_frame(h: int = 200, w: int = 300) -> np.ndarray:
    """生成合成 BGR 帧."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def make_bbox(center_x: float, center_y: float, size: float = 50.0) -> np.ndarray:
    """生成 bbox [x1, y1, x2, y2]."""
    half = size / 2.0
    return np.array(
        [center_x - half, center_y - half, center_x + half, center_y + half],
        dtype=np.float32,
    )


def make_normalized_embedding(seed: int, dim: int = 512) -> np.ndarray:
    """生成 L2 归一化的合成 embedding."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-12 else emb


def make_reid_mock(output_dim: int = 512) -> MagicMock:
    """构造 mock BoxMOT ReID 运行时."""
    reid = MagicMock()
    # __call__(inputs, boxes=None) → 返回 L2 归一化特征 (N, dim)
    def call(inputs, boxes=None, **kwargs):
        if boxes is not None:
            n = len(boxes) if hasattr(boxes, "__len__") else 1
        else:
            # crops 列表
            n = len(inputs) if isinstance(inputs, list) else 1
        # 生成确定性特征（基于 n）
        rng = np.random.default_rng(seed=n)
        feats = rng.standard_normal((max(n, 0), output_dim)).astype(np.float32)
        norms = np.linalg.norm(feats, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms
    reid.side_effect = call
    return reid


# ===== cosine_similarity 测试 =====

class TestCosineSimilarity:
    """cosine_similarity 函数测试."""

    def test_identical_vectors(self):
        """相同向量相似度 = 1."""
        a = make_normalized_embedding(42, dim=64).reshape(1, -1)
        sim = cosine_similarity(a, a)
        assert sim.shape == (1, 1)
        assert sim[0, 0] == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        """正交向量相似度 = 0."""
        a = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert sim[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors(self):
        """相反向量相似度 = -1."""
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[-1.0, 0.0]], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert sim[0, 0] == pytest.approx(-1.0, abs=1e-5)

    def test_matrix_shape(self):
        """矩阵形状 (M, N)."""
        a = make_normalized_embedding(1, dim=32).reshape(1, -1)
        a = np.tile(a, (3, 1))  # (3, 32)
        b = make_normalized_embedding(2, dim=32).reshape(1, -1)
        b = np.tile(b, (5, 1))  # (5, 32)
        sim = cosine_similarity(a, b)
        assert sim.shape == (3, 5)

    def test_empty_input(self):
        """空输入返回 (0, 0)."""
        a = np.zeros((0, 64), dtype=np.float32)
        b = np.zeros((0, 64), dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert sim.shape == (0, 0)


# ===== ReIDExtractor 测试 =====

class TestReIDExtractor:
    """ReIDExtractor 测试."""

    def test_aggregate_track_mean(self):
        """均值池化."""
        feats = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        agg = ReIDExtractor.aggregate_track(feats, strategy="mean")
        # 均值后 L2 归一化
        expected = np.array([1.0, 1.0, 1.0], dtype=np.float32) / np.sqrt(3)
        np.testing.assert_allclose(agg, expected, atol=1e-5)

    def test_aggregate_track_max(self):
        """最大池化."""
        feats = np.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            dtype=np.float32,
        )
        agg = ReIDExtractor.aggregate_track(feats, strategy="max")
        # max 后 L2 归一化
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(agg, expected, atol=1e-5)

    def test_aggregate_track_median(self):
        """中位数聚合."""
        feats = np.array(
            [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]],
            dtype=np.float32,
        )
        agg = ReIDExtractor.aggregate_track(feats, strategy="median")
        # median = [3.0, 0.0], L2 归一化后 = [1.0, 0.0]
        np.testing.assert_allclose(agg, [1.0, 0.0], atol=1e-5)

    def test_aggregate_track_empty(self):
        """空输入."""
        feats = np.zeros((0, 64), dtype=np.float32)
        agg = ReIDExtractor.aggregate_track(feats)
        assert agg.shape == (64,)
        np.testing.assert_allclose(agg, 0.0)

    def test_aggregate_track_invalid_strategy(self):
        """未知策略抛 ValueError."""
        feats = np.ones((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="未知聚合策略"):
            ReIDExtractor.aggregate_track(feats, strategy="invalid")

    def test_aggregate_track_l2_normalized(self):
        """聚合后 L2 归一化."""
        rng = np.random.default_rng(42)
        feats = rng.standard_normal((10, 128)).astype(np.float32)
        agg = ReIDExtractor.aggregate_track(feats)
        norm = np.linalg.norm(agg)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_extract_from_boxes_with_mock(self):
        """使用 mock ReID 测试 extract_from_boxes."""
        extractor = ReIDExtractor()
        # 注入 mock
        extractor._reid = make_reid_mock(output_dim=512)

        img = make_synthetic_frame()
        boxes = np.array(
            [[10, 10, 100, 100], [150, 50, 250, 150]],
            dtype=np.float32,
        )
        feats = extractor.extract_from_boxes(img, boxes)
        assert feats.shape == (2, 512)
        # 验证 L2 归一化
        norms = np.linalg.norm(feats, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_extract_from_boxes_empty(self):
        """空 bbox 输入."""
        extractor = ReIDExtractor()
        extractor._reid = make_reid_mock()
        img = make_synthetic_frame()
        boxes = np.zeros((0, 4), dtype=np.float32)
        feats = extractor.extract_from_boxes(img, boxes)
        assert feats.shape == (0, 0)

    def test_extract_from_crops_with_mock(self):
        """使用 mock ReID 测试 extract_from_crops."""
        extractor = ReIDExtractor()
        extractor._reid = make_reid_mock(output_dim=256)

        crops = [make_synthetic_frame(100, 100) for _ in range(3)]
        feats = extractor.extract_from_crops(crops)
        assert feats.shape == (3, 256)

    def test_extract_from_crops_empty(self):
        """空 crops 输入."""
        extractor = ReIDExtractor()
        extractor._reid = make_reid_mock()
        feats = extractor.extract_from_crops([])
        assert feats.shape == (0, 0)


# ===== DogIdentityGallery 测试 =====

class TestDogIdentityGallery:
    """DogIdentityGallery 测试."""

    def test_empty_gallery_match(self):
        """空 gallery 匹配返回 (None, 0.0)."""
        gallery = DogIdentityGallery()
        query = make_normalized_embedding(1, dim=64)
        match_id, sim = gallery.match(query)
        assert match_id is None
        assert sim == 0.0

    def test_register_and_match_same(self):
        """注册后相同特征匹配."""
        gallery = DogIdentityGallery(sim_threshold=0.7)
        emb = make_normalized_embedding(42, dim=64)
        gallery.register(track_id=0, embedding=emb)

        match_id, sim = gallery.match(emb)
        assert match_id == 0
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_match_below_threshold(self):
        """相似度低于阈值不匹配."""
        gallery = DogIdentityGallery(sim_threshold=0.99)  # 高阈值
        emb_a = make_normalized_embedding(1, dim=64)
        emb_b = make_normalized_embedding(2, dim=64)  # 不同 seed，相似度 < 1
        gallery.register(track_id=0, embedding=emb_a)

        match_id, sim = gallery.match(emb_b)
        # 相似度可能为负或正但低于 0.99
        assert match_id is None
        assert sim < 0.99

    def test_match_exclude_ids(self):
        """排除 ID 后匹配."""
        gallery = DogIdentityGallery(sim_threshold=0.5)
        emb = make_normalized_embedding(42, dim=64)
        gallery.register(track_id=0, embedding=emb)
        gallery.register(track_id=1, embedding=emb)  # 相同 embedding

        # 排除 0，应匹配到 1
        match_id, sim = gallery.match(emb, exclude_ids=[0])
        assert match_id == 1
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_match_best_match(self):
        """多身份时返回最相似的."""
        gallery = DogIdentityGallery(sim_threshold=0.5)
        query = make_normalized_embedding(100, dim=64)
        # id=0 完全相同，id=1 不同
        gallery.register(track_id=0, embedding=query)
        gallery.register(track_id=1, embedding=make_normalized_embedding(2, dim=64))

        match_id, sim = gallery.match(query)
        assert match_id == 0
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_next_id(self):
        """next_id 单调递增."""
        gallery = DogIdentityGallery()
        assert gallery.next_id() == 0
        assert gallery.next_id() == 1
        assert gallery.next_id() == 2

    def test_register_with_crops(self):
        """注册裁剪（用于微调数据收集）."""
        gallery = DogIdentityGallery()
        emb = make_normalized_embedding(0, dim=32)
        crops = [make_synthetic_frame(50, 50) for _ in range(3)]
        gallery.register(track_id=0, embedding=emb, crops=crops)
        assert gallery.num_identities == 1
        assert 0 in gallery.track_ids

    def test_export_crops(self, tmp_path: Path):
        """导出裁剪到磁盘."""
        gallery = DogIdentityGallery()
        emb = make_normalized_embedding(0, dim=32)
        crops = [make_synthetic_frame(50, 50) for _ in range(3)]
        gallery.register(track_id=0, embedding=emb, crops=crops)

        out_dir = tmp_path / "crops_export"
        exported = gallery.export_crops(out_dir)
        assert len(exported) == 3
        # 验证文件存在
        for f in exported:
            assert f.exists()
            assert f.suffix == ".jpg"
        # 验证目录结构
        track_dir = out_dir / "track_0"
        assert track_dir.exists()
        assert len(list(track_dir.glob("*.jpg"))) == 3

    def test_export_crops_specific_id(self, tmp_path: Path):
        """导出指定 track_id 的裁剪."""
        gallery = DogIdentityGallery()
        crops_0 = [make_synthetic_frame(50, 50) for _ in range(2)]
        crops_1 = [make_synthetic_frame(50, 50) for _ in range(4)]
        gallery.register(0, make_normalized_embedding(0, dim=32), crops=crops_0)
        gallery.register(1, make_normalized_embedding(1, dim=32), crops=crops_1)

        out_dir = tmp_path / "crops_specific"
        exported = gallery.export_crops(out_dir, track_id=1)
        assert len(exported) == 4
        # 验证只导出了 track_1
        assert (out_dir / "track_1").exists()
        assert not (out_dir / "track_0").exists()

    def test_summary(self):
        """gallery 摘要."""
        gallery = DogIdentityGallery(sim_threshold=0.8)
        gallery.register(0, make_normalized_embedding(0, dim=32), crops=[np.zeros((10, 10, 3), dtype=np.uint8)] * 2)
        gallery.register(1, make_normalized_embedding(1, dim=32))
        summary = gallery.summary()
        assert summary["num_identities"] == 2
        assert sorted(summary["track_ids"]) == [0, 1]
        assert summary["total_crops"] == 2
        assert summary["sim_threshold"] == 0.8


# ===== IDSwitchMonitor 测试 =====

class TestIDSwitchMonitor:
    """IDSwitchMonitor 测试."""

    def test_single_dog_no_switch(self):
        """单犬场景无 ID switch."""
        result = MultiDogTrackingResult(total_frames=10)
        track = DogTrack(track_id=0)
        for i in range(10):
            track.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=0,
                    bbox=make_bbox(50, 50, 50),
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                )
            )
        result.tracks[0] = track

        monitor = IDSwitchMonitor()
        report = monitor.evaluate(result)
        assert report.total_events == 0
        assert report.switch_rate == 0.0
        assert not report.should_trigger_reid_finetune

    def test_two_dogs_no_overlap(self):
        """两犬不重叠（不同位置）无 ID switch."""
        result = MultiDogTrackingResult(total_frames=10)
        track0 = DogTrack(track_id=0)
        track1 = DogTrack(track_id=1)
        for i in range(10):
            # 犬 0 在左上，犬 1 在右下，无 IoU
            track0.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=0,
                    bbox=make_bbox(50, 50, 50),
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                )
            )
            track1.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=1,
                    bbox=make_bbox(300, 300, 50),
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                )
            )
        result.tracks[0] = track0
        result.tracks[1] = track1

        monitor = IDSwitchMonitor()
        report = monitor.evaluate(result)
        assert report.total_events == 0

    def test_two_dogs_high_iou_id_switch(self):
        """两犬高 IoU 重叠 — 检测 ID switch."""
        result = MultiDogTrackingResult(total_frames=10)
        track0 = DogTrack(track_id=0)
        track1 = DogTrack(track_id=1)
        # 两犬 bbox 完全重叠（IoU=1.0）持续 10 帧
        for i in range(10):
            bbox = make_bbox(100, 100, 80)
            track0.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=0, bbox=bbox,
                    keypoints=np.zeros((24, 3), dtype=np.float32), conf=0.9,
                )
            )
            track1.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=1, bbox=bbox.copy(),
                    keypoints=np.zeros((24, 3), dtype=np.float32), conf=0.85,
                )
            )
        result.tracks[0] = track0
        result.tracks[1] = track1

        monitor = IDSwitchMonitor(iou_threshold=0.5, min_overlap_frames=3)
        report = monitor.evaluate(result)
        # 应检测到 1 个 ID switch 事件（track pair (0, 1) 只记录一次）
        assert report.total_events == 1
        assert report.switch_rate == pytest.approx(1 / 10, abs=1e-5)
        assert (0, 1) in report.unique_track_pairs
        # 10 帧 1 次事件，rate=0.1，恰好触发阈值
        assert report.should_trigger_reid_finetune

    def test_min_overlap_frames_filter(self):
        """min_overlap_frames 过滤短时重叠."""
        result = MultiDogTrackingResult(total_frames=10)
        track0 = DogTrack(track_id=0)
        track1 = DogTrack(track_id=1)
        # 仅 2 帧重叠（低于 min_overlap_frames=3）
        for i in range(10):
            if i < 2:
                bbox = make_bbox(100, 100, 80)
            else:
                # 分开
                bbox = make_bbox(50 + i * 30 if 0 else 100, 100, 80) if i % 2 == 0 else make_bbox(300, 100, 80)
            if i % 2 == 0:
                track0.frames.append(
                    DogTrackFrame(
                        frame_idx=i, track_id=0,
                        bbox=make_bbox(100, 100, 80) if i < 2 else make_bbox(100, 100, 80),
                        keypoints=np.zeros((24, 3), dtype=np.float32),
                    )
                )
                track1.frames.append(
                    DogTrackFrame(
                        frame_idx=i, track_id=1,
                        bbox=make_bbox(300, 100, 80),
                        keypoints=np.zeros((24, 3), dtype=np.float32),
                    )
                )
        result.tracks[0] = track0
        result.tracks[1] = track1

        monitor = IDSwitchMonitor(iou_threshold=0.5, min_overlap_frames=3)
        report = monitor.evaluate(result)
        assert report.total_events == 0

    def test_iou_threshold_filter(self):
        """IoU 阈值过滤低重叠."""
        result = MultiDogTrackingResult(total_frames=10)
        track0 = DogTrack(track_id=0)
        track1 = DogTrack(track_id=1)
        for i in range(10):
            # bbox 略有重叠，IoU ≈ 0.2
            track0.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=0,
                    bbox=make_bbox(50, 50, 80),
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                )
            )
            track1.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=1,
                    bbox=make_bbox(100, 50, 80),  # 与 track0 部分重叠
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                )
            )
        result.tracks[0] = track0
        result.tracks[1] = track1

        monitor = IDSwitchMonitor(iou_threshold=0.5, min_overlap_frames=3)
        report = monitor.evaluate(result)
        # IoU 低于阈值，不应检测到事件
        assert report.total_events == 0

    def test_report_summary(self):
        """报告摘要结构."""
        result = MultiDogTrackingResult(total_frames=100)
        track0 = DogTrack(track_id=0)
        track1 = DogTrack(track_id=1)
        for i in range(100):
            bbox = make_bbox(100, 100, 80)
            track0.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=0, bbox=bbox,
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                )
            )
            track1.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=1, bbox=bbox.copy(),
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                )
            )
        result.tracks[0] = track0
        result.tracks[1] = track1

        monitor = IDSwitchMonitor(iou_threshold=0.5, min_overlap_frames=3)
        report = monitor.evaluate(result)
        summary = report.summary()
        assert summary["total_frames"] == 100
        assert summary["total_events"] == 1
        assert summary["switch_rate"] == pytest.approx(0.01, abs=1e-5)
        assert summary["trigger_threshold"] == DEFAULT_SWITCH_RATE_TRIGGER
        assert "should_trigger_reid_finetune" in summary

    def test_with_mock_reid_extractor(self):
        """使用 mock ReID 提取器测试 ReID 确认."""
        result = MultiDogTrackingResult(total_frames=10)
        track0 = DogTrack(track_id=0)
        track1 = DogTrack(track_id=1)
        for i in range(10):
            bbox = make_bbox(100, 100, 80)
            track0.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=0, bbox=bbox,
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                )
            )
            track1.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=1, bbox=bbox.copy(),
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                )
            )
        result.tracks[0] = track0
        result.tracks[1] = track1

        # mock ReID 提取器
        mock_extractor = MagicMock(spec=ReIDExtractor)
        # extract_from_crops 返回合成特征
        mock_extractor.extract_from_crops.side_effect = lambda crops: (
            make_normalized_embedding(len(crops), dim=512).reshape(1, -1).repeat(len(crops), axis=0)
            if crops else np.zeros((0, 512), dtype=np.float32)
        )
        # aggregate_track 返回归一化特征
        mock_extractor.aggregate_track.side_effect = lambda feats, **kw: (
            ReIDExtractor.aggregate_track(feats, strategy=kw.get("strategy", "mean"))
        )

        # 合成视频帧
        video_frames = [make_synthetic_frame(200, 300) for _ in range(10)]

        monitor = IDSwitchMonitor(iou_threshold=0.5, min_overlap_frames=3)
        report = monitor.evaluate(result, reid_extractor=mock_extractor, video_frames=video_frames)

        assert report.total_events == 1
        # ReID 相似度应被填充（因为两犬相同 embedding seed，similarity=1.0）
        event = report.events[0]
        assert event.reid_similarity is not None
        assert event.is_confirmed  # sim ≥ 0.7
        assert report.confirmed_events == 1


# ===== ReID 启用验证（编程式实例化）=====

class TestReIDEnabled:
    """验证 MultiDogTracker 显式启用 with_reid=True."""

    def test_with_reid_default_true(self, monkeypatch):
        """MultiDogTracker 默认 with_reid=True."""
        from backend.ml.tracking.multi_dog_tracker import MultiDogTracker, TrackerBackend

        # mock BoxMOT ReID 构造（避免真实下载 OSNet 权重）
        class FakeReIDModel:
            """模拟 ReID backend model 对象."""
            pass

        class FakeReID:
            def __init__(self, *args, **kwargs):
                self.model = FakeReIDModel()

        # 仅 mock boxmot.reid.core.reid.ReID，保留 boxmot.trackers.OccluBoost 真实加载
        import boxmot.reid.core.reid as reid_module
        monkeypatch.setattr(reid_module, "ReID", FakeReID)

        tracker = MultiDogTracker(
            pose_model_path="dummy.pt",
            tracker_backend=TrackerBackend.OCCLUBOOST,
        )
        assert tracker.tracker_kwargs == {}
        tracker._init_tracker()
        assert tracker._tracker is not None
        assert tracker._tracker.with_reid is True
        # reid_model 应是 FakeReIDModel 实例（非字符串）
        assert isinstance(tracker._tracker.reid_model, FakeReIDModel)

    def test_with_reid_user_override(self, monkeypatch):
        """用户可通过 tracker_kwargs 显式禁用 with_reid."""
        from backend.ml.tracking.multi_dog_tracker import MultiDogTracker, TrackerBackend

        # mock 防止 ReID 加载
        class FakeReID:
            def __init__(self, *args, **kwargs):
                self.model = None

        import boxmot.reid.core.reid as reid_module
        monkeypatch.setattr(reid_module, "ReID", FakeReID)

        tracker = MultiDogTracker(
            pose_model_path="dummy.pt",
            tracker_backend=TrackerBackend.OCCLUBOOST,
            tracker_kwargs={"with_reid": False},
        )
        tracker._init_tracker()
        assert tracker._tracker.with_reid is False
        # with_reid=False 时不构造 ReID 对象
        assert tracker._tracker.reid_model is None

    def test_with_reid_load_failure_fallback(self, monkeypatch):
        """ReID 加载失败时降级为 with_reid=False."""
        from backend.ml.tracking.multi_dog_tracker import MultiDogTracker, TrackerBackend

        # mock ReID 构造抛异常
        class FailingReID:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("network error")

        import boxmot.reid.core.reid as reid_module
        monkeypatch.setattr(reid_module, "ReID", FailingReID)

        tracker = MultiDogTracker(
            pose_model_path="dummy.pt",
            tracker_backend=TrackerBackend.OCCLUBOOST,
        )
        tracker._init_tracker()
        # 应降级为 with_reid=False（不抛异常）
        assert tracker._tracker.with_reid is False
        assert tracker._tracker.reid_model is None

    def test_backend_not_implemented(self):
        """非 OccluBoost 后端抛 NotImplementedError."""
        from backend.ml.tracking.multi_dog_tracker import MultiDogTracker, TrackerBackend
        tracker = MultiDogTracker(
            pose_model_path="dummy.pt",
            tracker_backend=TrackerBackend.BOTSORT,
        )
        with pytest.raises(NotImplementedError, match="暂未实现"):
            tracker._init_tracker()


# ===== CanineReIDDataset 测试 =====

class TestCanineReIDDataset:
    """CanineReIDDataset 犬只 ReID 微调数据接口测试."""

    def test_empty_dataset(self):
        """空数据集完整性检查."""
        from backend.ml.tracking.reid_finetune_dataset import CanineReIDDataset
        ds = CanineReIDDataset()
        report = ds.integrity_report()
        assert report["total_samples"] == 0
        assert report["num_identities"] == 0
        assert not report["is_complete"]
        assert len(report["issues"]) > 0  # 应有"身份数不足"问题

    def test_collect_from_tracking_result(self):
        """从追踪结果收集裁剪."""
        from backend.ml.tracking.reid_finetune_dataset import (
            CanineReIDDataset, MIN_CROPS_PER_IDENTITY,
        )
        # 构造合成追踪结果（2 犬各 30 帧）
        result = MultiDogTrackingResult(total_frames=30)
        for tid in [0, 1]:
            track = DogTrack(track_id=tid)
            for i in range(30):
                track.frames.append(
                    DogTrackFrame(
                        frame_idx=i, track_id=tid,
                        bbox=make_bbox(50 + tid * 100, 50, 60),
                        keypoints=np.zeros((24, 3), dtype=np.float32),
                        conf=0.85,
                    )
                )
            result.tracks[tid] = track

        # 合成视频帧
        video_frames = [make_synthetic_frame(200, 300) for _ in range(30)]

        ds = CanineReIDDataset()
        n = ds.collect_from_tracking_result(
            result=result,
            video_frames=video_frames,
            video_path="test.mp4",
            frame_step=2,  # 每 2 帧取 1 张
        )
        # 2 犬 × 15 帧 (30/2) = 30 样本
        assert n == 30
        assert len(ds.samples) == 30
        assert ds.identity_to_track_ids == {0: [0], 1: [1]}

    def test_identity_mapping(self):
        """identity_mapping 跨视频身份合并."""
        from backend.ml.tracking.reid_finetune_dataset import CanineReIDDataset
        # 视频 A: track_id=0
        # 视频 B: track_id=1（实际是同一物理犬只，映射到 identity_id=100）
        result_a = MultiDogTrackingResult(total_frames=10)
        track_a = DogTrack(track_id=0)
        for i in range(10):
            track_a.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=0,
                    bbox=make_bbox(50, 50, 60),
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                    conf=0.85,
                )
            )
        result_a.tracks[0] = track_a

        result_b = MultiDogTrackingResult(total_frames=10)
        track_b = DogTrack(track_id=1)
        for i in range(10):
            track_b.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=1,
                    bbox=make_bbox(150, 50, 60),
                    keypoints=np.zeros((24, 3), dtype=np.float32),
                    conf=0.85,
                )
            )
        result_b.tracks[1] = track_b

        video_frames = [make_synthetic_frame(200, 300) for _ in range(10)]

        ds = CanineReIDDataset()
        ds.collect_from_tracking_result(
            result_a, video_frames, "video_a.mp4",
            identity_mapping={0: 100},  # track 0 → identity 100
            frame_step=1,
        )
        ds.collect_from_tracking_result(
            result_b, video_frames, "video_b.mp4",
            identity_mapping={1: 100},  # track 1 → identity 100 (同一物理犬只)
            frame_step=1,
        )

        # 应合并为 1 个身份（identity_id=100），但来自 2 个 track_id
        assert 100 in ds.identity_to_track_ids
        assert sorted(ds.identity_to_track_ids[100]) == [0, 1]
        # 所有样本都属于 identity 100
        assert all(s.identity_id == 100 for s in ds.samples)

    def test_integrity_report_insufficient(self):
        """完整性检查 - 样本不足."""
        from backend.ml.tracking.reid_finetune_dataset import (
            CanineReIDDataset, CanineReIDSample,
        )
        ds = CanineReIDDataset()
        # 仅 5 个样本，1 个身份
        for i in range(5):
            ds.samples.append(
                CanineReIDSample(
                    crop=make_synthetic_frame(50, 50),
                    identity_id=0,
                    source_video="test.mp4",
                    frame_idx=i,
                    track_id=0,
                    bbox=make_bbox(50, 50, 40),
                )
            )
        report = ds.integrity_report()
        assert not report["is_complete"]
        assert any("身份数不足" in issue for issue in report["issues"])
        assert any("样本不足" in issue for issue in report["issues"])

    def test_integrity_report_sufficient(self):
        """完整性检查 - 数据充足."""
        from backend.ml.tracking.reid_finetune_dataset import (
            CanineReIDDataset, CanineReIDSample,
            MIN_CROPS_PER_IDENTITY, MIN_IDENTITIES,
        )
        ds = CanineReIDDataset()
        # 2 个身份，每个 25 个样本
        for iid in range(MIN_IDENTITIES):
            for i in range(MIN_CROPS_PER_IDENTITY + 5):
                ds.samples.append(
                    CanineReIDSample(
                        crop=make_synthetic_frame(50, 50),
                        identity_id=iid,
                        source_video="test.mp4",
                        frame_idx=i,
                        track_id=iid,
                        bbox=make_bbox(50, 50, 40),
                    )
                )
        report = ds.integrity_report()
        assert report["is_complete"]
        assert len(report["issues"]) == 0

    def test_export_boxmot_format(self, tmp_path: Path):
        """导出 BoxMOT 训练格式."""
        from backend.ml.tracking.reid_finetune_dataset import (
            CanineReIDDataset, CanineReIDSample,
        )
        ds = CanineReIDDataset()
        # 2 个身份各 10 个样本
        for iid in range(2):
            for i in range(10):
                ds.samples.append(
                    CanineReIDSample(
                        crop=make_synthetic_frame(50, 50),
                        identity_id=iid,
                        source_video="test.mp4",
                        frame_idx=i,
                        track_id=iid,
                        bbox=make_bbox(50, 50, 40),
                    )
                )

        out_dir = tmp_path / "canine_reid"
        exported = ds.export_boxmot_format(out_dir, train_ratio=0.7)

        # 验证目录结构
        assert (out_dir / "train").exists()
        assert (out_dir / "query").exists()
        assert (out_dir / "gallery").exists()
        # 验证 train 集每身份有裁剪
        for iid in range(2):
            train_dir = out_dir / "train" / str(iid)
            assert train_dir.exists()
            assert len(list(train_dir.glob("*.jpg"))) > 0
        # 验证文件总数
        total = len(exported["train"]) + len(exported["query"]) + len(exported["gallery"])
        assert total == 20

    def test_generate_boxmot_train_command(self, tmp_path: Path):
        """生成 BoxMOT 训练命令."""
        from backend.ml.tracking.reid_finetune_dataset import CanineReIDDataset
        ds = CanineReIDDataset()
        cmd = ds.generate_boxmot_train_command(
            dataset_dir=tmp_path,
            model_arch="osnet_x1_0",
            dataset_name="canine_reid",
            epochs=60,
            device=0,
        )
        assert "boxmot train" in cmd
        assert "--model osnet_x1_0" in cmd
        assert "--dataset canine_reid" in cmd
        assert "--epochs 60" in cmd

    def test_sample_save(self, tmp_path: Path):
        """CanineReIDSample 保存裁剪."""
        from backend.ml.tracking.reid_finetune_dataset import CanineReIDSample
        sample = CanineReIDSample(
            crop=make_synthetic_frame(40, 40),
            identity_id=0,
            source_video="test.mp4",
            frame_idx=5,
            track_id=0,
            bbox=make_bbox(50, 50, 40),
        )
        out_path = tmp_path / "test_crop.jpg"
        saved = sample.save(out_path)
        assert saved.exists()
        assert saved.suffix == ".jpg"
