"""多犬追踪模块单元测试（Phase 3.2b）.

Owner: ML 开发
Phase: 3.2b

测试策略:
    1. 数据类型测试（DogTrackFrame / DogTrack / MultiDogTrackingResult）
    2. YOLO 检测解析测试（mock results 对象）
    3. 关键点关联测试（_tracks_to_frames 核心逻辑）
    4. 集成测试（合成多犬场景模拟）

注意:
    - 不加载真实 YOLO 模型（重量级，留给端到端测试）
    - 使用 mock 对象模拟 BoxMOT TrackResults
    - 测试核心逻辑：det_ind → 关键点关联的正确性
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.ml.behavior.constants import NUM_KEYPOINTS
from backend.ml.tracking.types import (
    DogTrack,
    DogTrackFrame,
    MultiDogTrackingResult,
)
from backend.ml.tracking.multi_dog_tracker import (
    TrackerBackend,
    _parse_yolo_detections,
    _tracks_to_frames,
)
from backend.ml.tracking import (
    MultiDogTracker,
    DogTrackFrame as ImportedDogTrackFrame,
)


# ===== 辅助函数 =====

def make_mock_yolo_results(
    num_dets: int = 1,
    img_w: int = 640,
    img_h: int = 480,
) -> MagicMock:
    """构造 mock YOLO predict 结果对象.

    Args:
        num_dets: 检测框数量
        img_w, img_h: 图像尺寸

    Returns:
        MagicMock，模拟 ultralytics YOLO results[0]
    """
    results = MagicMock()

    if num_dets == 0:
        results.boxes = None
        results.keypoints = None
        return results

    # 检测框: (N, 4) xyxy
    xyxy = np.array(
        [[50 + i * 100, 50, 150 + i * 100, 200] for i in range(num_dets)],
        dtype=np.float32,
    )
    conf = np.array([0.9 - i * 0.1 for i in range(num_dets)], dtype=np.float32)
    cls = np.array([0] * num_dets, dtype=np.float32)  # dog 类

    results.boxes = MagicMock()
    # .cpu().numpy() 调用链：cpu() 返回 mock，再 .numpy() 返回 ndarray
    results.boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
    results.boxes.conf.cpu.return_value.numpy.return_value = conf
    results.boxes.cls.cpu.return_value.numpy.return_value = cls
    # len(results.boxes.xyxy) 检查
    results.boxes.__len__ = MagicMock(return_value=num_dets)
    results.boxes.xyxy.__len__ = MagicMock(return_value=num_dets)

    # 关键点: (N, 24, 3)
    kps = np.random.rand(num_dets, NUM_KEYPOINTS, 3).astype(np.float32)
    kps[..., 2] = 0.85  # 统一置信度
    results.keypoints = MagicMock()
    results.keypoints.data.cpu.return_value.numpy.return_value = kps
    results.keypoints.data.__len__ = MagicMock(return_value=num_dets)

    return results


def make_mock_tracks(
    track_specs: list,
) -> np.ndarray:
    """构造 mock BoxMOT tracks (普通 2D ndarray，匹配真实 BoxMOT 输出格式).

    真实 BoxMOT tracker.update() 返回 (M, 8) ndarray:
        [x1, y1, x2, y2, id, conf, cls, det_ind]

    Args:
        track_specs: [(track_id, det_ind, x1, y1, x2, y2, conf), ...]

    Returns:
        np.ndarray shape=(M, 8)
    """
    tracks = np.zeros((len(track_specs), 8), dtype=np.float32)
    for i, (tid, det_ind, x1, y1, x2, y2, conf) in enumerate(track_specs):
        tracks[i] = [x1, y1, x2, y2, tid, conf, 0, det_ind]
    return tracks


# ===== 数据类型测试 =====

class TestDogTrackFrame:
    """DogTrackFrame 数据类型测试."""

    def test_basic_construction(self):
        """基本构造."""
        bbox = np.array([10, 20, 100, 200], dtype=np.float32)
        kps = np.random.rand(NUM_KEYPOINTS, 3).astype(np.float32)
        frame = DogTrackFrame(
            frame_idx=5,
            track_id=1,
            bbox=bbox,
            keypoints=kps,
            conf=0.9,
            det_ind=0,
        )
        assert frame.frame_idx == 5
        assert frame.track_id == 1
        assert frame.conf == 0.9
        assert frame.det_ind == 0
        assert frame.bbox.shape == (4,)
        assert frame.keypoints.shape == (NUM_KEYPOINTS, 3)

    def test_list_input_auto_convert(self):
        """list 输入自动转 ndarray."""
        frame = DogTrackFrame(
            frame_idx=0,
            track_id=0,
            bbox=[10, 20, 100, 200],
            keypoints=[[0, 0, 0.5]] * NUM_KEYPOINTS,
            conf=0.5,
        )
        assert isinstance(frame.bbox, np.ndarray)
        assert isinstance(frame.keypoints, np.ndarray)
        assert frame.bbox.dtype == np.float32

    def test_keypoints_pad(self):
        """关键点数不足 24 时自动 pad."""
        short_kps = np.zeros((10, 3), dtype=np.float32)
        frame = DogTrackFrame(
            frame_idx=0, track_id=0,
            bbox=[0, 0, 10, 10], keypoints=short_kps,
        )
        assert frame.keypoints.shape == (NUM_KEYPOINTS, 3)


class TestDogTrack:
    """DogTrack 单犬轨迹测试."""

    def test_empty_track(self):
        track = DogTrack(track_id=1)
        assert track.num_frames == 0
        assert track.frame_indices == []

    def test_add_frames(self):
        track = DogTrack(track_id=1)
        for i in range(5):
            track.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=1,
                    bbox=[0, 0, 10, 10], keypoints=np.zeros((24, 3)),
                )
            )
        assert track.num_frames == 5
        assert track.frame_indices == [0, 1, 2, 3, 4]

    def test_get_keypoints_sequence_no_padding(self):
        """关键点序列（不填充）."""
        track = DogTrack(track_id=1)
        for i in range(3):
            kps = np.full((24, 3), float(i), dtype=np.float32)
            track.frames.append(
                DogTrackFrame(frame_idx=i, track_id=1, bbox=[0, 0, 10, 10], keypoints=kps)
            )
        seq = track.get_keypoints_sequence()
        assert seq.shape == (3, 24, 3)
        # 第 0 帧全 0，第 1 帧全 1，第 2 帧全 2
        np.testing.assert_allclose(seq[0], 0.0)
        np.testing.assert_allclose(seq[1], 1.0)
        np.testing.assert_allclose(seq[2], 2.0)

    def test_get_keypoints_sequence_padded(self):
        """关键点序列（填充到 total_frames）."""
        track = DogTrack(track_id=1)
        # 仅在第 1 帧和第 3 帧出现
        for i in [1, 3]:
            kps = np.full((24, 3), float(i), dtype=np.float32)
            track.frames.append(
                DogTrackFrame(frame_idx=i, track_id=1, bbox=[0, 0, 10, 10], keypoints=kps)
            )
        seq = track.get_keypoints_sequence(total_frames=5)
        assert seq.shape == (5, 24, 3)
        # 第 0/2/4 帧应为 0（未出现）
        np.testing.assert_allclose(seq[0], 0.0)
        np.testing.assert_allclose(seq[2], 0.0)
        np.testing.assert_allclose(seq[4], 0.0)
        # 第 1/3 帧应有值
        np.testing.assert_allclose(seq[1], 1.0)
        np.testing.assert_allclose(seq[3], 3.0)

    def test_get_bbox_sequence_padded(self):
        """bbox 序列（填充）."""
        track = DogTrack(track_id=1)
        track.frames.append(
            DogTrackFrame(frame_idx=2, track_id=1, bbox=[10, 20, 30, 40], keypoints=np.zeros((24, 3)))
        )
        seq = track.get_bbox_sequence(total_frames=5)
        assert seq.shape == (5, 4)
        np.testing.assert_allclose(seq[2], [10, 20, 30, 40])
        np.testing.assert_allclose(seq[0], 0.0)


class TestMultiDogTrackingResult:
    """MultiDogTrackingResult 多犬结果测试."""

    def test_empty_result(self):
        result = MultiDogTrackingResult()
        assert result.num_dogs == 0
        assert result.track_ids == []

    def test_multi_dog_tracks(self):
        """两只犬的轨迹."""
        result = MultiDogTrackingResult(total_frames=10)
        # 犬 0: 帧 0-9
        track0 = DogTrack(track_id=0)
        for i in range(10):
            track0.frames.append(
                DogTrackFrame(frame_idx=i, track_id=0, bbox=[0, 0, 10, 10],
                              keypoints=np.zeros((24, 3)))
            )
        # 犬 1: 帧 5-9
        track1 = DogTrack(track_id=1)
        for i in range(5, 10):
            track1.frames.append(
                DogTrackFrame(frame_idx=i, track_id=1, bbox=[100, 0, 110, 10],
                              keypoints=np.ones((24, 3)))
            )
        result.tracks[0] = track0
        result.tracks[1] = track1

        assert result.num_dogs == 2
        assert result.track_ids == [0, 1]
        assert result.get_track(0).num_frames == 10
        assert result.get_track(1).num_frames == 5
        assert result.get_track(99) is None  # 不存在

    def test_get_all_keypoints_sequences(self):
        """获取所有犬只关键点序列（不填充）."""
        result = MultiDogTrackingResult()
        track0 = DogTrack(track_id=0)
        track0.frames.append(
            DogTrackFrame(frame_idx=0, track_id=0, bbox=[0, 0, 10, 10],
                          keypoints=np.zeros((24, 3)))
        )
        result.tracks[0] = track0

        seqs = result.get_all_keypoints_sequences()
        assert 0 in seqs
        assert seqs[0].shape == (1, 24, 3)

    def test_get_padded_keypoints_sequences(self):
        """获取填充后的所有犬只序列."""
        result = MultiDogTrackingResult(total_frames=5)
        track0 = DogTrack(track_id=0)
        track0.frames.append(
            DogTrackFrame(frame_idx=2, track_id=0, bbox=[0, 0, 10, 10],
                          keypoints=np.full((24, 3), 0.5))
        )
        result.tracks[0] = track0

        seqs = result.get_padded_keypoints_sequences()
        assert seqs[0].shape == (5, 24, 3)
        # 第 2 帧有值
        np.testing.assert_allclose(seqs[0][2], 0.5)
        # 其他帧为 0
        np.testing.assert_allclose(seqs[0][0], 0.0)

    def test_summary(self):
        """摘要不报错."""
        result = MultiDogTrackingResult(total_frames=10, fps=30.0)
        result.tracks[0] = DogTrack(track_id=0)
        result.tracks[0].frames.append(
            DogTrackFrame(frame_idx=0, track_id=0, bbox=[0, 0, 10, 10],
                          keypoints=np.zeros((24, 3)))
        )
        s = result.summary()
        assert "MultiDogTrackingResult" in s
        assert "track_ids" in s.lower() or "track" in s.lower()


# ===== YOLO 检测解析测试 =====

class TestParseYoloDetections:
    """_parse_yolo_detections 函数测试."""

    def test_empty_detection(self):
        """空检测返回空数组."""
        results = make_mock_yolo_results(num_dets=0)
        dets, kps = _parse_yolo_detections(results)
        assert dets.shape == (0, 6)
        assert kps.shape == (0, NUM_KEYPOINTS, 3)

    def test_single_detection(self):
        """单犬检测."""
        results = make_mock_yolo_results(num_dets=1)
        dets, kps = _parse_yolo_detections(results)
        assert dets.shape == (1, 6)  # [x1,y1,x2,y2,conf,cls]
        assert kps.shape == (1, NUM_KEYPOINTS, 3)
        # conf 在 [0, 1]
        assert 0 <= dets[0, 4] <= 1.0

    def test_multi_detection(self):
        """多犬检测."""
        results = make_mock_yolo_results(num_dets=3)
        dets, kps = _parse_yolo_detections(results)
        assert dets.shape == (3, 6)
        assert kps.shape == (3, NUM_KEYPOINTS, 3)
        # 三个检测框 x 坐标应不同（mock 生成时 i*100 偏移）
        assert dets[0, 0] != dets[1, 0] != dets[2, 0]


# ===== 关键点关联测试 =====

class TestTracksToFrames:
    """_tracks_to_frames 关键点关联测试（核心逻辑）."""

    def test_empty_tracks(self):
        """空 tracks 返回空列表."""
        kps = np.random.rand(3, NUM_KEYPOINTS, 3).astype(np.float32)
        frames = _tracks_to_frames(np.array([]), kps, frame_idx=0)
        assert frames == []

    def test_single_track_keypoint_association(self):
        """单犬 track_id=0 关联到检测 0 的关键点."""
        # 3 个检测，track 0 关联到 det_ind=0
        kps = np.zeros((3, NUM_KEYPOINTS, 3), dtype=np.float32)
        kps[0] = 0.1  # det 0 的关键点设为 0.1
        kps[1] = 0.2  # det 1
        kps[2] = 0.3  # det 2

        tracks = make_mock_tracks([(0, 0, 50, 50, 150, 200, 0.9)])
        frames = _tracks_to_frames(tracks, kps, frame_idx=5)

        assert len(frames) == 1
        assert frames[0].track_id == 0
        assert frames[0].frame_idx == 5
        assert frames[0].det_ind == 0
        # 关键点应为 det 0 的值（0.1）
        np.testing.assert_allclose(frames[0].keypoints, 0.1)

    def test_multi_track_keypoint_association(self):
        """多犬 track 分别关联到不同检测的关键点."""
        kps = np.zeros((2, NUM_KEYPOINTS, 3), dtype=np.float32)
        kps[0] = 0.5  # det 0
        kps[1] = 0.8  # det 1

        tracks = make_mock_tracks([
            (0, 0, 50, 50, 150, 200, 0.9),   # track 0 → det 0
            (1, 1, 250, 50, 350, 200, 0.85), # track 1 → det 1
        ])
        frames = _tracks_to_frames(tracks, kps, frame_idx=10)

        assert len(frames) == 2
        # track 0 关联 det 0 的关键点
        f0 = [f for f in frames if f.track_id == 0][0]
        np.testing.assert_allclose(f0.keypoints, 0.5)
        # track 1 关联 det 1 的关键点
        f1 = [f for f in frames if f.track_id == 1][0]
        np.testing.assert_allclose(f1.keypoints, 0.8)

    def test_det_ind_out_of_range(self):
        """det_ind 越界时关键点填 0."""
        kps = np.zeros((1, NUM_KEYPOINTS, 3), dtype=np.float32)
        kps[0] = 0.5

        # det_ind=99 越界
        tracks = make_mock_tracks([(0, 99, 50, 50, 150, 200, 0.9)])
        frames = _tracks_to_frames(tracks, kps, frame_idx=0)

        assert len(frames) == 1
        # 关键点应为 0（越界兜底）
        np.testing.assert_allclose(frames[0].keypoints, 0.0)

    def test_bbox_passed_through(self):
        """bbox 正确传递."""
        kps = np.zeros((1, NUM_KEYPOINTS, 3), dtype=np.float32)
        tracks = make_mock_tracks([(0, 0, 10, 20, 30, 40, 0.95)])
        frames = _tracks_to_frames(tracks, kps, frame_idx=0)

        np.testing.assert_allclose(frames[0].bbox, [10, 20, 30, 40])
        assert frames[0].conf == pytest.approx(0.95)


# ===== MultiDogTracker 类测试（不加载模型）=====

class TestMultiDogTrackerClass:
    """MultiDogTracker 类基础测试（不加载真实模型）."""

    def test_construction(self):
        """构造不加载模型（延迟初始化）."""
        tracker = MultiDogTracker(
            pose_model_path="fake.pt",
            tracker_backend=TrackerBackend.OCCLUBOOST,
            reid_model="lmbn_n_duke",
            device=0,
        )
        assert tracker.pose_model_path == "fake.pt"
        assert tracker.tracker_backend == TrackerBackend.OCCLUBOOST
        assert tracker._pose_model is None  # 延迟初始化
        assert tracker._tracker is None

    def test_unsupported_backend(self):
        """非 OccluBoost 后端触发 NotImplementedError."""
        tracker = MultiDogTracker(
            pose_model_path="fake.pt",
            tracker_backend=TrackerBackend.BOTSORT,
        )
        with pytest.raises(NotImplementedError, match="暂未实现"):
            tracker._init_tracker()

    def test_tracker_backend_enum(self):
        """TrackerBackend 枚举值."""
        assert TrackerBackend.OCCLUBOOST.value == "occluboost"
        assert TrackerBackend.BOTSORT.value == "botsort"
        assert TrackerBackend.BYTETRACK.value == "bytetrack"


# ===== 集成测试：模拟多犬追踪场景 =====

class TestIntegrationSimulated:
    """模拟多犬追踪集成测试（不依赖真实模型/视频）."""

    def test_multi_dog_scenario_simulation(self):
        """模拟两犬交叉场景的关键点关联.

        场景: 5 帧视频，犬 A 在左侧，犬 B 在右侧
            - 帧 0-2: 犬 A (track 0) 在 (50,50)，犬 B (track 1) 在 (250,50)
            - 帧 3-4: 犬 A 移到 (200,50)，犬 B 移到 (100,50)（交叉）
        验证: det_ind 关联正确，每犬关键点独立
        """
        result = MultiDogTrackingResult(total_frames=5, fps=30.0)

        # 模拟 5 帧的追踪结果
        frame_data = [
            # (frame_idx, [(track_id, det_ind, bbox, kps_value), ...])
            (0, [(0, 0, [50, 50, 150, 200], 0.1), (1, 1, [250, 50, 350, 200], 0.2)]),
            (1, [(0, 0, [60, 50, 160, 200], 0.1), (1, 1, [240, 50, 340, 200], 0.2)]),
            (2, [(0, 0, [100, 50, 200, 200], 0.1), (1, 1, [200, 50, 300, 200], 0.2)]),
            (3, [(0, 1, [200, 50, 300, 200], 0.1), (1, 0, [100, 50, 200, 200], 0.2)]),  # det_ind 交换
            (4, [(0, 1, [210, 50, 310, 200], 0.1), (1, 0, [90, 50, 190, 200], 0.2)]),
        ]

        for frame_idx, tracks_in_frame in frame_data:
            # 构造 kps: det_ind 决定取哪个关键点
            kps = np.zeros((2, NUM_KEYPOINTS, 3), dtype=np.float32)
            kps[0] = 0.5  # det 0
            kps[1] = 0.8  # det 1

            track_specs = [
                (tid, det_ind, *bbox, 0.9)
                for tid, det_ind, bbox, _ in tracks_in_frame
            ]
            tracks = make_mock_tracks(track_specs)
            frames = _tracks_to_frames(tracks, kps, frame_idx=frame_idx)

            for df in frames:
                if df.track_id not in result.tracks:
                    result.tracks[df.track_id] = DogTrack(track_id=df.track_id)
                result.tracks[df.track_id].frames.append(df)

        # 验证: 两犬各 5 帧
        assert result.num_dogs == 2
        assert result.track_ids == [0, 1]
        assert result.tracks[0].num_frames == 5
        assert result.tracks[1].num_frames == 5

        # 验证: 关键点序列可正确提取
        seqs = result.get_padded_keypoints_sequences()
        assert seqs[0].shape == (5, 24, 3)
        assert seqs[1].shape == (5, 24, 3)

        # 验证: 犬 0 的帧索引覆盖 0-4
        assert result.tracks[0].frame_indices == [0, 1, 2, 3, 4]
        assert result.tracks[1].frame_indices == [0, 1, 2, 3, 4]

    def test_single_dog_degradation(self):
        """单犬场景退化为 1 个 track_id（向后兼容）."""
        result = MultiDogTrackingResult(total_frames=3)
        track = DogTrack(track_id=0)
        for i in range(3):
            track.frames.append(
                DogTrackFrame(
                    frame_idx=i, track_id=0,
                    bbox=[i * 10, 0, i * 10 + 100, 200],
                    keypoints=np.full((24, 3), float(i)),
                )
            )
        result.tracks[0] = track

        # 单犬场景验证
        assert result.num_dogs == 1
        seq = result.get_padded_keypoints_sequences()[0]
        assert seq.shape == (3, 24, 3)
        # 每帧关键点值应递增
        np.testing.assert_allclose(seq[0], 0.0)
        np.testing.assert_allclose(seq[1], 1.0)
        np.testing.assert_allclose(seq[2], 2.0)
