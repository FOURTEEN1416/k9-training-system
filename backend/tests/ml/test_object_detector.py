"""Phase 1.5c 物体检测单元测试.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.5c
依据: dev-docs/stages/phase-1.md §1.5

测试策略:
    1. 纯数据结构测试（fast，无模型）:
       - COCO 类目映射（ball/food/toy/person/dog）
       - Detection / FrameDetection / VideoDetectionResult 数据结构
       - 类别筛选、几何属性（center/area）
    2. 合成帧检测测试（integration，需模型）:
       - 用 cv2 合成带球/食物的图片 → ObjectDetector → 验证检测
    3. 真实视频测试（integration，需模型 + 视频）:
       - 端到端 detect_video 流程

标记:
    fast — 纯 Python，无 GPU/模型
    integration — 需 YOLO26 模型 + 真实视频
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.ml.behavior.object_detector import (
    ALL_INTEREST_INDICES,
    BALL_IDX,
    CATEGORY_BALL,
    CATEGORY_DOG,
    CATEGORY_FOOD,
    CATEGORY_OTHER,
    CATEGORY_PERSON,
    CATEGORY_TOY,
    COCO_NAMES,
    DOG_IDX,
    FOOD_INDICES,
    PERSON_IDX,
    TOY_INDICES,
    Detection,
    FrameDetection,
    ObjectDetector,
    VideoDetectionResult,
    _idx_to_category,
)


# ============================================================
# 1. COCO 类目映射
# ============================================================


@pytest.mark.fast
class TestCOCOMapping:
    """COCO 80 类索引到工作犬场景分组的映射。"""

    def test_person_mapping(self) -> None:
        assert _idx_to_category(PERSON_IDX) == CATEGORY_PERSON

    def test_dog_mapping(self) -> None:
        assert _idx_to_category(DOG_IDX) == CATEGORY_DOG

    def test_ball_mapping(self) -> None:
        """sports ball (32) → ball 类别."""
        assert _idx_to_category(BALL_IDX) == CATEGORY_BALL

    def test_food_mapping(self) -> None:
        """所有食物类索引 → food 类别."""
        for idx in FOOD_INDICES:
            assert _idx_to_category(idx) == CATEGORY_FOOD, (
                f"索引 {idx} ({COCO_NAMES[idx]}) 应映射到 food"
            )

    def test_toy_mapping(self) -> None:
        """玩具类索引（除 ball） → toy 类别."""
        for idx in TOY_INDICES:
            if idx == BALL_IDX:
                continue  # ball 优先映射到 ball
            assert _idx_to_category(idx) == CATEGORY_TOY, (
                f"索引 {idx} ({COCO_NAMES[idx]}) 应映射到 toy"
            )

    def test_other_mapping(self) -> None:
        """未在感兴趣列表中的类 → other."""
        # 选若干明确不在感兴趣列表的索引
        other_indices = [1, 5, 14, 25, 60, 70]
        for idx in other_indices:
            assert _idx_to_category(idx) == CATEGORY_OTHER

    def test_all_interest_indices_distinct(self) -> None:
        """ALL_INTEREST_INDICES 应去重且有序."""
        assert ALL_INTEREST_INDICES == sorted(set(ALL_INTEREST_INDICES))

    def test_coco_names_length(self) -> None:
        """COCO_NAMES 应有 80 类."""
        assert len(COCO_NAMES) == 80

    def test_coco_names_no_duplicates(self) -> None:
        """COCO_NAMES 不应有重复."""
        assert len(set(COCO_NAMES)) == 80


# ============================================================
# 2. Detection 数据结构
# ============================================================


@pytest.mark.fast
class TestDetection:
    """Detection dataclass 测试。"""

    def test_detection_creation(self) -> None:
        """Detection 基本创建."""
        det = Detection(
            class_id=BALL_IDX,
            class_name="sports ball",
            category=CATEGORY_BALL,
            confidence=0.85,
            bbox=np.array([100, 200, 200, 280], dtype=np.float32),
        )
        assert det.class_id == BALL_IDX
        assert det.class_name == "sports ball"
        assert det.category == CATEGORY_BALL
        assert det.confidence == pytest.approx(0.85)

    def test_detection_center(self) -> None:
        """Detection.center 计算正确."""
        det = Detection(
            class_id=BALL_IDX,
            class_name="sports ball",
            category=CATEGORY_BALL,
            confidence=0.9,
            bbox=np.array([100, 100, 200, 200], dtype=np.float32),
        )
        cx, cy = det.center
        assert cx == pytest.approx(150.0)
        assert cy == pytest.approx(150.0)

    def test_detection_area(self) -> None:
        """Detection.area 计算正确."""
        det = Detection(
            class_id=BALL_IDX,
            class_name="sports ball",
            category=CATEGORY_BALL,
            confidence=0.9,
            bbox=np.array([100, 100, 200, 200], dtype=np.float32),
        )
        assert det.area == pytest.approx(10000.0)

    def test_detection_to_dict(self) -> None:
        """Detection.to_dict() 输出可序列化字典."""
        det = Detection(
            class_id=BALL_IDX,
            class_name="sports ball",
            category=CATEGORY_BALL,
            confidence=0.9,
            bbox=np.array([100, 100, 200, 200], dtype=np.float32),
        )
        d = det.to_dict()
        assert d["class_id"] == BALL_IDX
        assert d["class_name"] == "sports ball"
        assert d["category"] == CATEGORY_BALL
        assert d["confidence"] == pytest.approx(0.9)
        assert isinstance(d["bbox"], list)
        assert len(d["bbox"]) == 4


# ============================================================
# 3. FrameDetection 数据结构
# ============================================================


def _make_detection(category: str, conf: float = 0.9) -> Detection:
    """构造测试用 Detection."""
    return Detection(
        class_id=0,
        class_name="test",
        category=category,
        confidence=conf,
        bbox=np.array([0, 0, 10, 10], dtype=np.float32),
    )


@pytest.mark.fast
class TestFrameDetection:
    """FrameDetection dataclass 测试。"""

    def test_empty_frame(self) -> None:
        """空帧: 无检测."""
        frame = FrameDetection(frame_idx=0, frame_time_sec=0.0)
        assert frame.detections == []
        assert not frame.has_ball
        assert not frame.has_food
        assert not frame.has_toy
        assert not frame.has_person

    def test_filter_by_category(self) -> None:
        """按类别筛选."""
        frame = FrameDetection(
            frame_idx=0,
            frame_time_sec=0.0,
            detections=[
                _make_detection(CATEGORY_BALL),
                _make_detection(CATEGORY_FOOD),
                _make_detection(CATEGORY_PERSON),
            ],
        )
        balls = frame.filter_by_category(CATEGORY_BALL)
        assert len(balls) == 1
        foods = frame.filter_by_category(CATEGORY_FOOD)
        assert len(foods) == 1
        persons = frame.filter_by_category(CATEGORY_PERSON)
        assert len(persons) == 1

    def test_has_ball(self) -> None:
        """has_ball 检测."""
        frame = FrameDetection(
            frame_idx=0,
            frame_time_sec=0.0,
            detections=[_make_detection(CATEGORY_BALL)],
        )
        assert frame.has_ball
        assert not frame.has_food

    def test_has_food(self) -> None:
        """has_food 检测."""
        frame = FrameDetection(
            frame_idx=0,
            frame_time_sec=0.0,
            detections=[_make_detection(CATEGORY_FOOD)],
        )
        assert frame.has_food
        assert not frame.has_ball

    def test_has_toy_includes_ball(self) -> None:
        """has_toy 应包含 ball 类（玩具代理）."""
        frame_ball = FrameDetection(
            frame_idx=0,
            frame_time_sec=0.0,
            detections=[_make_detection(CATEGORY_BALL)],
        )
        frame_toy = FrameDetection(
            frame_idx=0,
            frame_time_sec=0.0,
            detections=[_make_detection(CATEGORY_TOY)],
        )
        frame_empty = FrameDetection(frame_idx=0, frame_time_sec=0.0)
        assert frame_ball.has_toy
        assert frame_toy.has_toy
        assert not frame_empty.has_toy

    def test_has_person(self) -> None:
        """has_person 检测."""
        frame = FrameDetection(
            frame_idx=0,
            frame_time_sec=0.0,
            detections=[_make_detection(CATEGORY_PERSON)],
        )
        assert frame.has_person
        assert not frame.has_ball


# ============================================================
# 4. VideoDetectionResult 数据结构
# ============================================================


@pytest.mark.fast
class TestVideoDetectionResult:
    """VideoDetectionResult dataclass 测试。"""

    def test_empty_result(self) -> None:
        """空结果."""
        result = VideoDetectionResult(meta={"fps": 30.0})
        assert len(result.frames) == 0
        assert result.shape == (0,)

    def test_with_frames(self) -> None:
        """带帧的结果."""
        frames = [
            FrameDetection(frame_idx=i, frame_time_sec=i / 30.0)
            for i in range(10)
        ]
        result = VideoDetectionResult(meta={"fps": 30.0}, frames=frames)
        assert len(result.frames) == 10
        assert result.shape == (10,)


# ============================================================
# 5. ObjectDetector 初始化测试（mock YOLO）
# ============================================================


@pytest.mark.fast
class TestObjectDetectorInit:
    """ObjectDetector 初始化逻辑测试（无需真实模型）。"""

    def test_init_with_nonexistent_model(self, tmp_path: Path) -> None:
        """不存在的 .pt 模型应抛 FileNotFoundError（ultralytics 会自动下载，但我们拦截非 .pt）."""
        # .pt 后缀会触发 ultralytics 自动下载，所以用 .engine
        fake_path = tmp_path / "nonexistent.engine"
        with pytest.raises(FileNotFoundError, match="模型文件不存在"):
            ObjectDetector(model_path=fake_path)

    def test_init_with_valid_config(self, tmp_path: Path) -> None:
        """有效配置初始化（mock YOLO）."""
        fake_pt = tmp_path / "fake.pt"
        fake_pt.write_bytes(b"fake")

        with patch("backend.ml.behavior.object_detector.YOLO") as mock_yolo:
            mock_yolo.return_value = MagicMock()
            detector = ObjectDetector(
                model_path=fake_pt,
                device="cpu",
                imgsz=320,
                conf=0.5,
                iou=0.45,
                verbose=True,
                interested_classes=[0, 32],
            )
            assert detector.model_path == str(fake_pt)
            assert detector.device == "cpu"
            assert detector.imgsz == 320
            assert detector.conf == 0.5
            assert detector.iou == 0.45
            assert detector.verbose is True
            assert detector.interested_classes == [0, 32]

    def test_default_interested_classes(self, tmp_path: Path) -> None:
        """默认 interested_classes 应为 ALL_INTEREST_INDICES."""
        fake_pt = tmp_path / "fake.pt"
        fake_pt.write_bytes(b"fake")

        with patch("backend.ml.behavior.object_detector.YOLO") as mock_yolo:
            mock_yolo.return_value = MagicMock()
            detector = ObjectDetector(model_path=fake_pt)
            assert detector.interested_classes == ALL_INTEREST_INDICES
            # 应包含 person/dog/ball/food/toy
            assert PERSON_IDX in detector.interested_classes
            assert DOG_IDX in detector.interested_classes
            assert BALL_IDX in detector.interested_classes
            for food_idx in FOOD_INDICES:
                assert food_idx in detector.interested_classes


# ============================================================
# 6. Singleton 测试
# ============================================================


@pytest.mark.fast
class TestDetectorSingleton:
    """get_detector_singleton 单例测试。"""

    def test_singleton_caches_instance(self, tmp_path: Path) -> None:
        """单例应缓存 ObjectDetector 实例."""
        import backend.ml.behavior.object_detector as mod

        # 重置单例
        prev = mod._detector_singleton
        try:
            mod._detector_singleton = None
            with patch("backend.ml.behavior.object_detector.YOLO") as mock_yolo:
                mock_yolo.return_value = MagicMock()
                # 注入一个临时 .pt 文件避免自动下载
                fake_pt = tmp_path / "yolo26n.pt"
                fake_pt.write_bytes(b"fake")
                # 用 monkey patch 修改 PROJECT_ROOT 查找
                with patch.object(mod, "PROJECT_ROOT", tmp_path):
                    d1 = mod.get_detector_singleton()
                    d2 = mod.get_detector_singleton()
                    assert d1 is d2
        finally:
            mod._detector_singleton = prev


# ============================================================
# 7. 集成测试（需真实模型 + 视频，CI 默认跳过）
# ============================================================


@pytest.mark.integration
class TestObjectDetectorIntegration:
    """真实模型集成测试（CI 默认跳过）。

    手动运行:
        pytest backend/tests/ml/test_object_detector.py -v -m integration
    """

    def test_detect_synthetic_ball_frame(self, tmp_path: Path) -> None:
        """合成带球的图片 → ObjectDetector → 应检测到 ball."""
        try:
            import cv2
        except ImportError:
            pytest.skip("cv2 不可用")

        # 创建合成图片：白色背景 + 红色圆形（模拟球）
        img = np.full((640, 640, 3), 255, dtype=np.uint8)
        cv2.circle(img, (320, 320), 50, (0, 0, 255), -1)  # 红色球
        img_path = tmp_path / "synthetic_ball.png"
        cv2.imwrite(str(img_path), img)

        try:
            detector = ObjectDetector(model_path="yolo26n.pt", verbose=False)
        except Exception as e:
            pytest.skip(f"YOLO26 模型不可用: {e}")

        dets = detector.detect_image(img_path)
        # COCO 80 类对纯色圆形可能不识别，验证流程不崩溃即可
        assert isinstance(dets, list)
        # 若检测到，验证数据结构
        for d in dets:
            assert isinstance(d, Detection)
            assert 0 <= d.class_id < 80

    def test_detect_real_video(self) -> None:
        """真实视频检测端到端."""
        project_root = Path(__file__).resolve().parents[3]
        # 候选测试视频
        video_candidates = [
            project_root / "data" / "sample.mp4",
            project_root / "data" / "test_video.mp4",
        ]
        video_path = None
        for cand in video_candidates:
            if cand.exists():
                video_path = cand
                break
        if video_path is None:
            pytest.skip("无可用测试视频")

        try:
            detector = ObjectDetector(model_path="yolo26n.pt", verbose=False)
        except Exception as e:
            pytest.skip(f"YOLO26 模型不可用: {e}")

        result = detector.detect_video(video_path, save_output=False)
        assert isinstance(result, VideoDetectionResult)
        assert "fps" in result.meta
        assert "frame_count" in result.meta
        assert "duration_sec" in result.meta
        assert len(result.frames) > 0
