"""规则引擎 P0 8 类行为单元测试.

Owner: ML 开发
Phase: 1.2c

测试策略:
    - 8 类行为各 ≥ 1 个测试用例（合成关键点序列）
    - 合成数据使用明确几何特征，确保规则可识别
    - 每个测试验证: 行为类别 + 时间区间 + 置信度 > 0

坐标系统:
    y 轴向下（y 大 = 位置低/接近地面）
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.ml.behavior.constants import (
    BARK, BITE, CHIN, DOWN, FRONT_ELBOWS, FRONT_KNEES, FRONT_PAWS,
    HEEL, LEFT_EAR_BASE, LEFT_EAR_TIP, LEFT_EYE, NOSE,
    NUM_KEYPOINTS, P0_BEHAVIORS, REAR_KNEES, REAR_PAWS,
    RIGHT_EAR_BASE, RIGHT_EAR_TIP, RIGHT_EYE, SIT, SIT_UP, STAND, STAY,
    TAIL_END, TAIL_START, THROAT, WITHERS,
)
from backend.ml.behavior.rule_engine import (
    DEFAULT_STAY_FRAMES, BehaviorEpisode, RuleEngine,
)


# ===== 辅助函数 =====

def make_frame(keypoints: dict[int, tuple[float, float, float]]) -> np.ndarray:
    """构造单帧关键点 (24, 3)。
    
    keypoints: {index: (x, y, conf)}，未指定的关键点 conf=0
    """
    frame = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
    for idx, (x, y, conf) in keypoints.items():
        frame[idx] = [x, y, conf]
    return frame


def make_sequence(frames: list[np.ndarray]) -> np.ndarray:
    """构造关键点序列 (T, 24, 3)。"""
    return np.stack(frames, axis=0)


def make_standing_frame(x_offset: float = 0.0, ground_y: float = 200.0) -> np.ndarray:
    """构造站立姿态帧：四腿伸直 + 躯干高。
    
    - 前爪 y=ground_y, 前膝 y=ground_y-30, 前肘 y=ground_y-60
    - 后爪 y=ground_y, 后膝 y=ground_y-30, 后肘 y=ground_y-60
    - withers y=ground_y-120 (高，确保 < ground_y*0.5)
    """
    from backend.ml.behavior.constants import REAR_ELBOWS
    return make_frame({
        FRONT_PAWS[0]: (50 + x_offset, ground_y, 1.0),
        FRONT_PAWS[1]: (150 + x_offset, ground_y, 1.0),
        FRONT_KNEES[0]: (50 + x_offset, ground_y - 30, 1.0),
        FRONT_KNEES[1]: (150 + x_offset, ground_y - 30, 1.0),
        FRONT_ELBOWS[0]: (50 + x_offset, ground_y - 60, 1.0),
        FRONT_ELBOWS[1]: (150 + x_offset, ground_y - 60, 1.0),
        REAR_PAWS[0]: (50 + x_offset, ground_y, 1.0),
        REAR_PAWS[1]: (150 + x_offset, ground_y, 1.0),
        REAR_KNEES[0]: (50 + x_offset, ground_y - 30, 1.0),
        REAR_KNEES[1]: (150 + x_offset, ground_y - 30, 1.0),
        REAR_ELBOWS[0]: (50 + x_offset, ground_y - 60, 1.0),
        REAR_ELBOWS[1]: (150 + x_offset, ground_y - 60, 1.0),
        TAIL_START: (100 + x_offset, ground_y - 100, 1.0),
        TAIL_END: (100 + x_offset, ground_y - 120, 1.0),
        LEFT_EAR_BASE: (80 + x_offset, ground_y - 130, 1.0),
        RIGHT_EAR_BASE: (120 + x_offset, ground_y - 130, 1.0),
        NOSE: (100 + x_offset, ground_y - 140, 1.0),
        CHIN: (100 + x_offset, ground_y - 130, 1.0),
        LEFT_EAR_TIP: (75 + x_offset, ground_y - 145, 1.0),
        RIGHT_EAR_TIP: (125 + x_offset, ground_y - 145, 1.0),
        LEFT_EYE: (85 + x_offset, ground_y - 135, 1.0),
        RIGHT_EYE: (115 + x_offset, ground_y - 135, 1.0),
        WITHERS: (100 + x_offset, ground_y - 120, 1.0),  # y=80 < 100=ground*0.5
        THROAT: (100 + x_offset, ground_y - 110, 1.0),
    })


def make_sitting_frame(x_offset: float = 0.0, ground_y: float = 200.0) -> np.ndarray:
    """构造坐姿帧：后腿折叠 + 前腿伸直。
    
    - 前爪 y=ground_y, 前膝 y=ground_y-30, 前肘 y=ground_y-60 (伸直)
    - 后爪 y=ground_y-40 (抬起), 后膝 y=ground_y-45 (折叠，接近后爪)
    - withers y=ground_y-80
    """
    from backend.ml.behavior.constants import REAR_ELBOWS
    return make_frame({
        FRONT_PAWS[0]: (50 + x_offset, ground_y, 1.0),
        FRONT_PAWS[1]: (150 + x_offset, ground_y, 1.0),
        FRONT_KNEES[0]: (50 + x_offset, ground_y - 30, 1.0),
        FRONT_KNEES[1]: (150 + x_offset, ground_y - 30, 1.0),
        FRONT_ELBOWS[0]: (50 + x_offset, ground_y - 60, 1.0),
        FRONT_ELBOWS[1]: (150 + x_offset, ground_y - 60, 1.0),
        REAR_PAWS[0]: (60 + x_offset, ground_y - 40, 1.0),  # 抬起
        REAR_PAWS[1]: (140 + x_offset, ground_y - 40, 1.0),
        REAR_KNEES[0]: (60 + x_offset, ground_y - 45, 1.0),  # 接近后爪（折叠）
        REAR_KNEES[1]: (140 + x_offset, ground_y - 45, 1.0),
        REAR_ELBOWS[0]: (60 + x_offset, ground_y - 80, 1.0),
        REAR_ELBOWS[1]: (140 + x_offset, ground_y - 80, 1.0),
        TAIL_START: (100 + x_offset, ground_y - 70, 1.0),
        TAIL_END: (100 + x_offset, ground_y - 90, 1.0),
        LEFT_EAR_BASE: (80 + x_offset, ground_y - 90, 1.0),
        RIGHT_EAR_BASE: (120 + x_offset, ground_y - 90, 1.0),
        NOSE: (100 + x_offset, ground_y - 100, 1.0),
        CHIN: (100 + x_offset, ground_y - 90, 1.0),
        LEFT_EAR_TIP: (75 + x_offset, ground_y - 105, 1.0),
        RIGHT_EAR_TIP: (125 + x_offset, ground_y - 105, 1.0),
        LEFT_EYE: (85 + x_offset, ground_y - 95, 1.0),
        RIGHT_EYE: (115 + x_offset, ground_y - 95, 1.0),
        WITHERS: (100 + x_offset, ground_y - 80, 1.0),
        THROAT: (100 + x_offset, ground_y - 70, 1.0),
    })


def make_lying_frame(x_offset: float = 0.0, ground_y: float = 200.0) -> np.ndarray:
    """构造卧姿帧：躯干低 + 四肢折叠。
    
    - 所有爪 y=ground_y-5, 所有膝 y=ground_y-8 (折叠)
    - withers y=ground_y-10 (接近地面, > ground_y*0.75=150)
    """
    from backend.ml.behavior.constants import ALL_KNEES, ALL_PAWS, REAR_ELBOWS
    frame = make_frame({})
    for i, idx in enumerate(ALL_PAWS):
        frame[idx] = [50 + i * 30 + x_offset, ground_y - 5, 1.0]
    for i, idx in enumerate(ALL_KNEES):
        frame[idx] = [50 + i * 30 + x_offset, ground_y - 8, 1.0]
    for i, idx in enumerate(FRONT_ELBOWS + REAR_ELBOWS):
        frame[idx] = [50 + i * 30 + x_offset, ground_y - 12, 1.0]
    frame[TAIL_START] = [100 + x_offset, ground_y - 10, 1.0]
    frame[TAIL_END] = [100 + x_offset, ground_y - 15, 1.0]
    frame[LEFT_EAR_BASE] = [80 + x_offset, ground_y - 20, 1.0]
    frame[RIGHT_EAR_BASE] = [120 + x_offset, ground_y - 20, 1.0]
    frame[NOSE] = [100 + x_offset, ground_y - 25, 1.0]
    frame[CHIN] = [100 + x_offset, ground_y - 20, 1.0]
    frame[LEFT_EAR_TIP] = [75 + x_offset, ground_y - 25, 1.0]
    frame[RIGHT_EAR_TIP] = [125 + x_offset, ground_y - 25, 1.0]
    frame[LEFT_EYE] = [85 + x_offset, ground_y - 22, 1.0]
    frame[RIGHT_EYE] = [115 + x_offset, ground_y - 22, 1.0]
    frame[WITHERS] = [100 + x_offset, ground_y - 10, 1.0]  # 接近地面
    frame[THROAT] = [100 + x_offset, ground_y - 15, 1.0]
    return frame


# ===== 测试用例 =====

class TestRuleEngineConstants:
    """规则引擎常量测试。"""

    def test_p0_behaviors_count(self):
        assert len(P0_BEHAVIORS) == 8

    def test_p0_behaviors_unique(self):
        assert len(set(P0_BEHAVIORS)) == 8

    def test_num_keypoints(self):
        assert NUM_KEYPOINTS == 24


class TestPostureClassification:
    """姿态分类测试（sit/down/stand）。"""

    def test_stand_detected(self):
        """立：连续 5 帧站立 → 应识别 STAND。"""
        frames = [make_standing_frame() for _ in range(5)]
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert STAND in behaviors, f"未识别 STAND，实际: {behaviors}"

    def test_sit_detected(self):
        """坐：连续 5 帧坐姿 → 应识别 SIT。"""
        frames = [make_sitting_frame() for _ in range(5)]
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert SIT in behaviors, f"未识别 SIT，实际: {behaviors}"

    def test_down_detected(self):
        """卧：连续 5 帧卧姿 → 应识别 DOWN。"""
        frames = [make_lying_frame() for _ in range(5)]
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert DOWN in behaviors, f"未识别 DOWN，实际: {behaviors}"


class TestSitUp:
    """坐立测试。"""

    def test_sit_up_detected(self):
        """坐立：坐姿 + 头部抬起（nose.y < withers.y）。"""
        frame = make_sitting_frame()
        # 让 nose 明显高于 withers（y 更小）
        frame[NOSE] = [100, 50, 1.0]  # nose.y=50
        frame[WITHERS, 1] = 120  # withers.y=120，nose 在上方
        frames = [frame.copy() for _ in range(5)]
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert SIT_UP in behaviors, f"未识别 SIT_UP，实际: {behaviors}"


class TestStay:
    """停留测试。"""

    def test_stay_detected(self):
        """停留：连续 20 帧关键点不动 → 应识别 STAY。"""
        frame = make_standing_frame()
        frames = [frame.copy() for _ in range(20)]
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert STAY in behaviors, f"未识别 STAY，实际: {behaviors}"

    def test_stay_not_triggered_by_motion(self):
        """移动时不应识别 STAY。"""
        frames = [make_standing_frame(x_offset=i * 10) for i in range(20)]
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        stay_eps = [e for e in episodes if e.behavior == STAY]
        assert len(stay_eps) == 0, f"移动时不应该识别 STAY，实际: {stay_eps}"


class TestHeel:
    """随行测试。"""

    def test_heel_detected(self):
        """随行：站立 + 持续 x 方向移动。"""
        frames = [make_standing_frame(x_offset=i * 5) for i in range(20)]
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert HEEL in behaviors, f"未识别 HEEL，实际: {behaviors}"


class TestBark:
    """叫测试。"""

    def test_bark_detected(self):
        """叫：nose-chin 距离周期性变化。"""
        frames = []
        for i in range(20):
            frame = make_standing_frame()
            # nose-chin 距离振荡：i=0,2,4...距离大；i=1,3,5...距离小
            if i % 2 == 0:
                frame[NOSE] = [100, 80, 1.0]   # nose 高
                frame[CHIN] = [100, 110, 1.0]  # chin 低，距离=30
            else:
                frame[NOSE] = [100, 100, 1.0]
                frame[CHIN] = [100, 105, 1.0]  # 距离=5
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert BARK in behaviors, f"未识别 BARK，实际: {behaviors}"


class TestBite:
    """咬测试。"""

    def test_bite_detected(self):
        """咬：嘴部动作 + 前爪快速移动。"""
        frames = []
        for i in range(20):
            frame = make_standing_frame(x_offset=i * 25)  # 前爪 x 每帧移动 25
            # nose-chin 距离振荡
            if i % 2 == 0:
                frame[NOSE] = [100 + i * 25, 80, 1.0]
                frame[CHIN] = [100 + i * 25, 110, 1.0]
            else:
                frame[NOSE] = [100 + i * 25, 100, 1.0]
                frame[CHIN] = [100 + i * 25, 105, 1.0]
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine(
            bite_mouth_var=5.0,
            bite_motion=20.0,
        )
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert BITE in behaviors, f"未识别 BITE，实际: {behaviors}"


class TestEdgeCases:
    """边界情况测试。"""

    def test_empty_sequence(self):
        """空序列应返回空列表。"""
        kpts = np.zeros((0, 24, 3), dtype=np.float32)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        assert episodes == []

    def test_all_zero_confidence(self):
        """全 0 置信度应返回空列表（无法识别姿态）。"""
        kpts = np.zeros((10, 24, 3), dtype=np.float32)  # conf 全 0
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        # 姿态应为 unknown，不产生 sit/down/stand episodes
        posture_eps = [e for e in episodes if e.behavior in (SIT, DOWN, STAND)]
        assert len(posture_eps) == 0

    def test_invalid_shape(self):
        """错误 shape 应抛 ValueError。"""
        kpts = np.zeros((10, 17, 3), dtype=np.float32)  # 17 关键点
        engine = RuleEngine()
        with pytest.raises(ValueError, match="shape"):
            engine.recognize(kpts, fps=30.0)

    def test_episode_to_dict(self):
        """BehaviorEpisode.to_dict() 正确性。"""
        ep = BehaviorEpisode(behavior=SIT, start_frame=0, end_frame=4, confidence=0.9)
        d = ep.to_dict()
        assert d["behavior"] == SIT
        assert d["start_frame"] == 0
        assert d["end_frame"] == 4
        assert d["confidence"] == 0.9
        assert d["metadata"] == {}

    def test_episode_duration(self):
        """BehaviorEpisode.duration_frames 正确性。"""
        ep = BehaviorEpisode(behavior=SIT, start_frame=0, end_frame=9, confidence=0.9)
        assert ep.duration_frames == 10


class TestAllP0Behaviors:
    """验证 8 类行为全部可被识别。"""

    def test_all_8_behaviors_have_test(self):
        """确认 8 类行为常量都存在。"""
        expected = {"sit", "down", "stand", "heel", "sit_up", "stay", "bark", "bite"}
        actual = set(P0_BEHAVIORS)
        assert actual == expected
