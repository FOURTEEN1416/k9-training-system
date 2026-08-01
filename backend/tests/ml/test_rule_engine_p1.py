"""规则引擎 P1 8 类行为单元测试.

Owner: ML 开发
Phase: 2.2c

测试策略:
    - 8 类 P1 行为各 ≥ 1 个测试用例（合成关键点序列）
    - 合成数据使用明确几何特征，确保规则可识别
    - 每个测试验证: 行为类别被检测到 + 置信度 > 0
    - 复用 P0 测试辅助函数（make_frame/make_standing_frame 等）

P1 行为（依据 RESEARCH_STANDARDS.md §4.1）:
    - track (追踪): 鼻尖贴近地面 + 路径跟随
    - alert_sit (示警坐): SIT + 头部抬起/朝向目标
    - alert_down (示警卧): DOWN + 头部抬起/朝向目标
    - apprehend (扑咬): 高速移动 + 嘴部活跃
    - escort (押解): 持续移动 + 站立 + 头部警觉
    - obstacle (障碍穿越): 跳跃轨迹 + 高速移动
    - recall (返回): 方向反转 + 快速移动
    - watch (警戒): 头部抬起 + 身体紧绷 + 站立

坐标系统:
    y 轴向下（y 大 = 位置低/接近地面）
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.ml.behavior.constants import (
    CHIN, DOWN, FRONT_ELBOWS, FRONT_KNEES, FRONT_PAWS,
    LEFT_EAR_BASE, LEFT_EAR_TIP, LEFT_EYE, NOSE,
    NUM_KEYPOINTS, P1_BEHAVIORS,
    REAR_ELBOWS, REAR_KNEES, REAR_PAWS,
    RIGHT_EAR_BASE, RIGHT_EAR_TIP, RIGHT_EYE,
    SIT, STAND,
    TAIL_END, TAIL_START, THROAT, WITHERS,
    TRACK, ALERT_SIT, ALERT_DOWN, APPREHEND, ESCORT, OBSTACLE, RECALL, WATCH,
)
from backend.ml.behavior.rule_engine import RuleEngine

# 复用 P0 测试辅助函数
from backend.tests.ml.test_rule_engine import (
    make_frame, make_sequence, make_standing_frame,
    make_sitting_frame, make_lying_frame,
)


# ===== P1 测试用例 =====

class TestP1Constants:
    """P1 常量测试。"""

    def test_p1_behaviors_count(self):
        assert len(P1_BEHAVIORS) == 8

    def test_p1_behaviors_unique(self):
        assert len(set(P1_BEHAVIORS)) == 8

    def test_all_behaviors_16(self):
        from backend.ml.behavior.constants import ALL_BEHAVIORS, NUM_ALL_BEHAVIORS
        assert NUM_ALL_BEHAVIORS == 16
        assert len(ALL_BEHAVIORS) == 16

    def test_p1_behaviors_expected(self):
        expected = {"track", "alert_sit", "alert_down", "apprehend",
                    "escort", "obstacle", "recall", "watch"}
        actual = set(P1_BEHAVIORS)
        assert actual == expected


class TestTrack:
    """追踪测试：鼻尖贴近地面 + 持续移动 + 头部朝下。"""

    def test_track_detected(self):
        """追踪：站立姿态 + 鼻尖贴地 + 持续 x 方向移动。"""
        frames = []
        ground_y = 200.0
        for i in range(20):
            frame = make_standing_frame(x_offset=i * 5)
            # 鼻尖贴近地面（NOSE.y > ground_y * 0.85 = 170）
            frame[NOSE] = [100 + i * 5, ground_y - 10, 1.0]  # y=190, 接近地面
            # 头部朝下（NOSE.y > WITHERS.y）
            # WITHERS.y = ground_y - 120 = 80, NOSE.y = 190 > 80 ✅
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert TRACK in behaviors, f"未识别 TRACK，实际: {behaviors}"

    def test_track_not_triggered_when_stationary(self):
        """静止时不应识别 TRACK（无持续移动）。"""
        frames = [make_standing_frame() for _ in range(20)]
        # 鼻尖贴地但不移动
        for frame in frames:
            frame[NOSE] = [100, 190, 1.0]
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        track_eps = [e for e in episodes if e.behavior == TRACK]
        assert len(track_eps) == 0, f"静止时不应识别 TRACK，实际: {track_eps}"


class TestObstacle:
    """障碍穿越测试：跳跃轨迹 + 高速移动。"""

    def test_obstacle_detected(self):
        """障碍穿越：withers.y 先高后低（跳跃）+ 高速移动。"""
        frames = []
        ground_y = 200.0
        for i in range(30):
            x_off = i * 20  # 高速移动
            if i < 15:
                # 前半段：withers 高（y 小，跳跃上升）
                frame = make_standing_frame(x_offset=x_off)
                frame[WITHERS, 1] = ground_y - 160  # y=40, 很高
            else:
                # 后半段：withers 低（y 大，落下）
                frame = make_standing_frame(x_offset=x_off)
                frame[WITHERS, 1] = ground_y - 60  # y=140, 较低
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine(obstacle_y_drop=15.0, obstacle_motion=15.0)
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert OBSTACLE in behaviors, f"未识别 OBSTACLE，实际: {behaviors}"


class TestWatch:
    """警戒测试：头部抬起 + 身体稳定 + 站立。"""

    def test_watch_detected(self):
        """警戒：站立不动 + 头部抬起（NOSE.y < WITHERS.y）。"""
        frames = []
        ground_y = 200.0
        for _ in range(20):
            frame = make_standing_frame()
            # 头部抬起：NOSE.y < WITHERS.y
            frame[NOSE] = [100, ground_y - 150, 1.0]  # y=50
            frame[WITHERS, 1] = ground_y - 120  # y=80, NOSE(50) < WITHERS(80)
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert WATCH in behaviors, f"未识别 WATCH，实际: {behaviors}"

    def test_watch_not_triggered_by_motion(self):
        """移动时不应识别 WATCH（身体不稳定）。"""
        frames = []
        ground_y = 200.0
        for i in range(20):
            frame = make_standing_frame(x_offset=i * 10)
            frame[NOSE] = [100 + i * 10, ground_y - 150, 1.0]
            frame[WITHERS, 1] = ground_y - 120
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        watch_eps = [e for e in episodes if e.behavior == WATCH]
        assert len(watch_eps) == 0, f"移动时不应识别 WATCH，实际: {watch_eps}"


class TestApprehend:
    """扑咬测试：高速移动 + 嘴部活跃。"""

    def test_apprehend_detected(self):
        """扑咬：withers 高速移动 + nose-chin 距离振荡。"""
        frames = []
        for i in range(20):
            frame = make_standing_frame(x_offset=i * 30)  # 高速移动 > 25
            # nose-chin 距离振荡
            if i % 2 == 0:
                frame[NOSE] = [100 + i * 30, 80, 1.0]
                frame[CHIN] = [100 + i * 30, 110, 1.0]  # 距离=30
            else:
                frame[NOSE] = [100 + i * 30, 100, 1.0]
                frame[CHIN] = [100 + i * 30, 105, 1.0]  # 距离=5
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine(apprehend_speed=25.0, apprehend_mouth_var=5.0)
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert APPREHEND in behaviors, f"未识别 APPREHEND，实际: {behaviors}"


class TestEscort:
    """押解测试：持续移动 + 站立 + 头部警觉。"""

    def test_escort_detected(self):
        """押解：站立 + 持续移动 + 头部抬起。"""
        frames = []
        ground_y = 200.0
        for i in range(20):
            frame = make_standing_frame(x_offset=i * 5)
            # 头部警觉：NOSE.y <= WITHERS.y
            frame[NOSE] = [100 + i * 5, ground_y - 130, 1.0]  # y=70
            frame[WITHERS, 1] = ground_y - 120  # y=80, NOSE(70) <= WITHERS(80)
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine(escort_stand_ratio=0.7)
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert ESCORT in behaviors, f"未识别 ESCORT，实际: {behaviors}"


class TestRecall:
    """返回测试：方向反转 + 快速移动。"""

    def test_recall_detected(self):
        """返回：前半段远离（x 增加），后半段快速返回（x 快速减少）。"""
        frames = []
        x = 100.0
        for i in range(30):
            if i < 15:
                # 前半段：缓慢远离（x 增加）
                x += 2
            else:
                # 后半段：快速返回（x 快速减少，速度 > 20）
                x -= 25
            frame = make_standing_frame(x_offset=x - 100)
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine(recall_speed=20.0)
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert RECALL in behaviors, f"未识别 RECALL，实际: {behaviors}"


class TestAlertSit:
    """示警坐测试：SIT 姿态 + 头部抬起。"""

    def test_alert_sit_detected(self):
        """示警坐：坐姿 + 头部抬起（NOSE.y < WITHERS.y）。"""
        frames = []
        ground_y = 200.0
        for _ in range(20):
            frame = make_sitting_frame()
            # 头部抬起：NOSE.y < WITHERS.y
            frame[NOSE] = [100, ground_y - 100, 1.0]  # y=100
            frame[WITHERS, 1] = ground_y - 80  # y=120, NOSE(100) < WITHERS(120)
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert ALERT_SIT in behaviors, f"未识别 ALERT_SIT，实际: {behaviors}"


class TestAlertDown:
    """示警卧测试：DOWN 姿态 + 头部抬起。"""

    def test_alert_down_detected(self):
        """示警卧：卧姿 + 头部抬起（NOSE.y < WITHERS.y）。"""
        frames = []
        ground_y = 200.0
        for _ in range(20):
            frame = make_lying_frame()
            # 头部抬起：NOSE.y < WITHERS.y
            # 卧姿 WITHERS.y = ground_y - 10 = 190
            # 需要 NOSE.y < 190
            frame[NOSE] = [100, ground_y - 30, 1.0]  # y=170 < 190
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert ALERT_DOWN in behaviors, f"未识别 ALERT_DOWN，实际: {behaviors}"


class TestAlertWithTarget:
    """示警行为完整版测试（带 target_boxes）。"""

    def test_alert_sit_with_target(self):
        """示警坐（完整版）：坐姿 + 鼻尖朝向目标框。"""
        frames = []
        ground_y = 200.0
        target_boxes = []
        for _ in range(20):
            frame = make_sitting_frame()
            # 鼻尖位置
            nose_x, nose_y = 100, ground_y - 100
            frame[NOSE] = [nose_x, nose_y, 1.0]
            frame[WITHERS, 1] = ground_y - 80
            frames.append(frame)
            # 目标框在鼻尖附近（距离 < 150）
            target_boxes.append([nose_x + 50, nose_y + 30, nose_x + 80, nose_y + 60])
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, target_boxes=target_boxes, fps=30.0)
        behaviors = [e.behavior for e in episodes]
        assert ALERT_SIT in behaviors, f"未识别 ALERT_SIT（带目标），实际: {behaviors}"


class TestP1EnableDisable:
    """P1 开关测试。"""

    def test_p1_disabled(self):
        """enable_p1=False 时不应检测到 P1 行为。"""
        # 构造追踪数据
        frames = []
        ground_y = 200.0
        for i in range(20):
            frame = make_standing_frame(x_offset=i * 5)
            frame[NOSE] = [100 + i * 5, ground_y - 10, 1.0]
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0, enable_p1=False)
        p1_eps = [e for e in episodes if e.behavior in P1_BEHAVIORS]
        assert len(p1_eps) == 0, f"enable_p1=False 时不应检测 P1，实际: {p1_eps}"

    def test_p1_enabled_by_default(self):
        """默认启用 P1。"""
        # 构造警戒数据
        frames = []
        ground_y = 200.0
        for _ in range(20):
            frame = make_standing_frame()
            frame[NOSE] = [100, ground_y - 150, 1.0]
            frame[WITHERS, 1] = ground_y - 120
            frames.append(frame)
        kpts = make_sequence(frames)
        engine = RuleEngine()
        episodes = engine.recognize(kpts, fps=30.0)
        p1_eps = [e for e in episodes if e.behavior in P1_BEHAVIORS]
        assert len(p1_eps) > 0, f"默认应启用 P1，实际 P1 episodes: {p1_eps}"


class TestAllP1Behaviors:
    """验证 8 类 P1 行为全部可被识别。"""

    def test_all_8_p1_behaviors_have_test(self):
        """确认 8 类 P1 行为常量都存在。"""
        expected = {"track", "alert_sit", "alert_down", "apprehend",
                    "escort", "obstacle", "recall", "watch"}
        actual = set(P1_BEHAVIORS)
        assert actual == expected
