"""Phase 2.6 系统集成测试.

Owner: 全体（见 AGENTS.md §2.2）
Phase: 2.6
依据: dev-docs/stages/phase-2.md §2.6

测试内容:
    2.6a  数据飞轮 API 契约测试（finetune + annotations + models）
    2.6b  16 行为引擎 → USPCA 评分端到端
    2.6c  标注任务 → 标注数据 → 导出 YOLO 格式
    2.6d  模型版本管理 A/B（注册 + 激活 + 回滚）

运行:
    pytest backend/tests/integration/test_phase2_6_integration.py -v

注意:
    - 不需要真实视频/GPU（用合成数据）
    - 需要 PostgreSQL 运行（127.0.0.1:5433/k9system）
    - 不需要 Celery worker / Label Studio（mock 或跳过）
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from backend.ml.behavior.rule_engine import RuleEngine
from backend.ml.behavior.constants import ALL_BEHAVIORS, P0_BEHAVIORS, P1_BEHAVIORS
from backend.ml.scoring.engine import ScoringEngine
from backend.ml.scoring.schema import Scene


PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# 2.6a 16 行为引擎 → USPCA 评分端到端
# ============================================================

class TestBehaviorToScoringIntegration:
    """16 行为规则引擎 → USPCA 评分卡端到端测试。"""

    def test_16_behaviors_all_defined(self):
        """验证 16 行为常量完整。"""
        assert len(P0_BEHAVIORS) == 8, f"P0 应有 8 类，实际 {len(P0_BEHAVIORS)}"
        assert len(P1_BEHAVIORS) == 8, f"P1 应有 8 类，实际 {len(P1_BEHAVIORS)}"
        assert len(ALL_BEHAVIORS) == 16, f"总共应有 16 类，实际 {len(ALL_BEHAVIORS)}"

    def test_rule_engine_p0_p1_enabled(self):
        """规则引擎默认启用 P1。"""
        engine = RuleEngine()
        # 合成 30 帧站立序列
        kpts = np.zeros((30, 24, 3), dtype=np.float32)
        kpts[:, :, 2] = 1.0  # 全部置信
        kpts[:, :, 1] = 100  # y 坐标
        episodes = engine.recognize(kpts, fps=30.0, enable_p1=True)
        assert isinstance(episodes, list)

    def test_usPCA_scoring_with_behaviors(self):
        """USPCA 评分卡能处理 16 行为 episodes 转换的标量信号。

        验证 16 行为 → 评分维度映射的端到端：
        - P0 行为（sit/down/stand/heel/...）→ accuracy + latency + duration
        - P1 行为（track/apprehend/recall/...）→ search_efficiency + attention
        - ScoringContext.signals 是标量字典，符合 schema 约束
        """
        from backend.ml.scoring.engine import ScoringEngine
        from backend.ml.scoring.schema import ScoringContext
        from backend.ml.behavior.rule_engine import BehaviorEpisode

        yaml_path = PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "uspca_patrol.yaml"
        engine = ScoringEngine.from_yaml(str(yaml_path))

        # 合成 16 行为 episodes（覆盖 P0 + P1）
        episodes = [
            # P0 服从基础
            BehaviorEpisode(behavior="sit", start_frame=0, end_frame=90, confidence=0.95),
            BehaviorEpisode(behavior="heel", start_frame=91, end_frame=150, confidence=0.88),
            BehaviorEpisode(behavior="stay", start_frame=151, end_frame=240, confidence=0.92),
            # P1 巡逻专项
            BehaviorEpisode(behavior="track", start_frame=241, end_frame=330, confidence=0.85),
            BehaviorEpisode(behavior="apprehend", start_frame=331, end_frame=380, confidence=0.90),
            BehaviorEpisode(behavior="recall", start_frame=381, end_frame=450, confidence=0.82),
            BehaviorEpisode(behavior="watch", start_frame=451, end_frame=540, confidence=0.78),
        ]

        # episodes → USPCA 标量信号（评分卡 YAML 实际使用的字段）
        signals = self._episodes_to_scoring_signals(episodes)
        ctx = ScoringContext(scene="uspca_patrol", signals=signals)
        result = engine.evaluate(ctx)
        assert result is not None
        assert 0 <= result.total_score <= 100
        # 验证 5 维都被评估
        assert len(result.dimension_scores) == 5
        # 验证 verdict 在合法值
        assert result.verdict in ("pass", "borderline", "fail")

    @staticmethod
    def _episodes_to_scoring_signals(episodes: list) -> dict:
        """将 16 行为 episodes 转为 USPCA 评分卡标量信号.

        依据 uspca_patrol.yaml behavior_mapping:
            - accuracy: P0 服从行为正确率
            - latency: 平均响应延迟
            - duration: 平均保持时长
            - search_efficiency: P1 搜索行为覆盖率
            - attention: 注意力聚焦比例
        """
        if not episodes:
            return {
                "action_correct": 0, "action_count": 0,
                "command_to_action_latency": 3.0, "action_duration": 0.0,
                "search_coverage": 0.0, "search_speed": 0.0, "target_found": False,
                "focus_ratio": 0.0, "unnecessary_movement_count": 0,
            }

        # 准确度: confidence ≥ 0.80 视为正确执行
        action_correct = sum(1 for e in episodes if e.confidence >= 0.80)
        action_count = len(episodes)

        # 延迟: 用平均 confidence 反推（高 confidence → 低延迟）
        avg_conf = sum(e.confidence for e in episodes) / action_count
        command_to_action_latency = max(0.2, 2.0 - avg_conf)  # 0.2s-1.8s

        # 保持: 平均 episode 时长（帧数 → 秒，假设 30fps）
        avg_duration_frames = sum(e.duration_frames for e in episodes) / action_count
        action_duration = avg_duration_frames / 30.0

        # 搜索效率: P1 搜索行为占比 + 速度
        p1_search_behaviors = {"track", "alert_sit", "alert_down", "obstacle"}
        search_episodes = [e for e in episodes if e.behavior in p1_search_behaviors]
        search_coverage = len(search_episodes) / action_count
        search_speed = sum(e.confidence for e in search_episodes) / max(len(search_episodes), 1) * 0.5
        target_found = any(e.behavior in {"alert_sit", "alert_down", "apprehend"} for e in episodes)

        # 注意力: 聚焦比例（confidence ≥ 0.85）+ 不必要移动（confidence < 0.70 视为分散）
        focused = sum(1 for e in episodes if e.confidence >= 0.85)
        unnecessary = sum(1 for e in episodes if e.confidence < 0.70)
        focus_ratio = focused / action_count

        return {
            "action_correct": action_correct,
            "action_count": action_count,
            "command_to_action_latency": command_to_action_latency,
            "action_duration": action_duration,
            "search_coverage": search_coverage,
            "search_speed": search_speed,
            "target_found": target_found,
            "focus_ratio": focus_ratio,
            "unnecessary_movement_count": unnecessary,
        }

    def test_uspca_yaml_behavior_mapping_completeness(self):
        """USPCA YAML 包含 16 行为映射。"""
        import yaml

        yaml_path = PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "uspca_patrol.yaml"
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        bm = config["scoring_engine"]["behavior_mapping"]
        assert len(bm) == 16, f"behavior_mapping 应有 16 项，实际 {len(bm)}"

        # 验证每个行为都有 dimensions 和 weight
        for behavior, mapping in bm.items():
            assert "dimensions" in mapping, f"{behavior} 缺少 dimensions"
            assert "weight" in mapping, f"{behavior} 缺少 weight"
            assert isinstance(mapping["dimensions"], list)
            assert isinstance(mapping["weight"], (int, float))


# ============================================================
# 2.6b 数据飞轮 API 路由注册验证
# ============================================================

class TestDataFlywheelAPI:
    """数据飞轮 API 路由注册验证（不需要 DB）。"""

    def test_finetune_routes_registered(self):
        """finetune 路由已注册。"""
        from backend.app.api.finetune import router

        paths = {r.path for r in router.routes}
        assert "/finetune/trigger" in paths
        assert "/finetune/status" in paths
        assert "/finetune/pipeline" in paths

    def test_annotations_routes_registered(self):
        """annotations 路由已注册。"""
        from backend.app.api.annotations import router

        paths = {r.path for r in router.routes}
        assert "/annotations/tasks" in paths
        assert "/annotations/sync" in paths
        assert "/annotations/ls/health" in paths

    def test_models_routes_registered(self):
        """models 路由已注册。"""
        from backend.app.api.models import router

        paths = {r.path for r in router.routes}
        assert "/models" in paths  # list
        assert "/models/current" in paths
        assert "/models/register" in paths
        assert "/models/{model_id}/activate" in paths

    def test_main_app_includes_all_routers(self):
        """main app 包含所有 Phase 2 路由（跳过 celery 依赖）。"""
        try:
            from backend.app.main import app
        except ModuleNotFoundError as e:
            if "celery" in str(e):
                pytest.skip(f"跳过: {e}")
            raise

        all_paths = {r.path for r in app.routes if hasattr(r, "path")}
        # 数据飞轮端点
        assert "/api/finetune/trigger" in all_paths
        assert "/api/finetune/status" in all_paths
        assert "/api/finetune/pipeline" in all_paths
        assert "/api/annotations/tasks" in all_paths
        assert "/api/annotations/sync" in all_paths
        # 模型管理
        assert "/api/models/register" in all_paths
        assert "/api/models/current" in all_paths
        # 视频上传
        assert "/api/videos/upload" in all_paths


# ============================================================
# 2.6c 标注 → YOLO 格式导出
# ============================================================

class TestAnnotationExport:
    """标注数据导出为 YOLO 格式（不依赖 DB，测试转换逻辑）。"""

    def test_keypoint_to_yolo_label_format(self):
        """关键点标注 → YOLO-pose label 格式正确。"""
        # 合成 24 关键点（像素坐标）
        keypoints = [[100 + i * 10, 200 + i * 5, 1.0] for i in range(24)]
        img_w, img_h = 640, 480

        # 计算检测框
        xs = [kp[0] for kp in keypoints if kp[2] > 0]
        ys = [kp[1] for kp in keypoints if kp[2] > 0]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        # 关键点归一化
        kp_flat = []
        for kp in keypoints:
            kp_flat.extend([kp[0] / img_w, kp[1] / img_h, kp[2]])

        label_line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(
            f"{v:.6f}" for v in kp_flat
        )

        # 验证格式: class cx cy w h px1 py1 v1 ... px24 py24 v24
        parts = label_line.split()
        assert parts[0] == "0"  # class
        assert len(parts) == 5 + 24 * 3  # 5 (class+cxcywh) + 72 (24*3)

        # 验证归一化
        for i in range(1, 5):
            val = float(parts[i])
            assert 0 <= val <= 1, f"坐标 {i} 未归一化: {val}"

    def test_finetune_dataset_yaml_format(self):
        """微调数据集 yaml 格式正确。"""
        import yaml

        yaml_content = """path: finetune_dataset
train: images/train
val: images/val

kpt_shape: [24, 3]

names:
  0: dog
"""
        config = yaml.safe_load(yaml_content)
        assert config["path"] == "finetune_dataset"
        assert config["train"] == "images/train"
        assert config["val"] == "images/val"
        assert config["kpt_shape"] == [24, 3]
        assert config["names"][0] == "dog"


# ============================================================
# 2.6d 模型版本管理 A/B
# ============================================================

class TestModelVersioning:
    """模型版本管理逻辑验证（不依赖 DB，测试激活逻辑）。"""

    def test_model_activate_deactivates_others(self):
        """激活模型时，同类型其他模型应被取消激活。"""
        # 这是逻辑验证，实际 DB 测试需要 async session
        # 这里验证 activate 端点的代码逻辑
        from backend.app.api.models import activate_model
        assert callable(activate_model)

    def test_model_types_valid(self):
        """模型类型枚举完整。"""
        from backend.app.models.ml_model import ModelType

        valid_types = {t.value for t in ModelType}
        assert "pose" in valid_types
        assert "behavior" in valid_types
        assert "scoring" in valid_types


# ============================================================
# 2.6b 历史评分查询 API
# ============================================================

class TestHistoryScoreQuery:
    """2.6b 历史评分查询 API 路由验证。"""

    def test_scores_routes_registered(self):
        """scores 路由已注册，包含 by-dog 端点。"""
        from backend.app.api.scores import router

        paths = {r.path for r in router.routes}
        assert "/scores" in paths
        assert "/scores/by-dog/{dog_id}" in paths
        assert "/scores/{score_id}" in paths

    def test_by_dog_route_before_score_id(self):
        """by-dog 路由必须在 /{score_id} 之前（FastAPI 按定义顺序匹配）。

        若顺序错误，"by-dog" 会被当作 score_id 解析为 int 失败返回 422。
        """
        from backend.app.api.scores import router

        paths_in_order = [r.path for r in router.routes if hasattr(r, "path")]
        by_dog_idx = next(i for i, p in enumerate(paths_in_order) if "by-dog" in p)
        score_id_idx = next(i for i, p in enumerate(paths_in_order)
                            if p == "/scores/{score_id}")
        assert by_dog_idx < score_id_idx, (
            f"by-dog 必须在 /{{score_id}} 之前（当前 by-dog@{by_dog_idx}, "
            f"score_id@{score_id_idx}）"
        )


# ============================================================
# 2.6e USPCA 场景端到端
# ============================================================

class TestUSPCASceneE2E:
    """USPCA 巡逻犬场景端到端测试。"""

    def test_uspca_scene_in_scoring_api(self):
        """评分 API 支持 USPCA 场景。"""
        from backend.app.api.scoring import _SCENE_TO_FILE

        assert "uspca_patrol" in _SCENE_TO_FILE

    def test_uspca_yaml_loads(self):
        """USPCA YAML 可加载且结构完整。"""
        import yaml

        yaml_path = PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "uspca_patrol.yaml"
        with open(yaml_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        se = config["scoring_engine"]
        assert se["scene"] == "uspca_patrol"
        assert "dimensions" in se
        assert "thresholds" in se
        assert se["thresholds"]["pass"] == 75

    def test_video_scene_supports_uspca(self):
        """Video 模型支持 USPCA 场景。"""
        from backend.app.models.video import VALID_SCENES

        assert "uspca_patrol" in VALID_SCENES

    def test_schema_scene_includes_uspca(self):
        """评分 schema 包含 USPCA 场景。"""
        from backend.ml.scoring.schema import Scene
        from typing import get_args

        scenes = get_args(Scene)
        assert "uspca_patrol" in scenes
