"""FCI-IGP 7 维评分信号提取（从 22 行为 episodes 提取）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.4
依据: dev-docs/stages/phase-3.md §3.4b
      backend/ml/scoring/configs/fci_igp.yaml

独立模块原因:
    原先 _signals_from_fci_igp_episodes 定义在 backend/workers/tasks.py，
    但 tasks.py 顶部 import celery 导致单元测试环境（无 celery）无法导入。
    将纯 ML 逻辑提取到本模块，消除 celery 依赖，提升可测试性。

信号映射（与 fci_igp.yaml 对齐）:
    - accuracy: action_correct / action_count
    - latency: command_to_action_latency（第一个行为起始时间）
    - duration: action_duration（行为总持续时长）
    - search: search_coverage / search_speed / target_found
    - attention: focus_ratio / unnecessary_movement_count
    - courage: courage_score / avoidance_detected
    - gait: gait_symmetry / pace_change_smoothness
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from backend.ml.behavior.rule_engine import BehaviorEpisode


def signals_from_fci_igp_episodes(
    episodes: List[BehaviorEpisode],
    fps: float,
    duration_sec: float,
) -> Dict:
    """从 22 行为 episodes 提取 FCI-IGP 7 维评分信号.

    Args:
        episodes: BehaviorEpisode 列表（22 类行为之一）
        fps: 视频帧率
        duration_sec: 视频时长（秒）

    Returns:
        信号字典，字段与 fci_igp.yaml 规则条件对齐
    """
    if not episodes:
        return {
            "action_correct": 0,
            "action_count": 0,
            "command_to_action_latency": 99.0,
            "action_duration": 0.0,
            "search_coverage": 0.0,
            "search_speed": 0.0,
            "target_found": False,
            "focus_ratio": 0.0,
            "unnecessary_movement_count": 0,
            "courage_score": 0.0,
            "avoidance_detected": False,
            "gait_symmetry": 0.0,
            "pace_change_smoothness": 0.0,
        }

    action_count = len(episodes)
    action_correct = sum(1 for e in episodes if e.confidence >= 0.5)
    first_start_sec = episodes[0].start_frame / fps if fps > 0 else 0.0
    total_duration = sum(
        (e.end_frame - e.start_frame + 1) / fps if fps > 0 else 0.0
        for e in episodes
    )
    focus_ratio = float(np.mean([e.confidence for e in episodes]))

    # 搜索相关: track/search_blind 行为视为搜索
    search_behaviors = [e for e in episodes if e.behavior in ("track", "search_blind")]
    search_coverage = min(1.0, total_duration / max(duration_sec, 1.0)) if duration_sec > 0 else 0.0
    search_speed = float(np.mean([e.confidence for e in search_behaviors])) if search_behaviors else 0.0
    target_found = any(e.behavior in ("alert_sit", "alert_down", "apprehend") for e in episodes)

    # 胆量: apprehend/bite/guard 行为视为胆量表现
    courage_behaviors = [e for e in episodes if e.behavior in ("apprehend", "bite", "guard", "release")]
    courage_score = float(np.mean([e.confidence for e in courage_behaviors])) if courage_behaviors else 0.0
    avoidance_detected = False  # 规则引擎无法检测回避，需 ST-GCN+BC

    # 步态: heel/obstacle/jump/scale 行为视为步态相关
    gait_behaviors = [e for e in episodes if e.behavior in ("heel", "obstacle", "jump", "scale")]
    gait_symmetry = float(np.mean([e.confidence for e in gait_behaviors])) if gait_behaviors else 0.0
    pace_change_smoothness = gait_symmetry  # 简化：用步态置信度近似

    # 注意力: 不必要的动作 = 短时低置信度行为数
    unnecessary_movement_count = sum(
        1 for e in episodes
        if e.confidence < 0.4 and (e.end_frame - e.start_frame) < 5
    )

    return {
        "action_correct": action_correct,
        "action_count": action_count,
        "command_to_action_latency": float(first_start_sec),
        "action_duration": float(total_duration),
        "search_coverage": float(search_coverage),
        "search_speed": float(search_speed),
        "target_found": bool(target_found),
        "focus_ratio": float(focus_ratio),
        "unnecessary_movement_count": int(unnecessary_movement_count),
        "courage_score": float(courage_score),
        "avoidance_detected": bool(avoidance_detected),
        "gait_symmetry": float(gait_symmetry),
        "pace_change_smoothness": float(pace_change_smoothness),
    }
