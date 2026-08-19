"""行为识别多轨路由层（ST-GCN+BC + Mamba + 规则引擎）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.1e / 4.2

设计:
    - 多轨部署: ST-GCN+BC（主）+ Mamba（对比基线）+ 规则引擎（备）
    - ST-GCN+BC 三阶段切换: shadow → vote → primary_stgcn → 退役规则引擎
    - Mamba 独立模式: mamba_only / mamba_shadow / mamba_vote
    - 单一 owner: 所有行为识别调用统一通过本路由层
    - 接口与 RuleEngine.recognize() 一致，最小侵入 worker

集成:
    - 上游: backend.workers.tasks.*_pipeline 调用
    - 下游: 输出 list[BehaviorEpisode]（与 RuleEngine 格式一致）

切换条件:
    - shadow: 上线初期，收集对比数据（ST-GCN+BC 真实数据准确率 < 70%）
    - vote: ST-GCN+BC 真实数据准确率 ≥ 70%，投票/置信度融合
    - primary_stgcn: ST-GCN+BC 真实数据准确率 ≥ 85%（phase-3.md §3.1d 目标）
    - mamba_only: Phase 4+ Mamba 基线单独推理
    - mamba_shadow: Mamba 影子对比（对比 ST-GCN+BC/规则引擎）
    - 退役: Phase 4+ 稳定运行后，删除 RuleEngine
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import numpy as np

from backend.ml.behavior.rule_engine import BehaviorEpisode, RuleEngine
from backend.ml.behavior.stgcn_bc.inference import STGCNBCInferer
from backend.ml.behavior.mamba_inference import MambaInferer
from backend.ml.behavior.mamba_bc_inference import MambaBCInferer

logger = logging.getLogger(__name__)


class DeployMode(str, Enum):
    """部署模式.

    ST-GCN+BC 轨:
    - SHADOW: ST-GCN+BC 影子推理 + 规则引擎返回（仅记录对比）
    - VOTE: 双轨投票融合（高置信度优先）
    - PRIMARY_STGCN: ST-GCN+BC 主 + 规则引擎备（失败降级）

    Mamba 轨:
    - MAMBA_ONLY: 仅 Mamba 推理（无边界头，纯分类）
    - MAMBA_SHADOW: Mamba 影子推理 + 规则引擎返回（对比基线）
    - MAMBA_VOTE: Mamba + 规则引擎投票融合

    Mamba+BC 轨:
    - MAMBA_BC_ONLY: 仅 Mamba+BC 推理（含边界检测）
    - MAMBA_BC_SHADOW: Mamba+BC 影子推理 + 规则引擎返回
    - MAMBA_BC_VOTE: Mamba+BC + 规则引擎投票融合

    降级:
    - RULE_ONLY: 仅规则引擎（神经网络不可用时降级）
    """

    # ST-GCN+BC 轨
    SHADOW = "shadow"
    VOTE = "vote"
    PRIMARY_STGCN = "primary_stgcn"

    # Mamba 轨
    MAMBA_ONLY = "mamba_only"
    MAMBA_SHADOW = "mamba_shadow"
    MAMBA_VOTE = "mamba_vote"

    # Mamba+BC 轨
    MAMBA_BC_ONLY = "mamba_bc_only"
    MAMBA_BC_SHADOW = "mamba_bc_shadow"
    MAMBA_BC_VOTE = "mamba_bc_vote"

    # 降级
    RULE_ONLY = "rule_only"


@dataclass
class ShadowComparison:
    """影子模式对比记录（用于评估 ST-GCN+BC 与规则引擎一致性）."""

    stgcn_count: int = 0
    rule_count: int = 0
    common_behaviors: int = 0  # 两路都识别到的行为数
    stgcn_only: int = 0         # 仅 ST-GCN+BC 识别到
    rule_only: int = 0          # 仅规则引擎识别到
    stgcn_avg_conf: float = 0.0
    rule_avg_conf: float = 0.0


class BehaviorRecognizer:
    """行为识别路由器（多轨部署入口）.

    用法:
        # shadow 模式（默认，ST-GCN+BC 轨）
        recognizer = BehaviorRecognizer(mode=DeployMode.SHADOW)
        episodes = recognizer.recognize(kpts_seq, fps=fps)

        # primary_stgcn 模式
        recognizer = BehaviorRecognizer(
            mode=DeployMode.PRIMARY_STGCN,
            stgcn_inferer=STGCNBCInferer(onnx_path="data/models/stgcn_bc/stgcn_bc_dog24.onnx"),
        )

        # mamba_only 模式（Phase 4 基线）
        recognizer = BehaviorRecognizer(
            mode=DeployMode.MAMBA_ONLY,
            mamba_inferer=MambaInferer(checkpoint_path="runs/mamba_synthetic/best.pt"),
        )

        # mamba_bc_only 模式（Phase 4 MS-Temba 集成）
        recognizer = BehaviorRecognizer(
            mode=DeployMode.MAMBA_BC_ONLY,
            mamba_bc_inferer=MambaBCInferer(checkpoint_path="runs/mamba_bc_synthetic/best.pt"),
        )

        # mamba_bc_shadow 模式（Mamba+BC 对比基线）
        recognizer = BehaviorRecognizer(
            mode=DeployMode.MAMBA_BC_SHADOW,
            mamba_bc_inferer=MambaBCInferer(onnx_path="data/models/mamba_bc/mamba_bc_dog24.onnx"),
        )
    """

    def __init__(
        self,
        mode: Union[DeployMode, str] = DeployMode.SHADOW,
        stgcn_inferer: Optional[STGCNBCInferer] = None,
        mamba_inferer: Optional[MambaInferer] = None,
        mamba_bc_inferer: Optional[MambaBCInferer] = None,
        rule_engine: Optional[RuleEngine] = None,
        vote_conf_threshold: float = 0.6,
    ):
        """初始化路由器.

        Args:
            mode: 部署模式
            stgcn_inferer: ST-GCN+BC 推理器
            mamba_inferer: Mamba 推理器
            mamba_bc_inferer: Mamba+BC 推理器
            rule_engine: 规则引擎（None=按需懒加载）
            vote_conf_threshold: VOTE 模式下置信度阈值
        """
        self.mode = mode if isinstance(mode, DeployMode) else DeployMode(mode)
        self._stgcn_inferer = stgcn_inferer
        self._mamba_inferer = mamba_inferer
        self._mamba_bc_inferer = mamba_bc_inferer
        self._rule_engine = rule_engine
        self.vote_conf_threshold = vote_conf_threshold

        # 影子模式对比记录（最近一次）
        self._last_comparison: Optional[ShadowComparison] = None

    @property
    def stgcn_inferer(self) -> Optional[STGCNBCInferer]:
        """懒加载 ST-GCN+BC 推理器."""
        return self._stgcn_inferer

    @property
    def mamba_inferer(self) -> Optional[MambaInferer]:
        """懒加载 Mamba 推理器."""
        return self._mamba_inferer

    @property
    def mamba_bc_inferer(self) -> Optional[MambaBCInferer]:
        """懒加载 Mamba+BC 推理器."""
        return self._mamba_bc_inferer

    @property
    def rule_engine(self) -> RuleEngine:
        """懒加载规则引擎."""
        if self._rule_engine is None:
            self._rule_engine = RuleEngine()
        return self._rule_engine

    def recognize(
        self,
        keypoints_sequence: np.ndarray,
        fps: float = 30.0,
        boxes: Optional[list] = None,
        person_boxes: Optional[list] = None,
        target_boxes: Optional[list] = None,
        enable_p1: bool = True,
    ) -> List[BehaviorEpisode]:
        """行为识别入口（多轨路由）."""
        if self.mode == DeployMode.RULE_ONLY:
            return self._recognize_rule(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)

        # ST-GCN+BC 轨
        if self.mode in (DeployMode.SHADOW, DeployMode.VOTE, DeployMode.PRIMARY_STGCN):
            if self._stgcn_inferer is None:
                logger.warning(f"[BehaviorRecognizer] mode={self.mode} 但 stgcn_inferer=None，降级 RULE_ONLY")
                return self._recognize_rule(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)
            if self.mode == DeployMode.SHADOW:
                return self._recognize_shadow(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)
            elif self.mode == DeployMode.VOTE:
                return self._recognize_vote(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)
            elif self.mode == DeployMode.PRIMARY_STGCN:
                return self._recognize_primary_stgcn(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)

        # Mamba 轨
        if self.mode in (DeployMode.MAMBA_ONLY, DeployMode.MAMBA_SHADOW, DeployMode.MAMBA_VOTE):
            if self._mamba_inferer is None:
                logger.warning(f"[BehaviorRecognizer] mode={self.mode} 但 mamba_inferer=None，降级 RULE_ONLY")
                return self._recognize_rule(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)
            if self.mode == DeployMode.MAMBA_ONLY:
                return self._recognize_mamba_only(keypoints_sequence, fps)
            elif self.mode == DeployMode.MAMBA_SHADOW:
                return self._recognize_mamba_shadow(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)
            elif self.mode == DeployMode.MAMBA_VOTE:
                return self._recognize_mamba_vote(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)

        # Mamba+BC 轨
        if self.mode in (DeployMode.MAMBA_BC_ONLY, DeployMode.MAMBA_BC_SHADOW, DeployMode.MAMBA_BC_VOTE):
            if self._mamba_bc_inferer is None:
                logger.warning(f"[BehaviorRecognizer] mode={self.mode} 但 mamba_bc_inferer=None，降级 RULE_ONLY")
                return self._recognize_rule(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)
            if self.mode == DeployMode.MAMBA_BC_ONLY:
                return self._recognize_mamba_bc_only(keypoints_sequence, fps)
            elif self.mode == DeployMode.MAMBA_BC_SHADOW:
                return self._recognize_mamba_bc_shadow(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)
            elif self.mode == DeployMode.MAMBA_BC_VOTE:
                return self._recognize_mamba_bc_vote(keypoints_sequence, fps, boxes, person_boxes, target_boxes, enable_p1)

        raise ValueError(f"未知部署模式: {self.mode}")

    # ====================================================================
    # 各模式实现
    # ====================================================================

    def _recognize_rule(
        self,
        kpts_seq: np.ndarray,
        fps: float,
        boxes, person_boxes, target_boxes,
        enable_p1: bool,
    ) -> List[BehaviorEpisode]:
        """仅规则引擎."""
        return self.rule_engine.recognize(
            kpts_seq, boxes=boxes, person_boxes=person_boxes,
            target_boxes=target_boxes, fps=fps, enable_p1=enable_p1,
        )

    def _recognize_shadow(
        self,
        kpts_seq: np.ndarray,
        fps: float,
        boxes, person_boxes, target_boxes,
        enable_p1: bool,
    ) -> List[BehaviorEpisode]:
        """影子模式: ST-GCN+BC 推理 + 规则引擎返回 + 记录对比."""
        # 两路并行推理
        rule_episodes = self.rule_engine.recognize(
            kpts_seq, boxes=boxes, person_boxes=person_boxes,
            target_boxes=target_boxes, fps=fps, enable_p1=enable_p1,
        )
        try:
            stgcn_episodes = self._stgcn_inferer.predict(kpts_seq, fps=fps)
        except Exception as e:
            logger.warning(
                f"[BehaviorRecognizer] ST-GCN+BC 推理失败，返回规则引擎结果: {e}"
            )
            return rule_episodes

        # 记录对比
        self._last_comparison = self._build_comparison(stgcn_episodes, rule_episodes)
        logger.info(
            f"[BehaviorRecognizer] SHADOW 对比: "
            f"STGCN={self._last_comparison.stgcn_count} "
            f"RULE={self._last_comparison.rule_count} "
            f"common={self._last_comparison.common_behaviors} "
            f"stgcn_only={self._last_comparison.stgcn_only} "
            f"rule_only={self._last_comparison.rule_only}"
        )

        # 影子模式返回规则引擎结果（ST-GCN+BC 仅记录）
        return rule_episodes

    def _recognize_vote(
        self,
        kpts_seq: np.ndarray,
        fps: float,
        boxes, person_boxes, target_boxes,
        enable_p1: bool,
    ) -> List[BehaviorEpisode]:
        """投票模式: 两路结果融合（高置信度优先）."""
        rule_episodes = self.rule_engine.recognize(
            kpts_seq, boxes=boxes, person_boxes=person_boxes,
            target_boxes=target_boxes, fps=fps, enable_p1=enable_p1,
        )
        try:
            stgcn_episodes = self._stgcn_inferer.predict(kpts_seq, fps=fps)
        except Exception as e:
            logger.warning(
                f"[BehaviorRecognizer] ST-GCN+BC 推理失败，返回规则引擎结果: {e}"
            )
            return rule_episodes

        # 投票融合：高置信度 ST-GCN+BC 优先 + 规则引擎补充
        return self._vote_merge(stgcn_episodes, rule_episodes)

    def _recognize_primary_stgcn(
        self,
        kpts_seq: np.ndarray,
        fps: float,
        boxes, person_boxes, target_boxes,
        enable_p1: bool,
    ) -> List[BehaviorEpisode]:
        """主备模式: ST-GCN+BC 主 + 规则引擎备（失败降级）."""
        try:
            return self._stgcn_inferer.predict(kpts_seq, fps=fps)
        except Exception as e:
            logger.error(
                f"[BehaviorRecognizer] ST-GCN+BC 主推理失败，降级到规则引擎: {e}",
                exc_info=True,
            )
            return self.rule_engine.recognize(
                kpts_seq, boxes=boxes, person_boxes=person_boxes,
                target_boxes=target_boxes, fps=fps, enable_p1=enable_p1,
            )

    # ====================================================================
    # Mamba 轨实现
    # ====================================================================

    def _recognize_mamba_only(
        self,
        kpts_seq: np.ndarray,
        fps: float,
    ) -> List[BehaviorEpisode]:
        """仅 Mamba 推理（无边界头，纯分类）."""
        try:
            return self._mamba_inferer.predict(kpts_seq, fps=fps)
        except Exception as e:
            logger.error(f"[BehaviorRecognizer] Mamba 推理失败: {e}", exc_info=True)
            return []

    def _recognize_mamba_shadow(
        self,
        kpts_seq: np.ndarray,
        fps: float,
        boxes, person_boxes, target_boxes,
        enable_p1: bool,
    ) -> List[BehaviorEpisode]:
        """Mamba 影子模式: Mamba 推理 + 规则引擎返回 + 记录对比."""
        rule_episodes = self.rule_engine.recognize(
            kpts_seq, boxes=boxes, person_boxes=person_boxes,
            target_boxes=target_boxes, fps=fps, enable_p1=enable_p1,
        )
        try:
            mamba_episodes = self._mamba_inferer.predict(kpts_seq, fps=fps)
        except Exception as e:
            logger.warning(f"[BehaviorRecognizer] Mamba 推理失败，返回规则引擎结果: {e}")
            return rule_episodes
        self._last_comparison = self._build_comparison(mamba_episodes, rule_episodes)
        logger.info(f"[BehaviorRecognizer] MAMBA_SHADOW 对比: MAMBA={self._last_comparison.stgcn_count} RULE={self._last_comparison.rule_count}")
        return rule_episodes

    def _recognize_mamba_vote(
        self,
        kpts_seq: np.ndarray,
        fps: float,
        boxes, person_boxes, target_boxes,
        enable_p1: bool,
    ) -> List[BehaviorEpisode]:
        """Mamba 投票模式: Mamba + 规则引擎投票融合."""
        rule_episodes = self.rule_engine.recognize(
            kpts_seq, boxes=boxes, person_boxes=person_boxes,
            target_boxes=target_boxes, fps=fps, enable_p1=enable_p1,
        )
        try:
            mamba_episodes = self._mamba_inferer.predict(kpts_seq, fps=fps)
        except Exception as e:
            logger.warning(f"[BehaviorRecognizer] Mamba 推理失败，返回规则引擎结果: {e}")
            return rule_episodes
        return self._vote_merge(mamba_episodes, rule_episodes)

    # ====================================================================
    # Mamba+BC 轨实现（边界检测增强）
    # ====================================================================

    def _recognize_mamba_bc_only(
        self,
        kpts_seq: np.ndarray,
        fps: float,
    ) -> List[BehaviorEpisode]:
        """仅 Mamba+BC 推理（含边界检测头）."""
        try:
            return self._mamba_bc_inferer.predict(kpts_seq, fps=fps)
        except Exception as e:
            logger.error(f"[BehaviorRecognizer] Mamba+BC 推理失败: {e}", exc_info=True)
            return []

    def _recognize_mamba_bc_shadow(
        self,
        kpts_seq: np.ndarray,
        fps: float,
        boxes, person_boxes, target_boxes,
        enable_p1: bool,
    ) -> List[BehaviorEpisode]:
        """Mamba+BC 影子模式: Mamba+BC 推理 + 规则引擎返回 + 记录对比."""
        rule_episodes = self.rule_engine.recognize(
            kpts_seq, boxes=boxes, person_boxes=person_boxes,
            target_boxes=target_boxes, fps=fps, enable_p1=enable_p1,
        )
        try:
            mb_episodes = self._mamba_bc_inferer.predict(kpts_seq, fps=fps)
        except Exception as e:
            logger.warning(f"[BehaviorRecognizer] Mamba+BC 推理失败，返回规则引擎结果: {e}")
            return rule_episodes
        self._last_comparison = self._build_comparison(mb_episodes, rule_episodes)
        logger.info(f"[BehaviorRecognizer] MAMBA_BC_SHADOW 对比: MAMBA_BC={self._last_comparison.stgcn_count} RULE={self._last_comparison.rule_count}")
        return rule_episodes

    def _recognize_mamba_bc_vote(
        self,
        kpts_seq: np.ndarray,
        fps: float,
        boxes, person_boxes, target_boxes,
        enable_p1: bool,
    ) -> List[BehaviorEpisode]:
        """Mamba+BC 投票模式: Mamba+BC + 规则引擎投票融合."""
        rule_episodes = self.rule_engine.recognize(
            kpts_seq, boxes=boxes, person_boxes=person_boxes,
            target_boxes=target_boxes, fps=fps, enable_p1=enable_p1,
        )
        try:
            mb_episodes = self._mamba_bc_inferer.predict(kpts_seq, fps=fps)
        except Exception as e:
            logger.warning(f"[BehaviorRecognizer] Mamba+BC 推理失败，返回规则引擎结果: {e}")
            return rule_episodes
        return self._vote_merge(mb_episodes, rule_episodes)

    # ====================================================================
    # 融合策略
    # ====================================================================

    def _vote_merge(
        self,
        stgcn_episodes: List[BehaviorEpisode],
        rule_episodes: List[BehaviorEpisode],
    ) -> List[BehaviorEpisode]:
        """投票融合.

        策略:
            - ST-GCN+BC 高置信度（>= vote_conf_threshold）的 episode 优先
            - 规则引擎补充 ST-GCN+BC 未覆盖的时段
            - 时间重叠 > 50% 的同类 episode 取高置信度
        """
        merged: List[BehaviorEpisode] = []

        # 高置信度 ST-GCN+BC episode 直接采纳
        high_conf_stgcn = [
            ep for ep in stgcn_episodes if ep.confidence >= self.vote_conf_threshold
        ]
        merged.extend(high_conf_stgcn)

        # 规则引擎补充：剔除与高置信度 ST-GCN+BC 重叠 > 50% 的
        for rule_ep in rule_episodes:
            overlap_ratio = self._max_overlap_ratio(rule_ep, high_conf_stgcn)
            if overlap_ratio < 0.5:
                merged.append(rule_ep)

        # 按时间排序
        merged.sort(key=lambda e: e.start_frame)
        return merged

    @staticmethod
    def _max_overlap_ratio(
        target: BehaviorEpisode,
        others: List[BehaviorEpisode],
    ) -> float:
        """计算 target 与 others 的最大重叠比例."""
        if not others:
            return 0.0
        target_len = target.duration_frames
        if target_len == 0:
            return 0.0
        max_ratio = 0.0
        for other in others:
            overlap_start = max(target.start_frame, other.start_frame)
            overlap_end = min(target.end_frame, other.end_frame)
            if overlap_end >= overlap_start:
                overlap_len = overlap_end - overlap_start + 1
                ratio = overlap_len / target_len
                max_ratio = max(max_ratio, ratio)
        return max_ratio

    @staticmethod
    def _build_comparison(
        stgcn_episodes: List[BehaviorEpisode],
        rule_episodes: List[BehaviorEpisode],
    ) -> ShadowComparison:
        """构建影子模式对比记录."""
        stgcn_behaviors = {ep.behavior for ep in stgcn_episodes}
        rule_behaviors = {ep.behavior for ep in rule_episodes}
        common = stgcn_behaviors & rule_behaviors
        stgcn_only = stgcn_behaviors - rule_behaviors
        rule_only = rule_behaviors - stgcn_behaviors

        stgcn_avg = float(np.mean([ep.confidence for ep in stgcn_episodes])) if stgcn_episodes else 0.0
        rule_avg = float(np.mean([ep.confidence for ep in rule_episodes])) if rule_episodes else 0.0

        return ShadowComparison(
            stgcn_count=len(stgcn_episodes),
            rule_count=len(rule_episodes),
            common_behaviors=len(common),
            stgcn_only=len(stgcn_only),
            rule_only=len(rule_only),
            stgcn_avg_conf=round(stgcn_avg, 4),
            rule_avg_conf=round(rule_avg, 4),
        )

    @property
    def last_comparison(self) -> Optional[ShadowComparison]:
        """获取最近一次影子模式对比记录（None=非 shadow 模式或未调用）."""
        return self._last_comparison


__all__ = [
    "BehaviorRecognizer",
    "DeployMode",
    "ShadowComparison",
]
