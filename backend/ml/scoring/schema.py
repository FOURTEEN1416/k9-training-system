"""评分引擎 Pydantic 数据模型.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.4b（Phase 3.4b 扩展 FCI-IGP 7 维 + DQ + 5 级评级）
依据: dev-docs/complex-features/scoring-card-schema.md §5
      dev-docs/research/RESEARCH_FCI_IGP_STANDARD.md §6（FCI-IGP YAML 设计）

数据流:
    ScoringContext (输入) → ScoringEngine.evaluate() → ScoringResult (输出)
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# 场景标识
# - puppy_selection: 选育 3 维（Phase 1）
# - obedience_trial: 科目测评 5 维（Phase 1）
# - working_dog_trial: 工作犬综合训练 7 维（Phase 2.2e，16 行为）
# - uspca_patrol: USPCA PDI 5 维（Phase 2.3b，USPCA 标准映射）
# - fci_igp: FCI-IGP 国际工作犬 7 维 + DQ + 5 级评级（Phase 3.4b，FCI 2025 标准）
Scene = Literal[
    "puppy_selection",
    "obedience_trial",
    "working_dog_trial",
    "uspca_patrol",
    "fci_igp",
]

# 聚合方式
Aggregation = Literal["weighted_sum", "max", "min"]

# FCI-IGP 5 级评级（Phase 3.4b，RESEARCH_FCI_IGP_STANDARD.md §1.4）
# Excellent（优秀 96%+）/ Very Good（很好 90%+）/ Good（良好 80%+）
# Satisfactory（合格 70%+）/ Insufficient（不合格 <70%）
FciRating = Literal[
    "Excellent", "Very Good", "Good", "Satisfactory", "Insufficient",
]


class RuleSpec(BaseModel):
    """YAML 评分卡中的单条规则定义。"""

    id: str = Field(..., description="规则标识（snake_case）")
    condition: str = Field(..., description="条件表达式（见 schema §3）")
    score: int = Field(..., ge=0, le=100, description="命中分数 0-100")
    label: str = Field(..., description="命中标签（高/中/低 等）")


class DefaultSpec(BaseModel):
    """所有规则未命中时的默认分。"""

    score: int = Field(..., ge=0, le=100)
    label: str


class DimensionSpec(BaseModel):
    """YAML 评分卡中的单个评分维度。"""

    id: str = Field(..., description="维度标识（snake_case）")
    name: str = Field(..., description="维度名称（中文可读）")
    weight: float = Field(..., ge=0.0, le=1.0, description="权重 0-1")
    rules: list[RuleSpec] = Field(..., min_length=1, description="规则列表（按顺序匹配）")
    default: DefaultSpec | None = Field(None, description="可选默认分")

    @field_validator("rules")
    @classmethod
    def rules_must_have_default_or_true_rule(cls, v: list[RuleSpec]) -> list[RuleSpec]:
        """规则列表最后一条件应为 'True'（兜底）或显式提供 default。"""
        has_true = any(r.condition.strip() == "True" for r in v)
        # default 在 DimensionSpec 层校验，这里仅校验规则非空
        return v


class ThresholdsSpec(BaseModel):
    """总分阈值。"""

    pass_: int = Field(70, alias="pass", ge=0, le=100, description="合格线")
    borderline: int = Field(60, ge=0, le=100, description="基本合格线")
    fail: int = Field(0, ge=0, le=100, description="淘汰线")

    @field_validator("borderline")
    @classmethod
    def borderline_between_fail_and_pass(cls, v: int, info) -> int:
        pass_ = info.data.get("pass_", 70)
        fail = info.data.get("fail", 0)
        if not (fail <= v <= pass_):
            raise ValueError(f"borderline({v}) 必须在 fail({fail}) 和 pass({pass_}) 之间")
        return v


class DisqualificationSpec(BaseModel):
    """FCI-IGP DQ 硬约束（Phase 3.4b）.

    任一 DQ 规则命中 → 整段评分强制 fail（verdict="fail"）。
    依据 RESEARCH_FCI_IGP_STANDARD.md §3.2：
        - 枪怯 (gun-shy) = DQ + 已得分清零
        - 不放口 (no release) = DQ
        - 衔取 3 口令不吐 = DQ（不服从）
    """

    id: str = Field(..., description="DQ 规则标识（snake_case）")
    condition: str = Field(..., description="DQ 触发条件表达式")
    action: Literal["force_fail"] = Field(
        "force_fail", description="DQ 动作（仅支持 force_fail）"
    )
    label: str = Field(..., description="DQ 标签（人类可读）")


class BehaviorMappingEntry(BaseModel):
    """行为 → 评分维度映射条目（Phase 3.4b）.

    每个行为映射到多个评分维度 + 权重调整 + 可选 IGP 阶段标签。
    """

    dimensions: list[str] = Field(
        ..., description="该行为映射到的评分维度 ID 列表"
    )
    weight: float = Field(1.0, ge=0.0, description="行为权重（影响维度内汇总）")
    label: str = Field(..., description="行为中文名")
    igp_phase: Literal["A", "B", "C", "ALL"] | None = Field(
        None, description="FCI-IGP 阶段（A 追踪 / B 服从 / C 护卫 / ALL 全阶段）"
    )


class ScoringCardSpec(BaseModel):
    """完整评分卡 YAML Schema（顶层）。"""

    name: str
    version: str
    scene: Scene
    description: str | None = None
    dimensions: list[DimensionSpec] = Field(..., min_length=1)
    thresholds: ThresholdsSpec = Field(default_factory=lambda: ThresholdsSpec.model_validate({"pass": 70, "borderline": 60, "fail": 0}))
    aggregation: Aggregation = "weighted_sum"
    # Phase 3.4b: FCI-IGP 扩展字段（其它场景可选）
    igp_level: Literal["IGP1", "IGP2", "IGP3"] | None = Field(
        None, description="FCI-IGP 等级（仅 fci_igp 场景）"
    )
    disqualifications: list[DisqualificationSpec] = Field(
        default_factory=list, description="DQ 硬约束列表（fci_igp 场景）"
    )
    behavior_mapping: dict[str, BehaviorMappingEntry] = Field(
        default_factory=dict, description="行为 → 评分维度映射"
    )

    @field_validator("dimensions")
    @classmethod
    def weights_sum_to_one(cls, v: list[DimensionSpec]) -> list[DimensionSpec]:
        total = sum(d.weight for d in v)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"所有维度权重之和应为 1.0，实际为 {total:.4f}")
        return v

    @field_validator("disqualifications")
    @classmethod
    def disqualifications_only_for_fci_igp(
        cls, v: list[DisqualificationSpec], info
    ) -> list[DisqualificationSpec]:
        scene = info.data.get("scene")
        if v and scene != "fci_igp":
            raise ValueError(
                f"disqualifications 仅支持 fci_igp 场景，当前 scene={scene}"
            )
        return v

    @field_validator("igp_level")
    @classmethod
    def igp_level_only_for_fci_igp(
        cls, v: str | None, info
    ) -> str | None:
        scene = info.data.get("scene")
        if v is not None and scene != "fci_igp":
            raise ValueError(
                f"igp_level 仅支持 fci_igp 场景，当前 scene={scene}"
            )
        return v


class ScoringContext(BaseModel):
    """评分输入上下文。"""

    signals: dict[str, float | int | str | bool] = Field(
        ..., description="信号字典（来自姿态/行为识别输出）"
    )
    scene: Scene
    meta: dict[str, Any] = Field(default_factory=dict, description="可选元数据（video_id 等）")


class DimensionResult(BaseModel):
    """单维度评分结果。"""

    dimension_id: str
    dimension_name: str
    weight: float
    score: int
    label: str
    hit_rule_id: str | None = Field(None, description="命中的规则 ID（None 表示用 default）")


class ScoringResult(BaseModel):
    """评分输出结果。"""

    total_score: float = Field(..., ge=0.0, le=100.0, description="总分 0-100")
    dimension_scores: dict[str, int] = Field(..., description="各维度得分")
    dimension_labels: dict[str, str] = Field(..., description="各维度标签")
    dimension_details: list[DimensionResult] = Field(..., description="各维度详情")
    passed: bool = Field(..., description="是否合格（total >= thresholds.pass）")
    verdict: Literal["pass", "borderline", "fail"] = Field(..., description="判定结果")
    explanation: list[str] = Field(default_factory=list, description="人类可读的评分说明")
    scene: Scene
    card_name: str
    card_version: str
    # Phase 3.4b: FCI-IGP 扩展字段
    disqualified: bool = Field(
        False, description="是否触发 DQ 硬约束（fci_igp 场景）"
    )
    disqualification_hit: str | None = Field(
        None, description="命中的 DQ 规则 ID（无则 None）"
    )
    rating: FciRating | None = Field(
        None, description="FCI-IGP 5 级评级（仅 fci_igp 场景）"
    )
