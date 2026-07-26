"""评分结果。"""

from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin
from backend.app.models.training_session import ScoringStandard

if TYPE_CHECKING:
    from backend.app.models.video import Video


class Score(TimestampMixin, Base):
    """评分结果表。

    7 维评分：
    - accuracy（动作准确度）
    - response_latency（响应延迟）
    - duration（保持时长）
    - search_efficiency（搜索效率）
    - attention（注意力）
    - courage（胆量欲望）
    - gait_quality（步态质量）

    Phase 1 仅用 3 维（accuracy/duration/attention）+ GA-T 标准。
    Phase 2 加 response_latency + search_efficiency + USPCA。
    Phase 3 加 courage + gait_quality + FCI-IGP。

    每维 ∈ [0, 100]，overall 为加权平均。
    """

    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )

    standard: Mapped[ScoringStandard] = mapped_column(
        Enum(ScoringStandard, name="scoring_standard"),
        nullable=False,
        comment="评分标准",
    )

    # 7 维评分（每维 [0, 100]，未启用维度为 NULL）
    accuracy: Mapped[float] = mapped_column(Float, nullable=False, comment="动作准确度")
    response_latency: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="响应延迟（Phase 2）"
    )
    duration: Mapped[float] = mapped_column(Float, nullable=False, comment="保持时长")
    search_efficiency: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="搜索效率（Phase 2）"
    )
    attention: Mapped[float] = mapped_column(Float, nullable=False, comment="注意力")
    courage: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="胆量欲望（Phase 3）"
    )
    gait_quality: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="步态质量（Phase 3）"
    )

    # 综合分（加权平均，[0, 100]）
    overall: Mapped[float] = mapped_column(Float, nullable=False, comment="综合分")

    # 评分引擎元数据
    scoring_engine_version: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="评分引擎版本"
    )
    notes: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # 关系
    video: Mapped["Video"] = relationship("Video", back_populates="scores")

    def __repr__(self) -> str:
        return f"<Score id={self.id} video_id={self.video_id} overall={self.overall:.1f}>"
