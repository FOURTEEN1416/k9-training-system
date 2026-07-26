"""训练会话。"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.dog import Dog
    from backend.app.models.handler import Handler
    from backend.app.models.video import Video


class ScoringStandard(str, enum.Enum):
    """评分标准（三大标准 + 自定义）。"""

    GA_T = "GA-T"  # 公安标准
    USPCA = "USPCA"  # 美国警犬认证
    FCI_IGP = "FCI-IGP"  # FCI 国际工作犬
    CUSTOM = "CUSTOM"


class TrainingSession(TimestampMixin, Base):
    """训练会话表。

    一次训练会话可能包含多段视频。
    """

    __tablename__ = "training_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    dog_id: Mapped[int] = mapped_column(
        ForeignKey("dogs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    handler_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("handlers.id", ondelete="SET NULL"), nullable=True, index=True
    )
    session_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, comment="训练时间"
    )
    location: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    standard: Mapped[ScoringStandard] = mapped_column(
        Enum(ScoringStandard, name="scoring_standard"),
        nullable=False,
        default=ScoringStandard.GA_T,
        comment="评分标准",
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 关系
    dog: Mapped["Dog"] = relationship("Dog", back_populates="sessions")
    handler: Mapped[Optional["Handler"]] = relationship("Handler", back_populates="sessions")
    videos: Mapped[list["Video"]] = relationship(
        "Video", back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<TrainingSession id={self.id} dog_id={self.dog_id} date={self.session_date}>"
