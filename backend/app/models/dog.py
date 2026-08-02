"""犬只档案。"""

import enum
from datetime import date
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Date, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.base_entity import Base_
    from backend.app.models.handler import Handler
    from backend.app.models.training_session import TrainingSession
    from backend.app.models.video import Video


class Gender(str, enum.Enum):
    MALE = "male"
    FEMALE = "female"


class TrainingStage(str, enum.Enum):
    """训练阶段（对应 P0/P1/P2 行为分类阶段）。"""

    P0 = "P0"  # 基础 8 类
    P1 = "P1"  # 训练 8 类
    P2 = "P2"  # 高级 6 类


class Dog(TimestampMixin, Base):
    """犬只档案表。

    Phase 3.6a 扩展：home_base_id 用于多租户隔离。
    """

    __tablename__ = "dogs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment="犬名")
    breed: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, comment="品种（马犬/昆明犬/德牧等）"
    )
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True, comment="出生日期")
    gender: Mapped[Optional[Gender]] = mapped_column(
        Enum(Gender, name="dog_gender"), nullable=True
    )
    chip_id: Mapped[Optional[str]] = mapped_column(
        String(64), unique=True, nullable=True, index=True, comment="芯片号"
    )
    handler_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("handlers.id", ondelete="SET NULL"), nullable=True, index=True
    )
    training_stage: Mapped[TrainingStage] = mapped_column(
        Enum(TrainingStage, name="training_stage"),
        nullable=False,
        default=TrainingStage.P0,
    )
    notes: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # === Phase 3.6a 多租户字段 ===
    home_base_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("bases.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="犬只主基地 ID",
    )

    # 关系
    handler: Mapped[Optional["Handler"]] = relationship("Handler", back_populates="dogs")
    home_base: Mapped[Optional["Base_"]] = relationship("Base_", back_populates="dogs")
    sessions: Mapped[list["TrainingSession"]] = relationship(
        "TrainingSession", back_populates="dog"
    )
    videos: Mapped[list["Video"]] = relationship("Video", back_populates="dog")

    def __repr__(self) -> str:
        return f"<Dog id={self.id} name={self.name!r} breed={self.breed}>"
