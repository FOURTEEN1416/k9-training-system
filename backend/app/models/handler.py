"""训导员/用户档案。"""

import enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.annotation import AnnotationTask
    from backend.app.models.dog import Dog
    from backend.app.models.training_session import TrainingSession
    from backend.app.models.video import Video


class UserRole(str, enum.Enum):
    """用户角色枚举。"""

    HANDLER = "handler"  # 训导员
    MANAGER = "manager"  # 管理层
    RESEARCHER = "researcher"  # 科研人员
    ADMIN = "admin"  # 系统管理员


class Handler(TimestampMixin, Base):
    """训导员/用户档案表。

    Phase 0 仅建立结构，鉴权留待 Phase 1（OAuth2/JWT）。
    """

    __tablename__ = "handlers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment="姓名")
    email: Mapped[Optional[str]] = mapped_column(
        String(128), unique=True, nullable=True, index=True, comment="邮箱"
    )
    phone: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, comment="电话")
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"),
        nullable=False,
        default=UserRole.HANDLER,
        comment="角色",
    )
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    # 关系
    dogs: Mapped[list["Dog"]] = relationship(
        "Dog", back_populates="handler", cascade="all, delete-orphan"
    )
    sessions: Mapped[list["TrainingSession"]] = relationship(
        "TrainingSession", back_populates="handler"
    )
    videos: Mapped[list["Video"]] = relationship("Video", back_populates="handler")
    annotation_tasks: Mapped[list["AnnotationTask"]] = relationship(
        "AnnotationTask", back_populates="handler"
    )

    def __repr__(self) -> str:
        return f"<Handler id={self.id} name={self.name!r} role={self.role}>"
