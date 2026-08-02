"""训导员/用户档案。"""

import enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.annotation import AnnotationTask
    from backend.app.models.base_entity import Base_
    from backend.app.models.dog import Dog
    from backend.app.models.training_session import TrainingSession
    from backend.app.models.video import Video


class UserRole(str, enum.Enum):
    """用户角色枚举（Phase 3.6a 扩展 VIEWER）。"""

    HANDLER = "handler"  # 训导员：仅自有犬只/视频读写
    MANAGER = "manager"  # 基地管理员：本基地全部读写
    RESEARCHER = "researcher"  # 科研人员：本基地只读 + 标注
    ADMIN = "admin"  # 系统管理员：全局读写
    VIEWER = "viewer"  # 查看者：全基地只读（Phase 3.6a 新增）


# 角色权限层级（数值越大权限越大，用于 hierarchy check）
ROLE_HIERARCHY: dict[UserRole, int] = {
    UserRole.VIEWER: 10,
    UserRole.HANDLER: 20,
    UserRole.RESEARCHER: 30,
    UserRole.MANAGER: 40,
    UserRole.ADMIN: 100,
}


class Handler(TimestampMixin, Base):
    """训导员/用户档案表。

    Phase 0 仅建立结构，鉴权留待 Phase 1（OAuth2/JWT）。
    Phase 3.6a 扩展：password_hash + base_id + is_superuser + VIEWER 角色。
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

    # === Phase 3.6a RBAC 扩展字段 ===
    password_hash: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="bcrypt 哈希（passlib CryptContext）；NULL = 未启用登录",
    )
    base_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("bases.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="所属基地 ID（NULL = 全局账号）",
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="超管标志（绕过所有 RBAC 检查）",
    )

    # 关系
    base: Mapped[Optional["Base_"]] = relationship("Base_", back_populates="handlers")
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
