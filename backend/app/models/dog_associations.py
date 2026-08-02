"""多对多关联表（Phase 3.6a RBAC）。

Owner: 后端开发
Phase: 3.6a

两张关联表:
1. dog_handler_association: 一只犬可分配多名训导员（主训+副训）
2. dog_base_association: 一只犬可临时进入其他基地（跨基地协作）
"""
import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base

if TYPE_CHECKING:
    from backend.app.models.base_entity import Base_
    from backend.app.models.dog import Dog
    from backend.app.models.handler import Handler


class DogHandlerRole(str, enum.Enum):
    """训导员分配角色。"""

    PRIMARY = "primary"  # 主训导员
    SECONDARY = "secondary"  # 副训导员


class DogBaseAccess(str, enum.Enum):
    """犬只基地访问类型。"""

    PERMANENT = "permanent"  # 永久访问（主基地）
    TEMPORARY = "temporary"  # 临时访问（联合训练）


class DogHandlerAssociation(Base):
    """犬只-训导员关联表（多对多）。

    用例:
        - 警犬配备主训导员 + 副训导员
        - 交接期一只犬同时关联两人
        - 副训导员对犬只拥有同主训导员相同的资源级权限
    """

    __tablename__ = "dog_handler_association"

    dog_id: Mapped[int] = mapped_column(
        ForeignKey("dogs.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    handler_id: Mapped[int] = mapped_column(
        ForeignKey("handlers.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    assignment_role: Mapped[DogHandlerRole] = mapped_column(
        Enum(DogHandlerRole, name="dog_handler_role"),
        nullable=False,
        default=DogHandlerRole.PRIMARY,
        comment="主训导员/副训导员",
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    notes: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # 关系
    dog: Mapped["Dog"] = relationship("Dog", backref="handler_associations")
    handler: Mapped["Handler"] = relationship("Handler", backref="dog_associations")

    def __repr__(self) -> str:
        return (
            f"<DogHandlerAssociation dog_id={self.dog_id} "
            f"handler_id={self.handler_id} role={self.assignment_role}>"
        )


class DogBaseAssociation(Base):
    """犬只-基地关联表（多对多，跨基地协作）。

    用例:
        - 联合训练期间一只犬可临时进入其他基地
        - 临时访问有过期时间，到期自动失效
        - 基地管理员可查看本基地临时访问的犬只
    """

    __tablename__ = "dog_base_association"

    dog_id: Mapped[int] = mapped_column(
        ForeignKey("dogs.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    base_id: Mapped[int] = mapped_column(
        ForeignKey("bases.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    access_type: Mapped[DogBaseAccess] = mapped_column(
        Enum(DogBaseAccess, name="dog_base_access"),
        nullable=False,
        default=DogBaseAccess.PERMANENT,
    )
    granted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="临时访问过期时间"
    )

    # 关系
    dog: Mapped["Dog"] = relationship("Dog", backref="base_associations")
    base: Mapped["Base_"] = relationship("Base_", backref="dog_associations")

    def __repr__(self) -> str:
        return (
            f"<DogBaseAssociation dog_id={self.dog_id} "
            f"base_id={self.base_id} type={self.access_type}>"
        )
