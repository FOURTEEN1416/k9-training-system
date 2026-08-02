"""基地/多租户隔离单元。

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 3.6a RBAC 多租户基础表

每个基地是一个独立的数据隔离单元：
- 训导员只能访问本基地的犬只/视频/评分
- 基地管理员可管理本基地所有数据
- 超管（admin）可跨基地访问
- 跨基地协作通过 dog_base_association 表授权
"""
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.dog import Dog
    from backend.app.models.handler import Handler


class Base_(TimestampMixin, Base):
    """基地表（多租户隔离单位）。

    命名说明：使用 Base_ 后缀避免与 SQLAlchemy DeclarativeBase 冲突。
    Python 类引用为 Base_，DB 表名为 bases。
    """

    __tablename__ = "bases"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, comment="基地名称"
    )
    code: Mapped[str] = mapped_column(
        String(32), nullable=False, unique=True, index=True, comment="基地代号（唯一）"
    )
    description: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, comment="是否启用"
    )

    # 关系
    handlers: Mapped[list["Handler"]] = relationship("Handler", back_populates="base")
    dogs: Mapped[list["Dog"]] = relationship("Dog", back_populates="home_base")

    def __repr__(self) -> str:
        return f"<Base_ id={self.id} code={self.code!r} name={self.name!r}>"
