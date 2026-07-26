"""行为向量（pgvector）。"""

from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.behavior import Behavior


# PoseC3D 默认输出 26 维骨架特征向量
# ST-GCN+BC 输出 256 维图嵌入向量
# 这里使用可变维度（最大 512），实际由模型决定
BEHAVIOR_VECTOR_DIM = 256


class BehaviorVector(TimestampMixin, Base):
    """行为嵌入向量表（pgvector）。

    用于行为相似检索、聚类、异常检测。
    Phase 1 先用 PoseC3D 26 维；Phase 3 升级 ST-GCN+BC 256 维。
    """

    __tablename__ = "behavior_vectors"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    behavior_id: Mapped[int] = mapped_column(
        ForeignKey("behaviors.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    dim: Mapped[int] = mapped_column(Integer, nullable=False, comment="向量维度")
    embedding: Mapped[list[float]] = mapped_column(
        Vector(BEHAVIOR_VECTOR_DIM), nullable=False, comment="行为嵌入向量"
    )

    # 关系
    behavior: Mapped["Behavior"] = relationship("Behavior", back_populates="vector")

    def __repr__(self) -> str:
        return f"<BehaviorVector id={self.id} behavior_id={self.behavior_id} dim={self.dim}>"
