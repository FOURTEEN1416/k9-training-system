"""标注数据模型（数据飞轮基础设施）.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 2.1a
依据: dev-docs/stages/phase-2.md §2.1a

表结构:
    - annotation_tasks: 标注任务（视频 ↔ Label Studio 任务关联 + 进度追踪）
    - annotations: 标注数据（逐帧关键点 / 行为标签 / 检测框）

数据流:
    Video → AnnotationTask（LS task） → Annotation（逐帧标注）
                                          ↓
                                     模型微调数据集
"""

import enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Enum, ForeignKey, Integer, String, Text, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.handler import Handler
    from backend.app.models.video import Video


class AnnotationTaskStatus(str, enum.Enum):
    """标注任务状态机。"""

    PENDING = "pending"          # 已创建，待标注
    IN_PROGRESS = "in_progress"  # 标注中
    COMPLETED = "completed"      # 标注完成
    FAILED = "failed"            # 标注失败


class AnnotationType(str, enum.Enum):
    """标注类型。"""

    KEYPOINT = "keypoint"    # 关键点标注（24 个 Dog-Pose 关键点）
    BEHAVIOR = "behavior"    # 行为标签（16 类 P0+P1）
    BBOX = "bbox"            # 检测框（犬/人/目标）


class AnnotationSource(str, enum.Enum):
    """标注来源。"""

    HUMAN = "human"              # 人工标注
    MODEL_PRELABEL = "prelabel"  # 模型预标注（YOLO26-pose）
    IMPORTED = "imported"        # 外部导入（如 Animal Kingdom）


class AnnotationTask(TimestampMixin, Base):
    """标注任务表。

    一个视频对应一个标注任务，关联 Label Studio task。
    """

    __tablename__ = "annotation_tasks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    handler_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("handlers.id", ondelete="SET NULL"), nullable=True, index=True,
        comment="分配给的训导员",
    )

    # Label Studio 关联
    ls_project_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Label Studio project ID"
    )
    ls_task_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Label Studio task ID"
    )

    # 标注进度
    status: Mapped[AnnotationTaskStatus] = mapped_column(
        Enum(AnnotationTaskStatus, name="annotation_task_status"),
        nullable=False,
        default=AnnotationTaskStatus.PENDING,
        index=True,
    )
    total_frames: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="视频总帧数"
    )
    annotated_frames: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="已标注帧数"
    )

    # 标注类型配置（JSON: ["keypoint", "behavior"]）
    annotation_types: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True, comment="标注类型列表"
    )

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 关系
    video: Mapped["Video"] = relationship("Video", back_populates="annotation_tasks")
    handler: Mapped[Optional["Handler"]] = relationship("Handler", back_populates="annotation_tasks")
    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="task", cascade="all, delete-orphan"
    )

    @property
    def progress(self) -> float:
        """标注进度比例 (0.0 - 1.0)。"""
        if not self.total_frames or self.total_frames == 0:
            return 0.0
        return min(1.0, self.annotated_frames / self.total_frames)

    def __repr__(self) -> str:
        return (
            f"<AnnotationTask id={self.id} video_id={self.video_id} "
            f"status={self.status} progress={self.progress:.0%}>"
        )


class Annotation(TimestampMixin, Base):
    """标注数据表（逐帧/逐片段）。

    每条记录是一帧的一个标注（关键点集合 / 行为标签 / 检测框）。
    """

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(
        ForeignKey("annotation_tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # 帧索引（关键点标注用），行为标注可为 NULL（片段级）
    frame_idx: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, index=True, comment="帧索引（行为片段可为 NULL）"
    )

    # 标注类型
    annotation_type: Mapped[AnnotationType] = mapped_column(
        Enum(AnnotationType, name="annotation_type"), nullable=False, index=True
    )

    # 标注来源
    source: Mapped[AnnotationSource] = mapped_column(
        Enum(AnnotationSource, name="annotation_source"),
        nullable=False,
        default=AnnotationSource.HUMAN,
        index=True,
    )

    # 标注数据（JSON）:
    #   keypoint: [{"x": float, "y": float, "conf": float, "label": str}, ...]（24 点）
    #   behavior: {"behavior": str, "start_frame": int, "end_frame": int}
    #   bbox:     {"x": float, "y": float, "w": float, "h": float, "label": str}
    data_json: Mapped[dict] = mapped_column(
        JSON, nullable=False, comment="标注数据（格式依 annotation_type 而定）"
    )

    # 模型预标注置信度（仅 source=prelabel 时有效）
    confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True, comment="模型预标注置信度"
    )

    # 关系
    task: Mapped["AnnotationTask"] = relationship("AnnotationTask", back_populates="annotations")
    video: Mapped["Video"] = relationship("Video", back_populates="annotations")

    def __repr__(self) -> str:
        return (
            f"<Annotation id={self.id} task_id={self.task_id} "
            f"type={self.annotation_type} frame={self.frame_idx} source={self.source}>"
        )
