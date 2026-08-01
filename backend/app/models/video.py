"""训练视频元数据。"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.annotation import Annotation, AnnotationTask
    from backend.app.models.behavior import Behavior
    from backend.app.models.dog import Dog
    from backend.app.models.handler import Handler
    from backend.app.models.keypoint import Keypoint
    from backend.app.models.score import Score
    from backend.app.models.training_session import TrainingSession


class VideoStatus(str, enum.Enum):
    """视频处理状态机。"""

    UPLOADED = "uploaded"  # 已上传，待处理
    PROCESSING = "processing"  # 推理中
    COMPLETED = "completed"  # 推理完成
    FAILED = "failed"  # 推理失败


# 测试场景（与评分卡 scene 一致）
SCENE_PUPPY_SELECTION = "puppy_selection"
SCENE_OBEDIENCE_TRIAL = "obedience_trial"
SCENE_WORKING_DOG_TRIAL = "working_dog_trial"
SCENE_USPCA_PATROL = "uspca_patrol"
VALID_SCENES = (
    SCENE_PUPPY_SELECTION,
    SCENE_OBEDIENCE_TRIAL,
    SCENE_WORKING_DOG_TRIAL,
    SCENE_USPCA_PATROL,
)


class Video(TimestampMixin, Base):
    """训练视频元数据表。

    实际视频文件存储在 `data/uploads/`，数据库只存元数据。
    """

    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    dog_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("dogs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    handler_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("handlers.id", ondelete="SET NULL"), nullable=True, index=True
    )

    original_filename: Mapped[str] = mapped_column(String(256), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False, comment="相对 upload_dir 的路径")
    storage_filename: Mapped[str] = mapped_column(String(128), nullable=False, comment="存储文件名（uuid + 扩展名）")

    # 视频技术元数据
    duration_sec: Mapped[Optional[float]] = mapped_column(nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # 处理状态
    status: Mapped[VideoStatus] = mapped_column(
        Enum(VideoStatus, name="video_status"),
        nullable=False,
        default=VideoStatus.UPLOADED,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # 测试场景（puppy_selection / obedience_trial）
    # 决定走哪条推理 + 评分管线
    scene: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=SCENE_OBEDIENCE_TRIAL,
        comment="测试场景: puppy_selection / obedience_trial",
    )

    # PDF 报告相对路径（相对 reports_dir，完成后写入）
    report_path: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True, comment="PDF 报告相对 reports_dir 的路径"
    )

    # 时间戳
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default="now()"
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # 关系
    session: Mapped[Optional["TrainingSession"]] = relationship("TrainingSession", back_populates="videos")
    dog: Mapped[Optional["Dog"]] = relationship("Dog", back_populates="videos")
    handler: Mapped[Optional["Handler"]] = relationship("Handler", back_populates="videos")
    keypoints: Mapped[list["Keypoint"]] = relationship(
        "Keypoint", back_populates="video", cascade="all, delete-orphan"
    )
    behaviors: Mapped[list["Behavior"]] = relationship(
        "Behavior", back_populates="video", cascade="all, delete-orphan"
    )
    scores: Mapped[list["Score"]] = relationship(
        "Score", back_populates="video", cascade="all, delete-orphan"
    )
    annotation_tasks: Mapped[list["AnnotationTask"]] = relationship(
        "AnnotationTask", back_populates="video", cascade="all, delete-orphan"
    )
    annotations: Mapped[list["Annotation"]] = relationship(
        "Annotation", back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Video id={self.id} status={self.status} filename={self.original_filename!r}>"
