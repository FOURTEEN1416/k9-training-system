"""24 关键点序列。"""

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.video import Video


# YOLO26-pose Dog-Pose 24 关键点 schema（与 data/dog-pose.yaml kpt_names 一致）:
# 0-5: 左侧肢（前左爪/前左膝/前左肘 + 后左爪/后左膝/后左肘）
# 6-11: 右侧肢（前右爪/前右膝/前右肘 + 后右爪/后右膝/后右肘）
# 12-13: 尾（尾根/尾尖）
# 14-15: 耳根（左/右）
# 16-17: 头部中线（鼻/下巴）
# 18-19: 耳尖（左/右）
# 20-21: 眼（左/右）
# 22: 鬐甲（withers，肩峰）
# 23: 喉咙（throat）
NUM_KEYPOINTS = 24


class Keypoint(TimestampMixin, Base):
    """逐帧 24 关键点序列表。

    存储策略：每帧一行，keypoints_json 为 [[x, y, conf], ...] 长度 24。
    Phase 2 评估后可改用 columnar 存储（TimescaleDB / Parquet）以提升查询性能。
    """

    __tablename__ = "keypoints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame_idx: Mapped[int] = mapped_column(Integer, nullable=False, comment="帧序号")
    frame_time_sec: Mapped[float] = mapped_column(
        Float, nullable=False, comment="帧时间（秒）"
    )

    # 24 个关键点：[[x, y, conf], ...] × 24
    keypoints_json: Mapped[list] = mapped_column(
        JSON, nullable=False, comment="24 个 [x, y, confidence]"
    )

    # 检测置信度（整体）
    detection_confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    # 关系
    video: Mapped["Video"] = relationship("Video", back_populates="keypoints")

    def __repr__(self) -> str:
        return f"<Keypoint id={self.id} video_id={self.video_id} frame={self.frame_idx}>"
