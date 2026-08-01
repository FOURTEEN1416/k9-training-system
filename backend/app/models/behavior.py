"""行为识别结果。"""

import enum
from typing import TYPE_CHECKING

from sqlalchemy import Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from backend.app.models.behavior_vector import BehaviorVector
    from backend.app.models.video import Video


# 22 种行为分类（依据 RESEARCH_STANDARDS.md §4.1）
# P0 基础 8 类（Phase 1，与 rule_engine.P0_BEHAVIORS 对齐）
# P1 训练 8 类（Phase 2，与 rule_engine.P1_BEHAVIORS 对齐）
# P2 高级 6 类（Phase 3）
class BehaviorClass(str, enum.Enum):
    """22 种行为分类。"""

    # P0 基础 8 类（Phase 1 已验收，rule_engine 输出）
    SIT = "sit"  # 坐
    DOWN = "down"  # 卧
    STAND = "stand"  # 立
    HEEL = "heel"  # 随行
    SIT_UP = "sit_up"  # 坐立
    STAY = "stay"  # 停留
    BARK = "bark"  # 吠叫
    BITE = "bite"  # 咬（嘴部活跃动作）

    # P1 训练专项 8 类（Phase 2，依据 RESEARCH_STANDARDS.md §4.1）
    TRACK = "track"  # 追踪
    ALERT_SIT = "alert_sit"  # 示警坐
    ALERT_DOWN = "alert_down"  # 示警卧
    APPREHEND = "apprehend"  # 扑咬（USPCA Apprehension，区别于 P0 BITE）
    ESCORT = "escort"  # 押解
    OBSTACLE = "obstacle"  # 障碍穿越
    RECALL = "recall"  # 返回
    WATCH = "watch"  # 警戒

    # P2 高级评估 6 类（Phase 3，依据 RESEARCH_STANDARDS.md §4.1）
    FOOD_DRIVE = "food_drive"  # 食物欲望
    COURAGE = "courage"  # 胆量测试
    ATTENTION = "attention"  # 注意力测试
    SEARCH_EFFICIENCY = "search_efficiency"  # 搜索效率
    IMPULSE_CONTROL = "impulse_control"  # 冲动控制
    GAIT = "gait"  # 步态分析


class BehaviorDetector(str, enum.Enum):
    """行为检测器类型。"""

    RULE = "rule"  # 规则引擎
    POSEC3D = "posec3d"  # PoseC3D
    STGCN_BC = "stgcn_bc"  # ST-GCN+BC


class Behavior(TimestampMixin, Base):
    """行为识别结果表。

    每行表示一个行为区间（一段视频中检测到的同类行为片段）。
    """

    __tablename__ = "behaviors"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, index=True
    )

    behavior_class: Mapped[BehaviorClass] = mapped_column(
        Enum(BehaviorClass, name="behavior_class"),
        nullable=False,
        index=True,
        comment="行为类别",
    )
    detector: Mapped[BehaviorDetector] = mapped_column(
        Enum(BehaviorDetector, name="behavior_detector"),
        nullable=False,
        comment="检测器",
    )

    # 帧区间
    start_frame: Mapped[int] = mapped_column(Integer, nullable=False)
    end_frame: Mapped[int] = mapped_column(Integer, nullable=False)

    # 时间区间（秒）
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)

    # 置信度 [0, 1]
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # 关系
    video: Mapped["Video"] = relationship("Video", back_populates="behaviors")
    vector: Mapped["BehaviorVector | None"] = relationship(
        "BehaviorVector", back_populates="behavior", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Behavior id={self.id} video_id={self.video_id} "
            f"class={self.behavior_class} conf={self.confidence:.2f}>"
        )
