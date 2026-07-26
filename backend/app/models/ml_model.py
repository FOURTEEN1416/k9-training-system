"""ML 模型版本管理。"""

import enum
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, Enum, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.models.base import Base, TimestampMixin


class ModelType(str, enum.Enum):
    """模型类型。"""

    POSE = "pose"  # 姿态检测（YOLO26-pose）
    BEHAVIOR = "behavior"  # 行为识别（PoseC3D/ST-GCN+BC）
    SCORING = "scoring"  # 评分引擎


class ModelFramework(str, enum.Enum):
    """模型框架。"""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class MLModel(TimestampMixin, Base):
    """ML 模型版本表。

    Phase 0 仅建立结构；Phase 2 启用模型 A/B 对比与切换。
    """

    __tablename__ = "ml_models"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False, comment="模型名称")
    version: Mapped[str] = mapped_column(String(32), nullable=False, comment="语义版本号")
    type: Mapped[ModelType] = mapped_column(
        Enum(ModelType, name="model_type"), nullable=False, index=True
    )
    framework: Mapped[ModelFramework] = mapped_column(
        Enum(ModelFramework, name="model_framework"), nullable=False
    )
    storage_path: Mapped[str] = mapped_column(
        String(512), nullable=False, comment="相对 models_dir 的路径"
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True, comment="是否为当前激活版本"
    )
    metrics_json: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, comment="性能指标（mAP/精度/延迟等）"
    )
    description: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    def __repr__(self) -> str:
        return f"<MLModel id={self.id} name={self.name!r} version={self.version!r} active={self.is_active}>"
