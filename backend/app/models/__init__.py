"""SQLAlchemy 模型集合。

导入所有模型以使 SQLAlchemy 注册元数据（Alembic autogenerate 依赖）。
"""

from backend.app.models.base import Base, TimestampMixin
from backend.app.models.behavior import Behavior, BehaviorClass, BehaviorDetector
from backend.app.models.behavior_vector import BehaviorVector, BEHAVIOR_VECTOR_DIM
from backend.app.models.dog import Dog, Gender, TrainingStage
from backend.app.models.handler import Handler, UserRole
from backend.app.models.keypoint import Keypoint, NUM_KEYPOINTS
from backend.app.models.ml_model import MLModel, ModelFramework, ModelType
from backend.app.models.score import Score
from backend.app.models.training_session import ScoringStandard, TrainingSession
from backend.app.models.video import Video, VideoStatus

__all__ = [
    "Base",
    "TimestampMixin",
    # 模型
    "Handler",
    "Dog",
    "TrainingSession",
    "Video",
    "MLModel",
    "Keypoint",
    "Behavior",
    "BehaviorVector",
    "Score",
    # 枚举
    "UserRole",
    "Gender",
    "TrainingStage",
    "ScoringStandard",
    "VideoStatus",
    "ModelType",
    "ModelFramework",
    "BehaviorClass",
    "BehaviorDetector",
    # 常量
    "NUM_KEYPOINTS",
    "BEHAVIOR_VECTOR_DIM",
]
