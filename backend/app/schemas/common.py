"""Pydantic schemas（Phase 0 仅占位，Phase 1 完善验证规则）。"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ORMModel(BaseModel):
    """带 ORM 模式的基础 schema。"""

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str
    version: str
    environment: str


class HandlerBase(ORMModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    role: str = "handler"


class HandlerCreate(HandlerBase):
    pass


class HandlerRead(HandlerBase):
    id: int
    is_active: bool
    base_id: Optional[int] = None
    is_superuser: bool = False
    created_at: datetime


# === Phase 3.6b 鉴权 ===

class LoginRequest(BaseModel):
    """登录请求。"""

    email: str
    password: str


class TokenResponse(BaseModel):
    """登录成功响应（JWT + 用户信息）。"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int  # 秒
    handler: "HandlerRead"


class PasswordChangeRequest(BaseModel):
    """修改密码请求。"""

    old_password: str
    new_password: str


class HandlerRegisterRequest(BaseModel):
    """注册新用户请求（仅 ADMIN 可调用）。"""

    name: str
    email: str
    password: str
    role: str = "handler"  # handler/manager/researcher/viewer
    base_id: Optional[int] = None
    phone: Optional[str] = None
    is_superuser: bool = False


# === Phase 3.6a 基地管理 ===

class BaseCreate(BaseModel):
    """创建基地请求。"""

    name: str
    code: str
    description: Optional[str] = None
    is_active: bool = True


class BaseUpdate(BaseModel):
    """更新基地请求（partial update）。"""

    name: Optional[str] = None
    code: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class BaseRead(ORMModel):
    """基地响应。"""

    id: int
    name: str
    code: str
    description: Optional[str] = None
    is_active: bool
    created_at: datetime


# 避免前向引用问题
TokenResponse.model_rebuild()


class DogBase(ORMModel):
    name: str
    breed: Optional[str] = None
    gender: Optional[str] = None
    chip_id: Optional[str] = None
    handler_id: Optional[int] = None
    training_stage: str = "P0"


class DogCreate(DogBase):
    pass


class DogUpdate(ORMModel):
    """犬只档案更新（所有字段可选，partial update）。"""

    name: Optional[str] = None
    breed: Optional[str] = None
    gender: Optional[str] = None
    chip_id: Optional[str] = None
    handler_id: Optional[int] = None
    training_stage: Optional[str] = None


class DogRead(DogBase):
    id: int
    created_at: datetime


class VideoRead(ORMModel):
    id: int
    dog_id: Optional[int] = None
    handler_id: Optional[int] = None
    original_filename: str
    status: str
    scene: str = "obedience_trial"
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    report_path: Optional[str] = None


class VideoStatusRead(BaseModel):
    """视频状态轮询响应。"""

    id: int
    status: str
    scene: str
    error_message: Optional[str] = None
    report_path: Optional[str] = None
    processed_at: Optional[datetime] = None


class ScoreRead(ORMModel):
    id: int
    video_id: int
    standard: str
    accuracy: float
    response_latency: Optional[float] = None
    duration: float
    search_efficiency: Optional[float] = None
    attention: float
    courage: Optional[float] = None
    gait_quality: Optional[float] = None
    overall: float
    scoring_engine_version: str
    created_at: datetime


# === ML 模型管理 ===

class ModelCreate(ORMModel):
    """注册新模型版本。"""

    name: str
    version: str
    type: str  # pose / behavior / scoring
    framework: str  # pytorch / onnx / tensorrt
    storage_path: str
    is_active: bool = False
    metrics: Optional[dict] = None
    description: Optional[str] = None


class ModelRead(ORMModel):
    id: int
    name: str
    version: str
    type: str
    framework: str
    storage_path: str
    is_active: bool
    metrics_json: Optional[dict] = None
    description: Optional[str] = None
    created_at: datetime


# === 评分卡管理 ===

class ScoringConfigRead(BaseModel):
    """评分卡读取响应。"""

    scene: str
    content: str  # YAML 原文


class ScoringConfigUpdate(BaseModel):
    """评分卡更新请求。"""

    content: str  # YAML 原文


class ScoringEvaluateRequest(BaseModel):
    """评分请求。"""

    scene: str  # puppy_selection / obedience_trial
    signals: dict  # 信号字典


class ScoringEvaluateResponse(BaseModel):
    """评分响应。"""

    total_score: float
    verdict: str
    passed: bool
    dimension_labels: dict
    dimension_scores: dict
    explanation: list[str]
    scene: str
    card_name: str
    card_version: str


# === 标注管理（数据飞轮 2.1b） ===

class AnnotationTaskCreate(BaseModel):
    """创建标注任务请求。"""

    video_id: int
    handler_id: Optional[int] = None
    annotation_types: Optional[list[str]] = None  # ["keypoint", "behavior", "bbox"]


class AnnotationTaskRead(ORMModel):
    """标注任务响应。"""

    id: int
    video_id: int
    handler_id: Optional[int] = None
    ls_project_id: Optional[int] = None
    ls_task_id: Optional[int] = None
    status: str
    total_frames: Optional[int] = None
    annotated_frames: int
    annotation_types: Optional[list] = None
    error_message: Optional[str] = None
    progress: float
    created_at: datetime
    updated_at: datetime


class AnnotationRead(ORMModel):
    """标注数据响应。"""

    id: int
    task_id: int
    video_id: int
    frame_idx: Optional[int] = None
    annotation_type: str
    source: str
    data_json: dict
    confidence: Optional[float] = None
    created_at: datetime


class LabelStudioSyncResult(BaseModel):
    """LS 标注同步结果。"""

    project_id: int
    synced_tasks: int
    synced_annotations: int
    completed_tasks: int
    errors: list[str] = []
