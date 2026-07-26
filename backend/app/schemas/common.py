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
    created_at: datetime


class DogBase(ORMModel):
    name: str
    breed: Optional[str] = None
    gender: Optional[str] = None
    chip_id: Optional[str] = None
    handler_id: Optional[int] = None
    training_stage: str = "P0"


class DogCreate(DogBase):
    pass


class DogRead(DogBase):
    id: int
    created_at: datetime


class VideoRead(ORMModel):
    id: int
    dog_id: Optional[int] = None
    handler_id: Optional[int] = None
    original_filename: str
    status: str
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    error_message: Optional[str] = None


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
