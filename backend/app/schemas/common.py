"""Pydantic schemas（Phase 0 仅占位，Phase 1 完善验证规则）。"""

import enum
import re
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# P1-6: processing 超过该时长（秒）视为疑似卡死。
# celery visibility_timeout=3600（celery_app.py:33）的 2 倍，留足重投递余量。
PROCESSING_STALE_AFTER_SEC = 7200


class VideoErrorCode(str, enum.Enum):
    """视频处理失败原因（可判定错误码）.

    error_message 以 `[CODE] 中文说明` 形式落库，前端/统计据此分类；
    枚举值只增不改，未知码按原文展示（见 parse_error_message）。
    """

    VIDEO_NOT_FOUND = "VIDEO_NOT_FOUND"
    FILE_MISSING = "FILE_MISSING"
    VIDEO_UNREADABLE = "VIDEO_UNREADABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    UNSUPPORTED_SCENE = "UNSUPPORTED_SCENE"
    SCORING_FAILED = "SCORING_FAILED"
    REPORT_FAILED = "REPORT_FAILED"
    DB_WRITE_FAILED = "DB_WRITE_FAILED"
    TASK_DISPATCH_FAILED = "TASK_DISPATCH_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


ERROR_CODE_HINTS: dict[VideoErrorCode, str] = {
    VideoErrorCode.VIDEO_NOT_FOUND: "视频记录不存在，可能已被删除，请刷新列表。",
    VideoErrorCode.FILE_MISSING: "视频文件在服务器上缺失，请重新上传。",
    VideoErrorCode.VIDEO_UNREADABLE: "视频无法解码，请确认文件完整且编码受支持（MP4/AVI/MOV/MKV）。",
    VideoErrorCode.MODEL_UNAVAILABLE: "推理模型不可用或加载失败，请联系管理员检查模型文件/GPU。",
    VideoErrorCode.INFERENCE_FAILED: "推理过程异常，请重试或联系管理员查看日志。",
    VideoErrorCode.UNSUPPORTED_SCENE: "测试场景不受支持，请重新选择场景后上传。",
    VideoErrorCode.SCORING_FAILED: "评分计算失败，请重试或联系管理员检查评分卡。",
    VideoErrorCode.REPORT_FAILED: "PDF 报告生成失败，请联系管理员检查报告模板与磁盘空间。",
    VideoErrorCode.DB_WRITE_FAILED: "结果写入数据库失败，请稍后重试或联系管理员。",
    VideoErrorCode.TASK_DISPATCH_FAILED: "任务派发失败，请检查 Celery/Redis 后重试。",
    VideoErrorCode.INTERNAL_ERROR: "内部错误，请联系管理员查看日志。",
}


_ERROR_MESSAGE_PATTERN = re.compile(r"^\[([A-Z][A-Z0-9_]*)\]\s*(.*)$", re.DOTALL)


def parse_error_message(message: Optional[str]) -> tuple[Optional[str], str]:
    """解析 `[CODE] 说明` 形式的 error_message.

    Returns:
        (code, text)：code 为已登记枚举值，否则 None；text 为去掉前缀后的说明。
        不合法/未知码时返回 (None, 原文)，保证历史数据可读且不抛错。
    """
    if not message:
        return None, ""
    m = _ERROR_MESSAGE_PATTERN.match(message.strip())
    if not m:
        return None, message
    code, text = m.group(1), m.group(2).strip()
    if code not in VideoErrorCode.__members__:
        return None, message
    return code, text


def error_code_hint(code: Optional[str]) -> Optional[str]:
    """错误码 → 中文处置建议（未知码返回 None）。"""
    if not code:
        return None
    try:
        return ERROR_CODE_HINTS[VideoErrorCode(code)]
    except (KeyError, ValueError):
        return None


def _ensure_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """naive datetime 一律按 UTC 解释（DB 历史值/测试构造值均可能无 tzinfo）。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def processing_elapsed_ms(
    started_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> Optional[int]:
    """processing 已运行毫秒数；未开始返回 None，时钟回拨 clamp 到 0。"""
    start = _ensure_aware(started_at)
    if start is None:
        return None
    current = _ensure_aware(now) or datetime.now(timezone.utc)
    delta_ms = int((current - start).total_seconds() * 1000)
    return max(0, delta_ms)


def _is_processing_stale(
    status: str,
    started_at: Optional[datetime],
    uploaded_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> bool:
    """processing 且（未记录开始时间时上传已久 / 已运行超阈值）→ 疑似卡死。"""
    if status != "processing":
        return False
    current = _ensure_aware(now) or datetime.now(timezone.utc)
    if started_at is None:
        uploaded = _ensure_aware(uploaded_at)
        if uploaded is None:
            return False
        return (current - uploaded) > timedelta(seconds=PROCESSING_STALE_AFTER_SEC)
    start = _ensure_aware(started_at)
    return (current - start) > timedelta(seconds=PROCESSING_STALE_AFTER_SEC)


class ORMModel(BaseModel):
    """带 ORM 模式的基础 schema。"""

    model_config = ConfigDict(from_attributes=True)


class ComponentHealth(BaseModel):
    """单个组件健康状态（P0-2 /health 聚合）。"""

    status: Literal["ok", "degraded", "down"]
    detail: str = ""


class HealthResponse(BaseModel):
    """健康检查响应。

    P0-2 扩展：checks 聚合 DB / Redis / 模型路径状态；
    status=ok 仅当全部组件 ok，否则 degraded（HTTP 仍 200，语义见 checks）。
    """

    status: str
    version: str
    environment: str
    deploy_mode: str = ""  # 当前行为识别部署模式（P0-6 前端页头展示）
    checks: dict[str, ComponentHealth] = Field(default_factory=dict)


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
    # P1-6 处理进度与可判定失败
    processing_stage: Optional[str] = None
    processing_percent: Optional[float] = None
    processing_started_at: Optional[datetime] = None
    processing_duration_ms: Optional[int] = None
    error_code: Optional[str] = None
    error_hint: Optional[str] = None
    processing_stale: bool = False
    retry_scheduled: bool = False

    @model_validator(mode="after")
    def _derive_observability(self) -> "VideoRead":
        code, _ = parse_error_message(self.error_message)
        self.error_code = code
        self.error_hint = error_code_hint(code)
        self.processing_stale = _is_processing_stale(
            self.status, self.processing_started_at, self.uploaded_at
        )
        self.retry_scheduled = (
            self.status == "processing" and self.processing_stage == "retrying"
        )
        return self


class VideoStatusRead(ORMModel):
    """视频状态轮询响应（P1-6：含阶段进度/错误码/卡死与重试标记）。"""

    id: int
    status: str
    scene: str
    error_message: Optional[str] = None
    report_path: Optional[str] = None
    processed_at: Optional[datetime] = None
    uploaded_at: Optional[datetime] = None
    processing_stage: Optional[str] = None
    processing_percent: Optional[float] = None
    processing_started_at: Optional[datetime] = None
    processing_duration_ms: Optional[int] = None
    processing_elapsed_ms: Optional[int] = None
    error_code: Optional[str] = None
    error_hint: Optional[str] = None
    processing_stale: bool = False
    retry_scheduled: bool = False

    @model_validator(mode="after")
    def _derive_observability(self) -> "VideoStatusRead":
        code, _ = parse_error_message(self.error_message)
        self.error_code = code
        self.error_hint = error_code_hint(code)
        self.processing_elapsed_ms = (
            processing_elapsed_ms(self.processing_started_at)
            if self.status == "processing"
            else None
        )
        self.processing_stale = _is_processing_stale(
            self.status, self.processing_started_at, self.uploaded_at
        )
        self.retry_scheduled = (
            self.status == "processing" and self.processing_stage == "retrying"
        )
        return self


class ProcessingStatsRead(BaseModel):
    """处理时延统计（P1-6 运维口径）.

    双口径并排，不混用：
        - processing_*: processing_started_at → processed_at（纯处理耗时，含重试的本次尝试）
        - e2e_*:        uploaded_at → processed_at（端到端，含排队等待）
    """

    window_days: int
    since: datetime
    method: str
    sample_size: int
    p50_duration_ms: Optional[int] = None
    p95_duration_ms: Optional[int] = None
    max_duration_ms: Optional[int] = None
    e2e_sample_size: int = 0
    e2e_p50_ms: Optional[int] = None
    e2e_p95_ms: Optional[int] = None
    e2e_max_ms: Optional[int] = None
    completed: int = 0
    failed: int = 0
    failure_rate: Optional[float] = None
    stale_processing: int = 0
    reliable: bool = False


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


# === Phase 3.7a 训练历史对比 ===

class DogBrief(BaseModel):
    """犬只简要信息（对比响应内嵌）。"""

    id: int
    name: str
    breed: Optional[str] = None


class ScorePoint(BaseModel):
    """评分时序点（按时间排序）。"""

    score_id: int
    video_id: int
    created_at: datetime
    overall: float
    standard: str
    dimensions: dict[str, float]  # 仅包含非 None 的维度


class DogScoreSeries(BaseModel):
    """单犬评分时间序列。"""

    dog: DogBrief
    points: list[ScorePoint]


class DogScoreStats(BaseModel):
    """单犬评分统计。"""

    dog: DogBrief
    count: int
    avg_overall: float
    max_overall: float
    min_overall: float
    latest_overall: Optional[float] = None
    trend_slope: Optional[float] = None  # overall 随时间变化的线性回归斜率（正=提升）
    avg_dimensions: dict[str, float]  # 各维度平均值（仅包含至少一个非 None 值的维度）


class ScoreCompareResponse(BaseModel):
    """训练历史对比查询响应。"""

    dogs: list[DogBrief]
    series: list[DogScoreSeries]  # 每犬一条时序
    stats: list[DogScoreStats]  # 每犬一条统计
    dimension_labels: dict[str, str]  # 维度 key → 中文标签
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    standard: Optional[str] = None


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
