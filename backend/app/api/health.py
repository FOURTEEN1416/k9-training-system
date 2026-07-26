"""健康检查路由。"""

from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.schemas.common import HealthResponse

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """健康检查端点。"""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """根路径重定向到健康检查。"""
    return await health_check()
