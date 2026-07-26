"""FastAPI 应用入口。"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import dogs, health, models, scores, videos
from backend.app.core.config import settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时确保数据目录存在。"""
    settings.ensure_dirs()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="工作犬训练机器视觉识别系统 API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS（开发期允许前端 Vite 端口）
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
app.include_router(health.router, tags=["meta"])
app.include_router(dogs.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(scores.router, prefix="/api")
app.include_router(models.router, prefix="/api")
