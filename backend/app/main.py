"""FastAPI 应用入口。"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api import (
    annotations, auth, bases, dogs, finetune, health, models, scoring, scores, videos,
)
from backend.app.core.config import settings
from backend.app.core.security import AuthError


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


# ============================================================
# 异常处理：AuthError → 401/403 JSON 响应（Phase 3.6b）
# ============================================================

@app.exception_handler(AuthError)
async def auth_error_handler(request: Request, exc: AuthError) -> JSONResponse:
    """将 AuthError（含子类 InvalidTokenError / InsufficientPermissionError / AuthenticationFailedError）
    统一转换为 JSON 响应，避免暴露堆栈。
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message},
    )


# ============================================================
# 路由注册
# ============================================================

app.include_router(health.router, tags=["meta"])
# Phase 3.6b 鉴权 + 基地管理
app.include_router(auth.router, prefix="/api")
app.include_router(bases.router, prefix="/api")
# 业务路由
app.include_router(dogs.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(scores.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(scoring.router, prefix="/api")
app.include_router(annotations.router, prefix="/api")
app.include_router(finetune.router, prefix="/api")
