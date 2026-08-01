"""Application configuration via Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录（backend/ 的父目录）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """应用配置。

    配置来源优先级：环境变量 > .env 文件 > 默认值。
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === 应用 ===
    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    app_log_level: str = "INFO"
    app_name: str = "K9 Training Vision System"
    app_version: str = "0.1.0"

    # === 数据库 ===
    database_url: str = Field(
        default="postgresql+asyncpg://k9system:K9System2026!@127.0.0.1:5433/k9system",
        description="异步 SQLAlchemy 数据库 URL",
    )
    pg_dsn: str = Field(
        default="postgresql://k9system:K9System2026!@127.0.0.1:5433/k9system",
        description="同步 psycopg2 DSN（Alembic 用）",
    )
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_echo: bool = False

    # === Redis / Celery ===
    redis_url: str = "redis://127.0.0.1:6379/0"
    celery_broker_url: str = "redis://127.0.0.1:6379/1"
    celery_result_backend: str = "redis://127.0.0.1:6379/2"

    # === 路径 ===
    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    generated_dir: Path = PROJECT_ROOT / "data" / "generated"
    models_dir: Path = PROJECT_ROOT / "data" / "models_weights"

    # === 文件上传 ===
    upload_max_size_mb: int = 2048
    upload_allowed_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    # === CORS（开发期） ===
    cors_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
    )

    # === Label Studio（数据飞轮标注平台） ===
    ls_url: str = "http://127.0.0.1:8080"
    ls_email: str = "admin@k9.local"
    ls_password: str = "k9admin2026"
    ls_project_id: int = 1

    def ensure_dirs(self) -> None:
        """创建所有数据目录（如不存在）。"""
        for d in (self.data_dir, self.upload_dir, self.generated_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """单例 Settings。"""
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
