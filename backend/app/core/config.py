"""Application configuration via Pydantic Settings.

SECURITY (2026-09-19 fix):
    This file previously shipped LIVE credential defaults (DB password,
    Label Studio password, JWT secret) to a PUBLIC GitHub repository
    (github.com/FOURTEEN1416/k9-training-system). Those values must be
    treated as COMPROMISED and ROTATED — they remain in git history even
    after this change.

    Defaults are now placeholders only. Real credentials must come from
    environment variables / local `.env` (never committed).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录（backend/ 的父目录）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]

# Placeholders — never valid production credentials
CHANGE_ME = "CHANGE_ME"

# Credentials that appeared in PUBLIC git history. Refuse if still configured.
_LEAKED_DB_PASSWORD = "K9System2026!"
_LEAKED_LS_PASSWORD = "k9admin2026"
_LEAKED_JWT_SECRET = "k9-training-system-dev-secret-key-change-in-production-32chars"

# Explicit TEST-only fallback (pytest sets APP_ENV=test; see tests/conftest.py)
_TEST_AUTH_SECRET = "TEST-ONLY-k9-training-system-not-for-prod-32chars!!"
_TEST_DB_URL = "postgresql+asyncpg://k9system:TEST_PLACEHOLDER@127.0.0.1:5433/k9system"
_TEST_PG_DSN = "postgresql://k9system:TEST_PLACEHOLDER@127.0.0.1:5433/k9system"
_DEV_AUTH_SECRET = "DEV-ONLY-k9-training-system-set-AUTH_SECRET_KEY-before-deploy-0001"


def _is_weak_secret(value: str | None) -> bool:
    v = (value or "").strip()
    if not v:
        return True
    if "CHANGE_ME" in v:
        return True
    if v in {_LEAKED_JWT_SECRET, _LEAKED_DB_PASSWORD, _LEAKED_LS_PASSWORD}:
        return True
    return len(v) < 32


def _dsn_has_leaked_password(url: str) -> bool:
    return _LEAKED_DB_PASSWORD in (url or "")


def _dsn_is_placeholder(url: str) -> bool:
    u = url or ""
    return (not u) or ("CHANGE_ME" in u) or ("TEST_PLACEHOLDER" in u) or _dsn_has_leaked_password(u)


class Settings(BaseSettings):
    """应用配置。

    配置来源优先级：环境变量 > .env 文件 > 默认值。

    Secret policy:
        - Defaults never contain real credentials.
        - APP_ENV=production refuses placeholder / weak / leaked secrets.
        - APP_ENV=test uses explicit TEST_* fallbacks only.
        - APP_ENV=development allows a clearly marked DEV-ONLY JWT secret
          so local imports work; production deployments MUST override.
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
    # Placeholder password only — set DATABASE_URL via env/.env.
    # ROTATE any password that was ever used with the old public defaults.
    database_url: str = Field(
        default=f"postgresql+asyncpg://k9system:{CHANGE_ME}@127.0.0.1:5433/k9system",
        description="异步 SQLAlchemy 数据库 URL（必须经环境变量配置真实口令）",
    )
    pg_dsn: str = Field(
        default=f"postgresql://k9system:{CHANGE_ME}@127.0.0.1:5433/k9system",
        description="同步 psycopg2 DSN（Alembic 用；必须经环境变量配置真实口令）",
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

    # === Label Studio（已废弃，保留配置向后兼容） ===
    # YouTube 自标数据源已废弃（2026-08-19），Label Studio 标注管线不再使用
    # 改为 AimCLR+SMQ+TCL+主动学习管线（见 ADR 0012）
    # Default is placeholder only — ROTATE if you ever used the old public default.
    ls_url: str = "http://127.0.0.1:8080"
    ls_email: str = "admin@k9.local"
    ls_password: str = Field(default=CHANGE_ME, description="Label Studio 口令（已废弃管线；占位符）")
    ls_project_id: int = 1

    # === RBAC 鉴权（Phase 3.6b） ===
    # JWT secret: must be provided via environment for non-dev deployments.
    # The old public default is refused by model_validator.
    auth_secret_key: str = Field(
        default="",
        description=(
            "JWT 签名密钥。生产环境必须通过环境变量设置唯一值（≥32 字符）；"
            "测试环境见 conftest（APP_ENV=test）。严禁使用曾入库的旧默认值。"
        ),
    )
    auth_algorithm: str = "HS256"
    auth_access_token_expire_minutes: int = 480  # 8 小时工作日
    auth_bcrypt_rounds: int = 12  # bcrypt cost factor（10-14 推荐）

    # === LLM 行为解释器（Phase 4） ===
    # 外部 LLM API 配置
    # 支持 OpenAI、Azure OpenAI、通义千问、智谱等兼容 API
    llm_api_key: str = Field(
        default="",
        description="LLM API Key（如 OpenAI key 或国内兼容 API key）",
    )
    llm_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="LLM API Base URL",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="使用的 LLM 模型名称",
    )
    llm_timeout: int = Field(
        default=30,
        description="LLM API 请求超时时间（秒）",
    )

    # === Phase 4 Mamba 部署 ===
    behavior_deploy_mode: Literal[
        "shadow",
        "vote",
        "primary_stgcn",
        "mamba_only",
        "mamba_shadow",
        "mamba_vote",
        "mamba_bc_only",
        "mamba_bc_shadow",
        "mamba_bc_vote",
        "rule_only",
    ] = "shadow"

    @model_validator(mode="after")
    def _enforce_secret_policy(self) -> "Settings":
        """Refuse leaked/placeholder secrets outside the explicit TEST path.

        OPERATOR ACTION REQUIRED:
            Credentials that were ever the defaults in this file are in PUBLIC
            git history. Rotate DB password, Label Studio password, and JWT
            signing key even if you never deployed with them.
        """
        if _dsn_has_leaked_password(self.database_url) or _dsn_has_leaked_password(self.pg_dsn):
            if self.app_env != "test":
                raise ValueError(
                    "Rejected leaked DB password default (was public on GitHub). "
                    "Set DATABASE_URL / PG_DSN in environment/.env to a NEW password "
                    "and ROTATE the old one."
                )

        if self.ls_password == _LEAKED_LS_PASSWORD:
            if self.app_env != "test":
                raise ValueError(
                    "Rejected leaked Label Studio password default (was public on GitHub). "
                    "Set LS_PASSWORD to a new value or leave CHANGE_ME; ROTATE the old password."
                )

        if (self.auth_secret_key or "").strip() == _LEAKED_JWT_SECRET:
            raise ValueError(
                "Rejected leaked JWT secret default (was public on GitHub). "
                "Generate a new AUTH_SECRET_KEY (≥32 chars) and ROTATE all issued tokens."
            )

        if self.app_env == "test":
            # Explicit TEST path only — no production credentials required.
            if _dsn_has_leaked_password(self.database_url) or _dsn_is_placeholder(self.database_url):
                self.database_url = _TEST_DB_URL
            if _dsn_has_leaked_password(self.pg_dsn) or _dsn_is_placeholder(self.pg_dsn):
                self.pg_dsn = _TEST_PG_DSN
            if self.ls_password in {CHANGE_ME, _LEAKED_LS_PASSWORD, ""}:
                self.ls_password = "TEST_PLACEHOLDER"
            if _is_weak_secret(self.auth_secret_key):
                self.auth_secret_key = _TEST_AUTH_SECRET
            return self

        if self.app_env == "production":
            if _is_weak_secret(self.auth_secret_key):
                raise ValueError(
                    "AUTH_SECRET_KEY must be set via environment to a unique value "
                    "(≥32 chars, not CHANGE_ME / empty / historical default) when APP_ENV=production."
                )
            if _dsn_is_placeholder(self.database_url) or _dsn_is_placeholder(self.pg_dsn):
                raise ValueError(
                    "DATABASE_URL and PG_DSN must be set via environment with real credentials "
                    f"when APP_ENV=production (placeholder/CHANGE_ME not allowed)."
                )
            return self

        # development: allow import; JWT gets a clearly non-production fallback.
        if _is_weak_secret(self.auth_secret_key):
            self.auth_secret_key = _DEV_AUTH_SECRET
        return self

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
