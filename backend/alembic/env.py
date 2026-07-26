"""Alembic 迁移环境。"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# 导入所有模型以注册元数据
from backend.app.models import Base  # noqa: F401
from backend.app.models import *  # noqa: F401, F403  注册所有模型
from backend.app.core.config import settings

# Alembic 配置对象
config = context.config

# 日志
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# SQLAlchemy MetaData（autogenerate 用）
target_metadata = Base.metadata

# 从 settings 覆盖 URL（避免硬编码）
config.set_main_option("sqlalchemy.url", settings.pg_dsn)


def run_migrations_offline() -> None:
    """离线模式：生成 SQL 脚本而不连接数据库。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """在线模式：连接数据库执行迁移。"""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
