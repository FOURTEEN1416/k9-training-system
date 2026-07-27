"""SQLAlchemy 同步数据库引擎与 Session（Celery worker 用）.

Owner: 后端开发
Phase: 1.4d
依据: dev-docs/stages/phase-1.md §1.4d

设计:
    Celery 任务为同步函数，不能直接使用 AsyncSession。
    本模块提供同步 engine + SessionLocal，与 async 路径共用同一数据库。

用法:
    from backend.app.core.database_sync import SessionLocal

    with SessionLocal() as session:
        video = session.get(Video, video_id)
        session.commit()
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import settings

# 同步 engine（psycopg2，与 Alembic 共用 pg_dsn）
sync_engine = create_engine(
    settings.pg_dsn,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    echo=settings.db_echo,
    future=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
)


def get_sync_db() -> Generator[Session, None, None]:
    """提供同步数据库会话（Celery worker 用）。"""
    session = SyncSessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
