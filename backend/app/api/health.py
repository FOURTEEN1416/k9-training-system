"""健康检查路由（P0-2：聚合 DB / Redis / 模型路径状态）。

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: P0-2（dev-docs/production-readiness-gap-plan.md P0-2）

语义:
    - HTTP 恒 200（liveness）；组件级故障在 JSON checks 中指名（readiness）。
    - status=ok 仅当全部组件 ok，任一组件异常 → degraded。
    - 故意停 Redis 时应能在 checks.redis.detail 看到指名故障（gap-plan 验收线）。
"""

from fastapi import APIRouter
import redis.asyncio as aioredis
from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.core.database import engine
from backend.app.schemas.common import ComponentHealth, HealthResponse
from backend.app.services.readiness import model_presence

router = APIRouter(tags=["meta"])


async def _check_database() -> ComponentHealth:
    """SELECT 1 探活（异步引擎）。"""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return ComponentHealth(status="ok", detail="database reachable")
    except Exception as e:  # noqa: BLE001 — 健康检查需吞一切异常并转状态
        return ComponentHealth(status="down", detail=f"{type(e).__name__}: {e}")


async def _check_redis() -> ComponentHealth:
    """PING 探活（2s 连接超时，避免 health 挂死）。"""
    client = aioredis.from_url(
        settings.redis_url,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    try:
        await client.ping()
        # 不回显连接串（渗透用例 OBS-health-redis-url：避免向未认证调用方泄露内网地址）
        return ComponentHealth(status="ok", detail="redis reachable")
    except Exception as e:  # noqa: BLE001
        return ComponentHealth(status="down", detail=f"{type(e).__name__}: {e}")
    finally:
        await client.close()


def _check_models() -> ComponentHealth:
    """模型路径存在性盘点（轻量，不加载模型）。"""
    presence = model_presence()
    missing = [name for name, info in presence.items() if not info["present"]]
    if not missing:
        return ComponentHealth(status="ok", detail="all model groups present")
    # pose 是推理硬依赖 → down；研究线模型缺失 → degraded
    status = "down" if "pose_yolo" in missing else "degraded"
    return ComponentHealth(status=status, detail="missing: " + ", ".join(missing))


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """健康检查端点（聚合）。"""
    checks = {
        "database": await _check_database(),
        "redis": await _check_redis(),
        "models": _check_models(),
    }
    overall = "ok" if all(c.status == "ok" for c in checks.values()) else "degraded"
    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.app_env,
        deploy_mode=settings.behavior_deploy_mode or "",
        checks=checks,
    )


@router.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """根路径重定向到健康检查。"""
    return await health_check()
