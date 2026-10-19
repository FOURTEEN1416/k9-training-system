"""P0-2 /health 聚合测试。

Owner: 后端开发
Phase: P0-2（production-readiness-gap-plan.md P0-2 验收线：停 Redis 时 health 指名故障）

策略:
    - DB / Redis 检查函数以 monkeypatch 注入成功/失败，确定性不依赖本机服务
    - 模型盘点为纯路径检查，真实执行
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.api import health as health_module
from backend.app.main import app
from backend.app.schemas.common import ComponentHealth


def _ok_sync() -> ComponentHealth:
    return ComponentHealth(status="ok", detail="reachable")


def _down_sync() -> ComponentHealth:
    return ComponentHealth(status="down", detail="ConnectionRefused: simulated outage")


# 端点内 await 检查函数，桩必须 async
async def _ok() -> ComponentHealth:
    return _ok_sync()


async def _down() -> ComponentHealth:
    return _down_sync()


class TestHealthAggregate:
    """聚合语义：全 ok → overall ok；任一 down → overall degraded 且指名组件。"""

    def test_all_ok_overall_ok(self, monkeypatch):
        monkeypatch.setattr(health_module, "_check_database", _ok)
        monkeypatch.setattr(health_module, "_check_redis", _ok)

        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert set(body["checks"].keys()) == {"database", "redis", "models"}
        for c in body["checks"].values():
            assert c["status"] in {"ok", "degraded", "down"}

    def test_redis_down_names_component(self, monkeypatch):
        """验收线：Redis 故障时 overall=degraded 且 checks.redis 指名故障。"""
        monkeypatch.setattr(health_module, "_check_database", _ok)
        monkeypatch.setattr(health_module, "_check_redis", _down)

        client = TestClient(app)
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["checks"]["redis"]["status"] == "down"
        assert "simulated outage" in body["checks"]["redis"]["detail"]
        assert body["checks"]["database"]["status"] == "ok"

    def test_db_down_names_component(self, monkeypatch):
        monkeypatch.setattr(health_module, "_check_database", _down)
        monkeypatch.setattr(health_module, "_check_redis", _ok)

        client = TestClient(app)
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["checks"]["database"]["status"] == "down"

    def test_root_returns_same_aggregate(self, monkeypatch):
        monkeypatch.setattr(health_module, "_check_database", _ok)
        monkeypatch.setattr(health_module, "_check_redis", _ok)

        client = TestClient(app)
        body = client.get("/").json()
        assert "checks" in body
        assert body["environment"] == "test"
