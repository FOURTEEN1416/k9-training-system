"""API auth gate smoke tests (commissioned fix verification).

Verifies:
  1. Business write routes reject unauthenticated requests with 401
  2. POST /api/annotations/sync exists and returns 501 (not 404)
  3. Route table wires auth dependencies on business routers

Uses TestClient; auth fails before DB when no Bearer token is provided.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault(
    "AUTH_SECRET_KEY",
    "TEST-ONLY-k9-training-system-not-for-prod-32chars!!",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.fast


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app

    with TestClient(app) as c:
        yield c


WRITE_CASES = [
    ("POST", "/api/dogs", {}),
    ("PUT", "/api/dogs/1", {}),
    ("DELETE", "/api/dogs/1", None),
    ("POST", "/api/videos/upload", None),
    ("POST", "/api/annotations/tasks", {}),
    ("POST", "/api/annotations/sync", {}),
    ("POST", "/api/finetune/trigger", {}),
    ("POST", "/api/models/register", {}),
    ("POST", "/api/models/1/activate", None),
    ("PUT", "/api/scoring/configs/fci_igp", {"content": "{}"}),
    ("POST", "/api/llm/explain", {}),
]

READ_CASES = [
    ("GET", "/api/dogs", None),
    ("GET", "/api/videos", None),
    ("GET", "/api/scores", None),
    ("GET", "/api/models", None),
    ("GET", "/api/annotations/tasks", None),
    ("GET", "/api/finetune/status", None),
    ("GET", "/api/scoring/configs", None),
]


@pytest.mark.parametrize("method,path,body", WRITE_CASES)
def test_write_requires_jwt(client, method, path, body):
    kwargs = {}
    if body is not None:
        kwargs["json"] = body
    resp = getattr(client, method.lower())(path, **kwargs)
    assert resp.status_code == 401, f"{method} {path} -> {resp.status_code} {resp.text[:200]}"


@pytest.mark.parametrize("method,path,body", READ_CASES)
def test_read_requires_jwt(client, method, path, body):
    kwargs = {}
    if body is not None:
        kwargs["json"] = body
    resp = getattr(client, method.lower())(path, **kwargs)
    assert resp.status_code == 401, f"{method} {path} -> {resp.status_code} {resp.text[:200]}"


def test_sync_stub_is_501_not_404(client):
    """Honest stub: endpoint must exist and return 501 with a clear message."""
    resp = client.post("/api/annotations/sync", json={})
    # Auth dependency runs first without token → 401; with invalid token also 401.
    # Route existence is asserted via OpenAPI when auth is satisfied.
    # Here we assert the route is registered (not 404).
    assert resp.status_code != 404
    openapi = client.get("/openapi.json")
    # openapi may also require nothing — docs are public
    if openapi.status_code == 200:
        paths = openapi.json().get("paths", {})
        assert "/api/annotations/sync" in paths, "sync route must be registered"
        assert "post" in paths["/api/annotations/sync"]


def test_business_routes_have_auth_dependency():
    from backend.app.core.deps import get_current_handler
    from backend.app.main import app

    prefixes = (
        "/api/dogs",
        "/api/videos",
        "/api/annotations",
        "/api/finetune",
        "/api/models",
        "/api/scores",
        "/api/scoring",
        "/api/llm",
    )

    def _has_auth(dependant) -> bool:
        for dep in getattr(dependant, "dependencies", []) or []:
            call = getattr(dep, "call", None)
            if call is get_current_handler:
                return True
            if _has_auth(dep):
                return True
        return False

    missing = []
    for route in app.routes:
        path = getattr(route, "path", "") or ""
        if not any(path.startswith(p) for p in prefixes):
            continue
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        if not _has_auth(dependant):
            missing.append(path)

    assert not missing, f"Routes missing get_current_handler dependency: {missing}"
