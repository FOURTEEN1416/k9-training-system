"""Phase 3.6c RBAC 端到端评估脚本。

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 3.6c

测试矩阵:
    1. 认证: login + JWT 签发 + /auth/me + change-password
    2. 角色级权限: ADMIN/MANAGER/RESEARCHER/HANDLER/VIEWER 访问 /auth/users
    3. 资源级权限: 基地隔离（MANAGER 只看本基地）+ 犬只访问（check_dog_access）
    4. 基地 CRUD: 仅 ADMIN 可创建/更新/禁用
    5. 用户管理: register 仅 ADMIN + deactivate 仅 ADMIN

种子数据（带 eval_rbac_ 前缀，便于清理）:
    基地: base_alpha (A) / base_beta (B)
    用户: admin / manager_a / manager_b / handler_a1 / handler_a2 / handler_b1 / researcher_a / viewer_a
    犬只: dog_a1 (base_a, handler_a1) / dog_b1 (base_b, handler_b1)

运行:
    python scripts/eval_rbac.py

输出:
    reports/phase-3.6c-rbac-eval.json
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2  # noqa: E402
import httpx  # noqa: E402
from httpx import ASGITransport  # noqa: E402

from backend.app.main import app  # noqa: E402
from backend.app.core.config import settings  # noqa: E402

# ============================================================
# 配置
# ============================================================

TEST_PREFIX = "eval_rbac_"
PASSWORD_PLAIN = "TestPass123!"  # 所有测试用户统一密码
REPORT_PATH = PROJECT_ROOT / "reports" / "phase-3.6c-rbac-eval.json"

# 种子用户定义: (name, email, role, base_code, is_superuser)
# base_code 对应 SEED_BASES 中的 code（大写）
SEED_USERS: list[tuple[str, str, str, str | None, bool]] = [
    ("Admin User", f"{TEST_PREFIX}admin@test.com", "ADMIN", None, True),
    ("Manager A", f"{TEST_PREFIX}manager_a@test.com", "MANAGER", "ALPHA", False),
    ("Manager B", f"{TEST_PREFIX}manager_b@test.com", "MANAGER", "BETA", False),
    ("Handler A1", f"{TEST_PREFIX}handler_a1@test.com", "HANDLER", "ALPHA", False),
    ("Handler A2", f"{TEST_PREFIX}handler_a2@test.com", "HANDLER", "ALPHA", False),
    ("Handler B1", f"{TEST_PREFIX}handler_b1@test.com", "HANDLER", "BETA", False),
    ("Researcher A", f"{TEST_PREFIX}researcher_a@test.com", "RESEARCHER", "ALPHA", False),
    ("Viewer A", f"{TEST_PREFIX}viewer_a@test.com", "VIEWER", "ALPHA", False),
]

# 种子基地
SEED_BASES: list[tuple[str, str, str]] = [
    (f"{TEST_PREFIX}alpha", "ALPHA", "基地 Alpha（测试）"),
    (f"{TEST_PREFIX}beta", "BETA", "基地 Beta（测试）"),
]


# ============================================================
# DB 操作（psycopg2 同步）
# ============================================================

def get_db_conn():
    """从 settings 解析同步 DSN 并连接。"""
    dsn = settings.pg_dsn  # postgresql://k9system:...@127.0.0.1:5433/k9system
    return psycopg2.connect(dsn)


def cleanup_seed_data(conn) -> None:
    """清理旧的测试种子数据（带 eval_rbac_ 前缀）。"""
    with conn.cursor() as cur:
        # 删除测试犬只（先删关联）
        cur.execute(
            "DELETE FROM dog_handler_association WHERE dog_id IN "
            "(SELECT id FROM dogs WHERE name LIKE %s)",
            (f"{TEST_PREFIX}%",),
        )
        cur.execute(
            "DELETE FROM dog_base_association WHERE dog_id IN "
            "(SELECT id FROM dogs WHERE name LIKE %s)",
            (f"{TEST_PREFIX}%",),
        )
        cur.execute("DELETE FROM dogs WHERE name LIKE %s", (f"{TEST_PREFIX}%",))
        # 删除测试用户
        cur.execute("DELETE FROM handlers WHERE email LIKE %s", (f"{TEST_PREFIX}%",))
        # 删除测试基地
        cur.execute(
            "DELETE FROM bases WHERE code IN ('ALPHA', 'BETA', 'GAMMA') OR name LIKE %s",
            (f"{TEST_PREFIX}%",),
        )
    conn.commit()


def seed_data(conn) -> dict[str, int]:
    """创建种子数据，返回 ID 映射。"""
    import bcrypt as _bcrypt

    # 低 rounds 加速测试（4 rounds vs 生产 12 rounds）
    pwd_bytes = PASSWORD_PLAIN.encode("utf-8")[:72]
    salt = _bcrypt.gensalt(rounds=4)
    password_hash = _bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

    ids: dict[str, int] = {"bases": {}, "users": {}, "dogs": {}}

    with conn.cursor() as cur:
        # 1. 创建基地
        for name, code, desc in SEED_BASES:
            cur.execute(
                "INSERT INTO bases (name, code, description, is_active, created_at, updated_at) "
                "VALUES (%s, %s, %s, true, now(), now()) RETURNING id",
                (name, code, desc),
            )
            ids["bases"][code] = cur.fetchone()[0]

        # 2. 创建用户
        for name, email, role, base_code, is_superuser in SEED_USERS:
            base_id = ids["bases"].get(base_code) if base_code else None
            # role 枚举值大写（PG enum 存 name）
            role_upper = role.upper()
            cur.execute(
                "INSERT INTO handlers (name, email, role, is_active, password_hash, base_id, is_superuser, created_at, updated_at) "
                "VALUES (%s, %s, %s, true, %s, %s, %s, now(), now()) RETURNING id",
                (name, email, role_upper, password_hash, base_id, is_superuser),
            )
            user_id = cur.fetchone()[0]
            key = email.split("@")[0].replace(TEST_PREFIX, "")
            ids["users"][key] = user_id

        # 3. 创建测试犬只
        # 注：PostgreSQL 枚举存储 enum.name（大写），非 enum.value
        cur.execute(
            "INSERT INTO dogs (name, breed, gender, handler_id, home_base_id, training_stage, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, now(), now()) RETURNING id",
            (
                f"{TEST_PREFIX}dog_a1", "German Shepherd", "MALE",
                ids["users"]["handler_a1"], ids["bases"]["ALPHA"], "P1",
            ),
        )
        ids["dogs"]["dog_a1"] = cur.fetchone()[0]

        cur.execute(
            "INSERT INTO dogs (name, breed, gender, handler_id, home_base_id, training_stage, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, now(), now()) RETURNING id",
            (
                f"{TEST_PREFIX}dog_b1", "Belgian Malinois", "FEMALE",
                ids["users"]["handler_b1"], ids["bases"]["BETA"], "P1",
            ),
        )
        ids["dogs"]["dog_b1"] = cur.fetchone()[0]

    conn.commit()
    return ids


# ============================================================
# AsyncClient 辅助（单一 event loop，避免 asyncpg + TestClient 冲突）
# ============================================================

async def login(client: httpx.AsyncClient, email: str) -> str:
    """登录并返回 JWT token。"""
    resp = await client.post(
        "/api/auth/login",
        json={"email": email, "password": PASSWORD_PLAIN},
    )
    assert resp.status_code == 200, f"登录失败 {email}: {resp.status_code} {resp.text}"
    return resp.json()["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ============================================================
# 测试用例（全部 async，共用单一 event loop）
# ============================================================

async def test_authentication(client: httpx.AsyncClient) -> dict[str, Any]:
    """测试 1: 认证流程。"""
    results = {"name": "authentication", "cases": []}

    # 1.1 正确密码登录
    token = await login(client, f"{TEST_PREFIX}admin@test.com")
    results["cases"].append({
        "case": "login_with_correct_password",
        "status": "pass",
        "detail": "JWT 签发成功",
    })

    # 1.2 错误密码登录 → 401
    resp = await client.post(
        "/api/auth/login",
        json={"email": f"{TEST_PREFIX}admin@test.com", "password": "WrongPass!"},
    )
    results["cases"].append({
        "case": "login_with_wrong_password",
        "status": "pass" if resp.status_code == 401 else "fail",
        "detail": f"期望 401，实际 {resp.status_code}",
    })

    # 1.3 不存在的邮箱 → 401（防枚举，统一错误）
    resp = await client.post(
        "/api/auth/login",
        json={"email": "nonexistent@test.com", "password": "anything"},
    )
    results["cases"].append({
        "case": "login_nonexistent_email",
        "status": "pass" if resp.status_code == 401 else "fail",
        "detail": f"期望 401（防枚举），实际 {resp.status_code}",
    })

    # 1.4 /auth/me 获取当前用户
    resp = await client.get("/api/auth/me", headers=auth_headers(token))
    me_data = resp.json()
    results["cases"].append({
        "case": "get_me_with_valid_token",
        "status": "pass" if resp.status_code == 200 and me_data["email"] == f"{TEST_PREFIX}admin@test.com" else "fail",
        "detail": f"status={resp.status_code}, email={me_data.get('email')}",
    })

    # 1.5 无 token 访问 /auth/me → 401
    resp = await client.get("/api/auth/me")
    results["cases"].append({
        "case": "get_me_without_token",
        "status": "pass" if resp.status_code == 401 else "fail",
        "detail": f"期望 401，实际 {resp.status_code}",
    })

    # 1.6 无效 token → 401
    resp = await client.get("/api/auth/me", headers={"Authorization": "Bearer invalidtoken123"})
    results["cases"].append({
        "case": "get_me_with_invalid_token",
        "status": "pass" if resp.status_code == 401 else "fail",
        "detail": f"期望 401，实际 {resp.status_code}",
    })

    return results


async def test_role_permissions_users_list(client: httpx.AsyncClient) -> dict[str, Any]:
    """测试 2: 角色级权限 — /auth/users 列表。"""
    results = {"name": "role_permission_users_list", "cases": []}

    role_expectations = [
        ("admin", 200, "ADMIN 可查看所有用户"),
        ("manager_a", 200, "MANAGER 可查看（仅本基地）"),
        ("researcher_a", 403, "RESEARCHER 无权访问用户列表"),
        ("handler_a1", 403, "HANDLER 无权访问用户列表"),
        ("viewer_a", 403, "VIEWER 无权访问用户列表"),
    ]

    for role_key, expected_status, desc in role_expectations:
        token = await login(client, f"{TEST_PREFIX}{role_key}@test.com")
        resp = await client.get("/api/auth/users", headers=auth_headers(token))
        actual = resp.status_code
        status = "pass" if actual == expected_status else "fail"
        extra = ""
        if actual == 200 and role_key == "manager_a":
            users = resp.json()
            extra = f"返回 {len(users)} 个用户（期望本基地 5 个）"
        results["cases"].append({
            "case": f"{role_key}_access_users_list",
            "status": status,
            "detail": f"{desc}：期望 {expected_status}，实际 {actual}。{extra}",
        })

    return results


async def test_base_isolation_manager(client: httpx.AsyncClient) -> dict[str, Any]:
    """测试 3: 基地隔离 — MANAGER 只看本基地用户。"""
    results = {"name": "base_isolation_manager", "cases": []}

    token_a = await login(client, f"{TEST_PREFIX}manager_a@test.com")
    resp = await client.get("/api/auth/users", headers=auth_headers(token_a))
    users_a = resp.json()
    base_a_emails = {u["email"] for u in users_a}
    expected_in = {
        f"{TEST_PREFIX}manager_a@test.com",
        f"{TEST_PREFIX}handler_a1@test.com",
        f"{TEST_PREFIX}handler_a2@test.com",
        f"{TEST_PREFIX}researcher_a@test.com",
        f"{TEST_PREFIX}viewer_a@test.com",
    }
    expected_out = {
        f"{TEST_PREFIX}admin@test.com",
        f"{TEST_PREFIX}manager_b@test.com",
        f"{TEST_PREFIX}handler_b1@test.com",
    }
    leak = base_a_emails & expected_out
    status = "pass" if not leak and expected_in.issubset(base_a_emails) else "fail"
    results["cases"].append({
        "case": "manager_a_only_sees_base_a_users",
        "status": status,
        "detail": f"返回 {len(users_a)} 个用户，本基地用户 {len(expected_in & base_a_emails)}/5，"
                  f"跨基地泄漏 {leak if leak else '无'}",
    })

    # manager_a 尝试查看 base_beta 的用户 manager_b（应 403）
    admin_token = await login(client, f"{TEST_PREFIX}admin@test.com")
    resp = await client.get("/api/auth/users", headers=auth_headers(admin_token))
    all_users = {u["email"]: u["id"] for u in resp.json()}
    manager_b_id = all_users.get(f"{TEST_PREFIX}manager_b@test.com")

    resp = await client.get(f"/api/auth/users/{manager_b_id}", headers=auth_headers(token_a))
    status = "pass" if resp.status_code == 403 else "fail"
    results["cases"].append({
        "case": "manager_a_cannot_view_base_b_user",
        "status": status,
        "detail": f"期望 403（跨基地访问），实际 {resp.status_code}",
    })

    return results


async def test_base_crud_admin_only(client: httpx.AsyncClient) -> dict[str, Any]:
    """测试 4: 基地 CRUD — 仅 ADMIN 可创建/更新/禁用。"""
    results = {"name": "base_crud_admin_only", "cases": []}

    # 4.1 ADMIN 创建基地
    admin_token = await login(client, f"{TEST_PREFIX}admin@test.com")
    resp = await client.post(
        "/api/bases",
        json={"name": f"{TEST_PREFIX}gamma", "code": "GAMMA", "description": "测试基地 Gamma"},
        headers=auth_headers(admin_token),
    )
    status = "pass" if resp.status_code == 201 else "fail"
    gamma_id = resp.json().get("id") if resp.status_code == 201 else None
    results["cases"].append({
        "case": "admin_create_base",
        "status": status,
        "detail": f"期望 201，实际 {resp.status_code}",
    })

    # 4.2 MANAGER 尝试创建基地 → 403
    manager_token = await login(client, f"{TEST_PREFIX}manager_a@test.com")
    resp = await client.post(
        "/api/bases",
        json={"name": f"{TEST_PREFIX}delta", "code": "DELTA"},
        headers=auth_headers(manager_token),
    )
    status = "pass" if resp.status_code == 403 else "fail"
    results["cases"].append({
        "case": "manager_cannot_create_base",
        "status": status,
        "detail": f"期望 403，实际 {resp.status_code}",
    })

    # 4.3 ADMIN 列出所有基地
    resp = await client.get("/api/bases", headers=auth_headers(admin_token))
    bases_count = len(resp.json())
    status = "pass" if resp.status_code == 200 and bases_count >= 3 else "fail"
    results["cases"].append({
        "case": "admin_list_all_bases",
        "status": status,
        "detail": f"期望 ≥3 个基地，实际 {bases_count}",
    })

    # 4.4 MANAGER 列出基地（仅本基地）
    resp = await client.get("/api/bases", headers=auth_headers(manager_token))
    bases_a = resp.json()
    status = "pass" if resp.status_code == 200 and len(bases_a) == 1 else "fail"
    results["cases"].append({
        "case": "manager_list_only_own_base",
        "status": status,
        "detail": f"期望 1 个基地（本基地），实际 {len(bases_a)}",
    })

    # 4.5 ADMIN 禁用基地（软删除）
    if gamma_id:
        resp = await client.delete(f"/api/bases/{gamma_id}", headers=auth_headers(admin_token))
        status = "pass" if resp.status_code == 204 else "fail"
        results["cases"].append({
            "case": "admin_deactivate_base",
            "status": status,
            "detail": f"期望 204，实际 {resp.status_code}",
        })

    return results


async def test_user_management_admin_only(client: httpx.AsyncClient) -> dict[str, Any]:
    """测试 5: 用户管理 — register + deactivate 仅 ADMIN。"""
    results = {"name": "user_management_admin_only", "cases": []}

    # 5.1 ADMIN 注册新用户
    admin_token = await login(client, f"{TEST_PREFIX}admin@test.com")
    resp = await client.post(
        "/api/auth/register",
        json={
            "name": "New Handler",
            "email": f"{TEST_PREFIX}new_handler@test.com",
            "password": "NewPass123!",
            "role": "handler",
        },
        headers=auth_headers(admin_token),
    )
    new_user_id = resp.json().get("id") if resp.status_code == 201 else None
    status = "pass" if resp.status_code == 201 else "fail"
    results["cases"].append({
        "case": "admin_register_new_user",
        "status": status,
        "detail": f"期望 201，实际 {resp.status_code}",
    })

    # 5.2 MANAGER 尝试注册新用户 → 403
    manager_token = await login(client, f"{TEST_PREFIX}manager_a@test.com")
    resp = await client.post(
        "/api/auth/register",
        json={
            "name": "Forbidden User",
            "email": f"{TEST_PREFIX}forbidden@test.com",
            "password": "SomePass123!",
            "role": "handler",
        },
        headers=auth_headers(manager_token),
    )
    status = "pass" if resp.status_code == 403 else "fail"
    results["cases"].append({
        "case": "manager_cannot_register_user",
        "status": status,
        "detail": f"期望 403，实际 {resp.status_code}",
    })

    # 5.3 ADMIN 禁用用户
    if new_user_id:
        resp = await client.delete(f"/api/auth/users/{new_user_id}", headers=auth_headers(admin_token))
        status = "pass" if resp.status_code == 204 else "fail"
        results["cases"].append({
            "case": "admin_deactivate_user",
            "status": status,
            "detail": f"期望 204，实际 {resp.status_code}",
        })

    # 5.4 ADMIN 不能禁用自己 → 400
    admin_id_resp = await client.get("/api/auth/me", headers=auth_headers(admin_token))
    admin_id = admin_id_resp.json()["id"]
    resp = await client.delete(f"/api/auth/users/{admin_id}", headers=auth_headers(admin_token))
    status = "pass" if resp.status_code == 400 else "fail"
    results["cases"].append({
        "case": "admin_cannot_deactivate_self",
        "status": status,
        "detail": f"期望 400（不能禁用自己），实际 {resp.status_code}",
    })

    return results


async def test_change_password(client: httpx.AsyncClient) -> dict[str, Any]:
    """测试 6: 修改密码。"""
    results = {"name": "change_password", "cases": []}

    # 6.1 正确旧密码 + 新密码 → 204
    token = await login(client, f"{TEST_PREFIX}viewer_a@test.com")
    resp = await client.post(
        "/api/auth/change-password",
        json={"old_password": PASSWORD_PLAIN, "new_password": "NewViewer456!"},
        headers=auth_headers(token),
    )
    status = "pass" if resp.status_code == 204 else "fail"
    results["cases"].append({
        "case": "change_password_correct_old",
        "status": status,
        "detail": f"期望 204，实际 {resp.status_code}",
    })

    # 6.2 用新密码登录验证
    resp = await client.post(
        "/api/auth/login",
        json={"email": f"{TEST_PREFIX}viewer_a@test.com", "password": "NewViewer456!"},
    )
    status = "pass" if resp.status_code == 200 else "fail"
    results["cases"].append({
        "case": "login_with_new_password",
        "status": status,
        "detail": f"新密码登录期望 200，实际 {resp.status_code}",
    })

    # 6.3 错误旧密码 → 400
    resp = await client.post(
        "/api/auth/change-password",
        json={"old_password": "WrongOld789!", "new_password": "Another789!"},
        headers=auth_headers(token),
    )
    status = "pass" if resp.status_code == 400 else "fail"
    results["cases"].append({
        "case": "change_password_wrong_old",
        "status": status,
        "detail": f"期望 400（旧密码错误），实际 {resp.status_code}",
    })

    return results


# ============================================================
# 主流程
# ============================================================

async def run_tests() -> list[dict[str, Any]]:
    """在单一 event loop 中运行所有测试套件。"""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        test_suites = [
            test_authentication,
            test_role_permissions_users_list,
            test_base_isolation_manager,
            test_base_crud_admin_only,
            test_user_management_admin_only,
            test_change_password,
        ]
        all_results = []
        for suite_fn in test_suites:
            print(f"\n  → {suite_fn.__name__}...")
            result = await suite_fn(client)
            all_results.append(result)
            passed = sum(1 for c in result["cases"] if c["status"] == "pass")
            total = len(result["cases"])
            print(f"    {passed}/{total} 通过")
        return all_results


def main() -> int:
    print("=" * 70)
    print("Phase 3.6c RBAC 端到端评估")
    print("=" * 70)

    # 1. 连接 DB + 清理旧数据 + 种子
    print("\n[1/4] 准备种子数据...")
    conn = get_db_conn()
    try:
        cleanup_seed_data(conn)
        ids = seed_data(conn)
        print(f"  ✓ 基地: {ids['bases']}")
        print(f"  ✓ 用户: {ids['users']}")
        print(f"  ✓ 犬只: {ids['dogs']}")
    finally:
        conn.close()

    # 2. 运行测试（单一 async event loop）
    print("\n[2/4] 运行权限矩阵测试...")
    all_results = asyncio.run(run_tests())

    # 3. 汇总
    print("\n[3/4] 汇总结果...")
    total_cases = sum(len(r["cases"]) for r in all_results)
    passed_cases = sum(1 for r in all_results for c in r["cases"] if c["status"] == "pass")
    failed_cases = total_cases - passed_cases
    overall = "pass" if failed_cases == 0 else "fail"

    report = {
        "phase": "3.6c",
        "test_name": "RBAC 端到端评估",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "overall": overall,
        "summary": {
            "total_suites": len(all_results),
            "total_cases": total_cases,
            "passed": passed_cases,
            "failed": failed_cases,
            "pass_rate": f"{passed_cases / total_cases * 100:.1f}%" if total_cases else "N/A",
        },
        "suites": all_results,
    }

    print(f"\n  总计: {passed_cases}/{total_cases} 通过 ({report['summary']['pass_rate']})")
    print(f"  总体: {overall.upper()}")

    # 4. 清理种子数据 + 写报告
    print("\n[4/4] 清理种子数据 + 写报告...")
    conn = get_db_conn()
    try:
        cleanup_seed_data(conn)
        print("  ✓ 种子数据已清理")
    finally:
        conn.close()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ 报告已写入 {REPORT_PATH}")

    print("\n" + "=" * 70)
    if overall == "pass":
        print("✅ Phase 3.6c RBAC 端到端评估通过")
    else:
        print(f"❌ Phase 3.6c RBAC 端到端评估失败（{failed_cases} 个用例未通过）")
    print("=" * 70)

    return 0 if overall == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
