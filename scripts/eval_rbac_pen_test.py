"""P2-3 多基地 RBAC 活体渗透测试（protocol pen-v0.1）.

Owner: 稳健性线（渗透 + 发布 + 预案）
Phase: P2-3（dev-docs/production-readiness-gap-plan.md:200）

本脚本做三件事:
    1. 对**活体** HTTP 服务（默认 127.0.0.1:8000）做未认证扫描 / 跨基地隔离 /
       角色边界 / JWT 面渗透，产出可入库的脱敏 JSON 报告；
    2. 用 ``--self-check`` 做纯离线用例矩阵 lint（不联网、不加载配置、不是 pytest 测试）；
    3. 用 ``--write-baseline`` / ``--baseline`` 支持基线比对，避免“首次发现即永久红”。

重要边界（实现时已写入代码，不得在报告外宣称）:
    - ``--self-check`` 只是本脚本内部 lint，**不接入任何自动回归门**；
      自动门仍需 `backend/tests/` 下的 pytest 用例（本文件不是 pytest 测试）。
    - 报告会随仓库提交（``git check-ignore`` 对其返回 1），因此脱敏是硬门：
      任何疑似凭据/JWT/种子密码命中都会导致**不写报告并 exit 2**。
    - 被测服务没有 ``--reload``；脚本用「服务启动时间 vs 模块 mtime」判定
      模块是否陈旧。陈旧模块的相关用例标 ``invalid``，绝不判 pass。
    - 默认只允许 loopback；远程目标需 ``--allow-remote``，写探针另需
      ``--allow-remote-writes``。

退出码:
    0 = 执行成功且无达到 --fail-on 阈值的发现
    1 = 达到阈值
    2 = 前置失败（服务/健康门/DB/种子/404 前置/快照/清理/脱敏/远程未授权）

运行:
    .venv\\Scripts\\python.exe scripts/eval_rbac_pen_test.py --self-check
    .venv\\Scripts\\python.exe scripts/eval_rbac_pen_test.py --write-baseline reports/p2-3-rbac-pen-baseline.json
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPORTS_DIR = PROJECT_ROOT / "reports"
PROTOCOL_VERSION = "pen-v0.1"
TASK_ID = "P2-3"
SEED_PREFIX = "pen_rbac_"
SEED_PASSWORD = "TestPass123!"
VICTIM_PDF_NAME = "pen_rbac_victim_report.pdf"
DEV_AUTH_SECRET_MARKER = "DEV-ONLY-"

SEVERITY_RANK = {"NONE": 0, "INFO": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
SEVERITIES = ("CRITICAL", "HIGH", "MEDIUM", "INFO", "NONE")

SEVERITY_POLICY = {
    "CRITICAL": (
        "跨租户写/删成功；未认证访问受保护端点成功；"
        "伪造/篡改/禁用账号 token 被接受；非超管创建超管"
    ),
    "HIGH": "跨租户读/列表泄漏；畸形 token 触发 5xx；VIEWER 跨基地创建标注任务",
    "MEDIUM": "审计日志缺失（P2-3 验收缺口）",
    "INFO": "health 暴露 redis URL；openapi 公开；role claim 被忽略",
}

EXIT_POLICY = {
    "0": "执行成功且无达到 --fail-on 阈值的发现",
    "1": "达到阈值",
    "2": "前置失败（服务/健康门/DB/种子/404 前置/快照/清理/脱敏/远程未授权）",
    "fail_on": {
        "critical": "仅 CRITICAL 触发 1（默认）",
        "high": "CRITICAL + HIGH 触发 1",
        "any": "CRITICAL + HIGH + MEDIUM 触发 1",
        "none": "永不因发现触发 1（前置失败仍可 2）",
    },
    "baseline": (
        "提供 --baseline 时，只有新增或严重度升级的发现触发阈值；"
        "已修复项列入 resolved。--write-baseline 写完后仍按当前发现执行阈值。"
    ),
}

MODULE_PATHS = {
    "auth": "backend/app/api/auth.py",
    "dogs": "backend/app/api/dogs.py",
    "videos": "backend/app/api/videos.py",
    "scores": "backend/app/api/scores.py",
    "annotations": "backend/app/api/annotations.py",
    "bases": "backend/app/api/bases.py",
    "scoring": "backend/app/api/scoring.py",
    "models": "backend/app/api/models.py",
    "finetune": "backend/app/api/finetune.py",
    "health": "backend/app/api/health.py",
    "llm_explainer": "backend/app/api/llm_explainer.py",
    "deps": "backend/app/core/deps.py",
    "security": "backend/app/core/security.py",
    "main": "backend/app/main.py",
    "schemas": "backend/app/schemas/common.py",
}

# 被测模块逐文件快照（计划要求的最小集合 + llm_explainer）。
TRACKED_MODULES = [
    "auth", "dogs", "videos", "scores", "annotations", "bases", "scoring",
    "models", "finetune", "health", "llm_explainer", "deps", "security",
    "main", "schemas",
]

PUBLIC_OPERATIONS = {
    ("GET", "/"),
    ("GET", "/health"),
    ("GET", "/docs"),
    ("GET", "/redoc"),
    ("GET", "/openapi.json"),
    ("POST", "/api/auth/login"),
}

SEED_BASES = {
    "A": (SEED_PREFIX + "alpha", "PEN_RBAC_ALPHA"),
    "B": (SEED_PREFIX + "beta", "PEN_RBAC_BETA"),
}

# (key, name, role, base, is_superuser, is_active)
SEED_ACCOUNTS = [
    ("superadmin", "Pen Super", "ADMIN", None, True, True),
    ("admin_nosuper", "Pen Admin NoSuper", "ADMIN", "A", False, True),
    ("manager_a", "Pen Manager A", "MANAGER", "A", False, True),
    ("manager_b", "Pen Manager B", "MANAGER", "B", False, True),
    ("handler_a1", "Pen Handler A1", "HANDLER", "A", False, True),
    ("handler_b1", "Pen Handler B1", "HANDLER", "B", False, True),
    ("researcher_a", "Pen Researcher A", "RESEARCHER", "A", False, True),
    ("viewer_a", "Pen Viewer A", "VIEWER", "A", False, True),
    ("viewer_b", "Pen Viewer B", "VIEWER", "B", False, True),
    ("victim_user_b", "Pen Victim User B", "HANDLER", "B", False, True),
    ("disabled_a", "Pen Disabled A", "HANDLER", "A", False, False),
]

SPECIAL_IDENTITIES = {
    "anonymous", "jwt_no_header", "jwt_bare", "jwt_garbage", "jwt_wrong_key",
    "jwt_expired", "jwt_alg_none", "jwt_nonexistent_sub", "jwt_disabled",
    "jwt_missing_sub", "jwt_sub_non_numeric", "jwt_role_garbage",
    "jwt_tampered_base", "jwt_secret_match",
}

IDENTITY_SET = {a[0] for a in SEED_ACCOUNTS} | SPECIAL_IDENTITIES

CATEGORY_REMEDIATION = {
    "cross_base_read": (
        "在路由中调用 deps.check_dog_access / check_base_access（或按 me.base_id 过滤），"
        "对 404/403 语义统一，避免跨基地枚举。"
    ),
    "cross_base_write": (
        "写路由必须叠加资源级归属校验（check_dog_access(require_write=True)），"
        "不能只依赖角色门；跨基地删除/修改返回 403。"
    ),
    "list_leak": (
        "列表端点按 me.base_id / handler 归属过滤；ADMIN/超管才返回全量；"
        "MANAGER 只看本基地，HANDLER 只看自有或关联犬只。"
    ),
    "jwt_robust": (
        "decode_access_token 对缺失/非数字 sub 抛 InvalidTokenError（401），"
        "补充畸形 token 单元测试，禁止把 500 当作受保护。"
    ),
    "role_gate": "角色门应拒绝该身份；补充该角色的负向单元测试。",
    "audit": (
        "实现审计日志表 + 中间件记录写操作（actor/action/resource/时间/结果），"
        "覆盖 P2-3 的‘审计日志’验收半边。"
    ),
    "info": "按最小暴露原则收敛（health 不回显 redis URL；非开发环境关闭 openapi/docs）。",
    "positive": "正向对照，无需修复；若失败说明环境/种子/precondition 异常。",
}


# ============================================================
# 用例定义（静态矩阵；self-check 只 lint 这一部分 + 动态生成器契约）
# ============================================================


def _spec(
    case_id: str,
    suite: str,
    method: str,
    endpoint: str,
    identity: str,
    expected_status,
    severity_if_fail: str,
    modules: list[str],
    *,
    title: str = "",
    category: str = "positive",
    body_kind: str | None = None,
    check_kind: str | None = None,
    auth_kind: str = "identity",
    victim: bool = False,
    forbid: list[str] | None = None,
    dangerous_positive: bool = False,
    note: str = "",
) -> dict:
    return {
        "case_id": case_id,
        "suite": suite,
        "method": method,
        "endpoint": endpoint,
        "identity": identity,
        "expected_status": expected_status,
        "severity_if_fail": severity_if_fail,
        "modules": modules,
        "title": title or case_id,
        "category": category,
        "body_kind": body_kind,
        "check_kind": check_kind,
        "auth_kind": auth_kind,
        "victim": victim,
        "forbid": forbid or [],
        "dangerous_positive": dangerous_positive,
        "note": note,
    }


def static_case_specs() -> list[dict]:
    """静态用例矩阵（含 suite B / C）。suite A 由 OpenAPI 动态生成。"""
    cases: list[dict] = []
    a_identities = ["manager_a", "researcher_a", "handler_a1", "viewer_a"]

    # ---------- suite B: 跨基地隔离 ----------
    for ident in a_identities:
        cases.append(_spec(
            f"B-dog-get-{ident}", "B", "GET", "/api/dogs/{ids.victim_dog_b}",
            ident, 403, "HIGH", ["dogs"], victim=True,
            category="cross_base_read", title="跨基地读取犬只档案",
        ))
    cases.append(_spec(
        "B-dog-get-viewer_b", "B", "GET", "/api/dogs/{ids.victim_dog_b}",
        "viewer_b", 200, "NONE", ["dogs"], victim=True,
        category="positive", title="B 基地 viewer 读取本基地犬只（正向对照）",
    ))
    for ident in ["manager_a", "researcher_a", "handler_a1", "viewer_a"]:
        cases.append(_spec(
            f"B-dog-put-{ident}", "B", "PUT", "/api/dogs/{ids.victim_dog_b}",
            ident, 403, "CRITICAL", ["dogs"], victim=True,
            body_kind="dog_noop", category="cross_base_write",
            title="跨基地修改犬只档案",
        ))
    for ident in a_identities:
        cases.append(_spec(
            f"B-dog-delete-{ident}", "B", "DELETE",
            "/api/dogs/{ids.victim_dog_b_del_" + ident + "}", ident, 403,
            "CRITICAL", ["dogs"], victim=True, category="cross_base_write",
            title="跨基地删除犬只档案",
        ))
    for ident in ["manager_a", "viewer_a"]:
        cases.append(_spec(
            f"B-dog-list-{ident}", "B", "GET", "/api/dogs", ident, 200,
            "HIGH", ["dogs"], category="list_leak",
            check_kind="no_foreign_ids", forbid=["victim_dog_b"],
            title="犬只列表跨基地泄漏",
        ))

    # videos（运行期若 videos.py 陈旧则整块 invalid）
    for ident in a_identities:
        cases.append(_spec(
            f"B-video-get-{ident}", "B", "GET",
            "/api/videos/{ids.victim_video_b}", ident, 403, "HIGH",
            ["videos"], victim=True, category="cross_base_read",
            title="跨基地读取视频元数据",
        ))
    for ident in ["manager_a", "viewer_a"]:
        cases.append(_spec(
            f"B-video-status-{ident}", "B", "GET",
            "/api/videos/{ids.victim_video_b}/status", ident, 403, "HIGH",
            ["videos"], victim=True, category="cross_base_read",
            title="跨基地轮询视频状态",
        ))
        cases.append(_spec(
            f"B-video-report-{ident}", "B", "GET",
            "/api/videos/{ids.victim_video_b}/report", ident, 403, "HIGH",
            ["videos"], victim=True, category="cross_base_read",
            title="跨基地下载 PDF 报告",
        ))
    cases.append(_spec(
        "B-video-list-manager_a", "B", "GET",
        "/api/videos?dog_id={ids.victim_dog_b}", "manager_a", 200, "HIGH",
        ["videos"], victim=True, category="list_leak",
        check_kind="no_foreign_ids", forbid=["victim_video_b"],
        title="视频列表按 dog_id 跨基地泄漏",
    ))
    cases.append(_spec(
        "B-video-report-viewer_b", "B", "GET",
        "/api/videos/{ids.victim_video_b}/report", "viewer_b", 200, "NONE",
        ["videos"], victim=True, category="positive",
        check_kind="report_pdf", title="B 基地 viewer 下载本基地报告（正向对照）",
    ))

    # scores
    for ident in a_identities:
        cases.append(_spec(
            f"B-score-get-{ident}", "B", "GET",
            "/api/scores/{ids.victim_score_b}", ident, 403, "HIGH",
            ["scores"], victim=True, category="cross_base_read",
            title="跨基地读取单条评分",
        ))
        cases.append(_spec(
            f"B-score-by-dog-{ident}", "B", "GET",
            "/api/scores/by-dog/{ids.victim_dog_b}", ident, 403, "HIGH",
            ["scores"], victim=True, category="cross_base_read",
            title="跨基地读取犬只历史评分",
        ))
    for ident in ["manager_a", "viewer_a"]:
        cases.append(_spec(
            f"B-score-compare-{ident}", "B", "GET",
            "/api/scores/compare?dog_ids={ids.victim_dog_b}", ident, 403,
            "HIGH", ["scores"], victim=True, category="cross_base_read",
            title="跨基地训练对比查询",
        ))
        cases.append(_spec(
            f"B-score-list-{ident}", "B", "GET", "/api/scores", ident, 200,
            "HIGH", ["scores"], category="list_leak",
            check_kind="no_foreign_ids", forbid=["victim_score_b"],
            title="评分列表跨基地泄漏",
        ))
    cases.append(_spec(
        "B-score-get-viewer_b", "B", "GET",
        "/api/scores/{ids.victim_score_b}", "viewer_b", 200, "NONE",
        ["scores"], victim=True, category="positive",
        title="B 基地 viewer 读取本基地评分（正向对照）",
    ))

    # annotations（运行期若 annotations.py 陈旧则整块 invalid）
    for ident in ["manager_a", "viewer_a"]:
        cases.append(_spec(
            f"B-anno-task-get-{ident}", "B", "GET",
            "/api/annotations/tasks/{ids.victim_task_b}", ident, 403, "HIGH",
            ["annotations"], victim=True, category="cross_base_read",
            title="跨基地读取标注任务",
        ))
        cases.append(_spec(
            f"B-anno-task-annotations-{ident}", "B", "GET",
            "/api/annotations/tasks/{ids.victim_task_b}/annotations", ident,
            403, "HIGH", ["annotations"], victim=True,
            category="cross_base_read", title="跨基地读取标注数据",
        ))
        cases.append(_spec(
            f"B-anno-task-list-{ident}", "B", "GET",
            "/api/annotations/tasks", ident, 200, "HIGH", ["annotations"],
            category="list_leak", check_kind="no_foreign_ids",
            forbid=["victim_task_b"], title="标注任务列表跨基地泄漏",
        ))
    for ident in ["admin_nosuper", "manager_a", "handler_a1"]:
        cases.append(_spec(
            f"B-anno-task-create-{ident}", "B", "POST",
            "/api/annotations/tasks", ident, 201, "NONE", ["annotations"],
            victim=True, body_kind="annotation_task", category="positive",
            title="有权角色创建标注任务（正向对照）",
        ))
    cases.append(_spec(
        "B-anno-task-create-viewer_a", "B", "POST", "/api/annotations/tasks",
        "viewer_a", 403, "HIGH", ["annotations"], victim=True,
        body_kind="annotation_task", category="cross_base_write",
        title="VIEWER 跨基地创建标注任务",
    ))

    # 正向回归（这些应继续被拒绝）
    cases.extend([
        _spec("B-users-get-manager_a", "B", "GET",
              "/api/auth/users/{ids.manager_b}", "manager_a", 403, "HIGH",
              ["auth"], category="cross_base_read",
              title="MANAGER 跨基地读取用户详情"),
        _spec("B-base-get-manager_a", "B", "GET",
              "/api/bases/{ids.base_b}", "manager_a", 403, "HIGH",
              ["bases"], category="cross_base_read",
              title="MANAGER 跨基地读取基地详情"),
        _spec("B-users-put-manager_a", "B", "PUT",
              "/api/auth/users/{ids.victim_user_b}", "manager_a", 403, "HIGH",
              ["auth"], body_kind="user_noop", category="cross_base_write",
              title="MANAGER 跨基地修改用户"),
        _spec("B-users-delete-manager_a", "B", "DELETE",
              "/api/auth/users/{ids.victim_user_b}", "manager_a", 403,
              "CRITICAL", ["auth"], category="cross_base_write",
              title="MANAGER 跨基地禁用用户"),
        _spec("B-users-list-manager_a", "B", "GET", "/api/auth/users",
              "manager_a", 200, "HIGH", ["auth"], category="list_leak",
              check_kind="no_foreign_ids",
              forbid=["manager_b", "handler_b1", "viewer_b", "victim_user_b"],
              title="用户列表跨基地泄漏"),
    ])

    # ---------- suite C: 角色边界 + JWT 面 ----------
    cases.extend([
        _spec("C-register-superadmin", "C", "POST", "/api/auth/register",
              "superadmin", 201, "HIGH", ["auth"],
              body_kind="register_normal", category="role_gate",
              title="超管创建普通用户"),
        _spec("C-register-admin_nosuper", "C", "POST", "/api/auth/register",
              "admin_nosuper", 201, "CRITICAL", ["auth"],
              body_kind="register_super_attempt", category="role_gate",
              check_kind="is_superuser_false",
              title="非超管尝试创建超管（响应必须降级为 false）"),
        _spec("C-register-manager_a", "C", "POST", "/api/auth/register",
              "manager_a", 403, "HIGH", ["auth"],
              body_kind="register_normal", category="role_gate",
              title="MANAGER 创建用户"),
        _spec("C-register-handler_a1", "C", "POST", "/api/auth/register",
              "handler_a1", 403, "HIGH", ["auth"],
              body_kind="register_normal", category="role_gate",
              title="HANDLER 创建用户"),
        _spec("C-register-viewer_a", "C", "POST", "/api/auth/register",
              "viewer_a", 403, "HIGH", ["auth"],
              body_kind="register_normal", category="role_gate",
              title="VIEWER 创建用户"),
        _spec("C-register-disabled_a", "C", "POST", "/api/auth/register",
              "disabled_a", 401, "HIGH", ["auth"],
              body_kind="register_normal", category="role_gate",
              title="禁用账号创建用户"),

        _spec("C-users-list-superadmin", "C", "GET", "/api/auth/users",
              "superadmin", 200, "NONE", ["auth"], category="positive",
              title="超管列出用户"),
        _spec("C-users-list-admin_nosuper", "C", "GET", "/api/auth/users",
              "admin_nosuper", 200, "NONE", ["auth"], category="positive",
              title="ADMIN 列出用户"),
        _spec("C-users-list-manager_a", "C", "GET", "/api/auth/users",
              "manager_a", 200, "NONE", ["auth"], category="positive",
              title="MANAGER 列出本基地用户"),
        _spec("C-users-list-handler_a1", "C", "GET", "/api/auth/users",
              "handler_a1", 403, "HIGH", ["auth"], category="role_gate",
              title="HANDLER 列出用户"),
        _spec("C-users-list-viewer_a", "C", "GET", "/api/auth/users",
              "viewer_a", 403, "HIGH", ["auth"], category="role_gate",
              title="VIEWER 列出用户"),
        _spec("C-users-list-disabled_a", "C", "GET", "/api/auth/users",
              "disabled_a", 401, "HIGH", ["auth"], category="role_gate",
              title="禁用账号列出用户"),

        _spec("C-user-delete-superadmin", "C", "DELETE",
              "/api/auth/users/{ids.del_super}", "superadmin", 204, "HIGH",
              ["auth"], category="positive", title="超管禁用临时用户"),
        _spec("C-user-delete-admin_nosuper", "C", "DELETE",
              "/api/auth/users/{ids.del_admin}", "admin_nosuper", 204, "HIGH",
              ["auth"], category="positive", title="ADMIN 禁用临时用户"),
        _spec("C-user-delete-manager_a", "C", "DELETE",
              "/api/auth/users/{ids.victim_user_b}", "manager_a", 403,
              "CRITICAL", ["auth"], category="role_gate",
              title="MANAGER 禁用用户"),

        _spec("C-base-create-superadmin", "C", "POST", "/api/bases",
              "superadmin", 201, "HIGH", ["bases"],
              body_kind="base_create_super", category="role_gate",
              title="超管创建基地（临时 pen_rbac_ 前缀）"),
        _spec("C-base-create-admin_nosuper", "C", "POST", "/api/bases",
              "admin_nosuper", 201, "HIGH", ["bases"],
              body_kind="base_create_admin", category="role_gate",
              title="ADMIN 创建基地（临时 pen_rbac_ 前缀）"),
        _spec("C-base-create-manager_a", "C", "POST", "/api/bases",
              "manager_a", 403, "HIGH", ["bases"],
              body_kind="base_create_manager", category="role_gate",
              title="MANAGER 创建基地"),
        _spec("C-base-create-handler_a1", "C", "POST", "/api/bases",
              "handler_a1", 403, "HIGH", ["bases"],
              body_kind="base_create_handler", category="role_gate",
              title="HANDLER 创建基地"),

        _spec("C-dog-create-handler_a1", "C", "POST", "/api/dogs",
              "handler_a1", 201, "HIGH", ["dogs"],
              body_kind="dog_create", category="positive",
              title="HANDLER 创建犬只（临时 pen_rbac_ 前缀）"),
        _spec("C-dog-create-viewer_a", "C", "POST", "/api/dogs",
              "viewer_a", 403, "CRITICAL", ["dogs"],
              body_kind="dog_create_viewer", category="role_gate",
              title="VIEWER 创建犬只"),

        _spec("C-scoring-put-superadmin", "C", "PUT",
              "/api/scoring/configs/obedience_trial", "superadmin", 200,
              "CRITICAL", ["scoring"], body_kind="scoring_put",
              dangerous_positive=True, category="role_gate",
              title="超管写评分卡（危险正向，SKIP）"),
        _spec("C-scoring-put-admin_nosuper", "C", "PUT",
              "/api/scoring/configs/obedience_trial", "admin_nosuper", 200,
              "CRITICAL", ["scoring"], body_kind="scoring_put",
              dangerous_positive=True, category="role_gate",
              title="ADMIN 写评分卡（危险正向，SKIP）"),
        _spec("C-scoring-put-manager_a", "C", "PUT",
              "/api/scoring/configs/obedience_trial", "manager_a", 403,
              "CRITICAL", ["scoring"], body_kind="scoring_put",
              category="role_gate", title="MANAGER 写评分卡"),
        _spec("C-scoring-put-viewer_a", "C", "PUT",
              "/api/scoring/configs/obedience_trial", "viewer_a", 403,
              "CRITICAL", ["scoring"], body_kind="scoring_put",
              category="role_gate", title="VIEWER 写评分卡"),

        _spec("C-model-activate-superadmin", "C", "POST",
              "/api/models/{ids.model_pose}/activate", "superadmin", 200,
              "CRITICAL", ["models"], dangerous_positive=True,
              category="role_gate", title="超管激活模型（危险正向，SKIP）"),
        _spec("C-model-activate-manager_a", "C", "POST",
              "/api/models/{ids.model_pose}/activate", "manager_a", 200,
              "CRITICAL", ["models"], dangerous_positive=True,
              category="role_gate", title="MANAGER 激活模型（危险正向，SKIP）"),
        _spec("C-model-activate-viewer_a", "C", "POST",
              "/api/models/{ids.model_pose}/activate", "viewer_a", 403,
              "CRITICAL", ["models"], category="role_gate",
              title="VIEWER 激活模型"),
        _spec("C-model-activate-handler_a1", "C", "POST",
              "/api/models/{ids.model_pose}/activate", "handler_a1", 403,
              "CRITICAL", ["models"], category="role_gate",
              title="HANDLER 激活模型"),

        _spec("C-finetune-status-superadmin", "C", "GET",
              "/api/finetune/status", "superadmin", 200, "NONE", ["finetune"],
              category="positive", title="超管查询飞轮状态"),
        _spec("C-finetune-status-handler_a1", "C", "GET",
              "/api/finetune/status", "handler_a1", 200, "NONE", ["finetune"],
              category="positive", title="HANDLER 查询飞轮状态（只读）"),
        _spec("C-finetune-status-viewer_a", "C", "GET",
              "/api/finetune/status", "viewer_a", 200, "NONE", ["finetune"],
              category="positive", title="VIEWER 查询飞轮状态（只读）"),
        _spec("C-finetune-trigger-superadmin", "C", "POST",
              "/api/finetune/trigger", "superadmin", 202, "HIGH",
              ["finetune"], body_kind="finetune_trigger",
              dangerous_positive=True, category="role_gate",
              title="超管触发微调（危险正向，SKIP）"),
        _spec("C-finetune-trigger-manager_a", "C", "POST",
              "/api/finetune/trigger", "manager_a", 202, "HIGH",
              ["finetune"], body_kind="finetune_trigger",
              dangerous_positive=True, category="role_gate",
              title="MANAGER 触发微调（危险正向，SKIP）"),
        _spec("C-finetune-trigger-viewer_a", "C", "POST",
              "/api/finetune/trigger", "viewer_a", 403, "HIGH",
              ["finetune"], body_kind="finetune_trigger",
              category="role_gate", title="VIEWER 触发微调"),
        _spec("C-finetune-trigger-handler_a1", "C", "POST",
              "/api/finetune/trigger", "handler_a1", 403, "HIGH",
              ["finetune"], body_kind="finetune_trigger",
              category="role_gate", title="HANDLER 触发微调"),

        _spec("C-video-upload-superadmin", "C", "POST", "/api/videos/upload",
              "superadmin", 201, "HIGH", ["videos"], body_kind="upload",
              dangerous_positive=True, category="role_gate",
              title="超管上传视频（危险正向，SKIP）"),
        _spec("C-video-upload-viewer_a", "C", "POST", "/api/videos/upload",
              "viewer_a", 403, "CRITICAL", ["videos"], body_kind="upload",
              category="role_gate", title="VIEWER 上传视频"),

        _spec("C-anno-create-admin_nosuper", "C", "POST",
              "/api/annotations/tasks", "admin_nosuper", 201, "NONE",
              ["annotations"], victim=True, body_kind="annotation_task",
              category="positive", title="ADMIN 创建标注任务（正向对照）"),
        _spec("C-anno-create-manager_a", "C", "POST",
              "/api/annotations/tasks", "manager_a", 201, "NONE",
              ["annotations"], victim=True, body_kind="annotation_task",
              category="positive", title="MANAGER 创建标注任务（正向对照）"),
        _spec("C-anno-create-handler_a1", "C", "POST",
              "/api/annotations/tasks", "handler_a1", 201, "NONE",
              ["annotations"], victim=True, body_kind="annotation_task",
              category="positive", title="HANDLER 创建标注任务（正向对照）"),
        _spec("C-anno-create-viewer_a", "C", "POST",
              "/api/annotations/tasks", "viewer_a", 403, "HIGH",
              ["annotations"], victim=True, body_kind="annotation_task",
              category="role_gate", title="VIEWER 创建标注任务"),
    ])

    # JWT 面
    jwt_cases = [
        ("C-jwt-login-superadmin", "POST", "/api/auth/login", "superadmin",
         200, "HIGH", ["auth"], "jwt_secret_match", "login_super",
         "positive", "真实登录获取 token", "has_token"),
        ("C-jwt-me-positive", "GET", "/api/auth/me", "jwt_secret_match",
         200, "NONE", ["auth"], "jwt_secret_match", None, "positive",
         "本地 secret 签发的超管 token 通过 /me（secret 匹配正向对照）", None),
        ("C-jwt-no-header", "GET", "/api/auth/me", "jwt_no_header",
         401, "HIGH", ["deps", "security"], "none", None, "jwt_robust",
         "无 Authorization 头", None),
        ("C-jwt-bare-bearer", "GET", "/api/auth/me", "jwt_bare",
         401, "HIGH", ["deps", "security"], "bare", None, "jwt_robust",
         "Authorization: Bearer（无令牌、无尾空格）", None),
        ("C-jwt-garbage", "GET", "/api/auth/me", "jwt_garbage",
         401, "HIGH", ["deps", "security"], "garbage", None, "jwt_robust",
         "垃圾 token", None),
        ("C-jwt-wrong-key", "GET", "/api/auth/me", "jwt_wrong_key",
         401, "CRITICAL", ["deps", "security"], "wrong_key", None, "jwt_robust",
         "错误密钥签名 token", None),
        ("C-jwt-expired", "GET", "/api/auth/me", "jwt_expired",
         401, "CRITICAL", ["deps", "security"], "expired", None, "jwt_robust",
         "过期 token", None),
        ("C-jwt-alg-none", "GET", "/api/auth/me", "jwt_alg_none",
         401, "CRITICAL", ["deps", "security"], "alg_none", None, "jwt_robust",
         "alg=none 手工拼装 token", None),
        ("C-jwt-nonexistent-sub", "GET", "/api/auth/me",
         "jwt_nonexistent_sub", 401, "CRITICAL", ["deps", "security"],
         "nonexistent_sub", None, "jwt_robust", "不存在用户 sub", None),
        ("C-jwt-disabled", "GET", "/api/auth/me", "jwt_disabled",
         401, "CRITICAL", ["deps", "security"], "disabled", None, "jwt_robust",
         "禁用账号正确签名 token", None),
        ("C-jwt-missing-sub", "GET", "/api/auth/me", "jwt_missing_sub",
         401, "HIGH", ["deps", "security"], "missing_sub", None, "jwt_robust",
         "缺失 sub claim（实测 500）", None),
        ("C-jwt-sub-non-numeric", "GET", "/api/auth/me",
         "jwt_sub_non_numeric", 401, "HIGH", ["deps", "security"],
         "sub_non_numeric", None, "jwt_robust", "sub 非数字（实测 500）", None),
        ("C-jwt-role-garbage", "GET", "/api/auth/me", "jwt_role_garbage",
         200, "INFO", ["deps", "security"], "role_garbage", None, "jwt_robust",
         "role claim 垃圾值（DB role 为准，实测 200）", None),
        ("C-jwt-tampered-base-bases", "GET", "/api/bases/{ids.base_b}",
         "jwt_tampered_base", 403, "CRITICAL", ["bases", "deps", "security"],
         "tampered_base", None, "jwt_robust",
         "manager_a token 篡改 base_id=base_b 后读取 B 基地", None),
        ("C-jwt-tampered-base-users", "GET", "/api/auth/users",
         "jwt_tampered_base", 200, "CRITICAL", ["auth", "deps", "security"],
         "tampered_base", None, "jwt_robust",
         "manager_a token 篡改 base_id=base_b 后列用户（不得含 B 用户）",
         "tampered_no_b_users"),
    ]
    for (cid, method, endpoint, ident, expected, sev, mods, auth_kind,
         body_kind, cat, title, check) in jwt_cases:
        cases.append(_spec(
            cid, "C", method, endpoint, ident, expected, sev, mods,
            title=title, category=cat, body_kind=body_kind,
            check_kind=check, auth_kind=auth_kind,
        ))

    return cases


def dynamic_anonymous_specs(openapi: dict) -> list[dict]:
    """由 OpenAPI 动态生成未认证扫描用例（不硬编码端点数量）。"""
    specs: list[dict] = []
    paths = openapi.get("paths") or {}
    for path, item in sorted(paths.items()):
        if not isinstance(item, dict):
            continue
        for method in ("get", "post", "put", "delete", "patch"):
            if method not in item:
                continue
            upper = method.upper()
            if (upper, path) in PUBLIC_OPERATIONS:
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")
            # 路径参数用固定占位值：未认证请求在依赖层即返回 401，
            # 不会触达真实资源；避免把 {video_id} 当作 id 映射键。
            concrete_path = re.sub(r"\{[^}]+\}", "1", path)
            specs.append(_spec(
                f"A-unauth-{method}-{slug}", "A", upper, concrete_path,
                "anonymous",
                401, "CRITICAL", [module_for_path(path)],
                body_kind=None if upper == "GET" else "empty_json",
                category="cross_base_read",
                title=f"未认证访问 {upper} {path}",
            ))
    return specs


def module_for_path(path: str) -> str:
    if path.startswith("/api/auth"):
        return "auth"
    if path.startswith("/api/dogs"):
        return "dogs"
    if path.startswith("/api/videos"):
        return "videos"
    if path.startswith("/api/scores"):
        return "scores"
    if path.startswith("/api/annotations"):
        return "annotations"
    if path.startswith("/api/bases"):
        return "bases"
    if path.startswith("/api/scoring"):
        return "scoring"
    if path.startswith("/api/models"):
        return "models"
    if path.startswith("/api/finetune"):
        return "finetune"
    if path.startswith("/api/llm"):
        return "llm_explainer"
    return "health"


# ============================================================
# 离线 self-check
# ============================================================


def run_self_check() -> int:
    """纯离线 lint：不联网、不加载 settings、不连 DB。"""
    errors: list[str] = []
    specs = static_case_specs()
    seen: set[str] = set()
    suites = {"A", "B", "C"}

    for s in specs:
        cid = s.get("case_id", "<missing>")
        if cid in seen:
            errors.append(f"duplicate case_id: {cid}")
        seen.add(cid)
        for key in ("case_id", "suite", "identity", "method", "endpoint",
                    "expected_status", "severity_if_fail", "modules"):
            if key not in s:
                errors.append(f"{cid}: missing field {key}")
        if s.get("suite") not in suites:
            errors.append(f"{cid}: unknown suite {s.get('suite')!r}")
        if s.get("identity") not in IDENTITY_SET:
            errors.append(f"{cid}: unknown identity {s.get('identity')!r}")
        endpoint = s.get("endpoint", "")
        if not endpoint.startswith("/"):
            errors.append(f"{cid}: endpoint must start with /")
        if "Bearer " in endpoint:
            errors.append(f"{cid}: endpoint contains literal 'Bearer ' (forbidden)")
        leftovers = re.sub(r"\{ids\.[A-Za-z0-9_]+\}", "", endpoint)
        if "{" in leftovers or "}" in leftovers:
            errors.append(f"{cid}: unresolved placeholder in {endpoint!r}")
        if re.search(r"/\d+", endpoint):
            errors.append(f"{cid}: endpoint hardcodes a numeric id ({endpoint})")
        if s.get("severity_if_fail") not in SEVERITIES:
            errors.append(f"{cid}: bad severity {s.get('severity_if_fail')!r}")
        expected = s.get("expected_status")
        if not (isinstance(expected, int) or
                (isinstance(expected, list) and all(isinstance(x, int) for x in expected))):
            errors.append(f"{cid}: expected_status must be int or list[int]")
        mods = s.get("modules") or []
        if not mods:
            errors.append(f"{cid}: modules must not be empty")
        for m in mods:
            if m not in MODULE_PATHS:
                errors.append(f"{cid}: unknown module {m!r}")
        if s.get("body_kind") not in (None, "empty_json") and s.get("method") == "GET":
            errors.append(f"{cid}: GET must not carry a body_kind")

    # 动态生成器契约（不联网）：用最小假 OpenAPI 验证不再硬编码 47。
    fake_openapi = {
        "paths": {
            "/api/dogs": {"get": {}, "post": {}},
            "/api/auth/login": {"post": {}},
            "/health": {"get": {}},
        }
    }
    dyn = dynamic_anonymous_specs(fake_openapi)
    if len(dyn) != 2:
        errors.append(f"dynamic generator returned {len(dyn)} specs, expected 2")
    for s in dyn:
        if s["identity"] != "anonymous":
            errors.append(f"{s['case_id']}: dynamic identity must be anonymous")
        if s["expected_status"] != 401:
            errors.append(f"{s['case_id']}: dynamic expected_status must be 401")
        if s["suite"] != "A":
            errors.append(f"{s['case_id']}: dynamic suite must be A")

    # 关键用例必须存在（防止误删覆盖）
    required = {
        "B-dog-delete-manager_a", "B-dog-put-manager_a",
        "B-score-get-manager_a", "B-anno-task-get-manager_a",
        "C-jwt-missing-sub", "C-jwt-sub-non-numeric",
        "C-jwt-tampered-base-bases", "C-jwt-tampered-base-users",
        "B-dog-get-viewer_b", "B-video-report-viewer_b",
    }
    missing = required - seen
    if missing:
        errors.append(f"required cases missing: {sorted(missing)}")

    # 占位符解析必须只按 seed id 做 dict 查找（{ids.key} 不是 format 的 getattr）
    try:
        resolved = resolve_endpoint("/api/dogs/{ids.foo}", {"foo": 7})
        if resolved != "/api/dogs/7":
            errors.append(f"resolve_endpoint wrong: {resolved!r}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"resolve_endpoint failed: {exc}")

    print(f"[self-check] static cases: {len(specs)}, dynamic sample: {len(dyn)}")
    print("[self-check] 说明: 本检查只做矩阵 lint，不联网、不加载配置、不是自动回归门。")
    if errors:
        print(f"[self-check] FAIL ({len(errors)} error(s))")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("[self-check] PASS")
    return 0


# ============================================================
# 通用工具
# ============================================================


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)


def utc_date() -> str:
    return utc_now().strftime("%Y-%m-%d")


def is_loopback_host(host: str) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def sanitize_text(value) -> str:
    """凭据类文本兜底脱敏（报告入库前还会做正则硬门）。"""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("Bearer ", "Bearer[REDACTED]")
    text = text.replace(SEED_PASSWORD, "[REDACTED]")
    text = text.replace("AUTH_SECRET" + "_KEY", "AUTH_SECRET[REDACTED]")
    if len(text) > 512:
        text = text[:512] + "...[truncated]"
    return text


SENSITIVE_KEYS = {
    "access_token", "token", "password", "new_password", "old_password",
    "authorization", "cookie", "set-cookie", "auth_secret_key", "secret",
    "password_hash",
}


def redact_obj(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k).lower() in SENSITIVE_KEYS:
                out[k] = "[REDACTED]"
            else:
                out[k] = redact_obj(v)
        return out
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, str):
        return sanitize_text(obj)
    return obj


def report_contains_credentials(serialized: str) -> list[str]:
    hits: list[str] = []
    if re.search(r"eyJ[A-Za-z0-9_-]{10,}\.", serialized):
        hits.append("jwt_like")
    if "Bearer " in serialized:
        hits.append("bearer_scheme")
    if SEED_PASSWORD in serialized:
        hits.append("seed_password")
    if ("AUTH_SECRET" + "_KEY") in serialized:
        hits.append("auth_secret_marker")
    return hits


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                    dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def resolve_endpoint(template: str, ids: dict) -> str:
    """把 ``{ids.key}`` 占位符替换为本次种子 id。

    注意: 不能用 ``str.format`` —— ``{ids.key}`` 在 format 语义下是 getattr，
    对 dict 会抛 AttributeError；这里显式按 dict 键查找。
    """
    def _repl(match: re.Match) -> str:
        key = match.group(1)
        if key not in ids:
            raise KeyError(f"seed id missing: {key}")
        return str(ids[key])

    return re.sub(r"\{ids\.([A-Za-z0-9_]+)\}", _repl, template)


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def run_cmd(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            args, cwd=str(PROJECT_ROOT), capture_output=True, text=True,
            timeout=30, encoding="utf-8", errors="replace",
        )
        # 只去掉尾部空白：`git status --porcelain` 的首行前导空格有语义
        return (proc.stdout or "").rstrip()
    except Exception:
        return ""


# ============================================================
# 快照
# ============================================================


def _service_processes() -> list[dict]:
    """返回 uvicorn backend.app.main 进程的 pid/create_time/cmdline。"""
    result: list[dict] = []
    try:
        import psutil  # 延迟导入：self-check 不触发
    except Exception:
        return result
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "exe"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            joined = " ".join(str(x) for x in cmdline)
            if "uvicorn" not in joined or "backend.app.main" not in joined:
                continue
            # 只保留真正跑 uvicorn 的 python 进程
            name = (proc.info.get("name") or "").lower()
            if "python" not in name:
                continue
            result.append({
                "pid": proc.info["pid"],
                "create_time": proc.info.get("create_time"),
                "cmdline": joined,
                "exe": proc.info.get("exe") or "",
            })
        except Exception:
            continue
    return result


def _fallback_process_info() -> dict | None:
    """psutil 不可用时的 PowerShell 回退（不失败的尽力而为）。"""
    try:
        out = run_cmd([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -match 'uvicorn' -and "
            "$_.CommandLine -match 'backend.app.main' } | "
            "Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json",
        ])
        if not out:
            return None
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        if not data:
            return None
        first = data[0]
        created = first.get("CreationDate")
        epoch = None
        if isinstance(created, str):
            # /Date(169...)/ 形式
            m = re.search(r"(\d{10,})", created)
            if m:
                epoch = float(m.group(1)) / 1000.0
        return {
            "pid": first.get("ProcessId"),
            "create_time": epoch,
            "cmdline": first.get("CommandLine") or "",
        }
    except Exception:
        return None


def build_snapshot(client, health: dict) -> dict:
    head = run_cmd(["git", "rev-parse", "HEAD"])
    dirty_raw = run_cmd(["git", "status", "--porcelain"])
    dirty_paths = [line[3:] for line in dirty_raw.splitlines() if line.strip()]
    tags_raw = run_cmd(["git", "tag", "--list"])
    tags = [t for t in tags_raw.splitlines() if t.strip()]

    procs = _service_processes()
    process_info = None
    if procs:
        # 优先选择工程 .venv 内的解释器
        chosen = next(
            (p for p in procs if ".venv" in (p.get("exe") or "").lower()),
            procs[0],
        )
        process_info = {
            "pids": [p["pid"] for p in procs],
            "pid": chosen["pid"],
            "create_time": chosen["create_time"],
            "create_time_utc": (
                _dt.datetime.fromtimestamp(chosen["create_time"], tz=_dt.timezone.utc).isoformat()
                if chosen.get("create_time") else None
            ),
            "cmdline": chosen.get("cmdline", ""),
        }
    else:
        fallback = _fallback_process_info()
        if fallback:
            process_info = {
                "pids": [fallback["pid"]],
                "pid": fallback["pid"],
                "create_time": fallback["create_time"],
                "create_time_utc": (
                    _dt.datetime.fromtimestamp(fallback["create_time"], tz=_dt.timezone.utc).isoformat()
                    if fallback.get("create_time") else None
                ),
                "cmdline": fallback.get("cmdline", ""),
                "source": "powershell_fallback",
            }

    reload_enabled = bool(process_info and "--reload" in (process_info.get("cmdline") or ""))

    openapi_bytes = b""
    try:
        resp = client.get("/openapi.json")
        if resp.status_code == 200:
            openapi_bytes = resp.content
    except Exception:
        openapi_bytes = b""
    openapi_sha = hashlib.sha256(openapi_bytes).hexdigest() if openapi_bytes else None

    start_epoch = process_info.get("create_time") if process_info else None
    files = []
    stale_modules = []
    for key in TRACKED_MODULES:
        rel = MODULE_PATHS[key]
        path = PROJECT_ROOT / rel
        try:
            st = path.stat()
            mtime = st.st_mtime
            mtime_utc = _dt.datetime.fromtimestamp(mtime, tz=_dt.timezone.utc).isoformat()
        except OSError:
            mtime = None
            mtime_utc = None
        digest = sha256_file(path)
        files.append({
            "module": key,
            "path": rel,
            "sha256": digest,
            "mtime_utc": mtime_utc,
            "exists": mtime is not None,
        })
        if start_epoch and mtime and mtime > start_epoch:
            stale_modules.append(key)

    warning_parts = []
    if stale_modules:
        warning_parts.append(
            "module_changed_after_service_start: " + ", ".join(stale_modules)
        )
    if process_info and not reload_enabled:
        warning_parts.append("service_runs_without_reload")
    if not process_info:
        warning_parts.append("service_process_not_identified")
    if process_info and not process_info.get("create_time"):
        warning_parts.append("service_start_time_unavailable")

    return {
        "git_head": head or None,
        "git_dirty_count": len(dirty_paths),
        "git_dirty_paths": dirty_paths,
        "git_tags_count": len(tags),
        "git_tags": tags,
        "service": process_info,
        "reload_enabled": reload_enabled,
        "openapi_sha256": openapi_sha,
        "health": health,
        "tracked_files": files,
        "stale_modules": stale_modules,
        "snapshot_warning": "; ".join(warning_parts) if warning_parts else None,
        "methodology": (
            "陈旧判定 = 模块 mtime > 服务进程 create_time；按 endpoint 所属模块归因。"
            "schemas/common.py 等共享依赖的改动记录在 tracked_files，但不单独归因到每个用例。"
        ),
    }


# ============================================================
# 健康门
# ============================================================


def check_health(client) -> dict:
    """健康门：HTTP 200 + status==ok + checks.database.status==ok。"""
    info = {
        "ok": False,
        "http_status": None,
        "status": None,
        "version": None,
        "environment": None,
        "deploy_mode": None,
        "checks": {},
        "reason": None,
    }
    try:
        resp = client.get("/health")
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"health_unreachable: {type(exc).__name__}: {exc}"
        return info

    info["http_status"] = resp.status_code
    if resp.status_code != 200:
        info["reason"] = f"health_http_{resp.status_code}"
        return info
    try:
        body = resp.json()
    except Exception:
        info["reason"] = "health_body_not_json"
        return info

    info["status"] = body.get("status")
    info["version"] = body.get("version")
    info["environment"] = body.get("environment")
    info["deploy_mode"] = body.get("deploy_mode")
    checks = body.get("checks") or {}
    for key in ("database", "redis", "models"):
        comp = checks.get(key) or {}
        info["checks"][key] = {
            "status": comp.get("status"),
            "detail": sanitize_text(comp.get("detail")),
        }

    db_component = info["checks"].get("database", {}).get("status")
    if info["status"] != "ok":
        bad = [k for k, v in info["checks"].items() if v.get("status") != "ok"]
        info["reason"] = "health_status_not_ok: " + (",".join(bad) or "unknown")
        return info
    if db_component != "ok":
        info["reason"] = f"health_database_not_ok: {db_component}"
        return info
    info["ok"] = True
    return info


# ============================================================
# DB 种子 / 清理
# ============================================================


def _hash_password(password: str) -> str:
    import bcrypt  # 延迟导入
    return bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt(rounds=4)
    ).decode("utf-8")


def cleanup_seed_data(conn, seed_ids: dict | None = None) -> dict:
    """按前缀 + 已记录 id 清理；返回各表残留计数。"""
    seed_ids = seed_ids or {}
    cur = conn.cursor()

    # 先删除本次上传的视频文件（若存在）
    try:
        cur.execute(
            "SELECT storage_path FROM videos WHERE original_filename LIKE %s",
            (SEED_PREFIX + "%",),
        )
        from backend.app.core.config import settings as _settings
        for (rel_path,) in cur.fetchall():
            if not rel_path:
                continue
            candidate = (_settings.upload_dir / rel_path).resolve()
            try:
                if candidate.is_file():
                    candidate.unlink()
            except OSError:
                pass
    except Exception:
        pass

    video_ids = list(seed_ids.get("video_ids", []))
    task_ids = list(seed_ids.get("task_ids", []))
    ids_sql_v = "SELECT unnest(%s::int[])" if video_ids else None
    ids_sql_t = "SELECT unnest(%s::int[])" if task_ids else None

    # 1. annotations
    if video_ids:
        cur.execute(
            "DELETE FROM annotations WHERE video_id IN (" + ids_sql_v + ") "
            "OR task_id IN (SELECT id FROM annotation_tasks WHERE video_id IN ("
            + ids_sql_v + "))",
            (video_ids, video_ids),
        )
    else:
        cur.execute(
            "DELETE FROM annotations WHERE video_id IN ("
            "SELECT id FROM videos WHERE original_filename LIKE %s)",
            (SEED_PREFIX + "%",),
        )
    # 2. annotation_tasks
    if video_ids:
        cur.execute(
            "DELETE FROM annotation_tasks WHERE video_id IN (" + ids_sql_v + ") "
            "OR handler_id IN (SELECT id FROM handlers WHERE email LIKE %s)",
            (video_ids, SEED_PREFIX + "%"),
        )
    else:
        cur.execute(
            "DELETE FROM annotation_tasks WHERE video_id IN ("
            "SELECT id FROM videos WHERE original_filename LIKE %s)",
            (SEED_PREFIX + "%",),
        )
    # 3. scores
    if video_ids:
        cur.execute(
            "DELETE FROM scores WHERE video_id IN (" + ids_sql_v + ")",
            (video_ids,),
        )
    else:
        cur.execute(
            "DELETE FROM scores WHERE video_id IN ("
            "SELECT id FROM videos WHERE original_filename LIKE %s)",
            (SEED_PREFIX + "%",),
        )
    # 4. videos
    cur.execute("DELETE FROM videos WHERE original_filename LIKE %s",
                (SEED_PREFIX + "%",))
    # 5. associations
    cur.execute(
        "DELETE FROM dog_handler_association WHERE "
        "dog_id IN (SELECT id FROM dogs WHERE name LIKE %s) "
        "OR handler_id IN (SELECT id FROM handlers WHERE email LIKE %s)",
        (SEED_PREFIX + "%", SEED_PREFIX + "%"),
    )
    cur.execute(
        "DELETE FROM dog_base_association WHERE "
        "dog_id IN (SELECT id FROM dogs WHERE name LIKE %s) OR "
        "base_id IN (SELECT id FROM bases WHERE name LIKE %s OR code LIKE %s)",
        (SEED_PREFIX + "%", SEED_PREFIX + "%", "PEN_RBAC_%"),
    )
    # 6. dogs
    cur.execute("DELETE FROM dogs WHERE name LIKE %s", (SEED_PREFIX + "%",))
    # 7. handlers
    cur.execute("DELETE FROM handlers WHERE email LIKE %s", (SEED_PREFIX + "%",))
    # 8. bases
    cur.execute(
        "DELETE FROM bases WHERE name LIKE %s OR code LIKE %s",
        (SEED_PREFIX + "%", "PEN_RBAC_%"),
    )
    conn.commit()

    residuals: dict[str, int] = {}
    probes = [
        ("handlers", "SELECT count(*) FROM handlers WHERE email LIKE %s",
         (SEED_PREFIX + "%",)),
        ("bases", "SELECT count(*) FROM bases WHERE name LIKE %s OR code LIKE %s",
         (SEED_PREFIX + "%", "PEN_RBAC_%")),
        ("dogs", "SELECT count(*) FROM dogs WHERE name LIKE %s",
         (SEED_PREFIX + "%",)),
        ("videos", "SELECT count(*) FROM videos WHERE original_filename LIKE %s",
         (SEED_PREFIX + "%",)),
    ]
    for name, sql, params in probes:
        cur.execute(sql, params)
        residuals[name] = int(cur.fetchone()[0])
    if video_ids:
        cur.execute("SELECT count(*) FROM scores WHERE video_id = ANY(%s)",
                    (video_ids,))
        residuals["scores"] = int(cur.fetchone()[0])
        cur.execute(
            "SELECT count(*) FROM annotation_tasks WHERE video_id = ANY(%s)",
            (video_ids,),
        )
        residuals["annotation_tasks"] = int(cur.fetchone()[0])
        cur.execute(
            "SELECT count(*) FROM annotations WHERE video_id = ANY(%s)",
            (video_ids,),
        )
        residuals["annotations"] = int(cur.fetchone()[0])
    else:
        residuals["scores"] = 0
        residuals["annotation_tasks"] = 0
        residuals["annotations"] = 0
    cur.close()
    return residuals


def seed_data(conn) -> dict:
    """创建 pen_rbac_ 种子；返回 id 映射（只含测试自建 id）。"""
    cur = conn.cursor()
    ids: dict = {}
    seed_ids = {"video_ids": [], "task_ids": []}

    for key in ("A", "B"):
        name, code = SEED_BASES[key]
        cur.execute(
            "INSERT INTO bases (name, code, description, is_active) "
            "VALUES (%s, %s, %s, true) RETURNING id",
            (name, code, "RBAC penetration test base"),
        )
        ids[f"base_{key.lower()}"] = cur.fetchone()[0]

    password_hash = _hash_password(SEED_PASSWORD)
    for key, display, role, base, is_superuser, is_active in SEED_ACCOUNTS:
        base_id = ids.get(f"base_{base.lower()}") if base else None
        cur.execute(
            "INSERT INTO handlers "
            "(name, email, phone, role, is_active, password_hash, base_id, is_superuser) "
            "VALUES (%s, %s, NULL, %s, %s, %s, %s, %s) RETURNING id",
            (display, f"{SEED_PREFIX}{key}@test.com", role, is_active,
             password_hash, base_id, is_superuser),
        )
        ids[key] = cur.fetchone()[0]

    # 供 DELETE /auth/users 正/负向用例使用的临时账号
    for key in ("del_super", "del_admin"):
        cur.execute(
            "INSERT INTO handlers "
            "(name, email, phone, role, is_active, password_hash, base_id, is_superuser) "
            "VALUES (%s, %s, NULL, 'HANDLER', true, %s, %s, false) RETURNING id",
            (f"Pen temp {key}", f"{SEED_PREFIX}{key}@test.com",
             password_hash, ids["base_a"]),
        )
        ids[key] = cur.fetchone()[0]

    # 犬只（基地 B）：DELETE 探针每身份一只独立 victim，避免首个越权删除
    # 成功（真 CRITICAL）后污染后续用例造成 404 假象。
    dog_specs = [
        ("victim_dog_b", "pen_rbac_victim_dog_b"),
    ]
    for ident in ("manager_a", "researcher_a", "handler_a1", "viewer_a"):
        dog_specs.append((
            f"victim_dog_b_del_{ident}",
            f"pen_rbac_victim_dog_b_del_{ident}",
        ))
    for key, name in dog_specs:
        cur.execute(
            "INSERT INTO dogs "
            "(name, breed, birth_date, gender, chip_id, handler_id, training_stage, "
            " notes, home_base_id) "
            "VALUES (%s, NULL, NULL, 'MALE', NULL, %s, 'P0', %s, %s) RETURNING id",
            (name, ids["handler_b1"], "RBAC penetration victim", ids["base_b"]),
        )
        ids[key] = cur.fetchone()[0]

    # 视频（COMPLETED + 真实 PDF）
    cur.execute(
        "INSERT INTO videos "
        "(dog_id, handler_id, original_filename, storage_path, storage_filename, "
        " duration_sec, fps, width, height, size_bytes, status, scene, report_path, "
        " uploaded_at, processed_at) "
        "VALUES (%s, %s, %s, %s, %s, 10.0, 25.0, 640, 480, 1024, 'COMPLETED', "
        " 'obedience_trial', %s, now(), now()) RETURNING id",
        (ids["victim_dog_b"], ids["handler_b1"], VICTIM_PDF_NAME,
         SEED_PREFIX + "victim.mp4", SEED_PREFIX + "victim.mp4", VICTIM_PDF_NAME),
    )
    ids["victim_video_b"] = cur.fetchone()[0]
    seed_ids["video_ids"].append(ids["victim_video_b"])

    # 评分
    cur.execute(
        "INSERT INTO scores "
        "(video_id, standard, accuracy, response_latency, duration, search_efficiency, "
        " attention, courage, gait_quality, overall, scoring_engine_version, notes) "
        "VALUES (%s, 'GA_T', 80.0, NULL, 70.0, NULL, 75.0, NULL, NULL, 75.0, "
        " 'pen-v0.1', 'pen_rbac_seed') RETURNING id",
        (ids["victim_video_b"],),
    )
    ids["victim_score_b"] = cur.fetchone()[0]

    # 标注任务 + 标注
    cur.execute(
        "INSERT INTO annotation_tasks "
        "(video_id, handler_id, ls_project_id, ls_task_id, status, total_frames, "
        " annotated_frames, annotation_types) "
        "VALUES (%s, %s, NULL, NULL, 'COMPLETED', 10, 1, %s) RETURNING id",
        (ids["victim_video_b"], ids["handler_b1"],
         json.dumps(["behavior"])),
    )
    ids["victim_task_b"] = cur.fetchone()[0]
    seed_ids["task_ids"].append(ids["victim_task_b"])

    cur.execute(
        "INSERT INTO annotations "
        "(task_id, video_id, frame_idx, annotation_type, annotation_source, "
        " data_json, confidence) "
        "VALUES (%s, %s, 0, 'BEHAVIOR', 'HUMAN', %s, NULL) RETURNING id",
        (ids["victim_task_b"], ids["victim_video_b"],
         json.dumps({"behavior": "SIT", "start_frame": 0, "end_frame": 10})),
    )
    ids["victim_annotation"] = cur.fetchone()[0]

    # 现有激活模型（只读引用，不修改）
    cur.execute("SELECT id FROM ml_models WHERE is_active = true "
                "AND type = 'POSE' ORDER BY id ASC LIMIT 1")
    row = cur.fetchone()
    if row:
        ids["model_pose"] = row[0]

    conn.commit()
    cur.close()
    return {"ids": ids, "seed_ids": seed_ids}


def write_victim_pdf() -> Path:
    pdf = REPORTS_DIR / VICTIM_PDF_NAME
    pdf.parent.mkdir(parents=True, exist_ok=True)
    body = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"trailer<</Root 1 0 R>>\n"
        b"%%EOF\n"
    )
    pdf.write_bytes(body)
    return pdf


# ============================================================
# 运行时上下文 / 请求
# ============================================================


class RunContext:
    def __init__(self) -> None:
        self.ids: dict = {}
        self.seed_ids: dict = {"video_ids": [], "task_ids": []}
        self.tokens: dict[str, str] = {}
        self.identity_base: dict[str, int | None] = {}
        self.secret: str = ""
        self.algorithm: str = "HS256"
        self.secret_matches_service: bool = False
        self.remote_writes_allowed: bool = True
        self.precondition_failed: bool = False
        self.precondition_notes: list[str] = []
        self.stale_modules: set[str] = set()
        self.login_status: dict[str, int] = {}

    def fail_precondition(self, message: str) -> None:
        self.precondition_failed = True
        self.precondition_notes.append(message)


def build_body(kind: str, ids: dict, ctx: RunContext) -> dict:
    if kind == "empty_json":
        return {}
    if kind == "dog_noop":
        return {"name": "pen_rbac_victim_dog_b"}
    if kind == "user_noop":
        return {
            "name": "Pen Victim User B",
            "email": f"{SEED_PREFIX}victim_user_b@test.com",
            "password": SEED_PASSWORD,
            "role": "handler",
            "base_id": ids["base_b"],
            "is_superuser": False,
        }
    if kind == "register_normal":
        return {
            "name": "Pen Registered",
            "email": f"{SEED_PREFIX}registered@test.com",
            "password": SEED_PASSWORD,
            "role": "handler",
            "base_id": ids["base_a"],
            "is_superuser": False,
        }
    if kind == "register_super_attempt":
        return {
            "name": "Pen Registered Super",
            "email": f"{SEED_PREFIX}registered_super@test.com",
            "password": SEED_PASSWORD,
            "role": "admin",
            "base_id": ids["base_a"],
            "is_superuser": True,
        }
    if kind == "register_super":
        return {
            "name": "Pen Registered Super",
            "email": f"{SEED_PREFIX}registered_super@test.com",
            "password": SEED_PASSWORD,
            "role": "admin",
            "is_superuser": True,
        }
    if kind == "base_create_super":
        return {"name": SEED_PREFIX + "api_base_super",
                "code": "PEN_RBAC_API_SUPER", "description": "pen test",
                "is_active": True}
    if kind == "base_create_admin":
        return {"name": SEED_PREFIX + "api_base_admin",
                "code": "PEN_RBAC_API_ADMIN", "description": "pen test",
                "is_active": True}
    if kind == "base_create_manager":
        return {"name": SEED_PREFIX + "api_base_manager",
                "code": "PEN_RBAC_API_MANAGER", "description": "pen test",
                "is_active": True}
    if kind == "base_create_handler":
        return {"name": SEED_PREFIX + "api_base_handler",
                "code": "PEN_RBAC_API_HANDLER", "description": "pen test",
                "is_active": True}
    if kind == "dog_create":
        return {"name": SEED_PREFIX + "api_dog_handler",
                "breed": None, "gender": "MALE", "chip_id": None,
                "handler_id": ids["handler_a1"], "training_stage": "P0"}
    if kind == "dog_create_viewer":
        return {"name": SEED_PREFIX + "api_dog_viewer",
                "breed": None, "gender": "MALE", "chip_id": None,
                "handler_id": ids["handler_a1"], "training_stage": "P0"}
    if kind == "scoring_put":
        return {"content": "scoring_engine: {}\n"}
    if kind == "finetune_trigger":
        return {"epochs": 1, "batch": 1, "imgsz": 64}
    if kind == "annotation_task":
        return {"video_id": ids["victim_video_b"],
                "handler_id": ids["handler_b1"],
                "annotation_types": ["behavior"]}
    if kind == "login_super":
        return {"email": f"{SEED_PREFIX}superadmin@test.com",
                "password": SEED_PASSWORD}
    raise ValueError(f"unknown body kind: {kind}")


def _b64url_json(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def build_jwt_token(auth_kind: str, ctx: RunContext, identity: str) -> str | None:
    import jwt  # 延迟导入

    now = utc_now()
    exp = int((now + _dt.timedelta(minutes=10)).timestamp())
    iat = int(now.timestamp())

    def sign(payload: dict, key: str | None = None) -> str:
        return jwt.encode(payload, key or ctx.secret, algorithm=ctx.algorithm)

    if auth_kind in ("secret_match", "jwt_secret_match"):
        return sign({
            "sub": str(ctx.ids["superadmin"]), "role": "ADMIN",
            "base_id": None, "is_superuser": True, "exp": exp, "iat": iat,
        })
    if auth_kind == "wrong_key":
        return sign({
            "sub": str(ctx.ids["superadmin"]), "role": "ADMIN",
            "base_id": None, "is_superuser": True, "exp": exp, "iat": iat,
        }, key="x" * 40)
    if auth_kind == "expired":
        return sign({
            "sub": str(ctx.ids["superadmin"]), "role": "ADMIN",
            "base_id": None, "is_superuser": True,
            "exp": int((now - _dt.timedelta(minutes=10)).timestamp()),
            "iat": int((now - _dt.timedelta(minutes=20)).timestamp()),
        })
    if auth_kind == "alg_none":
        header = _b64url_json({"alg": "none", "typ": "JWT"})
        payload = _b64url_json({
            "sub": str(ctx.ids["superadmin"]), "role": "ADMIN",
            "exp": exp, "iat": iat,
        })
        return f"{header}.{payload}."
    if auth_kind == "nonexistent_sub":
        return sign({
            "sub": "99999999", "role": "ADMIN", "base_id": None,
            "is_superuser": True, "exp": exp, "iat": iat,
        })
    if auth_kind == "disabled":
        return sign({
            "sub": str(ctx.ids["disabled_a"]), "role": "HANDLER",
            "base_id": ctx.ids["base_a"], "is_superuser": False,
            "exp": exp, "iat": iat,
        })
    if auth_kind == "missing_sub":
        return sign({
            "role": "ADMIN", "base_id": None, "is_superuser": True,
            "exp": exp, "iat": iat,
        })
    if auth_kind == "sub_non_numeric":
        return sign({
            "sub": "not-a-number", "role": "ADMIN", "base_id": None,
            "is_superuser": True, "exp": exp, "iat": iat,
        })
    if auth_kind == "role_garbage":
        return sign({
            "sub": str(ctx.ids["superadmin"]), "role": "GARBAGE",
            "base_id": None, "is_superuser": False, "exp": exp, "iat": iat,
        })
    if auth_kind == "tampered_base":
        return sign({
            "sub": str(ctx.ids["manager_a"]), "role": "MANAGER",
            "base_id": ctx.ids["base_b"], "is_superuser": False,
            "exp": exp, "iat": iat,
        })
    return None


def auth_header_for(spec: dict, ctx: RunContext) -> dict | None:
    auth_kind = spec.get("auth_kind") or "identity"
    identity = spec["identity"]
    if auth_kind == "none":
        return None
    if auth_kind == "bare":
        return {"Authorization": "Bearer"}
    if auth_kind == "garbage":
        return {"Authorization": "Bearer not-a-token"}
    if auth_kind == "identity":
        token = ctx.tokens.get(identity)
        if not token:
            return None
        return {"Authorization": f"Bearer {token}"}
    token = build_jwt_token(auth_kind, ctx, identity)
    if not token:
        return None
    return {"Authorization": f"Bearer {token}"}


def perform_request(client, spec: dict, ctx: RunContext) -> dict:
    endpoint = resolve_endpoint(spec["endpoint"], ctx.ids)
    headers = auth_header_for(spec, ctx)
    body_kind = spec.get("body_kind")
    result = {
        "endpoint": endpoint,
        "headers_sent": bool(headers),
        "status_code": None,
        "json": None,
        "text": "",
        "content_type": None,
        "content_length": None,
        "error": None,
    }
    def _send():
        if body_kind == "upload":
            files = {"file": ("pen_rbac_tiny.mp4", b"x", "video/mp4")}
            data = {"scene": "obedience_trial"}
            return client.request(spec["method"], endpoint, headers=headers,
                                  files=files, data=data)
        if body_kind:
            return client.request(spec["method"], endpoint, headers=headers,
                                  json=build_body(body_kind, ctx.ids, ctx))
        if spec["method"] in ("POST", "PUT", "DELETE", "PATCH"):
            return client.request(spec["method"], endpoint, headers=headers,
                                  json={})
        return client.request(spec["method"], endpoint, headers=headers)

    try:
        try:
            resp = _send()
        except Exception as exc:  # 5xx 后 uvicorn 偶发连接重置：重试一次
            if "ReadError" in type(exc).__name__ or "ConnectError" in type(exc).__name__ \
                    or "RemoteProtocolError" in type(exc).__name__:
                import time as _time
                _time.sleep(0.4)
                resp = _send()
            else:
                raise
        result["status_code"] = resp.status_code
        result["content_type"] = resp.headers.get("content-type")
        result["content_length"] = resp.headers.get("content-length")
        try:
            result["json"] = resp.json()
        except Exception:
            result["json"] = None
        if result["json"] is None:
            result["text"] = sanitize_text(resp.text)
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {sanitize_text(exc)}"
    return result


# ============================================================
# 判定
# ============================================================


def collect_ids(payload) -> set[int]:
    found: set[int] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in ("id", "dog_id", "video_id", "score_id", "task_id") \
                    and isinstance(value, int):
                found.add(value)
            found |= collect_ids(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= collect_ids(item)
    return found


def _top_level_total(payload) -> int | None:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("items", "series", "data", "results"):
            if isinstance(payload.get(key), list):
                return len(payload[key])
    return None


def _foreign_real_rows(payload, seed_ids: set[int],
                       identity_base: int | None) -> int | None:
    if not isinstance(payload, list):
        return None
    count = 0
    for item in payload:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if item_id in seed_ids:
            continue
        if "home_base_id" in item and item.get("home_base_id") == identity_base:
            continue
        count += 1
    return count


def evaluate(spec: dict, result: dict, ctx: RunContext) -> tuple[str, str | None, dict]:
    """返回 (status, fail_reason, evidence)。"""
    status = result["status_code"]
    expected = spec["expected_status"]
    expected_set = expected if isinstance(expected, list) else [expected]
    evidence: dict = {
        "endpoint": result["endpoint"],
        "expected_status": expected,
        "actual_status": status,
        "response_detail": None,
        "returned_total": None,
        "seed_ids_present": [],
        "foreign_real_rows_count": None,
        "content_type": result["content_type"],
        "content_length": result["content_length"],
    }

    if result["error"]:
        return "client_error", result["error"], evidence

    # 404 一律 precondition 失败（victim 目标被删/未创建）
    if status == 404 and spec.get("victim"):
        return "precondition_failed", "victim_endpoint_returned_404", evidence

    detail = None
    if isinstance(result["json"], dict):
        detail = result["json"].get("detail")
    if detail is not None:
        evidence["response_detail"] = sanitize_text(detail)

    check = spec.get("check_kind")
    payload = result["json"]
    seed_ids = set(ctx.ids.values())  # 本次自建 id 集合

    if check == "has_token":
        ok = (status in expected_set and isinstance(payload, dict)
              and bool(payload.get("access_token")))
        evidence["token_received"] = bool(
            isinstance(payload, dict) and payload.get("access_token")
        )
        return ("pass" if ok else "fail"), (None if ok else "no_access_token"), evidence

    if check == "is_superuser_false":
        ok = (status in expected_set and isinstance(payload, dict)
              and payload.get("is_superuser") is False)
        evidence["is_superuser"] = (
            payload.get("is_superuser") if isinstance(payload, dict) else None
        )
        return ("pass" if ok else "fail"), (
            None if ok else "non_superuser_created_superuser"), evidence

    if check == "report_pdf":
        ok = (status in expected_set
              and "application/pdf" in (result["content_type"] or ""))
        evidence["is_pdf"] = "application/pdf" in (result["content_type"] or "")
        return ("pass" if ok else "fail"), (None if ok else "report_not_pdf"), evidence

    if check in ("no_foreign_ids", "tampered_no_b_users"):
        present = collect_ids(payload)
        forbid_ids = [ctx.ids.get(k) for k in spec.get("forbid", [])]
        forbid_ids = [x for x in forbid_ids if isinstance(x, int)]
        hits = [x for x in forbid_ids if x in present]
        evidence["returned_total"] = _top_level_total(payload)
        evidence["seed_ids_present"] = sorted(hits)
        identity_base = ctx.identity_base.get(spec["identity"])
        evidence["foreign_real_rows_count"] = _foreign_real_rows(
            payload, seed_ids, identity_base
        )
        ok = status in expected_set and not hits
        return ("pass" if ok else "fail"), (
            None if ok else "cross_base_ids_present"), evidence

    # 默认：状态码判定
    if status in expected_set:
        return "pass", None, evidence
    if status is not None and 500 <= status < 600:
        return "fail", f"server_error_{status}", evidence
    if status == 422:
        return "inconclusive", "validation_error_422", evidence
    return "fail", f"unexpected_status_{status}", evidence


# ============================================================
# 套件执行
# ============================================================


def _module_is_stale(spec: dict, ctx: RunContext) -> bool:
    return any(m in ctx.stale_modules for m in spec.get("modules", []))


def execute_suite(client, suite_name: str, specs: list[dict], ctx: RunContext,
                  health_ok: bool) -> dict:
    cases: list[dict] = []
    if not health_ok:
        for spec in specs:
            cases.append(_case_record(
                spec, "invalid", "health_gate_failed_after_suite_start",
                {"skipped": True}, "NONE", ctx,
            ))
        return {
            "name": suite_name,
            "category": suite_name,
            "status": "invalid",
            "invalid_reason": "health_gate_failed",
            "cases": cases,
        }

    for spec in specs:
        if spec.get("dangerous_positive"):
            cases.append(_case_record(
                spec, "skipped", "dangerous_positive_write_disabled",
                {"skipped": True, "reason": "would mutate real scoring card / "
                                            "activate global model / trigger "
                                            "finetune / upload"},
                "NONE", ctx,
            ))
            continue
        if spec["method"] in ("POST", "PUT", "DELETE", "PATCH") and \
                spec.get("body_kind") != "login_super" and \
                not ctx.remote_writes_allowed:
            cases.append(_case_record(
                spec, "skipped", "remote_writes_not_allowed",
                {"skipped": True}, "NONE", ctx,
            ))
            continue
        if _module_is_stale(spec, ctx):
            cases.append(_case_record(
                spec, "invalid",
                "module_changed_after_service_start: " + ",".join(
                    m for m in spec["modules"] if m in ctx.stale_modules
                ),
                {"skipped": True, "stale_modules": [
                    m for m in spec["modules"] if m in ctx.stale_modules
                ]},
                "NONE", ctx,
            ))
            continue
        if spec.get("body_kind") == "upload" and not ctx.remote_writes_allowed:
            cases.append(_case_record(
                spec, "skipped", "remote_writes_not_allowed",
                {"skipped": True}, "NONE", ctx,
            ))
            continue

        result = perform_request(client, spec, ctx)
        status, reason, evidence = evaluate(spec, result, ctx)
        severity = spec["severity_if_fail"] if status == "fail" else "NONE"
        if status == "precondition_failed":
            ctx.fail_precondition(
                f"{spec['case_id']}: {reason} ({result['endpoint']})"
            )
        cases.append(_case_record(spec, status, reason, evidence, severity, ctx))

    if any(c["status"] == "precondition_failed" for c in cases):
        suite_status = "precondition_failed"
    elif any(c["status"] == "fail" for c in cases):
        suite_status = "fail"
    elif all(c["status"] in ("invalid", "skipped") for c in cases):
        suite_status = "invalid"
    elif any(c["status"] in ("invalid", "skipped") for c in cases):
        # 部分用例因陈旧快照/危险写被排除：不得整体判 pass
        suite_status = "partial"
    else:
        suite_status = "pass"

    invalid_reasons = sorted({
        c["invalid_reason"] for c in cases if c["invalid_reason"]
    })
    return {
        "name": suite_name,
        "category": suite_name,
        "status": suite_status,
        "invalid_reason": "; ".join(invalid_reasons) if invalid_reasons else None,
        "cases": cases,
    }


def _case_record(spec: dict, status: str, reason: str | None,
                 evidence: dict, severity: str, ctx: RunContext) -> dict:
    try:
        resolved_endpoint = resolve_endpoint(spec["endpoint"], ctx.ids)
    except KeyError:
        resolved_endpoint = spec["endpoint"]
    return {
        "case_id": spec["case_id"],
        "title": spec["title"],
        "method": spec["method"],
        "endpoint": resolved_endpoint,
        "endpoint_template": spec["endpoint"],
        "identity": spec["identity"],
        "expected_status": spec["expected_status"],
        "actual_status": evidence.get("actual_status"),
        "status": status,
        "invalid_reason": reason if status in ("invalid", "skipped") else None,
        "fail_reason": reason if status in ("fail", "client_error") else None,
        "severity": severity,
        "category": spec["category"],
        "evidence": redact_obj(evidence),
        "remediation": (
            CATEGORY_REMEDIATION.get(spec["category"], "")
            if status == "fail" else ""
        ),
    }


# ============================================================
# 审计日志核查
# ============================================================


def audit_static_check(cur) -> dict:
    hits = 0
    files_with_hits: list[str] = []
    for base in ("backend/app", "backend/workers"):
        root = PROJECT_ROOT / base
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            count = len(re.findall(r"audit", text, re.I))
            if count:
                hits += count
                files_with_hits.append(str(path.relative_to(PROJECT_ROOT)))
    cur.execute("SELECT tablename FROM pg_tables WHERE tablename ILIKE %s",
                ("%audit%",))
    tables = [r[0] for r in cur.fetchall()]
    return {
        "code_hits": hits,
        "files_with_hits": files_with_hits,
        "audit_tables": tables,
        "implemented": bool(hits or tables),
    }


# ============================================================
# 报告 / 基线
# ============================================================


def build_findings(suites: list[dict], observations: list[dict]) -> list[dict]:
    findings: list[dict] = []
    for suite in suites:
        for case in suite["cases"]:
            if case["status"] == "fail" and case["severity"] != "NONE":
                findings.append({
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "severity": case["severity"],
                    "category": case["category"],
                    "evidence": case["evidence"],
                    "remediation": case["remediation"],
                })
    findings.extend(observations)
    return findings


def _severity_trigger(fail_on: str, severities: set[str]) -> bool:
    if fail_on == "none":
        return False
    if fail_on == "critical":
        return "CRITICAL" in severities
    if fail_on == "high":
        return bool(severities & {"CRITICAL", "HIGH"})
    if fail_on == "any":
        return bool(severities & {"CRITICAL", "HIGH", "MEDIUM"})
    return False


def write_report(report: dict, path: Path) -> bool:
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    hits = report_contains_credentials(serialized)
    report["redaction"] = {
        "patterns_scanned": [
            "jwt_like", "bearer_scheme", "seed_password", "auth_secret_marker",
        ],
        "contains_no_credentials": not hits,
        "hits": hits,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if hits:
        print(f"[redaction] HIT: {hits} — refusing to write report")
        return False
    atomic_write_text(path, serialized)
    return True


def load_baseline(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("findings") or {}


def apply_baseline(findings: list[dict], baseline: dict) -> dict:
    current = {f["case_id"]: f["severity"] for f in findings}
    new = {}
    upgraded = {}
    resolved = []
    for cid, sev in current.items():
        if cid not in baseline:
            new[cid] = sev
        elif SEVERITY_RANK[sev] > SEVERITY_RANK.get(baseline[cid], 0):
            upgraded[cid] = sev
    for cid in baseline:
        if cid not in current:
            resolved.append(cid)
    triggering = set(new.values()) | set(upgraded.values())
    return {
        "new": new,
        "upgraded": upgraded,
        "resolved": sorted(resolved),
        "triggering_severities": sorted(triggering),
    }


# ============================================================
# 主流程
# ============================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P2-3 RBAC 活体渗透测试（protocol pen-v0.1）",
    )
    parser.add_argument("--base-url", default=None,
                        help="目标服务（默认 http://127.0.0.1:8000）")
    parser.add_argument("--self-check", action="store_true",
                        help="纯离线用例矩阵 lint（不联网）")
    parser.add_argument("--write-baseline", default=None,
                        help="把当前发现写入基线 JSON")
    parser.add_argument("--baseline", default=None,
                        help="与基线比对；只对新增/升级的发现执行阈值")
    parser.add_argument("--fail-on", default="critical",
                        choices=["critical", "high", "any", "none"])
    parser.add_argument("--out", default=None, help="报告输出路径")
    parser.add_argument("--allow-remote", action="store_true",
                        help="允许非 loopback 目标")
    parser.add_argument("--allow-remote-writes", action="store_true",
                        help="允许对非 loopback 目标执行写探针")
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_check:
        if (args.base_url is not None or args.write_baseline or args.baseline
                or args.out or args.allow_remote or args.allow_remote_writes):
            print("[self-check] FAIL: --self-check 与联网/输出参数互斥")
            return 1
        return run_self_check()

    base_url = (args.base_url or "http://127.0.0.1:8000").rstrip("/")
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.hostname or ""
    if parsed.scheme not in ("http", "https"):
        print(f"[fatal] unsupported scheme: {parsed.scheme}")
        return 2
    loopback = is_loopback_host(host)
    if not loopback and not args.allow_remote:
        print(f"[fatal] non-loopback target rejected without --allow-remote: {base_url}")
        return 2
    remote_writes_allowed = loopback or args.allow_remote_writes
    print(f"[target] {base_url} (loopback={loopback}, "
          f"remote_writes_allowed={remote_writes_allowed})")

    # 延迟导入：self-check 绝不触发配置加载
    import httpx  # noqa: WPS433
    import psycopg2  # noqa: WPS433
    from backend.app.core.config import settings  # noqa: WPS433

    report_path = Path(args.out) if args.out else (
        REPORTS_DIR / f"p2-3-rbac-pen-test-{utc_date()}.json"
    )
    baseline_path = Path(args.baseline) if args.baseline else None
    write_baseline_path = Path(args.write_baseline) if args.write_baseline else None

    if baseline_path is not None and not baseline_path.exists():
        print(f"[fatal] baseline not found: {baseline_path}")
        return 2

    ctx = RunContext()
    ctx.remote_writes_allowed = remote_writes_allowed
    ctx.secret = settings.auth_secret_key
    ctx.algorithm = settings.auth_algorithm

    client = httpx.Client(base_url=base_url, timeout=args.timeout,
                          follow_redirects=False)

    report: dict = {
        "protocol_version": PROTOCOL_VERSION,
        "task": TASK_ID,
        "timestamp": utc_now().isoformat(),
        "target": {
            "base_url": base_url,
            "transport": "live-http",
            "health": None,
        },
        "snapshot": None,
        "preconditions": {},
        "severity_policy": SEVERITY_POLICY,
        "exit_policy": EXIT_POLICY,
        "suites": [],
        "findings": [],
        "observations": [],
        "audit_static": {},
        "cleanup": {},
        "redaction": {},
        "summary": {},
    }

    conn = None
    exit_code = 0
    try:
        # ---------- 健康门 + 快照 ----------
        health = check_health(client)
        report["target"]["health"] = health
        if not health["ok"]:
            print(f"[precondition] health gate failed: {health['reason']}")
            ctx.fail_precondition(f"health_gate: {health['reason']}")
            exit_code = 2
        else:
            print(f"[health] ok version={health['version']} "
                  f"env={health['environment']} mode={health['deploy_mode']}")

        snapshot = build_snapshot(client, health)
        report["snapshot"] = snapshot
        ctx.stale_modules = set(snapshot.get("stale_modules") or [])
        if snapshot.get("snapshot_warning"):
            print(f"[snapshot] {snapshot['snapshot_warning']}")
        if ctx.stale_modules:
            print(f"[snapshot] stale modules (invalid cases): "
                  f"{sorted(ctx.stale_modules)}")
        if not snapshot.get("service") or not snapshot["service"].get("create_time"):
            ctx.fail_precondition("service_start_time_unavailable")
            exit_code = 2

        # ---------- DB ----------
        try:
            conn = psycopg2.connect(settings.pg_dsn)
        except Exception as exc:  # noqa: BLE001
            ctx.fail_precondition(f"db_connect: {type(exc).__name__}: {exc}")
            exit_code = 2
            raise RuntimeError("db_connect_failed")

        # 启动时先清理历史残留
        startup_residuals = cleanup_seed_data(conn, None)
        if any(startup_residuals.values()):
            print(f"[cleanup] pre-existing pen_rbac_ residue removed: "
                  f"{startup_residuals}")

        write_victim_pdf()
        seeded = seed_data(conn)
        ctx.ids = seeded["ids"]
        ctx.seed_ids = seeded["seed_ids"]
        ctx.identity_base = {
            "superadmin": None,
            "admin_nosuper": ctx.ids["base_a"],
            "manager_a": ctx.ids["base_a"],
            "manager_b": ctx.ids["base_b"],
            "handler_a1": ctx.ids["base_a"],
            "handler_b1": ctx.ids["base_b"],
            "researcher_a": ctx.ids["base_a"],
            "viewer_a": ctx.ids["base_a"],
            "viewer_b": ctx.ids["base_b"],
            "victim_user_b": ctx.ids["base_b"],
            "disabled_a": ctx.ids["base_a"],
        }
        print(f"[seed] ids={json.dumps(ctx.ids, ensure_ascii=False)}")

        # ---------- 登录 ----------
        for key, _display, _role, _base, _super, active in SEED_ACCOUNTS:
            email = f"{SEED_PREFIX}{key}@test.com"
            try:
                resp = client.post("/api/auth/login",
                                   json={"email": email, "password": SEED_PASSWORD})
                ctx.login_status[key] = resp.status_code
                if resp.status_code == 200:
                    ctx.tokens[key] = resp.json().get("access_token")
            except Exception as exc:  # noqa: BLE001
                ctx.login_status[key] = -1
                ctx.fail_precondition(f"login_exception {key}: {exc}")

        if "superadmin" not in ctx.tokens:
            ctx.fail_precondition("seed_login_failed: superadmin")
        for key, _display, _role, _base, _super, active in SEED_ACCOUNTS:
            if not active:
                if ctx.login_status.get(key) != 401:
                    ctx.fail_precondition(
                        f"disabled_login_expected_401_got_{ctx.login_status.get(key)}"
                    )
            elif key not in ctx.tokens:
                ctx.fail_precondition(f"seed_login_failed: {key}")

        # secret 匹配（仅当本地手签 token 被接受时才判定期望）
        if "superadmin" in ctx.tokens:
            probe = build_jwt_token("secret_match", ctx, "jwt_secret_match")
            try:
                resp = client.get("/api/auth/me",
                                  headers={"Authorization": f"Bearer {probe}"})
                ctx.secret_matches_service = resp.status_code == 200
            except Exception:
                ctx.secret_matches_service = False
        if not ctx.secret_matches_service:
            ctx.fail_precondition("secret_matches_service=false "
                                  "(forged-token cases inconclusive)")
            exit_code = 2

        # victime 正向探测（陈旧模块的探测不构成前置失败；
        # 其用例会在 execute_suite 中被标 invalid）
        victim_probes = []
        if "superadmin" in ctx.tokens:
            auth = {"Authorization": f"Bearer {ctx.tokens['superadmin']}"}
            probes = [
                ("dog", "dogs", f"/api/dogs/{ctx.ids['victim_dog_b']}"),
                ("video", "videos",
                 f"/api/videos/{ctx.ids['victim_video_b']}"),
                ("video_status", "videos",
                 f"/api/videos/{ctx.ids['victim_video_b']}/status"),
                ("video_report", "videos",
                 f"/api/videos/{ctx.ids['victim_video_b']}/report"),
                ("score", "scores",
                 f"/api/scores/{ctx.ids['victim_score_b']}"),
                ("task", "annotations",
                 f"/api/annotations/tasks/{ctx.ids['victim_task_b']}"),
                ("task_annotations", "annotations",
                 f"/api/annotations/tasks/{ctx.ids['victim_task_b']}/annotations"),
            ]
            for name, module, endpoint in probes:
                try:
                    resp = client.get(endpoint, headers=auth)
                    stale = module in ctx.stale_modules
                    victim_probes.append({
                        "probe": name, "module": module,
                        "status": resp.status_code,
                        "stale_module": stale,
                    })
                    if resp.status_code == 200:
                        continue
                    if stale:
                        # 陈旧快照下的异常不构成前置失败：对应套件用例会标 invalid
                        continue
                    ctx.fail_precondition(
                        f"victim_probe_{name}_{resp.status_code}"
                    )
                except Exception as exc:  # noqa: BLE001
                    stale = module in ctx.stale_modules
                    victim_probes.append({
                        "probe": name, "module": module, "status": None,
                        "stale_module": stale,
                        "error": sanitize_text(exc),
                    })
                    if not stale:
                        ctx.fail_precondition(f"victim_probe_{name}_exception")

        report["preconditions"] = {
            "health": health,
            "db_connected": conn is not None,
            "ids": redact_obj(ctx.ids),
            "login_status": ctx.login_status,
            "secret_matches_service": ctx.secret_matches_service,
            "victim_probes": victim_probes,
            "failures": list(ctx.precondition_notes),
        }

        if ctx.precondition_failed:
            exit_code = 2

        # ---------- 套件 ----------
        suites: list[dict] = []
        if not ctx.precondition_failed:
            # suite A（动态）：OpenAPI 拉取允许一次重试，避免并行线重启瞬间的
            # 连接重置把整轮前置判死（重试仍失败才记前置失败）。
            openapi = {}
            openapi_error = None
            for attempt in (1, 2):
                try:
                    resp = client.get("/openapi.json")
                    if resp.status_code == 200:
                        openapi = resp.json()
                        openapi_error = None
                        break
                    openapi_error = f"http_status_{resp.status_code}"
                except Exception as exc:  # noqa: BLE001
                    openapi_error = f"{type(exc).__name__}: {sanitize_text(exc)}"
                if attempt == 1:
                    import time as _time
                    _time.sleep(1.5)
            if openapi_error:
                ctx.fail_precondition(f"openapi_unavailable: {openapi_error}")
                print(f"[suite A] invalid: openapi_unavailable ({openapi_error})")
                suites.append({
                    "name": "A-unauthorized-scan",
                    "category": "A",
                    "status": "invalid",
                    "invalid_reason": f"openapi_unavailable: {openapi_error}",
                    "cases": [],
                })
            else:
                a_specs = dynamic_anonymous_specs(openapi)
                print(f"[suite A] dynamic operations: {len(a_specs)}")
                suites.append(execute_suite(client, "A-unauthorized-scan",
                                            a_specs, ctx, health_ok=True))

            health_b = check_health(client)
            if not health_b["ok"]:
                ctx.fail_precondition(f"health_gate_before_B: {health_b['reason']}")
            b_specs = [s for s in static_case_specs() if s["suite"] == "B"]
            print(f"[suite B] cases: {len(b_specs)}")
            suites.append(execute_suite(client, "B-cross-base-isolation",
                                        b_specs, ctx, health_ok=health_b["ok"]))

            health_c = check_health(client)
            if not health_c["ok"]:
                ctx.fail_precondition(f"health_gate_before_C: {health_c['reason']}")
            c_specs = [s for s in static_case_specs() if s["suite"] == "C"]
            print(f"[suite C] cases: {len(c_specs)}")
            suites.append(execute_suite(client, "C-role-and-jwt", c_specs,
                                        ctx, health_ok=health_c["ok"]))
        else:
            print("[skip] 前置失败，不执行渗透套件（不产出安全结论）")

        report["suites"] = suites

        # ---------- 审计静态核查 ----------
        cur = conn.cursor()
        report["audit_static"] = audit_static_check(cur)
        cur.close()
    except Exception as exc:  # noqa: BLE001
        ctx.fail_precondition(f"fatal: {type(exc).__name__}: {exc}")
        exit_code = 2
        print(f"[fatal] {type(exc).__name__}: {exc}")
    finally:
        # ---------- 清理 ----------
        cleanup_info = {"ok": False, "residual_counts": {}, "pdf_removed": False}
        try:
            if conn is None:
                conn = psycopg2.connect(settings.pg_dsn)
            cleanup_info["residual_counts"] = cleanup_seed_data(
                conn, ctx.seed_ids
            )
            cleanup_info["ok"] = all(
                v == 0 for v in cleanup_info["residual_counts"].values()
            )
            if not cleanup_info["ok"]:
                ctx.fail_precondition(
                    f"cleanup_residual: {cleanup_info['residual_counts']}"
                )
                exit_code = 2
            pdf = REPORTS_DIR / VICTIM_PDF_NAME
            try:
                if pdf.exists():
                    pdf.unlink()
                cleanup_info["pdf_removed"] = not pdf.exists()
            except OSError:
                cleanup_info["pdf_removed"] = False
                ctx.fail_precondition("victim_pdf_cleanup_failed")
                exit_code = 2
        except Exception as exc:  # noqa: BLE001
            cleanup_info["error"] = sanitize_text(exc)
            ctx.fail_precondition(f"cleanup_exception: {exc}")
            exit_code = 2
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        report["cleanup"] = cleanup_info

        # ---------- 观察项 ----------
        observations: list[dict] = []
        health = report.get("target", {}).get("health") or {}
        redis_detail = (health.get("checks", {}).get("redis", {})
                        .get("detail") or "")
        if "redis://" in redis_detail:
            observations.append({
                "case_id": "OBS-health-redis-url",
                "title": "health 回显 redis URL",
                "severity": "INFO",
                "category": "info",
                "evidence": {"component": "redis", "detail": redis_detail},
                "remediation": CATEGORY_REMEDIATION["info"],
            })
        observations.append({
            "case_id": "OBS-openapi-public",
            "title": "openapi.json 对匿名公开",
            "severity": "INFO",
            "category": "info",
            "evidence": {"path": "/openapi.json"},
            "remediation": CATEGORY_REMEDIATION["info"],
        })
        audit = report.get("audit_static") or {}
        if audit and not audit.get("implemented"):
            observations.append({
                "case_id": "OBS-audit-log-missing",
                "title": "P2-3 审计日志未实现",
                "severity": "MEDIUM",
                "category": "audit",
                "evidence": {
                    "code_hits": audit.get("code_hits", 0),
                    "audit_tables": audit.get("audit_tables", []),
                },
                "remediation": CATEGORY_REMEDIATION["audit"],
            })
        for suite in report.get("suites", []):
            for case in suite["cases"]:
                if case["case_id"] == "C-jwt-role-garbage" and \
                        case["status"] == "pass":
                    observations.append({
                        "case_id": "OBS-role-claim-ignored",
                        "title": "role claim 被忽略（DB role 为准）",
                        "severity": "INFO",
                        "category": "info",
                        "evidence": {"actual_status": case["actual_status"]},
                        "remediation": (
                            "不是漏洞；若希望 token claim 与 DB 强一致，可加一致性断言。"
                        ),
                    })
        report["observations"] = observations

        # ---------- 汇总 / findings ----------
        findings = build_findings(report.get("suites", []), observations)
        report["findings"] = findings

        counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        for suite in report.get("suites", []):
            for case in suite["cases"]:
                status_counts[case["status"]] = \
                    status_counts.get(case["status"], 0) + 1
        for finding in findings:
            counts[finding["severity"]] = counts.get(finding["severity"], 0) + 1
        report["summary"] = {
            "suites": len(report.get("suites", [])),
            "cases": sum(len(s["cases"]) for s in report.get("suites", [])),
            "case_status": status_counts,
            "findings_by_severity": counts,
            "cleanup_ok": cleanup_info.get("ok"),
            "secret_matches_service": ctx.secret_matches_service,
        }

        # ---------- 基线 / 阈值 ----------
        triggering_severities = {f["severity"] for f in findings}
        baseline_info = None
        if baseline_path is not None:
            loaded = load_baseline(baseline_path)
            if loaded is None:
                print(f"[fatal] baseline not found: {baseline_path}")
                exit_code = 2
            else:
                baseline_info = apply_baseline(findings, loaded)
                triggering_severities = set(
                    baseline_info["triggering_severities"]
                )
                report["baseline_comparison"] = baseline_info

        if _severity_trigger(args.fail_on, triggering_severities):
            if exit_code == 0:
                exit_code = 1

        # ---------- 写报告 ----------
        if not write_report(report, report_path):
            return 2
        print(f"[report] {report_path}")
        print(f"[summary] cases={report['summary']['cases']} "
              f"findings={report['summary']['findings_by_severity']} "
              f"cleanup_ok={cleanup_info.get('ok')} exit={exit_code}")

        # ---------- 写基线 ----------
        if write_baseline_path is not None:
            baseline_payload = {
                "protocol_version": PROTOCOL_VERSION,
                "task": TASK_ID,
                "generated_at": utc_now().isoformat(),
                "findings": {f["case_id"]: f["severity"] for f in findings},
            }
            baseline_serialized = json.dumps(baseline_payload,
                                             ensure_ascii=False, indent=2)
            hits = report_contains_credentials(baseline_serialized)
            if hits:
                print(f"[redaction] baseline HIT: {hits} — not written")
                return 2
            atomic_write_text(write_baseline_path, baseline_serialized)
            print(f"[baseline] {write_baseline_path}")

        if ctx.precondition_failed:
            print(f"[precondition] failures: {ctx.precondition_notes}")
            exit_code = 2

        return exit_code


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
