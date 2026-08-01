"""Label Studio 客户端服务（数据飞轮基础设施）.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 2.1b
依据: dev-docs/stages/phase-2.md §2.1b + ADR 0008 §2.3.2

认证方式: Django session cookie（CSRF + form POST）
- Label Studio 1.23 默认禁用 legacy DRF token
- /user/login/ 表单 POST → session cookie + X-CSRFToken header

能力:
    - login(): 会话登录
    - list_projects(): 列出项目
    - get_project(): 获取项目详情
    - list_tasks(): 列出标注任务
    - get_task(): 获取单个任务（含标注结果）
    - create_task(): 创建预标注任务
    - upload_file(): 上传视频/图片
    - list_annotations(): 获取已完成的标注
    - export_annotations(): 导出标注数据（JSON）
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class LabelStudioError(RuntimeError):
    """Label Studio API 错误。"""


class LabelStudioClient:
    """Label Studio HTTP 客户端（session 认证）。

    用法:
        client = LabelStudioClient()
        client.login()
        tasks = client.list_tasks(project_id=1)
    """

    def __init__(
        self,
        base_url: str | None = None,
        email: str | None = None,
        password: str | None = None,
    ) -> None:
        self.base_url = (base_url or settings.ls_url).rstrip("/")
        self.email = email or settings.ls_email
        self.password = password or settings.ls_password
        self._session: requests.Session | None = None

    @property
    def session(self) -> requests.Session:
        """已认证的 requests session（惰性登录）。"""
        if self._session is None:
            self.login()
        assert self._session is not None
        return self._session

    def login(self) -> None:
        """Django session 表单登录。"""
        s = requests.Session()

        # 1. GET 登录页获取 CSRF cookie
        r = s.get(f"{self.base_url}/user/login/", timeout=15)
        if r.status_code != 200:
            raise LabelStudioError(f"登录页不可达: {r.status_code}")

        csrf = s.cookies.get("csrftoken")
        if not csrf:
            raise LabelStudioError("未获取到 csrftoken cookie")

        # 2. POST 表单登录
        r = s.post(
            f"{self.base_url}/user/login/",
            data={
                "email": self.email,
                "password": self.password,
                "persist_session": "on",
                "csrfmiddlewaretoken": csrf,
            },
            headers={"Referer": f"{self.base_url}/user/login/"},
            timeout=15,
            allow_redirects=False,
        )
        if r.status_code not in (302, 303):
            raise LabelStudioError(f"登录失败: status={r.status_code} body={r.text[:300]}")

        # 3. 验证会话
        r = s.get(f"{self.base_url}/api/projects/?page=1&page_size=1", timeout=10)
        if r.status_code != 200:
            raise LabelStudioError(f"会话未认证: {r.status_code}")

        self._session = s
        logger.info("Label Studio 登录成功: %s", self.base_url)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
        data: dict | None = None,
        timeout: int = 30,
    ) -> Any:
        """统一请求封装（自动加 CSRF header）。"""
        url = f"{self.base_url}{path}"
        headers = {}
        csrf = self.session.cookies.get("csrftoken")
        if csrf and method.upper() in ("POST", "PATCH", "PUT", "DELETE"):
            headers["X-CSRFToken"] = csrf

        r = self.session.request(
            method, url,
            json=json, params=params, files=files, data=data,
            headers=headers, timeout=timeout,
        )
        if r.status_code >= 400:
            raise LabelStudioError(
                f"{method} {path} -> {r.status_code}: {r.text[:500]}"
            )
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    # ===== 项目管理 =====

    def list_projects(self, page: int = 1, page_size: int = 50) -> list[dict]:
        """列出所有项目。"""
        result = self._request(
            "GET", "/api/projects/",
            params={"page": page, "page_size": page_size},
        )
        return result.get("results", []) if isinstance(result, dict) else result

    def get_project(self, project_id: int) -> dict:
        """获取项目详情。"""
        return self._request("GET", f"/api/projects/{project_id}/")

    # ===== 任务管理 =====

    def list_tasks(
        self,
        project_id: int,
        page: int = 1,
        page_size: int = 100,
        fields: str = "all",
    ) -> list[dict]:
        """列出项目下的标注任务。"""
        result = self._request(
            "GET", f"/api/projects/{project_id}/tasks/",
            params={"page": page, "page_size": page_size, "fields": fields},
        )
        return result.get("tasks", []) if isinstance(result, dict) else result

    def get_task(self, project_id: int, task_id: int) -> dict:
        """获取单个任务（含 predictions/annotations）。"""
        return self._request(
            "GET", f"/api/projects/{project_id}/tasks/{task_id}/",
        )

    def create_task(self, project_id: int, task_data: dict) -> dict:
        """创建预标注任务（JSON import）。

        task_data 格式:
            {
                "data": {"video": "/data/upload/xxx.mp4", "video_id": "..."},
                "predictions": [{"result": [...], "model_version": "yolo26-pose"}]
            }
        """
        return self._request(
            "POST", f"/api/projects/{project_id}/import",
            json=task_data,
        )

    def upload_file(self, project_id: int, file_path: str, filename: str | None = None) -> dict:
        """上传文件（multipart）→ 返回 FileUpload 记录。"""
        import os
        name = filename or os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {"file": (name, f)}
            return self._request(
                "POST", f"/api/projects/{project_id}/upload",
                files=files,
            )

    def list_file_uploads(self, project_id: int) -> list[dict]:
        """列出项目下的文件上传记录。"""
        result = self._request(
            "GET", f"/api/projects/{project_id}/file-uploads",
            params={"all": "true"},
        )
        return result if isinstance(result, list) else result.get("results", [])

    def patch_task(self, project_id: int, task_id: int, data: dict) -> dict:
        """更新任务（如修复视频 URL）。"""
        return self._request(
            "PATCH", f"/api/projects/{project_id}/tasks/{task_id}/",
            json=data,
        )

    # ===== 标注导出 =====

    def list_annotations(self, project_id: int, task_id: int) -> list[dict]:
        """获取任务的所有标注结果。"""
        task = self.get_task(project_id, task_id)
        return task.get("annotations", [])

    def export_project(self, project_id: int, export_type: str = "JSON") -> list[dict]:
        """导出项目所有标注（JSON 格式）。

        返回每个任务一条记录，含 completed annotations。
        """
        # 触发导出
        r = self.session.post(
            f"{self.base_url}/api/projects/{project_id}/export",
            json={"exportType": export_type},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        # 新版 LS 需要先创建导出快照
        r = self.session.post(
            f"{self.base_url}/api/projects/{project_id}/exports",
            json={"exportType": export_type, "serializeUITasks": True},
            timeout=30,
        )
        if r.status_code >= 400:
            raise LabelStudioError(f"导出失败: {r.status_code} {r.text[:300]}")
        export_id = r.json().get("id")
        # 轮询直到完成
        import time
        for _ in range(30):
            r = self.session.get(
                f"{self.base_url}/api/projects/{project_id}/exports/{export_id}",
                timeout=10,
            )
            status = r.json().get("status", "")
            if status == "completed":
                break
            time.sleep(1)
        # 下载
        r = self.session.get(
            f"{self.base_url}/api/projects/{project_id}/exports/{export_id}/download",
            timeout=60,
        )
        if r.status_code != 200:
            raise LabelStudioError(f"下载导出失败: {r.status_code}")
        return r.json()

    # ===== 健康检查 =====

    def health(self) -> bool:
        """检查 LS 是否可达且已认证。"""
        try:
            r = self.session.get(
                f"{self.base_url}/api/projects/?page=1&page_size=1",
                timeout=5,
            )
            return r.status_code == 200
        except Exception:
            return False
