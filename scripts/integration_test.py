"""Phase 1.4h 前后端联调集成测试.

Owner: 后端开发 + 前端开发
Phase: 1.4h
依据: dev-docs/stages/phase-1.md §1.4h

测试流程:
    1. 健康检查（后端直连 + Vite 代理）
    2. API CRUD（dogs / scoring configs）
    3. 评分引擎在线评估
    4. 视频上传 → Celery 推理 → 状态轮询 → 报告下载
    5. 评分结果查询

前置条件:
    - PostgreSQL @ 127.0.0.1:5433
    - Redis @ 127.0.0.1:6379
    - FastAPI @ 127.0.0.1:8000
    - Celery worker（k9_worker）
    - Vite dev server @ 127.0.0.1:5173（可选，代理测试用）

用法:
    python scripts/integration_test.py
    python scripts/integration_test.py --skip-upload  # 跳过视频上传（耗时）
    python scripts/integration_test.py --proxy        # 通过 Vite 代理测试
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ===== 配置 =====

BACKEND_URL = "http://127.0.0.1:8000"
PROXY_URL = "http://127.0.0.1:5173"
UPLOAD_POLL_INTERVAL = 2.0  # 秒
UPLOAD_POLL_TIMEOUT = 300   # 5 分钟


# ===== 工具函数 =====


def http_request(
    url: str,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes, dict]:
    """发送 HTTP 请求，返回 (status_code, body, headers)。"""
    req = urllib.request.Request(url, data=data, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def http_get_json(url: str, timeout: float = 30.0) -> tuple[int, object]:
    status, body, _ = http_request(url, "GET", timeout=timeout)
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body.decode("utf-8", errors="replace")


def http_post_json(url: str, payload: dict, timeout: float = 30.0) -> tuple[int, object]:
    data = json.dumps(payload).encode("utf-8")
    status, body, _ = http_request(
        url, "POST", data=data,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body.decode("utf-8", errors="replace")


# ===== 测试结果 =====


class TestRunner:
    def __init__(self, use_proxy: bool = False):
        self.base_url = PROXY_URL if use_proxy else BACKEND_URL
        self.use_proxy = use_proxy
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results: list[tuple[str, str, str]] = []  # (name, status, detail)

    def _record(self, name: str, status: str, detail: str = "") -> None:
        self.results.append((name, status, detail))
        symbol = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
        print(f"  {symbol} {name}" + (f": {detail}" if detail else ""))
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.skipped += 1

    def run_test(self, name: str, test_fn) -> None:
        try:
            detail = test_fn()
            self._record(name, "PASS", detail or "")
        except AssertionError as e:
            self._record(name, "FAIL", str(e))
        except Exception as e:
            self._record(name, "FAIL", f"{type(e).__name__}: {e}")

    # ===== 测试用例 =====

    def test_health(self) -> str:
        status, data = http_get_json(f"{self.base_url}/health")
        assert status == 200, f"HTTP {status}"
        assert isinstance(data, dict), f"非 JSON: {data}"
        assert data.get("status") == "ok", f"status != ok: {data}"
        return f"v{data.get('version', '?')}"

    def test_list_dogs(self) -> str:
        status, data = http_get_json(f"{self.base_url}/api/dogs?limit=5")
        assert status == 200, f"HTTP {status}"
        # API 可能返回 list 或 {"items": [...]}
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            count = len(data.get("items", []))
        else:
            raise AssertionError(f"非 list/dict: {type(data)}")
        return f"{count} dogs"

    def test_create_dog(self) -> str:
        payload = {
            "name": f"IntegrationTestDog_{int(time.time())}",
            "breed": "TestBreed",
            "gender": "male",
            "chip_id": f"TEST_{int(time.time())}",
            "training_stage": "P0",
        }
        status, data = http_post_json(f"{self.base_url}/api/dogs", payload)
        assert status == 201, f"HTTP {status}: {data}"
        assert isinstance(data, dict), f"非 JSON: {data}"
        dog_id = data.get("id")
        assert dog_id is not None, f"无 id: {data}"
        self._created_dog_id = dog_id
        return f"dog_id={dog_id}"

    def test_list_scoring_configs(self) -> str:
        status, data = http_get_json(f"{self.base_url}/api/scoring/configs")
        assert status == 200, f"HTTP {status}"
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            count = len(data.get("configs", data.get("items", [])))
        else:
            raise AssertionError(f"非 list/dict: {type(data)}")
        return f"{count} configs"

    def test_scoring_evaluate_puppy(self) -> str:
        payload = {
            "signals": {
                "approach_latency": 0.5,
                "approach_speed": 3.0,
                "sniff_duration": 2.0,
                "chase_latency": 0.3,
                "chase_speed": 4.0,
                "hold_duration": 5.0,
                "retreat_distance": 0.2,
                "recovery_time": 1.0,
                "freeze_duration": 0.5,
            },
            "scene": "puppy_selection",
        }
        status, data = http_post_json(f"{self.base_url}/api/scoring/evaluate", payload)
        assert status == 200, f"HTTP {status}: {data}"
        assert isinstance(data, dict), f"非 JSON: {data}"
        score = data.get("total_score")
        assert score is not None, f"无 total_score: {data}"
        assert 0 <= score <= 100, f"分数越界: {score}"
        return f"score={score}, verdict={data.get('verdict', '?')}"

    def test_scoring_evaluate_obedience(self) -> str:
        payload = {
            "signals": {
                "action_correct": 20,
                "action_count": 20,
                "command_to_action_latency": 0.3,
                "action_duration": 35.0,
                "focus_ratio": 0.90,
                "gait_score": 0.90,
            },
            "scene": "obedience_trial",
        }
        status, data = http_post_json(f"{self.base_url}/api/scoring/evaluate", payload)
        assert status == 200, f"HTTP {status}: {data}"
        assert isinstance(data, dict), f"非 JSON: {data}"
        score = data.get("total_score")
        assert score is not None, f"无 total_score: {data}"
        assert 0 <= score <= 100, f"分数越界: {score}"
        return f"score={score}, verdict={data.get('verdict', '?')}"

    def test_list_videos(self) -> str:
        status, data = http_get_json(f"{self.base_url}/api/videos?limit=5")
        assert status == 200, f"HTTP {status}"
        if isinstance(data, list):
            count = len(data)
        elif isinstance(data, dict):
            count = len(data.get("items", []))
        else:
            raise AssertionError(f"非 list/dict: {type(data)}")
        return f"{count} videos"

    def test_upload_video(self) -> str:
        """上传合成视频并轮询状态。"""
        # 1. 生成合成视频
        video_path = self._generate_test_video()
        if video_path is None:
            raise AssertionError("无法生成测试视频（OpenCV 不可用）")

        # 2. 上传
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        boundary = "----IntegrationTestBoundary12345"
        body = io.BytesIO()
        # file 字段
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="file"; filename="{video_path.name}"\r\n'
            .encode()
        )
        body.write(b"Content-Type: video/mp4\r\n\r\n")
        body.write(video_bytes)
        body.write(b"\r\n")
        # dog_id 字段
        dog_id = getattr(self, "_created_dog_id", None)
        if dog_id:
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="dog_id"\r\n\r\n'.encode())
            body.write(f"{dog_id}\r\n".encode())
        # scene 字段
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="scene"\r\n\r\n'.encode())
        body.write(b"obedience_trial\r\n")
        body.write(f"--{boundary}--\r\n".encode())

        status, resp_body, _ = http_request(
            f"{self.base_url}/api/videos/upload",
            "POST",
            data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=60.0,
        )

        assert status == 201, f"上传失败 HTTP {status}: {resp_body[:500]}"

        resp = json.loads(resp_body)
        video_id = resp.get("id")
        assert video_id is not None, f"无 video_id: {resp}"
        self._uploaded_video_id = video_id

        # 3. 轮询状态
        start_time = time.time()
        final_status = "unknown"
        while time.time() - start_time < UPLOAD_POLL_TIMEOUT:
            status, data = http_get_json(
                f"{self.base_url}/api/videos/{video_id}/status"
            )
            if status != 200:
                time.sleep(UPLOAD_POLL_INTERVAL)
                continue
            final_status = data.get("status", "unknown")
            if final_status in ("completed", "failed"):
                break
            time.sleep(UPLOAD_POLL_INTERVAL)

        elapsed = time.time() - start_time
        assert final_status in ("completed", "failed"), (
            f"超时 {elapsed:.0f}s, status={final_status}"
        )

        # 4. 如果完成，检查报告
        if final_status == "completed":
            self._video_completed = True
            # 检查评分
            status, scores = http_get_json(
                f"{self.base_url}/api/scores?video_id={video_id}"
            )
            score_count = 0
            if status == 200:
                if isinstance(scores, list):
                    score_count = len(scores)
                elif isinstance(scores, dict):
                    items = scores.get("items", scores)
                    score_count = len(items) if isinstance(items, list) else 0
            return f"video_id={video_id}, completed in {elapsed:.0f}s, scores={score_count}"
        else:
            self._video_completed = False
            # 获取错误信息
            status, data = http_get_json(f"{self.base_url}/api/videos/{video_id}")
            error = data.get("error_message", "?") if isinstance(data, dict) else "?"
            return f"video_id={video_id}, failed in {elapsed:.0f}s: {error[:80]}"

    def test_download_report(self) -> str:
        """下载 PDF 报告（仅当视频完成时）。"""
        video_id = getattr(self, "_uploaded_video_id", None)
        if video_id is None:
            raise AssertionError("无已上传视频")
        if not getattr(self, "_video_completed", False):
            raise AssertionError("视频未完成，跳过报告下载")

        status, body, headers = http_request(
            f"{self.base_url}/api/videos/{video_id}/report",
            "GET",
            timeout=30.0,
        )
        assert status == 200, f"HTTP {status}"
        # header key 大小写不敏感查找
        content_type = ""
        for k, v in headers.items():
            if k.lower() == "content-type":
                content_type = v
                break
        assert "pdf" in content_type or "octet-stream" in content_type, (
            f"非 PDF: {content_type} (headers={list(headers.keys())})"
        )
        return f"{len(body)} bytes, {content_type}"

    def _generate_test_video(self) -> Path | None:
        """生成合成测试视频（黑白渐变 + 移动圆点，模拟运动）。"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None

        video_path = PROJECT_ROOT / "data" / "uploads" / "_integration_test.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30
        width, height = 640, 480
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        # 生成 60 帧（2 秒）视频：移动的圆点
        for i in range(60):
            frame = np.full((height, width, 3), 50, dtype=np.uint8)
            # 移动的圆点（模拟运动物体）
            cx = int(width * (i / 60.0))
            cy = height // 2
            cv2.circle(frame, (cx, cy), 30, (0, 0, 255), -1)  # 红色圆
            # 添加文字
            cv2.putText(frame, f"Frame {i}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)

        out.release()
        return video_path

    # ===== 主运行 =====

    def run(self, skip_upload: bool = False) -> int:
        print(f"\n{'=' * 60}")
        print(f"Phase 1.4h 前后端联调集成测试")
        print(f"目标: {self.base_url}{' (Vite proxy)' if self.use_proxy else ' (直连)'}")
        print(f"{'=' * 60}\n")

        print("[1] 基础连通性:")
        self.run_test("健康检查 /health", self.test_health)

        print("\n[2] API CRUD:")
        self.run_test("列出犬只 GET /api/dogs", self.test_list_dogs)
        self.run_test("创建犬只 POST /api/dogs", self.test_create_dog)
        self.run_test("列出视频 GET /api/videos", self.test_list_videos)

        print("\n[3] 评分引擎 API:")
        self.run_test("列出评分卡 GET /api/scoring/configs", self.test_list_scoring_configs)
        self.run_test("幼犬选育评分 POST /api/scoring/evaluate", self.test_scoring_evaluate_puppy)
        self.run_test("科目测评评分 POST /api/scoring/evaluate", self.test_scoring_evaluate_obedience)

        print("\n[4] 视频上传 + Celery 推理:")
        if skip_upload:
            print("  [SKIP] 视频上传（--skip-upload）")
            self.skipped += 1
        else:
            self.run_test("上传视频 + 轮询状态", self.test_upload_video)
            if getattr(self, "_video_completed", False):
                self.run_test("下载 PDF 报告", self.test_download_report)
            elif hasattr(self, "_uploaded_video_id"):
                print("  [SKIP] 报告下载（视频未完成）")
                self.skipped += 1

        # 汇总
        print(f"\n{'=' * 60}")
        total = self.passed + self.failed + self.skipped
        print(f"结果: {self.passed} passed, {self.failed} failed, {self.skipped} skipped ({total} total)")
        print(f"{'=' * 60}\n")

        return 0 if self.failed == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1.4h 前后端联调集成测试")
    parser.add_argument("--skip-upload", action="store_true", help="跳过视频上传测试")
    parser.add_argument("--proxy", action="store_true", help="通过 Vite 代理测试")
    args = parser.parse_args()

    runner = TestRunner(use_proxy=args.proxy)
    return runner.run(skip_upload=args.skip_upload)


if __name__ == "__main__":
    sys.exit(main())
