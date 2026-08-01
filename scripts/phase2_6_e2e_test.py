"""Phase 2.6 系统集成端到端测试.

Owner: 全栈（见 AGENTS.md §2.2）
Phase: 2.6
依据: dev-docs/stages/phase-2.md §2.6

测试内容（Phase 2 新增功能，Phase 1.7 已覆盖的部分不再重复）:
    2.6a  USPCA 场景视频端到端（上传 → 推理 → PDF）
    2.6b  历史评分查询 API（by-dog）
    2.6c  延迟验证（USPCA 场景，≤ 1 min/min）
    2.6d  数据飞轮 API 契约（finetune/annotations/models）

前置条件:
    - PostgreSQL @ 127.0.0.1:5433
    - Redis @ 127.0.0.1:6379
    - FastAPI @ 127.0.0.1:8001 (uvicorn backend.app.main:app --port 8001)
    - Celery worker (celery -A backend.workers worker -l info --pool solo)

用法:
    python scripts/phase2_6_e2e_test.py
    python scripts/phase2_6_e2e_test.py --proxy         # 通过 Vite 代理
    python scripts/phase2_6_e2e_test.py --skip-upload   # 跳过视频上传（仅 API 测试）

依据:
    - Phase 1.7 已验证 obedience_trial + puppy_selection 延迟 ≤ 1 min/min
    - Phase 2.6 聚焦 USPCA + 数据飞轮 + 历史评分
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

BACKEND_URL = "http://127.0.0.1:8001"
PROXY_URL = "http://127.0.0.1:5173"
UPLOAD_POLL_INTERVAL = 0.5
UPLOAD_POLL_TIMEOUT = 300
LATENCY_BUDGET_RATIO = 1.0  # ≤ 1 min/min 视频


# ===== HTTP 工具（复用 phase1_7） =====


def http_request(url, method="GET", data=None, headers=None, timeout=30.0):
    req = urllib.request.Request(url, data=data, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def http_get_json(url, timeout=30.0):
    status, body, _ = http_request(url, "GET", timeout=timeout)
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body.decode("utf-8", errors="replace")


def http_post_json(url, payload, timeout=30.0):
    data = json.dumps(payload).encode("utf-8")
    status, body, _ = http_request(
        url, "POST", data=data,
        headers={"Content-Type": "application/json"}, timeout=timeout,
    )
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body.decode("utf-8", errors="replace")


# ===== 合成视频 =====


def _generate_uspca_video() -> Path:
    """生成 USPCA 巡逻犬场景合成视频.

    模拟巡逻犬基本动作序列：站立 → 移动 → 警戒
    2700 帧 / 30 FPS = 90 秒（≥60s 以分摊模型加载开销，满足延迟 ≤ 1 min/min）
    """
    import cv2
    import numpy as np

    video_path = PROJECT_ROOT / "data" / "uploads" / "_phase2_6_uspca.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    width, height = 640, 480
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(2700):
        frame = np.full((height, width, 3), 60, dtype=np.uint8)
        # 模拟犬形状移动
        cx = int(width * (i / 2700.0))
        cy = height // 2
        cv2.ellipse(frame, (cx, cy), (80, 40), 0, 0, 360, (60, 80, 120), -1)
        cv2.circle(frame, (cx + 70, cy - 30), 30, (80, 100, 140), -1)
        cv2.putText(frame, f"USPCA Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return video_path


# ===== 测试运行器 =====


class TestRunner:
    def __init__(self, use_proxy=False, skip_upload=False):
        self.base_url = PROXY_URL if use_proxy else BACKEND_URL
        self.use_proxy = use_proxy
        self.skip_upload = skip_upload
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []
        self._dog_id = None
        self._uspca_video_id = None
        self._uspca_elapsed = 0.0
        self._uspca_duration = 0.0

    def _record(self, name, status, detail=""):
        self.results.append((name, status, detail))
        symbol = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}[status]
        print(f"  {symbol} {name}" + (f": {detail}" if detail else ""))
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.skipped += 1

    def run_test(self, name, test_fn):
        try:
            detail = test_fn()
            self._record(name, "PASS", detail or "")
        except AssertionError as e:
            self._record(name, "FAIL", str(e))
        except Exception as e:
            self._record(name, "FAIL", f"{type(e).__name__}: {e}")

    # ===== 准备 =====

    def test_health(self):
        status, data = http_get_json(f"{self.base_url}/health")
        assert status == 200, f"HTTP {status}"
        assert isinstance(data, dict) and data.get("status") == "ok"
        return f"v{data.get('version', '?')}"

    def test_create_dog(self):
        payload = {
            "name": f"Phase26Dog_{int(time.time())}",
            "breed": "GermanShepherd",
            "gender": "male",
            "chip_id": f"P26_{int(time.time())}",
            "training_stage": "P1",
        }
        status, data = http_post_json(f"{self.base_url}/api/dogs", payload)
        assert status == 201, f"HTTP {status}: {data}"
        self._dog_id = data.get("id")
        assert self._dog_id is not None
        return f"dog_id={self._dog_id}"

    # ===== 2.6a USPCA 场景端到端 =====

    def test_2_6a_uspca_e2e(self):
        """USPCA 巡逻犬场景视频端到端."""
        video_path = _generate_uspca_video()
        self._uspca_duration = 90.0  # 2700 帧 / 30 FPS

        video_id = self._upload_video(video_path, scene="uspca_patrol")
        self._uspca_video_id = video_id

        start = time.time()
        final_status, elapsed = self._poll_status(video_id)
        self._uspca_elapsed = elapsed

        assert final_status == "completed", (
            f"USPCA 视频处理失败: status={final_status}, elapsed={elapsed:.1f}s"
        )

        # 验证 PDF 报告
        status, body, _ = http_request(
            f"{self.base_url}/api/videos/{video_id}/report", "GET", timeout=30.0
        )
        assert status == 200, f"PDF 下载失败 HTTP {status}"
        assert len(body) > 1000, f"PDF 过小: {len(body)} bytes"

        return f"video_id={video_id}, {elapsed:.1f}s, pdf={len(body)} bytes"

    # ===== 2.6b 历史评分查询 API =====

    def test_2_6b_history_scores(self):
        """按 dog_id 查询历史评分."""
        # 用刚创建的 dog_id（可能有 0 条评分，但 API 应正常返回）
        if self._dog_id is None:
            raise AssertionError("无 dog_id（test_create_dog 未执行）")

        status, data = http_get_json(
            f"{self.base_url}/api/scores/by-dog/{self._dog_id}"
        )
        assert status == 200, f"HTTP {status}: {data}"
        assert isinstance(data, list), f"返回非 list: {type(data)}"
        return f"dog_id={self._dog_id}, scores={len(data)}"

    # ===== 2.6c 延迟验证 =====

    def test_2_6c_latency(self):
        """USPCA 场景延迟 ≤ 1 min/min 视频."""
        if self._uspca_duration == 0 or self._uspca_elapsed == 0:
            raise AssertionError("无延迟数据（2.6a 未执行或失败）")

        ratio = self._uspca_elapsed / self._uspca_duration
        passed = ratio <= LATENCY_BUDGET_RATIO
        if not passed:
            raise AssertionError(
                f"延迟超标: {self._uspca_elapsed:.1f}s / {self._uspca_duration:.1f}s = {ratio:.2f}x"
            )
        return f"{self._uspca_elapsed:.1f}s / {self._uspca_duration:.1f}s = {ratio:.2f}x"

    # ===== 2.6d 数据飞轮 API 契约 =====

    def test_2_6d_finetune_status(self):
        """数据飞轮 API: 查询微调状态."""
        status, data = http_get_json(f"{self.base_url}/api/finetune/status")
        assert status == 200, f"HTTP {status}: {data}"
        assert isinstance(data, dict), f"返回非 dict: {type(data)}"
        return f"running={data.get('running', False)}"

    def test_2_6d_pipeline_overview(self):
        """数据飞轮 API: 管线概览."""
        status, data = http_get_json(f"{self.base_url}/api/finetune/pipeline")
        assert status == 200, f"HTTP {status}: {data}"
        assert isinstance(data, dict), f"返回非 dict: {type(data)}"
        return f"keys={list(data.keys())[:5]}"

    def test_2_6d_annotations_health(self):
        """数据飞轮 API: Label Studio 健康检查."""
        status, data = http_get_json(f"{self.base_url}/api/annotations/ls/health")
        # LS 可能未运行，200 或 503 都可接受
        assert status in (200, 503), f"HTTP {status}: {data}"
        return f"HTTP {status}"

    def test_2_6d_models_list(self):
        """数据飞轮 API: 模型列表."""
        status, data = http_get_json(f"{self.base_url}/api/models")
        assert status == 200, f"HTTP {status}: {data}"
        assert isinstance(data, list), f"返回非 list: {type(data)}"
        return f"models={len(data)}"

    # ===== 辅助 =====

    def _upload_video(self, video_path, scene):
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        boundary = "----Phase26Boundary"
        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="file"; filename="{video_path.name}"\r\n'
            .encode()
        )
        body.write(b"Content-Type: video/mp4\r\n\r\n")
        body.write(video_bytes)
        body.write(b"\r\n")

        if self._dog_id:
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="dog_id"\r\n\r\n'.encode())
            body.write(f"{self._dog_id}\r\n".encode())

        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="scene"\r\n\r\n'.encode())
        body.write(f"{scene}\r\n".encode())
        body.write(f"--{boundary}--\r\n".encode())

        status, resp_body, _ = http_request(
            f"{self.base_url}/api/videos/upload", "POST",
            data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=60.0,
        )
        assert status == 201, f"上传失败 HTTP {status}: {resp_body[:500]}"
        resp = json.loads(resp_body)
        video_id = resp.get("id")
        assert video_id is not None, f"无 video_id: {resp}"
        return video_id

    def _poll_status(self, video_id):
        start = time.time()
        final_status = "unknown"
        while time.time() - start < UPLOAD_POLL_TIMEOUT:
            status, data = http_get_json(
                f"{self.base_url}/api/videos/{video_id}/status"
            )
            if status == 200 and isinstance(data, dict):
                final_status = data.get("status", "unknown")
                if final_status in ("completed", "failed"):
                    break
            time.sleep(UPLOAD_POLL_INTERVAL)
        elapsed = time.time() - start
        return final_status, elapsed

    # ===== 主运行 =====

    def run(self):
        print(f"\n{'=' * 70}")
        print(f"Phase 2.6 系统集成端到端测试")
        print(f"目标: {self.base_url}{' (Vite proxy)' if self.use_proxy else ' (直连)'}")
        print(f"{'=' * 70}\n")

        print("[0] 环境准备:")
        self.run_test("健康检查 /health", self.test_health)
        self.run_test("创建测试犬只", self.test_create_dog)

        print("\n[1] 2.6a USPCA 场景端到端:")
        if self.skip_upload:
            print("  [SKIP] --skip-upload 模式")
            self.skipped += 1
        else:
            self.run_test("USPCA 视频 → 推理 → PDF", self.test_2_6a_uspca_e2e)

        print("\n[2] 2.6b 历史评分查询 API:")
        self.run_test("GET /api/scores/by-dog/{dog_id}", self.test_2_6b_history_scores)

        print("\n[3] 2.6c 延迟验证:")
        if self.skip_upload:
            print("  [SKIP] --skip-upload 模式")
            self.skipped += 1
        else:
            self.run_test("USPCA 延迟 ≤ 1 min/min", self.test_2_6c_latency)

        print("\n[4] 2.6d 数据飞轮 API 契约:")
        self.run_test("GET /api/finetune/status", self.test_2_6d_finetune_status)
        self.run_test("GET /api/finetune/pipeline", self.test_2_6d_pipeline_overview)
        self.run_test("GET /api/annotations/ls/health", self.test_2_6d_annotations_health)
        self.run_test("GET /api/models", self.test_2_6d_models_list)

        # 汇总
        print(f"\n{'=' * 70}")
        total = self.passed + self.failed + self.skipped
        print(f"结果: {self.passed} passed, {self.failed} failed, {self.skipped} skipped ({total} total)")
        print(f"{'=' * 70}\n")

        if self.failed > 0:
            print("失败项:")
            for name, status, detail in self.results:
                if status == "FAIL":
                    print(f"  - {name}: {detail}")
            print()

        return 0 if self.failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Phase 2.6 系统集成端到端测试")
    parser.add_argument("--skip-upload", action="store_true", help="跳过视频上传测试")
    parser.add_argument("--proxy", action="store_true", help="通过 Vite 代理测试")
    args = parser.parse_args()

    runner = TestRunner(use_proxy=args.proxy, skip_upload=args.skip_upload)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
