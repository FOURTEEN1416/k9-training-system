"""Phase 3.4 FCI-IGP 端到端视频推理验证.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.4
依据: dev-docs/stages/phase-3.md §3.4

测试内容:
    3.4a  FCI-IGP 场景注册（VALID_SCENES + 评分 API 映射 + YAML 存在）
    3.4b  FCI-IGP 评分卡 YAML Schema（7 维 + 22 行为 + 3 DQ + 5 级评级）
    3.4c  评分卡三档验证（excellent/borderline/failing + DQ）
    3.4d  视频端到端（上传 → 推理 → PDF）+ 延迟验证

前置条件:
    - PostgreSQL @ 127.0.0.1:5433
    - Redis @ 127.0.0.1:6379
    - FastAPI @ 127.0.0.1:8001 (uvicorn backend.app.main:app --port 8001)
    - Celery worker (celery -A backend.workers.celery_app worker -l info --pool=solo)

用法:
    python scripts/phase3_4_e2e_test.py
    python scripts/phase3_4_e2e_test.py --skip-upload   # 跳过视频上传（仅 API/Schema 测试）
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ===== 配置 =====

BACKEND_URL = "http://127.0.0.1:8001"
UPLOAD_POLL_INTERVAL = 0.5
UPLOAD_POLL_TIMEOUT = 300
LATENCY_BUDGET_RATIO = 1.5  # FCI-IGP 22 行为 + SHADOW 双轨，放宽至 1.5 min/min


# ===== HTTP 工具 =====


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


def _generate_fci_igp_video() -> Path:
    """生成 FCI-IGP 场景合成视频.

    模拟工作犬基本动作序列：站立 → 移动 → 警戒
    2700 帧 / 30 FPS = 90 秒（与 Phase 2.6 USPCA 对齐）
    """
    import cv2
    import numpy as np

    video_path = PROJECT_ROOT / "data" / "uploads" / "_phase3_4_fci_igp.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    width, height = 640, 480
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(2700):
        frame = np.full((height, width, 3), 50, dtype=np.uint8)
        # 模拟犬形状移动（从左到右）
        cx = int(width * (i / 2700.0))
        cy = height // 2
        cv2.ellipse(frame, (cx, cy), (80, 40), 0, 0, 360, (70, 90, 130), -1)
        cv2.circle(frame, (cx + 70, cy - 30), 30, (90, 110, 150), -1)
        cv2.putText(frame, f"FCI-IGP Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(frame)
    out.release()
    return video_path


# ===== 测试运行器 =====


class TestRunner:
    def __init__(self, skip_upload=False):
        self.base_url = BACKEND_URL
        self.skip_upload = skip_upload
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []
        self._dog_id = None
        self._fci_video_id = None
        self._fci_elapsed = 0.0
        self._fci_duration = 0.0

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
            "name": f"FCIIgpDog_{int(time.time())}",
            "breed": "GermanShepherd",
            "gender": "male",
            "chip_id": f"F34_{int(time.time())}",
            "training_stage": "P2",
        }
        status, data = http_post_json(f"{self.base_url}/api/dogs", payload)
        assert status == 201, f"HTTP {status}: {data}"
        self._dog_id = data.get("id")
        assert self._dog_id is not None
        return f"dog_id={self._dog_id}"

    # ===== 3.4a 场景注册 =====

    def test_3_4a_scene_registered(self):
        """FCI-IGP 场景在 Video 模型 + 评分 API 中注册。"""
        from backend.app.models.video import VALID_SCENES
        from backend.app.api.scoring import _SCENE_TO_FILE

        assert "fci_igp" in VALID_SCENES, f"fci_igp 未在 VALID_SCENES: {VALID_SCENES}"
        assert _SCENE_TO_FILE.get("fci_igp") == "fci_igp.yaml"
        return "VALID_SCENES + _SCENE_TO_FILE"

    # ===== 3.4b YAML Schema =====

    def test_3_4b_yaml_schema(self):
        """fci_igp.yaml Schema 校验（7 维 + 22 行为 + 3 DQ）。"""
        from backend.ml.scoring import ScoringEngine
        from backend.ml.behavior.constants import ALL_BEHAVIORS_22

        yaml_path = (
            PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "fci_igp.yaml"
        )
        engine = ScoringEngine.from_yaml(yaml_path)

        # 7 维
        assert len(engine.spec.dimensions) == 7, f"维度数: {len(engine.spec.dimensions)}"
        # 权重和 = 1.0
        total_w = sum(d.weight for d in engine.spec.dimensions)
        assert abs(total_w - 1.0) < 0.001, f"权重和: {total_w}"
        # 22 行为
        mapped = set(engine.spec.behavior_mapping.keys())
        assert mapped == set(ALL_BEHAVIORS_22), f"行为映射不匹配"
        # 3 DQ
        assert len(engine.spec.disqualifications) == 3, f"DQ 数: {len(engine.spec.disqualifications)}"
        # IGP 等级
        assert engine.spec.igp_level == "IGP1"
        return f"7维+22行为+3DQ+IGP1"

    # ===== 3.4c 三档评分验证 =====

    def test_3_4c_excellent(self):
        """Excellent 档：96+ 分 + pass + Excellent 评级。"""
        from backend.ml.scoring import ScoringContext, ScoringEngine

        yaml_path = (
            PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "fci_igp.yaml"
        )
        engine = ScoringEngine.from_yaml(yaml_path)
        signals = {
            "action_correct": 97, "action_count": 100,
            "command_to_action_latency": 0.3, "action_duration": 35.0,
            "search_coverage": 0.90, "search_speed": 0.60, "target_found": True,
            "focus_ratio": 0.85, "unnecessary_movement_count": 1,
            "courage_score": 0.92, "avoidance_detected": False,
            "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.rating == "Excellent", f"评级: {result.rating}"
        assert result.verdict == "pass"
        assert result.total_score >= 96, f"总分: {result.total_score}"
        return f"score={result.total_score:.1f} rating={result.rating}"

    def test_3_4c_failing(self):
        """Failing 档：< 70 分 + fail + Insufficient 评级。"""
        from backend.ml.scoring import ScoringContext, ScoringEngine

        yaml_path = (
            PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "fci_igp.yaml"
        )
        engine = ScoringEngine.from_yaml(yaml_path)
        signals = {
            "action_correct": 10, "action_count": 20,
            "command_to_action_latency": 4.0, "action_duration": 3.0,
            "search_coverage": 0.30, "search_speed": 0.20, "target_found": False,
            "focus_ratio": 0.40, "unnecessary_movement_count": 10,
            "courage_score": 0.40, "avoidance_detected": True,
            "gait_symmetry": 0.40, "pace_change_smoothness": 0.40,
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.rating == "Insufficient", f"评级: {result.rating}"
        assert result.verdict == "fail"
        assert result.total_score < 70, f"总分: {result.total_score}"
        return f"score={result.total_score:.1f} rating={result.rating}"

    def test_3_4c_dq_gunfire(self):
        """DQ 枪怯 → 总分清零 + disqualified。"""
        from backend.ml.scoring import ScoringContext, ScoringEngine

        yaml_path = (
            PROJECT_ROOT / "backend" / "ml" / "scoring" / "configs" / "fci_igp.yaml"
        )
        engine = ScoringEngine.from_yaml(yaml_path)
        signals = {
            "action_correct": 20, "action_count": 20,
            "command_to_action_latency": 0.3, "action_duration": 35.0,
            "search_coverage": 0.90, "search_speed": 0.60, "target_found": True,
            "focus_ratio": 0.85, "unnecessary_movement_count": 1,
            "courage_score": 0.92, "avoidance_detected": False,
            "gait_symmetry": 0.92, "pace_change_smoothness": 0.88,
            "gunshot_reaction": "shy",  # DQ 触发
        }
        result = engine.evaluate(ScoringContext(signals=signals, scene="fci_igp"))
        assert result.disqualified is True
        assert result.total_score == 0.0
        assert result.verdict == "fail"
        return f"DQ={result.disqualification_hit} score={result.total_score}"

    # ===== 3.4d 视频端到端 =====

    def test_3_4d_fci_igp_e2e(self):
        """FCI-IGP 场景视频端到端：上传 → 推理 → PDF。"""
        video_path = _generate_fci_igp_video()
        self._fci_duration = 90.0  # 2700 帧 / 30 FPS

        video_id = self._upload_video(video_path, scene="fci_igp")
        self._fci_video_id = video_id

        start = time.time()
        final_status, elapsed = self._poll_status(video_id)
        self._fci_elapsed = elapsed

        assert final_status == "completed", (
            f"FCI-IGP 视频处理失败: status={final_status}, elapsed={elapsed:.1f}s"
        )

        # 验证 PDF 报告
        status, body, _ = http_request(
            f"{self.base_url}/api/videos/{video_id}/report", "GET", timeout=30.0
        )
        assert status == 200, f"PDF 下载失败 HTTP {status}"
        assert len(body) > 1000, f"PDF 过小: {len(body)} bytes"

        # 验证评分结果
        status, data = http_get_json(f"{self.base_url}/api/videos/{video_id}")
        assert status == 200, f"视频详情 HTTP {status}"
        assert data.get("status") == "completed"
        assert data.get("scene") == "fci_igp"

        return f"video_id={video_id}, {elapsed:.1f}s, pdf={len(body)} bytes"

    def test_3_4d_latency(self):
        """FCI-IGP 延迟验证（≤ 1.5 min/min，SHADOW 双轨放宽）。"""
        if self._fci_duration == 0 or self._fci_elapsed == 0:
            raise AssertionError("无延迟数据（3.4d 未执行或失败）")

        ratio = self._fci_elapsed / self._fci_duration
        passed = ratio <= LATENCY_BUDGET_RATIO
        if not passed:
            raise AssertionError(
                f"延迟超标: {self._fci_elapsed:.1f}s / {self._fci_duration:.1f}s = {ratio:.2f}x"
            )
        return f"{self._fci_elapsed:.1f}s / {self._fci_duration:.1f}s = {ratio:.2f}x"

    # ===== 辅助 =====

    def _upload_video(self, video_path, scene):
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        boundary = "----Phase34Boundary"
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
        print(f"Phase 3.4 FCI-IGP 端到端验证")
        print(f"目标: {self.base_url}")
        print(f"{'=' * 70}\n")

        print("[0] 环境准备:")
        self.run_test("健康检查 /health", self.test_health)
        self.run_test("创建测试犬只", self.test_create_dog)

        print("\n[1] 3.4a 场景注册:")
        self.run_test("FCI-IGP 场景注册", self.test_3_4a_scene_registered)

        print("\n[2] 3.4b YAML Schema:")
        self.run_test("fci_igp.yaml Schema 校验", self.test_3_4b_yaml_schema)

        print("\n[3] 3.4c 三档评分验证:")
        self.run_test("Excellent 档", self.test_3_4c_excellent)
        self.run_test("Failing 档", self.test_3_4c_failing)
        self.run_test("DQ 枪怯", self.test_3_4c_dq_gunfire)

        print("\n[4] 3.4d 视频端到端:")
        if self.skip_upload:
            print("  [SKIP] --skip-upload 模式")
            self.skipped += 1
        else:
            self.run_test("FCI-IGP 视频 → 推理 → PDF", self.test_3_4d_fci_igp_e2e)
            self.run_test("延迟验证 ≤ 1.5 min/min", self.test_3_4d_latency)

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
    parser = argparse.ArgumentParser(description="Phase 3.4 FCI-IGP 端到端验证")
    parser.add_argument("--skip-upload", action="store_true", help="跳过视频上传测试")
    args = parser.parse_args()

    runner = TestRunner(skip_upload=args.skip_upload)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
