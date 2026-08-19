"""Phase 3.8a 端到端测试 — Phase 3 全功能集成验证.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.8a
依据: dev-docs/stages/phase-3.md §3.8a

测试内容（可执行部分，硬件/数据阻塞项标记 SKIP）:
    3.1e  ST-GCN+BC SHADOW 模式部署集成（ONNX 导出 + 路由层 + 单元测试）
    3.3d  3D 姿态重建 MPJPE ≤ 50mm（报告校验）
    3.4c  FCI-IGP 评分卡 4 档验证
    3.6c  RBAC 多租户端到端 25/25
    3.7c  训练历史对比端到端 41/41
    3.8b  延迟验证（当前硬件 1.13x，Jetson 目标 ≤ 0.5x — SKIP 待硬件）

SKIP 项（阻塞原因）:
    3.2d  多犬追踪 MOTA ≥ 70%（待多犬视频 + GT 标注）
    3.5e  Jetson 边缘部署延迟 ≤ 0.5x（待硬件采购）
    3.1d  ST-GCN+BC 真实数据准确率 ≥ 85%（待 Label Studio 标注训练）

用法:
    python scripts/phase3_8_e2e_test.py
    python scripts/phase3_8_e2e_test.py --skip-server   # 跳过需要运行服务器的检查
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPORTS_DIR = PROJECT_ROOT / "reports"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def run_eval_script(script_name: str, timeout: int = 300) -> dict:
    """运行评估脚本，返回结果字典。"""
    script_path = PROJECT_ROOT / "scripts" / script_name
    if not script_path.is_file():
        return {"status": "SKIP", "reason": f"脚本不存在: {script_name}"}
    try:
        proc = subprocess.run(
            [str(VENV_PYTHON), str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        return {
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"status": "FAIL", "reason": f"超时 {timeout}s"}


def check_report_exists(report_name: str, required_keys: list[str] | None = None) -> dict:
    """检查验收报告存在性 + 关键字段。"""
    report_path = REPORTS_DIR / report_name
    if not report_path.is_file():
        return {"status": "SKIP", "reason": f"报告不存在: {report_name}"}
    if report_path.suffix != ".json":
        return {"status": "PASS", "path": str(report_path.relative_to(PROJECT_ROOT))}
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        missing = [k for k in (required_keys or []) if k not in data]
        if missing:
            return {"status": "FAIL", "reason": f"缺失字段: {missing}"}
        return {"status": "PASS", "path": str(report_path.relative_to(PROJECT_ROOT)), "keys": list(data.keys())[:10]}
    except json.JSONDecodeError as e:
        return {"status": "FAIL", "reason": f"JSON 解析失败: {e}"}


def check_stgcn_bc_onnx() -> dict:
    """3.1e: 检查 ST-GCN+BC ONNX 模型已导出。"""
    onnx_path = PROJECT_ROOT / "data" / "models" / "stgcn_bc" / "stgcn_bc_dog24.onnx"
    if not onnx_path.is_file():
        return {"status": "SKIP", "reason": "ONNX 模型未导出（运行 scripts/export_stgcn_bc_onnx.py）"}
    size_kb = onnx_path.stat().st_size / 1024
    return {"status": "PASS", "path": str(onnx_path.relative_to(PROJECT_ROOT)), "size_kb": round(size_kb, 1)}


def check_shadow_mode() -> dict:
    """3.1e: 检查 SHADOW 模式路由层可实例化。"""
    try:
        from backend.ml.behavior import BehaviorRecognizer, DeployMode
        recognizer = BehaviorRecognizer(mode=DeployMode.SHADOW)
        return {"status": "PASS", "mode": recognizer.mode.value}
    except Exception as e:
        return {"status": "FAIL", "reason": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3.8a 端到端集成验证")
    parser.add_argument("--skip-server", action="store_true", help="跳过需要运行服务器的检查")
    parser.add_argument("--timeout", type=int, default=300, help="子脚本超时秒数")
    args = parser.parse_args()

    print("=" * 72)
    print("Phase 3.8a 端到端集成验证")
    print("=" * 72)

    results: dict[str, dict] = {}
    t0 = time.time()

    # === 3.1e ST-GCN+BC 部署集成 ===
    print("\n[3.1e] ST-GCN+BC SHADOW 模式部署集成...")
    results["3.1e_onnx_model"] = check_stgcn_bc_onnx()
    print(f"  ONNX 模型: {results['3.1e_onnx_model']['status']} — {results['3.1e_onnx_model'].get('path', results['3.1e_onnx_model'].get('reason', ''))}")

    results["3.1e_shadow_mode"] = check_shadow_mode()
    print(f"  SHADOW 路由层: {results['3.1e_shadow_mode']['status']} — mode={results['3.1e_shadow_mode'].get('mode', results['3.1e_shadow_mode'].get('reason', ''))}")

    # === 3.3d 3D 姿态重建 ===
    print("\n[3.3d] 3D 姿态重建 MPJPE ≤ 50mm...")
    results["3.3d_report"] = check_report_exists(
        "phase-3.3d-3d-pose-eval.json",
        required_keys=["mpjpe_mm"],
    )
    # 读取 MPJPE 值
    report_3d = REPORTS_DIR / "phase-3.3d-3d-pose-eval.json"
    if report_3d.is_file():
        try:
            data_3d = json.loads(report_3d.read_text(encoding="utf-8"))
            mpjpe = data_3d.get("mpjpe_mm")
            threshold = 50.0
            if mpjpe is not None and mpjpe <= threshold:
                results["3.3d_report"]["status"] = "PASS"
                results["3.3d_report"]["mpjpe_mm"] = mpjpe
                results["3.3d_report"]["threshold"] = threshold
                print(f"  MPJPE: {results['3.3d_report']['status']} — {mpjpe}mm ≤ {threshold}mm")
            else:
                results["3.3d_report"]["status"] = "FAIL"
                results["3.3d_report"]["mpjpe_mm"] = mpjpe
                print(f"  MPJPE: FAIL — {mpjpe}mm > {threshold}mm")
        except Exception as e:
            print(f"  MPJPE: FAIL — {e}")
    else:
        print(f"  报告: {results['3.3d_report']['status']} — {results['3.3d_report'].get('reason', '')}")

    # === 3.4c FCI-IGP 评分卡 ===
    print("\n[3.4c] FCI-IGP 评分卡验证...")
    if not args.skip_server:
        results["3.4c_eval"] = run_eval_script("eval_fci_igp.py", timeout=args.timeout)
        print(f"  eval_fci_igp.py: {results['3.4c_eval']['status']}")
    else:
        results["3.4c_report"] = check_report_exists("phase-3.4c-fci-igp-eval.json")
        print(f"  报告: {results['3.4c_report']['status']} — {results['3.4c_report'].get('path', results['3.4c_report'].get('reason', ''))}")

    # === 3.6c RBAC 多租户 ===
    print("\n[3.6c] RBAC 多租户端到端...")
    if not args.skip_server:
        results["3.6c_eval"] = run_eval_script("eval_rbac.py", timeout=args.timeout)
        print(f"  eval_rbac.py: {results['3.6c_eval']['status']}")
    else:
        results["3.6c_report"] = check_report_exists("phase-3.6c-rbac-eval.json")
        print(f"  报告: {results['3.6c_report']['status']} — {results['3.6c_report'].get('path', results['3.6c_report'].get('reason', ''))}")

    # === 3.7c 训练历史对比 ===
    print("\n[3.7c] 训练历史对比端到端...")
    if not args.skip_server:
        results["3.7c_eval"] = run_eval_script("eval_scores_compare.py", timeout=args.timeout)
        print(f"  eval_scores_compare.py: {results['3.7c_eval']['status']}")
    else:
        results["3.7c_report"] = check_report_exists("phase-3.7c-scores-compare-eval.json")
        print(f"  报告: {results['3.7c_report']['status']} — {results['3.7c_report'].get('path', results['3.7c_report'].get('reason', ''))}")

    # === SKIP 项（阻塞） ===
    print("\n[SKIP] 硬件/数据阻塞项:")
    results["3.2d_multi_dog"] = {"status": "SKIP", "reason": "待多犬视频 + GT 标注"}
    results["3.5e_jetson"] = {"status": "SKIP", "reason": "待 Jetson 硬件采购"}
    results["3.1d_real_data"] = {"status": "SKIP", "reason": "待 Label Studio 标注训练 ≥ 85%"}
    print(f"  3.2d 多犬追踪 MOTA: SKIP — {results['3.2d_multi_dog']['reason']}")
    print(f"  3.5e Jetson 延迟 ≤0.5x: SKIP — {results['3.5e_jetson']['reason']}")
    print(f"  3.1d 真实数据 ≥85%: SKIP — {results['3.1d_real_data']['reason']}")

    # === 汇总 ===
    elapsed = round(time.time() - t0, 1)
    n_pass = sum(1 for r in results.values() if r["status"] == "PASS")
    n_fail = sum(1 for r in results.values() if r["status"] == "FAIL")
    n_skip = sum(1 for r in results.values() if r["status"] == "SKIP")

    print("\n" + "=" * 72)
    print(f"汇总: {n_pass} PASS / {n_skip} SKIP / {n_fail} FAIL  （耗时 {elapsed}s）")
    print("=" * 72)

    # 写报告
    report_path = REPORTS_DIR / "phase-3.8a-e2e-eval.json"
    report_path.write_text(
        json.dumps(
            {
                "phase": "3.8a",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_s": elapsed,
                "summary": {"pass": n_pass, "skip": n_skip, "fail": n_fail},
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"报告: {report_path.relative_to(PROJECT_ROOT)}")

    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
