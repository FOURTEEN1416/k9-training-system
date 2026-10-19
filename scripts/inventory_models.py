"""模型与权重盘点脚本（P0-4）。

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: P0-4（dev-docs/production-readiness-gap-plan.md P0-4）

功能:
    1. 按 readiness.MODEL_GROUPS 的候选顺序解析在盘权重
    2. 逐个做加载验证（ONNX → InferenceSession；torch 系 → torch.load CPU）
    3. 产出 JSON 报告 reports/p0-4-model-inventory-<日期>.json
    4. pose 缺失视为硬失败（exit=1）；研究线模型缺失/损坏标 degraded（exit=0 但报告记录）

用法:
    .venv\\Scripts\\python.exe scripts/inventory_models.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.readiness import MODEL_GROUPS, resolve_first  # noqa: E402


def _verify(path: Path) -> dict[str, Any]:
    """按后缀做加载验证。"""
    suffix = path.suffix.lower()
    try:
        if suffix == ".onnx":
            import onnxruntime as ort

            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            return {
                "loadable": True,
                "engine": "onnxruntime",
                "detail": f"inputs={len(session.get_inputs())} providers={session.get_providers()}",
            }
        if suffix in {".pt", ".bin"} or path.name.endswith(".pth.tar"):
            import torch

            try:
                obj = torch.load(str(path), map_location="cpu", weights_only=True)
                kind = type(obj).__name__
            except Exception:
                # 旧版 checkpoint 可能含非张量对象，退回非安全模式仅做结构验证
                obj = torch.load(str(path), map_location="cpu", weights_only=False)
                kind = type(obj).__name__ + "(weights_only=False)"
            if isinstance(obj, dict):
                keys = list(obj.keys())[:5]
                detail = f"dict keys(sample)={keys}"
            else:
                detail = f"obj type={kind}"
            return {"loadable": True, "engine": "torch", "detail": detail}
        return {"loadable": False, "engine": "unknown", "detail": f"unsupported suffix: {suffix}"}
    except Exception as e:  # noqa: BLE001 — 盘点脚本须单点失败不拖垮全表
        return {"loadable": False, "engine": "error", "detail": f"{type(e).__name__}: {e}"}


def main() -> int:
    report: dict[str, Any] = {"date": str(date.today()), "groups": {}}
    hard_failures: list[str] = []
    degraded: list[str] = []

    for name, candidates in MODEL_GROUPS.items():
        hit = resolve_first(candidates)
        entry: dict[str, Any] = {
            "candidates": [str(c) for c in candidates],
            "resolved_path": str(hit) if hit else None,
            "present": hit is not None,
        }
        if hit is not None:
            entry.update(_verify(hit))
        else:
            entry.update({"loadable": False, "engine": None, "detail": "no candidate on disk"})
        report["groups"][name] = entry

        if not entry["present"]:
            (hard_failures if name == "pose_yolo" else degraded).append(name)
        elif not entry["loadable"]:
            degraded.append(f"{name}(corrupt)")

    out_path = PROJECT_ROOT / "reports" / f"p0-4-model-inventory-{date.today().isoformat()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== P0-4 模型盘点 ===")
    for name, e in report["groups"].items():
        if not e["present"]:
            mark, status = "MISSING", "red"
        elif not e["loadable"]:
            mark, status = "CORRUPT", "yellow"
        else:
            mark, status = "OK", "green"
        print(f"[{mark:7}] {name:15} {e['resolved_path'] or '-'}")
        print(f"          {e['detail']}")
    print(f"\n报告: {out_path}")

    if hard_failures:
        print(f"硬依赖缺失: {', '.join(hard_failures)} → 系统不可部署推理", file=sys.stderr)
        return 1
    if degraded:
        print(f"降级项: {', '.join(degraded)} → 对应模式将运行时降级（见 tasks.py 路由日志）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
