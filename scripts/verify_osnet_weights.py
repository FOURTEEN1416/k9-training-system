"""验证 boxmot ReID 能加载从 HuggingFace mirror 下载的 OSNet 权重.

用途:
    - 验证 kaiyangzhou/osnet 仓库的 .pth 权重能否被 boxmot 19.0 ReID 运行时加载
    - 验证特征提取功能（forward pass + L2 normalize）
    - 对比 x1.0 vs x0.25 输出维度

跑法:
    .venv/Scripts/python.exe scripts/verify_osnet_weights.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 走 mihomo 代理（避免 boxmot 内部触发 Google Drive 重试）
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")


def main() -> int:
    import numpy as np
    from boxmot.reid.core.reid import ReID

    print("=" * 70)
    print("OSNet weights loading verification")
    print("=" * 70)

    weights_dir = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages" / "models"
    candidates = [
        ("osnet_x1_0_msmt17.pt", "x1.0 MSMT17 combineall"),
        ("osnet_x0_25_msmt17.pt", "x0.25 MSMT17 combineall (boxmot default)"),
    ]

    results: list[dict] = []
    for fname, desc in candidates:
        path = weights_dir / fname
        print(f"\n--- {fname} ({desc}) ---")
        if not path.exists():
            print(f"  MISSING: {path}")
            results.append({"name": fname, "ok": False, "err": "missing"})
            continue
        size_mb = path.stat().st_size / 1_048_576
        print(f"  file: {path}")
        print(f"  size: {size_mb:.2f} MB")

        t0 = time.time()
        try:
            reid = ReID(path, device="cpu", half=False)
            elapsed = time.time() - t0
            print(f"  loaded in {elapsed:.2f}s")
            print(f"  backend model: {type(reid.model).__name__}")
            print(f"  input_shape: {reid.model.input_shape}")
            # 用一张随机图 + 检测框测试特征提取
            img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
            boxes = np.array([[10, 20, 100, 200]], dtype=np.float32)
            feats = reid(img, boxes=boxes)
            print(f"  forward OK: feats shape = {feats.shape}, dtype = {feats.dtype}")
            norm = float(np.linalg.norm(feats[0])) if feats.size else 0.0
            print(f"  L2 norm of first feature: {norm:.6f} (should be ~1.0)")
            ok = feats.size > 0 and abs(norm - 1.0) < 0.01
            print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
            results.append({"name": fname, "ok": ok, "shape": list(feats.shape), "norm": norm})
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.2f}s: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": fname, "ok": False, "err": str(e)})

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  [{status}] {r['name']}")
    n_pass = sum(1 for r in results if r["ok"])
    print(f"\n{n_pass}/{len(results)} weights loaded and forward-pass verified")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
