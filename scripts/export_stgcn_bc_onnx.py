"""ST-GCN+BC ONNX 导出命令行入口.

Owner: ML 开发
Phase: 3.1e

用法:
    python scripts/export_stgcn_bc_onnx.py \\
        --checkpoint runs/stgcn_bc_synthetic/best.pt \\
        --output data/models/stgcn_bc/stgcn_bc_dog24.onnx
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# 确保项目根在 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.ml.behavior.stgcn_bc.export_onnx import export_onnx


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ST-GCN+BC ONNX 导出")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="checkpoint 路径 (best.pt)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="ONNX 输出路径"
    )
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=30, help="trace 用时间长度")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = export_onnx(
        args.checkpoint,
        args.output,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        window_size=args.window_size,
        opset_version=args.opset,
        verify=not args.no_verify,
    )

    print("\n" + "=" * 60)
    print("ST-GCN+BC ONNX 导出完成")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
