"""Phase 2.0b: Label Studio 启动脚本（绕过 sandbox 限制）.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 2.0b
依据: dev-docs/stages/phase-2.md §2.0b

问题:
    TRAE sandbox 限制 C:\\Users\\<user>\\AppData\\Local\\ 写入，
    LS 默认 BASE_DATA_DIR 指向该路径，启动失败。

方案:
    在 import LS 前 monkey-patch get_data_dir，重定向到项目内 data/label_studio_data。

用法:
    python scripts/start_label_studio.py
    python scripts/start_label_studio.py --port 8080
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# === Monkey-patch: 在 import LS 前重定向 data_dir ===
LS_DATA_DIR = PROJECT_ROOT / "data" / "label_studio_data"
LS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# appdirs.user_data_dir 会返回 C:\Users\<user>\AppData\Local\label-studio
# 我们 monkey-patch label_studio.core.utils.io.get_data_dir
import label_studio.core.utils.io as _ls_io  # noqa: E402

_ls_io.get_data_dir = lambda: str(LS_DATA_DIR)
print(f"[ls] data_dir 重定向: {LS_DATA_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 Label Studio（sandbox 兼容）")
    parser.add_argument("--port", type=int, default=8080, help="端口")
    parser.add_argument("--host", default="127.0.0.1", help="主机")
    args = parser.parse_args()

    # 重新设置 argv 让 LS 正常解析
    sys.argv = [
        "label-studio",
        "start",
        "--port", str(args.port),
        "--host", args.host,
        "--data-dir", str(LS_DATA_DIR),
    ]

    print(f"[ls] 启动 Label Studio: http://{args.host}:{args.port}")
    print(f"[ls] data_dir: {LS_DATA_DIR}")

    from label_studio.server import main as ls_main
    return ls_main()


if __name__ == "__main__":
    sys.exit(main())
