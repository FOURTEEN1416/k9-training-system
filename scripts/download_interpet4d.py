"""InterPet4D 数据集下载脚本（Phase 2 启动前置）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.8 → Phase 2 启动条件
ADR: ADR 0006（DogMo 替代方案）+ ADR 0005（Phase 2 升级决策）

InterPet4D 数据集（HuggingFace ohi_carip/interpet4d，10.7 GB，227 clips）：
- 13 犬 × 多种互动场景
- SMAL 拟合 3D 关键点 kp_world: (T, 24, 3) ← 与 Dog-Pose 24 关键点对齐
- CC BY-NC 4.0 许可证（非商业研究用）

用途:
- 1.6d 验证：kp_world → puppy_signals.extract_puppy_signals() → 9 信号合理性
- 1.2f 复核：视频帧 → YOLO26-pose → 规则引擎 → sit/down/stand/come 准确率

依赖:
    pip install huggingface_hub>=0.24.0

用法:
    # 检查 + 下载指引
    python scripts/download_interpet4d.py

    # 下载（默认 data/interpet4d/，约 10.7 GB）
    python scripts/download_interpet4d.py --download

    # 自定义输出目录
    python scripts/download_interpet4d.py --download --output-dir D:/datasets/interpet4d

    # 校验已下载数据
    python scripts/download_interpet4d.py --verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "interpet4d"

# InterPet4D HuggingFace 仓库
REPO_ID = "ohicarip/interpet4d"
REPO_TYPE = "dataset"
EXPECTED_SIZE_GB = 10.7


def check_interpet4d_exists(output_dir: Path) -> bool:
    """检查 InterPet4D 数据集是否已下载。

    Args:
        output_dir: 输出目录

    Returns:
        True 若目录存在且非空
    """
    return output_dir.exists() and any(output_dir.iterdir())


def print_download_guide(output_dir: Path) -> None:
    """打印下载指引。"""
    print("=" * 72)
    print("InterPet4D 数据集下载指引")
    print("=" * 72)
    print()
    print(f"仓库: {REPO_ID} (HuggingFace Dataset, ~{EXPECTED_SIZE_GB} GB)")
    print(f"目标路径: {output_dir}")
    print(f"许可证: CC BY-NC 4.0（非商业研究用）")
    print()
    print("方式 1: 使用本脚本自动下载（推荐）")
    print("  python scripts/download_interpet4d.py --download")
    print()
    print("方式 2: 使用 huggingface-cli 手动下载")
    print("  pip install huggingface_hub>=0.24.0")
    print("  huggingface-cli login  # 需要 HF token")
    print(f"  huggingface-cli download {REPO_ID} --repo-type {REPO_TYPE} \\")
    print(f"      --local-dir {output_dir}")
    print()
    print("方式 3: 浏览器下载")
    print(f"  https://huggingface.co/datasets/{REPO_ID}")
    print()
    print("解压后目录结构预期（实际以仓库为准）:")
    print("  data/interpet4d/")
    print("    ├── smal_npy/          # SMAL 拟合 3D 关键点 (kp_world: (T, 24, 3))")
    print("    ├── videos/            # RGB 视频流")
    print("    ├── labels/            # 动作类别标签")
    print("    └── README.md          # 数据集说明")
    print()
    print("Phase 2 启动前置:")
    print("  1.6d: kp_world → puppy_signals 验证（scripts/validate_phase2_prereq.py --task 1.6d）")
    print("  1.2f: 视频帧 → YOLO26-pose → 规则引擎准确率（scripts/validate_phase2_prereq.py --task 1.2f）")
    print()


def download_interpet4d(output_dir: Path) -> int:
    """使用 huggingface_hub 下载 InterPet4D。

    Args:
        output_dir: 输出目录

    Returns:
        0 成功 / 1 失败
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] 未安装 huggingface_hub，请先执行:")
        print("  pip install huggingface_hub>=0.24.0")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 开始下载 {REPO_ID} → {output_dir}")
    print(f"[INFO] 预计大小 {EXPECTED_SIZE_GB} GB，请耐心等待...")

    try:
        local_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[OK] 下载完成: {local_path}")
        return 0
    except Exception as e:
        print(f"[ERROR] 下载失败: {type(e).__name__}: {e}")
        print()
        print("可能原因:")
        print("  1. 未登录 HuggingFace（运行 `huggingface-cli login`）")
        print("  2. 网络问题（考虑使用 HF_ENDPOINT=https://hf-mirror.com 镜像）")
        print("  3. 磁盘空间不足（需 ~11 GB 可用空间）")
        return 1


def verify_structure(output_dir: Path) -> int:
    """校验已下载的 InterPet4D 目录结构。

    Args:
        output_dir: 输出目录

    Returns:
        0 完整 / 1 目录不存在 / 2 部分缺失
    """
    if not output_dir.exists():
        print(f"[ERROR] InterPet4D 目录不存在: {output_dir}")
        return 1

    # 列出所有子目录与文件
    items = list(output_dir.iterdir())
    if not items:
        print(f"[ERROR] 目录为空: {output_dir}")
        return 1

    # 统计文件类型
    all_files: list[Path] = []
    for root, _, files in _walk(output_dir):
        for f in files:
            all_files.append(root / f)

    total_size_mb = sum(f.stat().st_size for f in all_files if f.exists()) / (1024 * 1024)
    npy_files = [f for f in all_files if f.suffix == ".npy"]
    video_files = [f for f in all_files if f.suffix in (".mp4", ".avi", ".mov", ".mkv")]
    label_files = [f for f in all_files if f.suffix in (".json", ".csv", ".txt")]
    readme_files = [f for f in all_files if f.name.lower() in ("readme.md", "readme.txt")]

    print(f"[INFO] InterPet4D 目录: {output_dir}")
    print(f"  总文件数: {len(all_files)}")
    print(f"  总大小: {total_size_mb:.1f} MB ({total_size_mb / 1024:.2f} GB)")
    print(f"  .npy 关键点文件: {len(npy_files)}")
    print(f"  视频文件: {len(video_files)}")
    print(f"  标签文件: {len(label_files)}")
    print(f"  README: {len(readme_files)}")

    # 探测 kp_world 样例
    if npy_files:
        try:
            import numpy as np
            sample = npy_files[0]
            arr = np.load(str(sample), allow_pickle=True)
            print(f"  样例 npy: {sample.name}  shape={arr.shape}  dtype={arr.dtype}")
            if hasattr(arr, "item"):
                # 可能是 dict 包装
                try:
                    obj = arr.item()
                    if isinstance(obj, dict) and "kp_world" in obj:
                        kp = obj["kp_world"]
                        print(f"    kp_world shape: {kp.shape}")
                except Exception:
                    pass
        except Exception as e:
            print(f"  [WARN] npy 加载失败: {e}")

    # 完整性判断
    if len(npy_files) == 0 and len(video_files) == 0:
        print("[WARN] 未发现 .npy 或视频文件，可能下载不完整")
        return 2

    if total_size_mb < 5000:
        print(f"[WARN] 总大小 {total_size_mb:.0f} MB < 5000 MB，可能下载不完整（预期 ~{EXPECTED_SIZE_GB * 1024:.0f} MB）")
        return 2

    print("[OK] InterPet4D 目录结构基本完整")
    return 0


def _walk(root: Path):
    """简化版 os.walk（Path 友好）。"""
    dirs: list[Path] = []
    files: list[Path] = []
    for p in root.iterdir():
        if p.is_dir():
            dirs.append(p)
        else:
            files.append(p)
    yield root, dirs, files
    for d in dirs:
        yield from _walk(d)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="InterPet4D 数据集下载/校验（Phase 2 启动前置）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="执行下载（默认仅打印指引）",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="校验已下载数据结构",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    args = parser.parse_args()

    if args.verify:
        return verify_structure(args.output_dir)

    if args.download:
        return download_interpet4d(args.output_dir)

    # 默认：检查 + 指引
    if check_interpet4d_exists(args.output_dir):
        print(f"[INFO] InterPet4D 数据已存在: {args.output_dir}")
        return verify_structure(args.output_dir)

    print_download_guide(args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
