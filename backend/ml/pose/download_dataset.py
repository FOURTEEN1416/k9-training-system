"""Dog-Pose 数据集下载脚本.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.0
依据: dev-docs/stages/phase-1.md §1.0a

下载并解压 Ultralytics 官方 Dog-Pose 数据集（337 MB）到项目内 data/dog-pose/。
24 关键点标注，6773 train / 1703 val。

使用 ultralytics.utils.downloads.safe_download，支持自动重试 + 进度条 + 解压。

用法:
    python -m backend.ml.pose.download_dataset
    python -m backend.ml.pose.download_dataset --force  # 强制重新下载
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics.utils.downloads import safe_download

# 项目根目录（backend/ml/pose/download_dataset.py → 上三级 = 项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dog-pose"
ZIP_PATH = DATA_DIR / "dog-pose.zip"

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/dog-pose.zip"
EXPECTED_TRAIN_IMAGES = 6773
EXPECTED_VAL_IMAGES = 1703


def dataset_exists() -> bool:
    """检查数据集是否已下载且结构完整。"""
    if not DATASET_DIR.exists():
        return False
    train_dir = DATASET_DIR / "images" / "train"
    val_dir = DATASET_DIR / "images" / "val"
    if not train_dir.exists() or not val_dir.exists():
        return False
    train_count = sum(1 for _ in train_dir.glob("*.jpg"))
    val_count = sum(1 for _ in val_dir.glob("*.jpg"))
    return train_count == EXPECTED_TRAIN_IMAGES and val_count == EXPECTED_VAL_IMAGES


def download_and_extract() -> None:
    """下载并解压 dog-pose.zip 到 data/dog-pose/。

    使用 ultralytics safe_download：
    - retry=5（默认 3，提到 5 应对 GitHub release 偶发断流）
    - unzip=True（自动解压）
    - delete=True（解压后删除 zip）
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1/2] 下载 {DATASET_URL}", flush=True)
    print(f"      目标目录: {DATA_DIR}", flush=True)
    # safe_download 会下到 data/dog-pose.zip，解压到 data/dog-pose/
    # 注意：dir 必须传 Path 对象，safe_download 内部 (dir or f.parent).resolve() 要求 Path
    safe_download(
        url=DATASET_URL,
        dir=DATA_DIR,  # Path 对象，不是 str
        unzip=True,
        delete=True,  # 解压后删除 zip 释放空间
        retry=5,
        min_bytes=300 * 1024 * 1024,  # 至少 300MB，防止半截文件
        progress=True,
    )
    print(f"  下载并解压完成", flush=True)


def verify_dataset() -> None:
    """验证数据集完整性。"""
    print("[2/2] 验证数据集完整性", flush=True)
    if not DATASET_DIR.exists():
        # 兜底：检查 data/ 下实际结构
        entries = [p.name for p in DATA_DIR.iterdir()]
        raise RuntimeError(f"解压后未找到 {DATASET_DIR}，data/ 内容: {entries}")

    train_dir = DATASET_DIR / "images" / "train"
    val_dir = DATASET_DIR / "images" / "val"
    train_labels = DATASET_DIR / "labels" / "train"
    val_labels = DATASET_DIR / "labels" / "val"

    train_count = sum(1 for _ in train_dir.glob("*.jpg"))
    val_count = sum(1 for _ in val_dir.glob("*.jpg"))
    train_label_count = sum(1 for _ in train_labels.glob("*.txt"))
    val_label_count = sum(1 for _ in val_labels.glob("*.txt"))

    print(f"  train images: {train_count} (期望 {EXPECTED_TRAIN_IMAGES})", flush=True)
    print(f"  val   images: {val_count} (期望 {EXPECTED_VAL_IMAGES})", flush=True)
    print(f"  train labels: {train_label_count}", flush=True)
    print(f"  val   labels: {val_label_count}", flush=True)

    if train_count != EXPECTED_TRAIN_IMAGES or val_count != EXPECTED_VAL_IMAGES:
        raise RuntimeError(
            f"数据集图像数量不匹配: train={train_count}(期望{EXPECTED_TRAIN_IMAGES}), "
            f"val={val_count}(期望{EXPECTED_VAL_IMAGES})"
        )
    if train_count != train_label_count or val_count != val_label_count:
        raise RuntimeError(
            f"图像与标签数量不匹配: train img/label={train_count}/{train_label_count}, "
            f"val img/label={val_count}/{val_label_count}"
        )
    print(f"  ✅ 数据集验证通过: {DATASET_DIR}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="下载并解压 Dog-Pose 数据集")
    parser.add_argument(
        "--force", action="store_true", help="强制重新下载（即使数据集已存在）"
    )
    args = parser.parse_args()

    if not args.force and dataset_exists():
        print(f"✅ 数据集已存在且完整: {DATASET_DIR}", flush=True)
        print("   使用 --force 强制重新下载", flush=True)
        return 0

    # --force 时清理旧目录
    if args.force and DATASET_DIR.exists():
        print(f"[force] 清理旧目录: {DATASET_DIR}", flush=True)
        shutil.rmtree(DATASET_DIR)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    try:
        download_and_extract()
        verify_dataset()
        print(f"\n✅ Dog-Pose 数据集准备完成: {DATASET_DIR}", flush=True)
        print(f"   配置文件: {DATA_DIR / 'dog-pose.yaml'}", flush=True)
        print(f"   下一步: python -m backend.ml.pose.train", flush=True)
        return 0
    except Exception as e:
        print(f"\n❌ 下载失败: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
