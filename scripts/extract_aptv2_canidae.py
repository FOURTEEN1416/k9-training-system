"""APTv2 犬科(Canidae)子集提取脚本.

Owner: ML 开发
Phase: 3.2d

功能:
- 从 APTv2 完整数据集中提取犬科(dog/wolf/fox)子集
- 用于多犬追踪评估 (MOTA/IDF1/ID-switch)
- 输出: 视频帧 + 对应标注 (MOT 格式)

用法:
    python scripts/extract_aptv2_canidae.py \
        --source data/APTv2 \
        --annotations data/aptv2_annotations/APTv2/annotations \
        --output data/aptv2_canidae
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 犬科类别 (在 APTv2 30 类中的 id)
# dog=1, wolf=28, fox=25 (从 val_annotations_Canidae.json 中确认)
CANIDAE_CATEGORIES = {"dog", "wolf", "fox"}


def load_annotations(annotation_file: Path) -> Dict:
    """加载 COCO 格式标注文件."""
    with open(annotation_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_canidae_videos(
    annotations: Dict,
    categories: Optional[set] = None,
) -> tuple:
    """筛选犬科类别的视频与标注.

    Returns:
        (canidae_video_ids, canidae_images, canidae_annotations)
    """
    cats = annotations.get("categories", [])
    cat_name_map = {c["id"]: c["name"] for c in cats}

    target = categories or CANIDAE_CATEGORIES
    target_cat_ids = {cid for cid, name in cat_name_map.items() if name in target}
    print(f"目标类别: {target}")
    print(f"对应 category_id: {target_cat_ids}")

    # 筛选标注
    canid_anns = [
        a for a in annotations.get("annotations", [])
        if a.get("category_id") in target_cat_ids
    ]
    print(f"犬科标注: {len(canid_anns)} 条")

    # 视频 ID 集合
    canid_video_ids = {a.get("video_id") for a in canid_anns}
    print(f"犬科视频: {len(canid_video_ids)} 个")

    # 筛选图片
    canid_images = [
        img for img in annotations.get("images", [])
        if img.get("video_id") in canid_video_ids
    ]
    print(f"犬科帧: {len(canid_images)} 张")

    return canid_video_ids, canid_images, canid_anns


def extract_canidae(
    source_dir: Path,
    annotations: Dict,
    output_dir: Path,
) -> Dict:
    """复制犬科视频帧到输出目录，并生成 MOT 格式 GT.

    Returns:
        统计信息 dict
    """
    canid_video_ids, canid_images, canid_anns = get_canidae_videos(annotations)

    # 创建输出目录
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 复制帧文件
    copied = 0
    for img in canid_images:
        src = source_dir / img["file_name"]
        if src.exists():
            dst = frames_dir / img["file_name"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    print(f"已复制 {copied}/{len(canid_images)} 帧")

    # 生成 MOT 格式 GT (frame_idx, track_id, x, y, w, h, conf, class, visibility)
    gt_lines = []
    # 视频内帧排序: 需要 image -> frame index 映射
    # APTv2 中每视频 15 帧，file_name 包含帧号
    for a in canid_anns:
        img_id = a.get("image_id")
        img = next((i for i in canid_images if i["id"] == img_id), None)
        if not img:
            continue
        # 从 file_name 提取帧索引 (0000.jpg -> 0)
        fname = img["file_name"]
        try:
            frame_idx = int(Path(fname).stem) + 1  # MOT 从 1 开始
        except ValueError:
            frame_idx = 1

        x, y, w, h = a["bbox"]
        track_id = a.get("track_id", 0)
        gt_lines.append(
            f"{frame_idx},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,1,1.0"
        )

    gt_path = output_dir / "gt.txt"
    with open(gt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(gt_lines) + "\n")

    print(f"GT 已生成: {gt_path} ({len(gt_lines)} 行)")

    return {
        "videos": len(canid_video_ids),
        "frames": copied,
        "annotations": len(canid_anns),
        "gt_lines": len(gt_lines),
    }


def main():
    parser = argparse.ArgumentParser(description="提取 APTv2 犬科子集")
    parser.add_argument("--source", type=Path, default=Path("data/APTv2/APTv2/data"),
                        help="APTv2 解压后的数据目录")
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/aptv2_annotations/APTv2/annotations"),
                        help="标注文件目录")
    parser.add_argument("--annotation-file", type=Path,
                        default=Path("val_annotations_hard.json"),
                        help="使用的标注文件 (相对 annotations 目录)")
    parser.add_argument("--output", type=Path, default=Path("data/aptv2_canidae"),
                        help="输出目录")
    args = parser.parse_args()

    # 加载标注
    ann_path = args.annotations / args.annotation_file
    print(f"加载标注: {ann_path}")
    if not ann_path.exists():
        print(f"错误: 标注文件不存在 {ann_path}")
        # 尝试其他标注文件
        for alt in ["val_annotations.json", "test_annotations.json"]:
            p = args.annotations / alt
            if p.exists():
                ann_path = p
                print(f"改用: {p}")
                break

    annotations = load_annotations(ann_path)

    # 提取
    stats = extract_canidae(args.source, annotations, args.output)

    print("\n=== 提取完成 ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()