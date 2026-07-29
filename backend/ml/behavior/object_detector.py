"""YOLO26 COCO 80 类物体检测器（幼犬选育场景）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.5
依据: dev-docs/stages/phase-1.md §1.5

功能:
    - 输入视频帧 → 输出球/食物/玩具类物体的检测框序列
    - 用于幼犬选育场景的 approach/chase/hold 信号提取（Phase 1.6）

设计:
    - 使用 Ultralytics YOLO26 默认 COCO 80 类权重（yolo26n.pt）
    - 默认仅保留与工作犬选育相关的类目（球类/食物类/玩具类/人）
    - 单例缓存：避免每次任务重新加载模型
    - 与 PoseInferenceEngine 风格一致：流式推理 + np.ndarray 输出

COCO 80 类相关索引:
    - 球类: 32 (sports ball)
    - 食物类: 46 (banana), 47 (apple), 48 (sandwich), 49 (orange),
              50 (broccoli), 51 (carrot), 54 (hot dog), 55 (pizza),
              56 (donut), 57 (cake)
    - 玩具类: COCO 无直接类，用 32 (sports ball) 替代
    - 人: 0 (person) — 用于 heel 行为辅助 + 惊吓源检测
    - 玩具替代扩展: 38 (baseball bat), 39 (baseball glove),
                   41 (tennis racket) — 球类交互配件
"""
from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# === COCO 80 类相关索引（工作犬选育场景）===
# 完整 COCO 类名（80 类，索引 0-79）
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# 工作犬选育场景相关类目分组
PERSON_IDX = 0
DOG_IDX = 16
BALL_IDX = 32  # sports ball — 主要的"猎物/玩具"代理
FOOD_INDICES = [46, 47, 48, 49, 50, 51, 54, 55, 56, 57]  # 食物类
TOY_INDICES = [32, 38, 39, 41]  # 玩具替代：球 + 球拍/手套
TOY_AND_BALL = sorted(set([BALL_IDX] + TOY_INDICES))  # 全部球/玩具类
ALL_INTEREST_INDICES = sorted(set([PERSON_IDX, DOG_IDX] + FOOD_INDICES + TOY_AND_BALL))

# 类别分组标签（用于信号提取）
CATEGORY_BALL = "ball"
CATEGORY_FOOD = "food"
CATEGORY_TOY = "toy"
CATEGORY_PERSON = "person"
CATEGORY_DOG = "dog"
CATEGORY_OTHER = "other"

# 索引 → 类别分组映射
def _idx_to_category(idx: int) -> str:
    if idx == PERSON_IDX:
        return CATEGORY_PERSON
    if idx == DOG_IDX:
        return CATEGORY_DOG
    if idx == BALL_IDX:
        return CATEGORY_BALL
    if idx in FOOD_INDICES:
        return CATEGORY_FOOD
    if idx in TOY_INDICES:
        return CATEGORY_TOY
    return CATEGORY_OTHER


@dataclass
class Detection:
    """单个物体检测结果。"""

    class_id: int              # COCO 类索引 0-79
    class_name: str            # COCO 类名
    category: str              # 工作犬场景分组（ball/food/toy/person/dog/other）
    confidence: float          # 置信度 [0, 1]
    bbox: np.ndarray           # (4,) [x1, y1, x2, y2] 像素坐标

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "category": self.category,
            "confidence": float(self.confidence),
            "bbox": self.bbox.tolist(),
        }

    @property
    def center(self) -> tuple[float, float]:
        """检测框中心点 (cx, cy)。"""
        x1, y1, x2, y2 = self.bbox
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    @property
    def area(self) -> float:
        """检测框面积（像素²）。"""
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))


@dataclass
class FrameDetection:
    """单帧检测结果。"""

    frame_idx: int
    frame_time_sec: float
    detections: list[Detection] = field(default_factory=list)

    def filter_by_category(self, category: str) -> list[Detection]:
        """按类别筛选检测。"""
        return [d for d in self.detections if d.category == category]

    @property
    def has_ball(self) -> bool:
        return any(d.category == CATEGORY_BALL for d in self.detections)

    @property
    def has_food(self) -> bool:
        return any(d.category == CATEGORY_FOOD for d in self.detections)

    @property
    def has_toy(self) -> bool:
        return any(d.category in (CATEGORY_TOY, CATEGORY_BALL)
                   for d in self.detections)

    @property
    def has_person(self) -> bool:
        return any(d.category == CATEGORY_PERSON for d in self.detections)


@dataclass
class VideoDetectionResult:
    """视频物体检测完整结果。"""

    meta: dict
    frames: list[FrameDetection] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int]:
        return (len(self.frames),)


class ObjectDetector:
    """YOLO26 COCO 80 类物体检测器.

    用法:
        detector = ObjectDetector("yolo26n.pt")
        result = detector.detect_video("data/puppy_test.mp4")
        for frame in result.frames:
            if frame.has_ball:
                print(f"帧 {frame.frame_idx} 检测到球")

    支持:
        - .pt: PyTorch 原生（默认）
        - .engine: TensorRT FP16
        - .onnx: ONNX Runtime
    """

    def __init__(
        self,
        model_path: str | Path = "yolo26n.pt",
        device: int | str = 0,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        verbose: bool = False,
        interested_classes: Optional[list[int]] = None,
    ) -> None:
        """
        Args:
            model_path: YOLO26 模型路径（.pt/.engine/.onnx）
            device: 推理设备（0/cpu）
            imgsz: 输入尺寸
            conf: 置信度阈值
            iou: NMS IoU 阈值
            interested_classes: 仅保留这些类（None = 全部 80 类，
                                默认 ALL_INTEREST_INDICES）
        """
        self.model_path = str(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.verbose = verbose
        self.interested_classes = (
            interested_classes if interested_classes is not None
            else ALL_INTEREST_INDICES
        )

        if not Path(model_path).exists() and not str(model_path).endswith(".pt"):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"[detector] 加载模型: {model_path}", flush=True)
        self.model = YOLO(model_path)

    def detect_video(
        self,
        video_path: str | Path,
        save_output: bool = False,
        output_path: Optional[str | Path] = None,
    ) -> VideoDetectionResult:
        """对整段视频做物体检测.

        Args:
            video_path: 视频文件路径
            save_output: 是否保存 pkl
            output_path: pkl 输出路径（默认与视频同目录 .detections.pkl）

        Returns:
            VideoDetectionResult
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0.0
        cap.release()

        print(f"[detector] 视频: {video_path}", flush=True)
        print(f"[detector] fps={fps:.2f} frames={frame_count} {width}x{height} "
              f"duration={duration_sec:.2f}s", flush=True)

        # 流式推理
        results_stream = self.model.predict(
            source=str(video_path),
            stream=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=self.verbose,
            save=False,
            classes=self.interested_classes,  # 仅保留感兴趣类
        )

        meta = {
            "video_path": str(video_path),
            "model_path": self.model_path,
            "fps": float(fps),
            "frame_count": int(frame_count),
            "width": int(width),
            "height": int(height),
            "duration_sec": float(duration_sec),
            "imgsz": self.imgsz,
            "conf_threshold": self.conf,
            "iou_threshold": self.iou,
            "interested_classes": list(self.interested_classes),
            "detector": "yolo26-coco80",
        }

        result = VideoDetectionResult(meta=meta)
        processed = 0

        for frame_idx, res in enumerate(results_stream):
            frame_time = frame_idx / fps if fps > 0 else 0.0
            dets: list[Detection] = []

            if res.boxes is not None and len(res.boxes) > 0:
                # 提取所有检测框
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # (N, 4)
                confs = res.boxes.conf.cpu().numpy()       # (N,)
                classes = res.boxes.cls.cpu().numpy().astype(int)  # (N,)

                for i in range(len(boxes_xyxy)):
                    cls_id = int(classes[i])
                    cls_name = (COCO_NAMES[cls_id]
                                if 0 <= cls_id < len(COCO_NAMES)
                                else f"class_{cls_id}")
                    dets.append(Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        category=_idx_to_category(cls_id),
                        confidence=float(confs[i]),
                        bbox=boxes_xyxy[i].astype(np.float32),
                    ))

            result.frames.append(FrameDetection(
                frame_idx=frame_idx,
                frame_time_sec=float(frame_time),
                detections=dets,
            ))
            processed += 1
            if processed % 100 == 0:
                print(f"[detector] 进度: {processed}/{frame_count} 帧",
                      flush=True)

        print(f"[detector] 完成: {processed} 帧, "
              f"总检测数={sum(len(f.detections) for f in result.frames)}",
              flush=True)

        if save_output:
            if output_path is None:
                output_path = video_path.with_suffix(".detections.pkl")
            self._save_pkl(result, output_path)

        return result

    def detect_image(self, image_path: str | Path) -> list[Detection]:
        """单图检测，返回检测列表."""
        results = self.model.predict(
            source=str(image_path),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=self.verbose,
            save=False,
            classes=self.interested_classes,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        dets: list[Detection] = []
        for i in range(len(boxes_xyxy)):
            cls_id = int(classes[i])
            cls_name = (COCO_NAMES[cls_id]
                        if 0 <= cls_id < len(COCO_NAMES)
                        else f"class_{cls_id}")
            dets.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                category=_idx_to_category(cls_id),
                confidence=float(confs[i]),
                bbox=boxes_xyxy[i].astype(np.float32),
            ))
        return dets

    @staticmethod
    def _save_pkl(result: VideoDetectionResult, output_path: str | Path) -> None:
        """保存检测结果到 pkl."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "meta": result.meta,
            "frames": [
                {
                    "frame_idx": f.frame_idx,
                    "frame_time_sec": f.frame_time_sec,
                    "detections": [d.to_dict() for d in f.detections],
                }
                for f in result.frames
            ],
        }
        with open(output_path, "wb") as fp:
            pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[detector] 已保存 pkl: {output_path} "
              f"({output_path.stat().st_size / 1024:.1f} KB)", flush=True)

    @staticmethod
    def load_pkl(pkl_path: str | Path) -> dict:
        """加载 pkl."""
        with open(pkl_path, "rb") as fp:
            return pickle.load(fp)


# ============================================================
# 单例缓存（供 Celery worker 使用）
# ============================================================

_detector_singleton: Optional[ObjectDetector] = None


def get_detector_singleton(
    model_path: Optional[str | Path] = None,
) -> ObjectDetector:
    """获取 ObjectDetector 单例.

    优先级:
        1. 显式传入的 model_path
        2. runs/train-2/weights/best.onnx（与 pose 共用权重不合适，应用独立 COCO 权重）
        3. yolo26n.pt（自动下载）
    """
    global _detector_singleton
    if _detector_singleton is None:
        if model_path is None:
            # 默认使用 yolo26n.pt（COCO 80 类）
            # 与 pose 模型分开：pose 用 Dog-Pose 微调权重，detector 用 COCO 原权重
            candidates = [
                PROJECT_ROOT / "yolo26n.pt",
            ]
            model_path = "yolo26n.pt"  # 自动下载
            for p in candidates:
                if p.exists():
                    model_path = p
                    break
        print(f"[detector] 初始化单例: {model_path}", flush=True)
        _detector_singleton = ObjectDetector(model_path=model_path, verbose=False)
    return _detector_singleton


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26 COCO 80 物体检测（Phase 1.5）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--model", default="yolo26n.pt",
                        help="模型路径（默认自动下载 yolo26n.pt）")
    parser.add_argument("--output", default=None, help="pkl 输出路径")
    parser.add_argument("--device", default=0, help="设备（0 / cpu）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--no-save", action="store_true", help="不保存 pkl")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detector = ObjectDetector(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=args.verbose,
    )
    result = detector.detect_video(
        video_path=args.video,
        save_output=not args.no_save,
        output_path=args.output,
    )

    # 摘要
    print(f"\n=== 检测摘要 ===", flush=True)
    print(f"视频: {result.meta['video_path']}", flush=True)
    print(f"模型: {result.meta['model_path']}", flush=True)
    print(f"帧数: {len(result.frames)}", flush=True)

    # 按类别统计
    cat_counts: dict[str, int] = {}
    for f in result.frames:
        for d in f.detections:
            cat_counts[d.category] = cat_counts.get(d.category, 0) + 1

    print(f"\n=== 类别统计 ===", flush=True)
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt} 次检测", flush=True)

    # 球/食物帧占比
    ball_frames = sum(1 for f in result.frames if f.has_ball)
    food_frames = sum(1 for f in result.frames if f.has_food)
    total = len(result.frames)
    print(f"\n球类出现帧数: {ball_frames}/{total} ({ball_frames/total*100:.1f}%)", flush=True)
    print(f"食物出现帧数: {food_frames}/{total} ({food_frames/total*100:.1f}%)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
