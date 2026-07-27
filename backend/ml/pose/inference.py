"""YOLO26-pose 视频推理模块.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.0
依据: dev-docs/stages/phase-1.md §1.0g

功能:
    1. 输入视频路径 → 输出逐帧 24 关键点序列
    2. 输出格式: pkl（dict: frames/box/pose/confidence/meta）
    3. 可选: 写入数据库 keypoints 表（Phase 1.4 集成时启用）

用法:
    # CLI 推理
    python -m backend.ml.pose.inference --video data/sample.mp4 --model best.pt

    # 输出 pkl 到指定路径
    python -m backend.ml.pose.inference --video data/sample.mp4 --output data/keypoints.pkl

    # Python API
    from backend.ml.pose.inference import PoseInferenceEngine
    engine = PoseInferenceEngine("runs/pose/train/weights/best.pt")
    result = engine.infer_video("data/sample.mp4")
    print(result.shape)  # (T, 24, 3)

输出 pkl 结构:
    {
        "meta": {
            "video_path": str,
            "model_path": str,
            "fps": float,
            "frame_count": int,
            "width": int,
            "height": int,
            "duration_sec": float,
            "num_keypoints": 24,
            "kpt_names": [...],  # 24 关键点名称
        },
        "frames": [
            {
                "frame_idx": int,
                "frame_time_sec": float,
                "keypoints": np.ndarray,  # (24, 3) [x, y, conf]
                "box": np.ndarray,        # (4,) [x1, y1, x2, y2] 或 None
                "box_conf": float,        # 检测置信度
            },
            ...
        ],
        "keypoints_sequence": np.ndarray,  # (T, 24, 3) 拼接后的序列
    }
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

# Dog-Pose 24 关键点名称（与 data/dog-pose.yaml 一致）
KPT_NAMES = [
    "front_left_paw", "front_left_knee", "front_left_elbow",
    "rear_left_paw", "rear_left_knee", "rear_left_elbow",
    "front_right_paw", "front_right_knee", "front_right_elbow",
    "rear_right_paw", "rear_right_knee", "rear_right_elbow",
    "tail_start", "tail_end",
    "left_ear_base", "right_ear_base",
    "nose", "chin",
    "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye",
    "withers", "throat",
]
NUM_KEYPOINTS = 24


@dataclass
class FrameResult:
    """单帧推理结果。"""

    frame_idx: int
    frame_time_sec: float
    keypoints: np.ndarray  # (24, 3) [x, y, conf]
    box: Optional[np.ndarray]  # (4,) [x1, y1, x2, y2] 或 None
    box_conf: float = 0.0


@dataclass
class VideoInferenceResult:
    """视频推理结果。"""

    meta: dict
    frames: list[FrameResult] = field(default_factory=list)

    @property
    def keypoints_sequence(self) -> np.ndarray:
        """(T, 24, 3) 关键点序列。"""
        if not self.frames:
            return np.zeros((0, NUM_KEYPOINTS, 3), dtype=np.float32)
        return np.stack([f.keypoints for f in self.frames], axis=0)

    @property
    def shape(self) -> tuple[int, int, int]:
        """序列 shape (T, 24, 3)。"""
        return self.keypoints_sequence.shape


class PoseInferenceEngine:
    """YOLO26-pose 推理引擎。

    支持三种模型格式:
        - .pt: PyTorch 原生（默认，GPU 17ms/frame @ RTX 5060）
        - .engine: TensorRT FP16（Phase 1.1，目标 ≤5ms/frame，需系统安装）
        - .onnx: ONNX Runtime GPU（Phase 1.1 回退方案，13ms/frame @ RTX 5060）

    YOLO() 自动按文件后缀选择后端，无需手动配置。
    """

    def __init__(
        self,
        model_path: str | Path,
        device: int | str = 0,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        verbose: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.verbose = verbose

        if not Path(model_path).exists() and not str(model_path).endswith(".pt"):
            # yolo26n-pose.pt 等会自动下载
            pass
        elif not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        print(f"[pose] 加载模型: {model_path}", flush=True)
        self.model = YOLO(model_path)

    def infer_video(
        self,
        video_path: str | Path,
        save_output: bool = True,
        output_path: Optional[str | Path] = None,
    ) -> VideoInferenceResult:
        """对整段视频做姿态推理。

        Args:
            video_path: 视频文件路径
            save_output: 是否保存 pkl
            output_path: pkl 输出路径（默认与视频同目录 .pkl）

        Returns:
            VideoInferenceResult
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频不存在: {video_path}")

        # 读取视频元数据
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0.0
        cap.release()

        print(f"[pose] 视频: {video_path}", flush=True)
        print(f"[pose] fps={fps:.2f} frames={frame_count} {width}x{height} "
              f"duration={duration_sec:.2f}s", flush=True)

        # Ultralytics stream 模式推理（流式，省内存）
        results_stream = self.model.predict(
            source=str(video_path),
            stream=True,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=self.verbose,
            save=False,
        )

        meta = {
            "video_path": str(video_path),
            "model_path": self.model_path,
            "fps": float(fps),
            "frame_count": int(frame_count),
            "width": int(width),
            "height": int(height),
            "duration_sec": float(duration_sec),
            "num_keypoints": NUM_KEYPOINTS,
            "kpt_names": KPT_NAMES,
            "imgsz": self.imgsz,
            "conf_threshold": self.conf,
            "iou_threshold": self.iou,
        }

        result = VideoInferenceResult(meta=meta)
        processed = 0

        for frame_idx, res in enumerate(results_stream):
            frame_time = frame_idx / fps if fps > 0 else 0.0

            # 提取关键点：results[0].keypoints.data shape = (N, 24, 3)
            # N = 检测到的狗数量，单犬场景取第 0 个
            if res.keypoints is not None and len(res.keypoints.data) > 0:
                # 取置信度最高的检测
                if len(res.keypoints.data) > 1:
                    box_confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.array([0.0])
                    pick = int(np.argmax(box_confs))
                else:
                    pick = 0

                kpts = res.keypoints.data[pick].cpu().numpy()  # (24, 3)
                # 兜底：若关键点数不足 24，pad 0
                if kpts.shape[0] < NUM_KEYPOINTS:
                    pad = np.zeros((NUM_KEYPOINTS - kpts.shape[0], 3), dtype=np.float32)
                    kpts = np.concatenate([kpts, pad], axis=0)
                elif kpts.shape[0] > NUM_KEYPOINTS:
                    kpts = kpts[:NUM_KEYPOINTS]

                # 边界框
                if res.boxes is not None and len(res.boxes.xyxy) > 0:
                    box = res.boxes.xyxy[pick].cpu().numpy()  # (4,)
                    box_conf = float(res.boxes.conf[pick].cpu().numpy())
                else:
                    box = None
                    box_conf = 0.0
            else:
                # 未检测到狗，全 0 关键点
                kpts = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
                box = None
                box_conf = 0.0

            result.frames.append(
                FrameResult(
                    frame_idx=frame_idx,
                    frame_time_sec=float(frame_time),
                    keypoints=kpts.astype(np.float32),
                    box=box.astype(np.float32) if box is not None else None,
                    box_conf=float(box_conf),
                )
            )
            processed += 1
            if processed % 100 == 0:
                print(f"[pose] 进度: {processed}/{frame_count} 帧", flush=True)

        print(f"[pose] 完成: {processed} 帧, shape={result.shape}", flush=True)

        if save_output:
            if output_path is None:
                output_path = video_path.with_suffix(".pkl")
            self._save_pkl(result, output_path)

        return result

    def infer_image(self, image_path: str | Path) -> np.ndarray:
        """单图推理，返回 (24, 3) 关键点。"""
        results = self.model.predict(
            source=str(image_path),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=self.verbose,
            save=False,
        )
        if not results or results[0].keypoints is None or len(results[0].keypoints.data) == 0:
            return np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        kpts = results[0].keypoints.data[0].cpu().numpy()  # (24, 3)
        if kpts.shape[0] < NUM_KEYPOINTS:
            pad = np.zeros((NUM_KEYPOINTS - kpts.shape[0], 3), dtype=np.float32)
            kpts = np.concatenate([kpts, pad], axis=0)
        elif kpts.shape[0] > NUM_KEYPOINTS:
            kpts = kpts[:NUM_KEYPOINTS]
        return kpts.astype(np.float32)

    @staticmethod
    def _save_pkl(result: VideoInferenceResult, output_path: str | Path) -> None:
        """保存推理结果到 pkl。"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 序列化：FrameResult 内 np.ndarray 直接 pickle
        payload = {
            "meta": result.meta,
            "frames": [
                {
                    "frame_idx": f.frame_idx,
                    "frame_time_sec": f.frame_time_sec,
                    "keypoints": f.keypoints,
                    "box": f.box,
                    "box_conf": f.box_conf,
                }
                for f in result.frames
            ],
            "keypoints_sequence": result.keypoints_sequence,
        }
        with open(output_path, "wb") as fp:
            pickle.dump(payload, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[pose] 已保存 pkl: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)",
              flush=True)

    @staticmethod
    def load_pkl(pkl_path: str | Path) -> dict:
        """加载 pkl。"""
        with open(pkl_path, "rb") as fp:
            return pickle.load(fp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26-pose 视频推理（Phase 1.0）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument(
        "--model", default="yolo26n-pose.pt",
        help="模型路径（.pt 或 .engine，默认自动下载 yolo26n-pose.pt）",
    )
    parser.add_argument("--output", default=None, help="pkl 输出路径（默认与视频同目录）")
    parser.add_argument("--device", default=0, help="设备（0 / cpu）")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    parser.add_argument("--no-save", action="store_true", help="不保存 pkl")
    parser.add_argument("--verbose", action="store_true", help="详细日志")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = PoseInferenceEngine(
        model_path=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=args.verbose,
    )
    result = engine.infer_video(
        video_path=args.video,
        save_output=not args.no_save,
        output_path=args.output,
    )
    # 摘要
    print(f"\n=== 推理摘要 ===", flush=True)
    print(f"视频: {result.meta['video_path']}", flush=True)
    print(f"模型: {result.meta['model_path']}", flush=True)
    print(f"帧数: {len(result.frames)}", flush=True)
    print(f"关键点序列 shape: {result.shape}  (期望 (T, 24, 3))", flush=True)
    if result.shape[1] != NUM_KEYPOINTS or result.shape[2] != 3:
        print(f"❌ shape 不符合预期 (T, 24, 3)", file=sys.stderr)
        return 1
    print(f"✅ Phase 1.0-d 推理 API 验证通过: shape={result.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
