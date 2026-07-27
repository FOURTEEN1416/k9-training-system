"""模型导出脚本: best.pt → best.engine (TensorRT FP16) 或 best.onnx (回退).

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.1
依据: dev-docs/stages/phase-1.md §1.1

导出策略:
    1. 优先导出 TensorRT FP16 engine（需 TensorRT 10.8 + CUDA 12.8）
    2. 失败回退 ONNX（需 onnxruntime-gpu）
    3. 两者都失败则保留 .pt 模式（无加速）

用法:
    # 尝试 TensorRT 导出（需先安装 TensorRT）
    python -m backend.ml.pose.export_engine --model runs/train-2/weights/best.pt --format engine

    # ONNX 导出（回退方案）
    python -m backend.ml.pose.export_engine --model runs/train-2/weights/best.pt --format onnx

    # 自动选择（先试 engine，失败回退 onnx）
    python -m backend.ml.pose.export_engine --model runs/train-2/weights/best.pt --format auto

    # 延迟验证
    python -m backend.ml.pose.export_engine --validate --model runs/train-2/weights/best.pt

TensorRT 10.8 安装步骤（不可逆，需用户授权）:
    1. 下载 TensorRT 10.8.0.6 Windows ZIP（CUDA 12.8）
       https://developer.nvidia.com/tensorrt/download/10x（需 NVIDIA 账号）
    2. 解压到 C:\\TensorRT-10.8.0.6\\
    3. 添加 C:\\TensorRT-10.8.0.6\\lib 到系统 PATH
    4. pip install C:\\TensorRT-10.8.0.6\\python\\tensorrt-10.8.0.6-cp312-cp312-win_amd64.whl
    5. 验证: python -c "import tensorrt; print(tensorrt.__version__)"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def check_tensorrt_available() -> bool:
    """检查 TensorRT 是否可用。"""
    try:
        import tensorrt  # noqa: F401
        return True
    except ImportError:
        return False


def check_onnxruntime_available() -> bool:
    """检查 onnxruntime-gpu 是否可用。"""
    try:
        import onnxruntime as ort
        # 检查是否有 GPU provider
        providers = ort.get_available_providers()
        return "CUDAExecutionProvider" in providers or "CPUExecutionProvider" in providers
    except ImportError:
        return False


def export_tensorrt_engine(
    model_path: str | Path,
    half: bool = True,
    imgsz: int = 640,
    batch: int = 1,
) -> Path:
    """导出 TensorRT FP16 engine。

    Args:
        model_path: .pt 模型路径
        half: FP16 量化
        imgsz: 输入尺寸
        batch: 批大小

    Returns:
        engine 文件路径

    Raises:
        RuntimeError: TensorRT 不可用或导出失败
    """
    if not check_tensorrt_available():
        raise RuntimeError(
            "TensorRT 不可用。请先安装 TensorRT 10.8（见模块 docstring 安装步骤）"
        )

    from ultralytics import YOLO

    model_path = Path(model_path)
    print(f"[export] TensorRT 导出: {model_path} (half={half}, imgsz={imgsz}, batch={batch})")

    model = YOLO(str(model_path))
    engine_path = model.export(
        format="engine",
        half=half,
        imgsz=imgsz,
        batch=batch,
        dynamic=False,
        simplify=True,
        workspace=4,  # 4GB workspace
    )

    engine_path = Path(engine_path)
    if not engine_path.exists():
        raise RuntimeError(f"导出后 engine 文件不存在: {engine_path}")

    size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"[export] ✅ TensorRT engine 导出成功: {engine_path} ({size_mb:.1f} MB)")
    return engine_path


def export_onnx(
    model_path: str | Path,
    imgsz: int = 640,
    batch: int = 1,
    dynamic: bool = True,
) -> Path:
    """导出 ONNX 模型（TensorRT 失败时的回退方案）。

    Args:
        model_path: .pt 模型路径
        imgsz: 输入尺寸
        batch: 批大小
        dynamic: 是否启用动态 batch

    Returns:
        onnx 文件路径
    """
    from ultralytics import YOLO

    model_path = Path(model_path)
    print(f"[export] ONNX 导出: {model_path} (imgsz={imgsz}, batch={batch}, dynamic={dynamic})")

    model = YOLO(str(model_path))
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        batch=batch,
        dynamic=dynamic,
        simplify=True,
        opset=17,
    )

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise RuntimeError(f"导出后 onnx 文件不存在: {onnx_path}")

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[export] ✅ ONNX 导出成功: {onnx_path} ({size_mb:.1f} MB)")
    return onnx_path


def validate_latency(
    model_path: str | Path,
    num_warmup: int = 10,
    num_iters: int = 100,
    imgsz: int = 640,
    device: int | str = 0,
) -> dict:
    """验证推理延迟。

    Args:
        model_path: 模型路径（.pt / .engine / .onnx）
        num_warmup: 预热次数
        num_iters: 正式测量次数
        imgsz: 输入尺寸
        device: 设备

    Returns:
        dict: latency_ms (mean/p50/p95/p99), fps, model_type
    """
    from ultralytics import YOLO

    model_path = Path(model_path)
    suffix = model_path.suffix.lower()
    model_type = {
        ".pt": "pytorch",
        ".engine": "tensorrt",
        ".onnx": "onnx",
    }.get(suffix, "unknown")

    print(f"[validate] 模型: {model_path} (类型: {model_type})")
    print(f"[validate] 预热 {num_warmup} 次 + 测量 {num_iters} 次")

    # 生成随机输入图像
    dummy_image = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # 加载模型
    model = YOLO(str(model_path))

    # 预热
    for _ in range(num_warmup):
        model.predict(source=dummy_image, imgsz=imgsz, device=device, verbose=False, save=False)

    # 正式测量
    latencies = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        model.predict(source=dummy_image, imgsz=imgsz, device=device, verbose=False, save=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    latencies = np.array(latencies)
    result = {
        "model_path": str(model_path),
        "model_type": model_type,
        "imgsz": imgsz,
        "num_iters": num_iters,
        "latency_mean_ms": float(latencies.mean()),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "fps_mean": float(1000.0 / latencies.mean()),
    }

    print(f"[validate] === 延迟结果 ===")
    print(f"[validate] 模型类型: {model_type}")
    print(f"[validate] 均值: {result['latency_mean_ms']:.2f} ms")
    print(f"[validate] P50:  {result['latency_p50_ms']:.2f} ms")
    print(f"[validate] P95:  {result['latency_p95_ms']:.2f} ms")
    print(f"[validate] P99:  {result['latency_p99_ms']:.2f} ms")
    print(f"[validate] FPS:  {result['fps_mean']:.1f}")

    # Phase 1.1 验收：≤ 5 ms/frame
    target = 5.0
    if model_type == "tensorrt":
        if result["latency_mean_ms"] <= target:
            print(f"[validate] ✅ TensorRT 延迟 ≤ {target} ms/frame（Phase 1.1 验收通过）")
        else:
            print(f"[validate] ⚠️ TensorRT 延迟 > {target} ms/frame（Phase 1.1 验收未达标）")

    return result


def export_auto(
    model_path: str | Path,
    imgsz: int = 640,
    batch: int = 1,
) -> Path:
    """自动选择导出格式：先试 TensorRT，失败回退 ONNX。

    Returns:
        导出后的模型路径
    """
    model_path = Path(model_path)

    # 先试 TensorRT
    if check_tensorrt_available():
        try:
            return export_tensorrt_engine(model_path, half=True, imgsz=imgsz, batch=batch)
        except Exception as e:
            print(f"[export] ⚠️ TensorRT 导出失败: {e}", file=sys.stderr)
            print(f"[export] 回退到 ONNX", file=sys.stderr)
    else:
        print("[export] TensorRT 不可用，使用 ONNX")

    # 回退 ONNX
    if not check_onnxruntime_available():
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime-gpu"])
        except Exception:
            print("[export] ⚠️ onnxruntime-gpu 安装失败，尝试 onnxruntime", file=sys.stderr)
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
            except Exception as e:
                raise RuntimeError(
                    f"ONNX Runtime 安装失败，无法导出: {e}"
                )

    return export_onnx(model_path, imgsz=imgsz, batch=batch, dynamic=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO26-pose 模型导出（TensorRT / ONNX）+ 延迟验证",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="runs/train-2/weights/best.pt",
        help="输入模型路径（.pt）",
    )
    parser.add_argument(
        "--format", choices=["engine", "onnx", "auto"], default="auto",
        help="导出格式（auto: 先 engine 失败回退 onnx）",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--batch", type=int, default=1, help="批大小")
    parser.add_argument("--no-half", action="store_true", help="禁用 FP16（仅 engine）")
    parser.add_argument(
        "--validate", action="store_true",
        help="仅验证延迟（不导出），需配合 --model 指定已导出的模型",
    )
    parser.add_argument(
        "--validate-iters", type=int, default=100,
        help="延迟测量次数",
    )
    parser.add_argument("--device", default=0, help="设备（0 / cpu）")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[export] ❌ 模型不存在: {model_path}", file=sys.stderr)
        return 1

    if args.validate:
        # 仅验证
        result = validate_latency(
            model_path=model_path,
            num_iters=args.validate_iters,
            imgsz=args.imgsz,
            device=args.device,
        )
        return 0

    # 导出
    if args.format == "engine":
        if not check_tensorrt_available():
            print("[export] ❌ TensorRT 不可用，请先安装", file=sys.stderr)
            print("[export] 安装步骤见模块 docstring", file=sys.stderr)
            return 1
        exported = export_tensorrt_engine(
            model_path, half=not args.no_half, imgsz=args.imgsz, batch=args.batch
        )
    elif args.format == "onnx":
        exported = export_onnx(model_path, imgsz=args.imgsz, batch=args.batch)
    else:  # auto
        exported = export_auto(model_path, imgsz=args.imgsz, batch=args.batch)

    # 导出后验证延迟
    print(f"\n[export] 验证导出模型延迟...")
    validate_latency(
        model_path=exported,
        num_iters=args.validate_iters,
        imgsz=args.imgsz,
        device=args.device,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
