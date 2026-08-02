"""TensorRT FP16 模型转换脚本（Phase 3.5 Jetson 边缘部署）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.5
依据: dev-docs/stages/phase-3.md §3.5
      dev-docs/research/RESEARCH_JETSON_DEPLOYMENT.md §3

功能:
    将 ONNX 模型转换为 TensorRT FP16 引擎，用于 Jetson Orin Nano Super 部署。
    支持 YOLO26-pose ONNX + ST-GCN+BC ONNX 双模型转换。

前置条件:
    - 目标设备为 Jetson Orin Nano Super + JetPack 6.1
    - 已安装 tensorrt (JetPack 自带) + onnx
    - ONNX 模型已导出（YOLO26-pose best.onnx + ST-GCN+BC stgcn_bc_dog24.onnx）

性能预期（RESEARCH_JETSON_DEPLOYMENT.md §4 实测）:
    - YOLO26-pose FP32: 2700 帧 / 98s（1.13x）
    - YOLO26-pose TRT FP16 + stride=3: 2700 帧 / 13s（0.15x，6.7x 加速）

用法（在 Jetson 设备上执行）:
    python scripts/convert_trt_fp16.py \\
        --onnx runs/train-2/weights/best.onnx \\
        --engine data/models/jetson/yolo26_pose_fp16.engine \\
        --fp16

    python scripts/convert_trt_fp16.py \\
        --onnx data/models/stgcn_bc/stgcn_bc_dog24.onnx \\
        --engine data/models/jetson/stgcn_bc_fp16.engine \\
        --fp16 \\
        --min-batch 1 \\
        --opt-batch 1 \\
        --max-batch 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def convert_onnx_to_trt(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
    int8: bool = False,
    workspace_size_gb: int = 4,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 1,
    dynamic_batch: bool = False,
) -> dict:
    """将 ONNX 模型转换为 TensorRT 引擎.

    Args:
        onnx_path: ONNX 模型路径
        engine_path: TensorRT 引擎输出路径
        fp16: 启用 FP16 精度
        int8: 启用 INT8 精度（需要校准数据）
        workspace_size_gb: TensorRT workspace 大小（GB）
        min_batch: 最小 batch size（动态 batch 模式）
        opt_batch: 最优 batch size
        max_batch: 最大 batch size
        dynamic_batch: 启用动态 batch

    Returns:
        转换结果字典
    """
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX 文件不存在: {onnx_path}")

    # 检查 tensorrt 是否可用
    try:
        import tensorrt as trt
    except ImportError:
        return {
            "success": False,
            "error": "tensorrt 未安装（需在 Jetson 设备上运行）",
            "onnx_path": str(onnx_path),
            "engine_path": str(engine_path),
        }

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)

    # 创建网络
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    # 解析 ONNX
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            return {
                "success": False,
                "error": "ONNX 解析失败: " + "; ".join(errors),
                "onnx_path": str(onnx_path),
            }

    # 配置 builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_size_gb * (1 << 30),
    )

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"[TRT] 启用 FP16 精度")
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print(f"[TRT] 启用 INT8 精度（注意：需要校准数据）")

    # 动态 batch 配置
    if dynamic_batch:
        profile = builder.create_optimization_profile()
        # 获取输入张量名称
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        input_shape = input_tensor.shape  # (-1, C, H, W) 或 (-1, T, K, D)

        # 设置动态 batch 维度
        min_shape = list(input_shape)
        opt_shape = list(input_shape)
        max_shape = list(input_shape)
        min_shape[0] = min_batch
        opt_shape[0] = opt_batch
        max_shape[0] = max_batch

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
        print(f"[TRT] 动态 batch: min={min_batch} opt={opt_batch} max={max_batch}")

    # 构建引擎
    print(f"[TRT] 构建引擎中...（可能需要几分钟）")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        return {
            "success": False,
            "error": "TensorRT 引擎构建失败",
            "onnx_path": str(onnx_path),
        }

    # 保存引擎
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    engine_size_mb = engine_path.stat().st_size / (1024 * 1024)
    print(f"[TRT] 引擎已保存: {engine_path} ({engine_size_mb:.1f} MB)")

    return {
        "success": True,
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "engine_size_mb": round(engine_size_mb, 2),
        "fp16": fp16,
        "int8": int8,
        "workspace_gb": workspace_size_gb,
        "dynamic_batch": dynamic_batch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TensorRT FP16 模型转换（Phase 3.5 Jetson 部署）"
    )
    parser.add_argument(
        "--onnx", type=str, required=True,
        help="ONNX 模型路径",
    )
    parser.add_argument(
        "--engine", type=str, required=True,
        help="TensorRT 引擎输出路径",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="启用 FP16 精度（默认开启）",
    )
    parser.add_argument(
        "--int8", action="store_true",
        help="启用 INT8 精度（需要校准数据）",
    )
    parser.add_argument(
        "--workspace-gb", type=int, default=4,
        help="TensorRT workspace 大小（GB，默认 4）",
    )
    parser.add_argument(
        "--dynamic-batch", action="store_true",
        help="启用动态 batch",
    )
    parser.add_argument(
        "--min-batch", type=int, default=1,
        help="最小 batch size",
    )
    parser.add_argument(
        "--opt-batch", type=int, default=1,
        help="最优 batch size",
    )
    parser.add_argument(
        "--max-batch", type=int, default=1,
        help="最大 batch size",
    )
    args = parser.parse_args()

    result = convert_onnx_to_trt(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=args.fp16,
        int8=args.int8,
        workspace_size_gb=args.workspace_gb,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        dynamic_batch=args.dynamic_batch,
    )

    if result["success"]:
        print(f"\n✅ 转换成功: {result['engine_path']}")
        sys.exit(0)
    else:
        print(f"\n❌ 转换失败: {result.get('error', '未知错误')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
