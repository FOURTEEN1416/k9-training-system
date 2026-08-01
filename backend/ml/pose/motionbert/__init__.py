"""MotionBERT 17→24 关键点适配模块（Phase 3.3c）.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 3.3c
依据: dev-docs/stages/phase-3.md §3.3c + ADR 0010

功能:
    1. DSTformer 17→24 关键点适配（pos_embed 重初始化 + 其他层迁移）
    2. InterPet4D 24 关键点微调管线（基于 lifting_pairing.py 数据）
    3. ONNX 导出 + 推理集成
    4. 与现有 pose 模块（YOLO26-pose 2D 检测 → MotionBERT 3D lifting）闭环

模块结构:
    - model.py: DSTformer 包装器 + 权重迁移
    - config.py: 配置加载（不依赖 easydict）
    - dataset.py: PyTorch Dataset 包装 lifting_pairing
    - train.py: 微调脚本
    - export_onnx.py: ONNX 导出
    - inference.py: 运行时推理（2D → 3D lifting）

设计原则:
    - 不复制 external/MotionBERT 源码，通过 sys.path 注入引用
    - DSTformer 原生支持 num_joints 参数，仅需 pos_embed 适配
    - 不引入兜底层（如随机初始化回退、数据插补）
"""
from backend.ml.pose.motionbert.model import (
    DSTformerWrapper,
    build_model_from_config,
    load_pretrained_weights,
)
from backend.ml.pose.motionbert.config import MotionBERTConfig, load_config
from backend.ml.pose.motionbert.inference import MotionBERTLifter

__all__ = [
    "DSTformerWrapper",
    "MotionBERTConfig",
    "MotionBERTLifter",
    "build_model_from_config",
    "load_config",
    "load_pretrained_weights",
]
