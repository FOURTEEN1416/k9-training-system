"""行为识别模块.

Owner: ML 开发（见 AGENTS.md §2.2）
Phase: 1.2（科目规则引擎 P0 8 类行为）→ 3.1（ST-GCN+BC 22 类）→ 3.1e（双轨部署）

子模块:
    - constants: 24 关键点索引 + 22 类行为类别
    - rule_engine: 基于几何规则的 16 类行为识别（P0 8 + P1 8）
    - stgcn_bc: ST-GCN+BC 自研模型（22 类 + 边界检测）
    - router: 双轨部署路由层（ST-GCN+BC + 规则引擎）
"""
from backend.ml.behavior.router import BehaviorRecognizer, DeployMode

__all__ = ["BehaviorRecognizer", "DeployMode"]

