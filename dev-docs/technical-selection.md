# Technical Selection — 技术选型 Truth

> Truth source: 基于 `research/` 目录下 8 份调研文档综合形成
> 状态: ✅ 已确认（基于 v2.0 PRD + 调研文档）
> 日期: 2026-07-26
> 决策状态: **primary recommendation**（待 foundation gate `--foundation-gate implementation` 验证后转 implementation）

## 1. 架构驱动因素

| 驱动因素 | 来源 | 优先级 |
|---------|------|--------|
| 本地部署、无云依赖 | PRD 硬约束 | P0 |
| 数据隐私（公安/海关场景） | PRD 硬约束 | P0 |
| 三大标准兼容（GA-T/USPCA/FCI-IGP） | PRD 硬约束 | P0 |
| 端到端视频→评分报告 | 核心价值 | P0 |
| Windows 优先部署 | 目标用户环境 | P1 |
| 边缘部署能力（Jetson） | Phase 3+ | P1 |
| 精度优先于速度 | 用户原则 | P1 |
| 可解释性 | 训导员使用需求 | P2 |

## 2. 当前外部证据

> 详见 `research/` 目录

### 2.1 姿态检测引擎

| 方案 | 证据来源 | 关键指标 | 适用性 |
|------|---------|---------|--------|
| **YOLO26-pose** | `research/RESEARCH_YOLO26_DEPLOY_DEEP.md` | 24 关键点、NMS-Free、双头、CPU 提速 43% | ✅ 直接可用 |
| YOLO11-pose | `research/RESEARCH_TECH_COMPARISON.md` | MIT 许可、但精度低于 YOLO26 | ⚠️ 备选（AGPL 风险时） |
| DeepLabCut (DLC) | `research/RESEARCH_OPENSOURCE.md` | ASBAR 框架使用、但重且慢 | ❌ 不推荐 |

### 2.2 行为分类

| 方案 | 证据来源 | 关键指标 | 适用性 |
|------|---------|---------|--------|
| **规则引擎** | `research/RESEARCH_IMPLEMENTATION.md` §5.1 | 坐/卧/立/走 4 类基础、可解释 | ✅ Phase 1 兜底 |
| **PoseC3D** | `research/RESEARCH_LITERATURE.md` | ASBAR 75.3%、抗噪强 | ✅ Phase 1-2 基线 |
| **ST-GCN+BC** | `research/BCST-GCN_ST-GCN_深度分析报告.md` | BCST-GCN 94.43% NTU RGB+D | ✅ Phase 3 升级 |
| Transformer/Mamba | 待调研（Phase 4） | 长序列建模 | ⏳ Phase 4 |

### 2.3 推理加速

| 方案 | 证据来源 | 关键指标 | 适用性 |
|------|---------|---------|--------|
| **TensorRT FP16** | `research/RESEARCH_YOLO26_DEPLOY_DEEP.md` | 2-3x 加速、<0.5% 精度损失 | ✅ Phase 1 |
| ONNX Runtime | 调研文档 | 跨平台但速度次于 TRT | ⚠️ 备选 |
| 原生 PyTorch | 调研文档 | 最简但最慢 | ❌ 仅开发期 |

### 2.4 端侧硬件

| 方案 | 证据来源 | 关键指标 | 适用性 |
|------|---------|---------|--------|
| **Jetson Orin Nano Super** | `research/RESEARCH_YOLO26_DEPLOY_DEEP.md` | $249、67 TOPS、219 FPS | ✅ Phase 3+ |
| Jetson Orin NX | 调研文档 | 更强但更贵 | ⚠️ 备选 |
| Windows + GPU | PRD | 开发环境 | ✅ Phase 0-2 |

## 3. 可信候选组合

### 候选 A（主推）：YOLO26-pose + FastAPI + PostgreSQL + Vue

**组合**：
- 姿态：YOLO26-pose + TensorRT FP16
- 行为：规则 → PoseC3D → ST-GCN+BC（三阶段递进）
- 后端：FastAPI + Celery + Redis
- 数据库：PostgreSQL 15 + pgvector
- 前端：Vue 3 + Naive UI + ECharts
- 部署：原生 Python（Windows 优先）+ Jetson（Phase 3+）

**优势**：
- 调研充分（8 份调研文档支撑）
- ASBAR 框架可直接改造（DLC→PoseC3D 管道）
- 本地部署、无云依赖
- pgvector 支持行为向量相似检索

**劣势**：
- YOLO26 AGPL-3.0 商用限制（Phase 5 前需解决）
- MMAction2 环境配置复杂（需 Docker 预配置）

**迁移悬崖**：
- YOLO26 → YOLO11（如商用许可无法解决）：API 兼容，但需重训
- PoseC3D → ST-GCN+BC：数据格式转换（关键点序列 → 图序列）

### 候选 B（备选）：YOLO11-pose + 同上

**差异**：
- 姿态：YOLO11-pose（MIT 许可，无商用限制）
- 精度略低于 YOLO26

**触发条件**：候选 A 的 AGPL 商用许可无法解决时

### 候选 C（备选）：DLC + ASBAR 原生路径

**差异**：
- 姿态：DeepLabCut（ASBAR 原生路径）
- 速度慢、重，但完全开源

**触发条件**：候选 A/B 均不满足时

## 4. 框架-架构匹配分析

### 4.1 前端框架匹配

| 候选 | 匹配度 | 理由 |
|------|--------|------|
| **Vue 3** | ✅ 高 | 中国生态友好、Naive UI 成熟、学习曲线低 |
| React | ⚠️ 中 | 生态更大但过度工程化 |
| Svelte | ⚠️ 中 | 轻但生态小 |

### 4.2 后端框架匹配

| 候选 | 匹配度 | 理由 |
|------|--------|------|
| **FastAPI** | ✅ 高 | 异步原生、PyTorch 集成好、自动文档 |
| Django | ⚠️ 中 | 同步、重 |
| Flask | ⚠️ 中 | 需大量扩展 |

### 4.3 数据库匹配

| 候选 | 匹配度 | 理由 |
|------|--------|------|
| **PostgreSQL + pgvector** | ✅ 高 | 关系+向量混合、成熟稳定 |
| MySQL | ⚠️ 中 | 无原生向量支持 |
| MongoDB | ❌ 低 | 事务弱、不适合评分数据 |

## 5. 主推组合（Primary Recommendation）

**候选 A：YOLO26-pose + FastAPI + PostgreSQL + Vue**

### 5.1 完整技术栈

| 层 | 选型 | 版本 | 证据 |
|----|------|------|------|
| Python | 3.12 | 3.12.x | 稳定、与 PyTorch 2.4 兼容 |
| PyTorch | 2.4 + CUDA 12.1 | 2.4.x | YOLO26 要求 |
| 姿态检测 | ultralytics 8.6+ | 8.6+ | YOLO26-pose |
| 行为识别 | mmaction2 | latest | PoseC3D/ST-GCN |
| 推理加速 | TensorRT | 10.x | FP16 加速 |
| 后端框架 | FastAPI | 0.110+ | 异步原生 |
| 任务队列 | Celery + Redis | 5.x | 异步推理 |
| ORM | SQLAlchemy async | 2.x | 异步 ORM |
| 数据库 | PostgreSQL 15 + pgvector | 15.x | 关系+向量 |
| 前端框架 | Vue 3 + Vite | 3.4+ | 中国生态 |
| UI 库 | Naive UI | 2.x | Vue 3 原生 |
| 图表 | ECharts | 5.x | 数据可视化 |
| 部署 | 原生 Python | - | Windows 优先 |

### 5.2 部署拓扑

```
开发环境（Phase 0-2）
  Windows + GPU + Python venv
  PostgreSQL 本地实例
  Redis 本地实例

边缘部署（Phase 3+）
  Jetson Orin Nano Super
  TensorRT FP16
  本地存储
```

## 6. 重新评估触发器

以下情况触发技术选型重新评估：

1. **YOLO26 AGPL 商用许可无法解决**：切换到候选 B（YOLO11）
2. **PoseC3D 在工作犬场景精度 <75%**：启动 ST-GCN+BC 提前
3. **MMAction2 环境配置失败**：评估自研行为分类
4. **PostgreSQL 性能不足**：评估 TimescaleDB 或 ClickHouse
5. **Phase 4 触发**：评估 LLM/Transformer-Mamba/RL 集成方案
6. **中文路径迁移失败**：评估 WSL2 或 Docker 部署

## 7. 未验证项

- ⏳ YOLO26-pose 在工作犬品种（马犬/昆明犬）的精度（需 Phase 2 微调验证）
- ⏳ PoseC3D 在俯视/低光/运动模糊场景的表现（需 Phase 2 数据验证）
- ⏳ TensorRT FP16 在 Jetson Orin Nano Super 的实际加速比（需 Phase 3 硬件验证）
- ⏳ MMAction2 在 Windows 原生部署的兼容性（需 Phase 0 环境验证）

## 8. 第三方文档 Gate

> 待 Phase 0 启动时完成 Third-Party Documentation Gate

| 第三方 | 官方文档 | 检查日期 | 版本 | 状态 |
|--------|---------|---------|------|------|
| ultralytics (YOLO26) | 待检查 | - | - | ⏳ |
| mmaction2 | 待检查 | - | - | ⏳ |
| FastAPI | 待检查 | - | - | ⏳ |
| PostgreSQL + pgvector | 待检查 | - | - | ⏳ |
| Vue 3 + Naive UI | 待检查 | - | - | ⏳ |
| TensorRT | 待检查 | - | - | ⏳ |
