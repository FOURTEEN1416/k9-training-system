# ADR 0003: Phase 0 → Phase 1 升级决策

> 状态: ✅ 已确认
> 日期: 2026-07-26
> Owner: 项目宪法 + 阶段计划 Truth
> 修改触发: Phase 1 启动 / 阶段范围变更
> 依据: [RESEARCH_PHASE1_STACK_DEEP.md](../research/RESEARCH_PHASE1_STACK_DEEP.md) + [phase-0 验收报告](../../reports/phase-0-validation.md)

## 1. 上下文

Phase 0 基础设施已于 2026-07-26 验收通过（见 `reports/phase-0-validation.md`），满足升级前置条件：
- Python 3.12 + PyT2.11+cu128 + Blackwell sm_120 验证通过
- PostgreSQL 17 + pgvector 0.8.0 + Redis 8.6.3 运行正常
- backend/ 骨架（FastAPI + Celery + SQLAlchemy + Alembic）齐备
- frontend/ 骨架（Vue3 + Vite + TS + Naive UI + Pinia + ECharts）齐备
- 9 张核心表 schema 已应用

Phase 1（MVP）有四项未验证技术栈：
1. mmcv 2.1.0 + PyTorch 2.11 + Blackwell 源码编译
2. PoseC3D 用于狗 24 关键点行为识别
3. TensorRT 10.8 + CUDA 12.8 + Windows
4. YOLO26-pose + Dog-Pose 24 关键点

针对上述四项已完成深度调研（见 `RESEARCH_PHASE1_STACK_DEEP.md`），本 ADR 记录基于调研结论的升级决策。

## 2. 决策

### 2.1 升级到 Phase 1（MVP）

**确认升级**。Phase 1 目标为端到端闭环：视频上传 → YOLO26-pose 关键点 → 行为识别 → 评分 → PDF 报告。

### 2.2 实施路径：分阶段降风险 + 规则引擎兜底

采用调研报告 §5.1 推荐的分阶段路径，**不要求所有技术栈同时验证通过**。

> **不设周期承诺**：子阶段按顺序推进，前序子阶段验收通过后才启动下一阶段。周期估算会随 GPU 实际性能、数据采集进度、编译结果变化，硬性周期承诺反而诱导赶工。各子阶段实际耗时在阶段验收报告中归档。

| 子阶段 | 顺序 | 任务 | 阻塞性 |
|--------|------|------|--------|
| Phase 1.0 | 第 1 | YOLO26-pose + Dog-Pose 微调 + 推理验证 + 工作犬精度评估 | 必达 |
| Phase 1.1 | 第 2 | TensorRT 10.8 FP16 加速 | 必达（失败则降级 ONNX Runtime GPU） |
| Phase 1.2 | 第 3 | 规则引擎（P0 8 类行为）+ 准确率评估 | 必达（MVP 兜底） |
| Phase 1.3 | 第 4 | mmaction2 + PoseC3D 尝试（**条件触发**，见 §2.4） | 条件性 |
| Phase 1.4 | 第 5 | 评分（3 维 + 公式 + 阈值表）+ 报告 + 前后端集成 | 必达 |
| Phase 1.5 | 第 6 | 端到端验收 + Phase 2 升级决策 | 必达 |

### 2.3 PoseC3D 定位：条件触发（保留可选 + 明确触发条件）

**触发条件**：Phase 1.2 规则引擎在自标工作犬测试集上 P0 8 类行为准确率 < 80% 时，启动 Phase 1.3 mmaction2 + PoseC3D。

**跳过条件**：规则引擎准确率 ≥ 80% 时跳过 Phase 1.3，Phase 1 仅用规则引擎交付 MVP，Phase 2 重新评估 ST-GCN+BC 直上路径（绕过 PoseC3D）。

**Phase 1.3 失败处置**（若触发后失败）：
- mmcv 编译失败或 PoseC3D 训练不收敛
- 失败原因记录到 `decisions/0004-*.md`
- Phase 1 仍用规则引擎交付 MVP
- Phase 2 直接评估 ST-GCN+BC，不再回头尝试 PoseC3D

### 2.4 工作犬数据采集：评估后决定（条件触发）

**Phase 1.0 启动时不采集**：先用 Dog-Pose 通用数据集微调 YOLO26-pose，跑通端到端闭环 + 在工作犬测试视频上评估精度。

**触发条件**：Phase 1.0 评估结果显示工作犬场景关键点 mAP50-95 < 60%（对应调研报告 §4.6 中"工作犬场景精度低"50% 概率风险），启动 300 张工作犬视频帧采集（10-20 小时人工标注，Roboflow YOLO Pose 格式）。

**跳过条件**：mAP50-95 ≥ 60% 时跳过采集，Phase 1.0-1.2 全程使用 Dog-Pose 微调模型。Phase 2 数据飞轮阶段再补充工作犬数据。

**采集工具**：Roboflow（云端协作 + YOLO Pose 自动导出），见 [RESEARCH_YOLO26_DEPLOY_DEEP.md §2.3](../research/RESEARCH_YOLO26_DEPLOY_DEEP.md)

### 2.5 评分范围：GA-T 3 维 + 细化逻辑（不扩维）

**维度保持 3 维**：动作准确度 / 响应延迟 / 保持时长（与 function-list.md F4 Phase 1 边界一致）。

**细化要求**：
- 每维必须给出**计算公式**（基于行为识别结果的可量化指标）
- 每维必须给出**阈值表**（0-100 分分段，对应 GA-T 标准）
- 必须给出**加权逻辑**（3 维如何合成总分）
- 评分逻辑文档化到 `dev-docs/scoring-design.md`（Phase 1.4 启动时创建）

**不扩维理由**：function-list.md 明确 Phase 1=3 维 / Phase 2=5 维 / Phase 3=7 维，扩维会破坏阶段边界。Phase 1 重点是把 3 维做深做透，而非扩维。

## 3. 风险与缓解

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| mmcv 编译失败 | 30% | 中 | 规则引擎兜底，PoseC3D 推迟 |
| PoseC3D 训练不收敛 | 50% | 中 | 规则引擎兜底，Phase 2 积累数据 |
| YOLO26-pose 工作犬精度 < 60% | 50% | 高 | Phase 1 末采集 300 张微调 |
| TensorRT engine 导出失败 | 10% | 低 | 退化为 ONNX Runtime GPU |
| 规则引擎精度 < 70% | 30% | 高 | 细化 P0 8 类关键点规则 |
| 端到端延迟 > 1min/min 视频 | 20% | 中 | TensorRT + 关键帧采样（每 3 帧采 1） |

## 4. 影响的 Truth 文档

| 文档 | 变更类型 | 说明 |
|------|---------|------|
| `stages/phase-1.md` | 新增 | Phase 1 阶段计划（子阶段 / 任务分解 / 验收清单） |
| `stages/phase-0.md` | 已完成 | Phase 0 状态已更新为 ✅ 完成 |
| `stage-plan.md` | 更新 | 项目阶段总览同步 Phase 1 启动 |
| `backend/requirements.txt` | 追加 | Phase 1 依赖（按子阶段分批追加） |
| `AGENTS.md` §7 | 已更新 | 阶段映射 Phase 1 状态 |

## 5. 出口条件（Phase 1 验收标准）

详见 `stages/phase-1.md` §6。核心指标：
- ✅ YOLO26-pose mAP50-95 ≥ 70%（Dog-Pose 验证集）
- ✅ 规则引擎 P0 8 类行为识别准确率 ≥ 80%（自标测试集，否则触发 Phase 1.3）
- ✅ 评分引擎 3 维均有公式 + 阈值表 + 加权逻辑
- ✅ 端到端延迟 ≤ 1 min / min 视频
- ✅ 1 段工作犬视频 → 评分 PDF 全流程跑通
- ✅ `reports/phase-1-validation.md` 验收报告归档

## 6. 不可逆操作清单

Phase 1 涉及以下不可逆操作，需在执行前再次确认：
- ⏳ 下载 Dog-Pose 数据集（337 MB，本地存储）
- ⏳ YOLO26-pose 微调训练（GPU 资源消耗 4-8 小时）
- ⏳ TensorRT 10.8 安装（系统 PATH 修改）
- ⏳ VS BuildTools 2022 安装（仅 Phase 1.3，约 5GB）
- ⏳ mmcv 源码编译（10-30 分钟 GPU+CPU）
- ⏳ Phase 1 末工作犬视频采集（300 张标注人工）

## 7. 未解决问题（沿用 AGENTS.md §9）

- ⏳ YOLO26 AGPL 商用许可（Phase 5 前解决）
- ⏳ 工作犬数据采集方案（Phase 1.3 启动前细化）
- ⏳ mmaction2 Windows 兼容性（Phase 1.3 实测验证）
- ⏳ 中文路径迁移时机（见 ADR 0001）

## 8. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-26 | 初始版本，基于 RESEARCH_PHASE1_STACK_DEEP.md 调研结论确认 Phase 1 升级 |
| v1.1 | 2026-07-26 | 项目体检后修订：删除周期承诺；PoseC3D 改为条件触发（< 80% 准确率）；数据采集改为评估后决定（< 60% mAP）；评分范围明确 3 维 + 公式 + 阈值表 + 加权逻辑 |
