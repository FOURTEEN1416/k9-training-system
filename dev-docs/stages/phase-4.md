# Phase 4 专业阶段计划

**状态**: 🔄 并行实施中 (2026-08-02 启动，2026-08-19 用户授权 Transformer-Mamba 纳入 + LLM 双阶段决策；P0 数据解阻为当前主线)
**Truth source**: `dev-docs/stages/phase-4.md` + `dev-docs/decisions/0011-phase-4-llm-explainer-adr.md`（v1.1）+ `dev-docs/stages/phase-4-transformer-mamba.md`
**当前子任务完成**: 4.1 LLM 行为解释器（Agnes API 实验室阶段）/ 4.2 Transformer-Mamba 基线 + mamba_ssm WSL 编译 + tasks.py 集成路由（用户 2026-08-19 认可纳入）

## 4.1 LLM行为解释器 (已完成)

### 依据
- `dev-docs/decisions/0011-phase-4-llm-explainer-adr.md`（v1.1：实验室 Agnes API + 实用期本地模型）

### 子任务

| 子任务 | 状态 | 说明 |
|--------|------|------|
| 4.1.1 环境搭建 | ✅ 已完成 | 实验室阶段：Agnes 2.5 Flash API 配置（LLM_API_KEY/LLM_API_BASE/LLM_MODEL 环境变量）；本地 Llama 3.2 1B 8bit 量化路径已验证可行（bitsandbytes），留待实用阶段切换 |
| 4.1.2 提示工程 | ✅ 已完成 | 行为分析专家系统提示词设计 (约200字中文报告结构) |
| 4.1.3 API 端点 | ✅ 已完成 | FastAPI `/api/llm/explain` POST 端点实现 |
| 4.1.4 单元测试 | ✅ 已完成 | 20/20 测试通过 (输入验证、输出格式、模板降级) |
| 4.1.5 集成验证 | ✅ 已完成 | 端到端测试通过 (真实概率输出 → 中文报告生成) |

### 实现细节
- **模型**: 实验室阶段 Agnes 2.5 Flash（外部 API）；实用阶段切换本地 Llama 3.2 1B（bitsandbytes 8bit 量化）
- **输入**: 22类行为概率分布 + 7维 FCI-IGP 评分
- **输出**: 中文行为分析报告 (≈200字，5章节结构)
- **后端**: FastAPI + 环境变量解耦的 LLM 客户端（切换模型仅改配置）
- **降级**: 当模型不可用时提供结构化模板报告

### API规范
- **端点**: `POST /api/llm/explain`
- **请求**:
  - `video_id`: 视频标识符
  - `duration_sec`: 视频时长(秒)
  - `probabilities`: 22元素列表 (0-1 概率)
  - `scores`: 7元素列表 (0-100 FCI-IGP评分)
  - `behaviors`: 可选行为名称列表
- **响应**:
  - `report`: 中文行为分析报告
  - `generation_time`: 生成耗时 (秒)
  - `model_version`: 模型版本字符串
  - `confidence`: Top概率值

## 4.2 Transformer-Mamba长序列分析 (✅ 已授权纳入，2026-08-19 用户认可)

### 依据
- 用户决策（2026-08-19）：认可已完成的 Mamba 基线 + 编译 + 集成工作纳入 Phase 4
- 详细方案与实验记录：`phase-4-transformer-mamba.md`

### 目标
- 探索 Transformer/Mamba 架构在长序列行为分析上的效用
- 与现有 ST-GCN+BC 基线对比

### 已完成（2026-08-17~19）
- ✅ 4.2.1 数据准备: 合成数据基线训练（22 类 × N 样本）
- ✅ 4.2.2 模型实现: Mamba 基线（合成数据 85.61% vs ST-GCN 46.97%，21K 参数）+ VideoMambaSkeleton 纯 PyTorch 回退实现 + mamba_ssm WSL 真实编译（RTX 5060 sm_120）
- ✅ 4.2.3 基线对比: tasks.py 集成路由（`behavior_deploy_mode` 10 种模式：mamba_only / mamba_shadow / mamba_vote / mamba_bc_* 等），605 测试零回归
- ⏳ 4.2.4 实验报告: 记录在 `reports/phase-4-llm-experiments.md`

### 待推进
- MS-Temba 多尺度膨胀 SSM 集成（阶段 3 改进实验）
- 动作边界检测损失
- 消融实验: Mamba-only vs Transformer-only vs Hybrid
- 真实数据训练（需标注数据集: 通过 AimCLR+SMQ+TCL+主动学习管线，见 ADR 0012）

## 4.3 RL评分优化 (待启动)

### 依据
- 由用户决定是否基于训导员反馈启动

### 目标
- 使用强化学习优化行为评分
- 基于训导员的标记反馈迭代改进评分卡

### 子任务
- 4.3.1 奖励函数设计: 培训师标记的质量评分
- 4.3.2 PPO/SFT 训练循环
- 4.3.3 评分卡改进: FCI-IGP 权重自适应调整

## 4.4 阶段过渡

### 完成标准
- [x] LLM行为解释器模块实现并验证
- [x] Phase 3 所有子任务验收通过
- [x] Stage plan 与 truth 文档同步
- [x] API 端点已注册并可调用
- [x] Transformer-Mamba 已授权纳入（2026-08-19 用户认可）

### 遗留 deferred 项
- [⏸️] LLM 实用阶段本地模型切换: 实验室 Agnes API → 本地 Llama 3.2 1B（数据隐私硬约束，列入 HANDOVER P1）
- [⏸️] Jetson 硬件部署: 已在 RTX 5060 上验证软件优化足够
- [⏸️] 真实数据训练: ST-GCN+BC / Mamba 均需主动学习标注数据升级（Label Studio 仅保留为通用人工复核工具；YouTube 数据源已废弃，见 ADR 0008 v1.7）

### 下一步
1. 继续 Transformer-Mamba 改进实验（MS-Temba + 消融，按需）
2. 收集训导员对 LLM报告质量的反馈
3. 根据反馈迭代优化提示词和报告结构
4. 学术副产物整理（可解释评分卡 + 端到端工作犬评估系统方向，素材：合成对比 + 端到端闭环 + FCI-IGP）

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-02 | Phase 4 计划创建（LLM 解释器 + Transformer-Mamba + RL 按需），LLM 解释器子任务完成 |
| v1.1 | 2026-08-19 | **漂移修正（用户决策）**：①头部"当前子任务完成"修正——移除 Phase 3 任务清单误复制，改为 Phase 4 自身子任务；②4.1 LLM 解释器改为双阶段描述（实验室 Agnes API + 实用期本地 Llama 3.2 1B，ADR 0011 v1.1）；③4.2 Transformer-Mamba ⏳→✅ 已授权纳入（2026-08-19 用户认可：Mamba 基线 85.61% + WSL 编译 + tasks.py 集成路由 10 模式 + 605 测试零回归）；④遗留项更新（3.2d 移除——已补充验证完成；新增 LLM 本地化 P1 跟踪） |
