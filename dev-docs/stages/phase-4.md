# Phase 4 专业阶段计划

**状态**: 🔄 实施中 (2026-08-02)
**Truth source**: `dev-docs/stages/phase-4.md` + `dev-docs/decisions/0011-phase-4-llm-explainer-adr.md`
**当前子任务完成**: 3.1b pyskl+K9Graph / 3.2b BoxMOT多犬追踪 / 3.2c ReID身份关联 / 3.3b InterPet4D配对管线 / 3.4b FCI-IGP评分卡 / 3.5c抽帧策略 + TRT FP16脚本 / 3.6a/b/c RBAC / 3.7训练历史对比 / 3.8a/b/c系统集成 / 3.8d用户手册

## 4.1 LLM行为解释器 (已完成)

### 依据
- `dev-docs/decisions/0011-phase-4-llm-explainer-adr.md`

### 子任务

| 子任务 | 状态 | 说明 |
|--------|------|------|
| 4.1.1 环境搭建 | ✅ 已完成 | Llama 3.2 1B 8bit量化模型下载与 bitsandbytes 配置 |
| 4.1.2 提示工程 | ✅ 已完成 | 行为分析专家系统提示词设计 (约200字中文报告结构) |
| 4.1.3 API 端点 | ✅ 已完成 | FastAPI `/api/llm/explain` POST 端点实现 |
| 4.1.4 单元测试 | ✅ 已完成 | 20/20 测试通过 (输入验证、输出格式、模板降级) |
| 4.1.5 集成验证 | ✅ 已完成 | 端到端测试通过 (真实概率输出 → 中文报告生成) |

### 实现细节
- **模型**: Llama 3.2 1B (bitsandbytes 8bit量化)
- **输入**: 22类行为概率分布 + 7维 FCI-IGP 评分
- **输出**: 中文行为分析报告 (≈200字，5章节结构)
- **后端**: FastAPI + 本地推理 (CPU ≈ 2s/次)
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

## 4.2 Transformer-Mamba长序列分析 (进行中)

### 依据
- 待用户决策是否启动

### 目标
- 探索 Transformer/Mamba 架构在长序列行为分析上的效用
- 与现有 ST-GCN+BC 基线对比

### 子任务
- 4.2.1 数据准备: YouTube自标长视频序列
- 4.2.2 模型实现: 基于 PyTorch 的Transformer/Mamba实现
- 4.2.3 基线对比: 与 ST-GCN+BC 的精度/延迟权衡
- 4.2.4 实验报告: 记录在 `reports/phase-4-llm-experiments.md`

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

### 遗留 deferred 项
- [⏸️] 3.2d 多犬 MOTA 测试: 缺少多犬视频+GT标注
- [⏸️] Jetson 硬件部署: 已在 RTX 5060 上验证软件优化足够
- [⏸️] Transformer-Mamba 实验: 根据用户决策启动或推迟

### 下一步
1. 继续 Transformer-Mamba 长序列实验 (如有需要)
2. 收集训导员对 LLM报告质量的反馈
3. 根据反馈迭代优化提示词和报告结构
