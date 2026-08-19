# Stage Plan — 工作犬训练系统阶段计划

> Truth source: 基于用户立项研讨（2026-07-27）+ 后续阶段验收
> 状态: ✅ 已确认（v1.8，2026-08-19 基座方向确认：ADR 0012 创建 + 数据解阻阶段启动）
> 日期: 2026-07-27（v1.0） / 2026-08-01（v1.1） / 2026-08-02（v1.2/v1.3/v1.4/v1.5/v1.6） / 2026-08-19（v1.7/v1.8）

## 1. 阶段总览

| 阶段 | 状态 | 周期 | 目标 |
|------|------|------|------|
| 立项 | ✅ 完成 | 2026-07-26 → 2026-07-27 | truth 文档 + ADR + 调研 |
| Phase 0 | ✅ 完成 | 2026-07-26 验收 | 环境 + DB schema + 前后端骨架 |
| Phase 1 | ✅ 完成 | 2026-07-28 验收 | MVP 端到端闭环（双场景 + YAML 评分） |
| Phase 2 | ✅ 完成 | 2026-07-30 验收 | 数据飞轮 + 16 行为 + USPCA + 1.2f 补强 |
| Phase 3 | ✅ 完成 | 2026-07-30 启动 | ST-GCN+BC + 多犬 + 3D + FCI-IGP + Jetson + 用户权限 + 训练历史对比（v3.3：3.1b/c/d/e + 3.2b/c/d + 3.3b/c/d + 3.4b/c + 3.5c 部分 + 3.6a/b/c + 3.7a/b/c + 3.8a/b/c/d 完成；3.2d 多犬追踪于 2026-08-17 补充验证：APTv2 数据集下载 25GB + Canidae 犬科子集提取 + YOLO26-pose fine-tune（Box mAP50=96.1%）+ 24 视频 MOTA/IDF1 评估（5 个双达标，最高 MOTA 93.33%/IDF1 96.55%）+ conf=0.35 优化（MOTA≥70% 达 9/24）；Git 检查点 `2e6aef3`，631 单元测试通过 + RBAC 端到端 25/25 + 训练历史对比端到端 41/41 + 端到端集成 6 PASS + 延迟验证 FCI-IGP 0.63x + 项目守卫 7 PASS + 用户手册 Phase 3 v0.3.0） |
| Phase 4 | 🔄 实施中 | 2026-08-02 启动 | LLM 行为解释器（实验室 Agnes API，实用期本地模型，ADR 0011 v1.1）+ Transformer-Mamba 长序列（2026-08-19 用户授权纳入：Mamba 基线 85.61% + mamba_ssm WSL 编译 + tasks.py 集成路由）+ RL 评分优化（按需） |
| **数据解阻（P0）** | 🔄 实施中 | 2026-08-19 启动 | **基座系统 L3 数据飞轮**：AimCLR 自监督预训练 + SMQ 无监督时序分割 + TCL 半监督自训练 + 主动学习 100-200 标注 → ST-GCN+BC ≥ 85%（ADR 0012） |
| 学术副产物 | 🔄 整理中（2026-08-19 用户确认主方向） | 与产品重心一致 | ACM MM / AAAI Application Track / Pattern Recognition：**可解释评分卡 + 端到端工作犬评估系统 + 物理-语义解耦基座** |

## 2. Phase 1 详细计划（3-4 周）

> 详细子阶段见 `stages/phase-1.md`

### Week 1（科目测评闭环）

**目标**：1 段测评视频 → 测评评分 PDF

- Phase 1.0 收尾（✅ best.pt 已完成，补充评估）
- Phase 1.1 TensorRT FP16 加速
- Phase 1.2 科目规则引擎 8 类行为
- Phase 1.4a-d 评分引擎（YAML 配置化）+ PDF 报告 + 前后端联调（科目部分）

### Week 2（幼犬选育原型）

**目标**：1 段选育视频 → 选育评分 PDF

- Phase 1.5 物体检测集成（COCO 球/食物类）
- Phase 1.6 选育信号提取器
- Phase 1.4e-h 选育评分卡 + 前端选育页面 + 联调

### Week 3（系统集成 + 评分卡动态配置）

**目标**：双场景端到端 + YAML 评分卡可动态修改

- Phase 1.7 评分卡管理 API（CRUD + 热加载）
- Phase 1.8 系统集成 + 端到端测试
- Phase 1.9 部署文档 + 用户手册

### Week 4（缓冲 + 学术整理）

**目标**：bug 修复 + 学术副产物初稿

- Bug 修复 + 工作犬精度优化（如有时间）
- 学术副产物方向选定 + 实验整理
- Phase 1 验收报告

## 3. 学术副产物轨道

**不阻塞主交付**。2026-08-19 用户确认主方向（与产品重心一致），进入整理中状态。

### 3.1 候选方向（2026-08-19 决策：主方向已定）

| 方向 | 创新点 | 目标会议 | 可行性 | 状态 |
|------|--------|---------|--------|------|
| **A. 可解释 AI 评分卡 + 端到端工作犬评估系统** | YAML 配置化 + 双场景端到端闭环 + 命中规则可解释 + FCI-IGP 标准映射 | ACM MM Application Track | 高 | ✅ **主方向（用户确认）** |
| B. 幼犬选育数据集 | 公开数据集 + 基准 + 评分规则 | 动物行为学交叉会议 | 中 | 备选 |
| C. 双场景统一框架 | 选育 + 测评统一评分引擎 | AAAI Application Track | 中 | 并入 A |
| D. 跨物种行为识别基准 | InterPet4D + Animal Kingdom | CVPR Workshop | 中 | 备选（消融章节素材） |

### 3.2 决策时机

~~Phase 1 完成后，根据实际产出选择最出彩的方向~~ → **2026-08-19 已决策**：主方向 A（可解释评分卡 + 端到端工作犬评估系统），与产品重心一致，素材最全（YAML 评分引擎 + 双场景闭环 + ST-GCN+BC/Mamba 对比 + APTv2 多犬验证 + FCI-IGP 0.63x 延迟）。

### 3.3 学术产出要求

- 开源代码（GitHub）
- 数据集（如自建）
- 论文（arxiv 预印 + 投稿）
- Demo 视频

## 4. Phase 2+ 远期规划

### Phase 2（+4-6 周）
- 基地合作落地（找 1-2 个警犬基地试点）
- 真实工作犬数据采集 + 标注
- 16 类行为扩展（P1 训练专项 8 种）
- USPCA 标准映射
- 数据飞轮（用户反馈 → 模型迭代）

### Phase 3（+8-12 周）
- ST-GCN+BC 行为识别（22 类）
- 多犬追踪 + ID 关联
- 3D 姿态重建（多视角融合）
- FCI-IGP 标准映射
- Jetson 边缘部署

### Phase 4（按需）
- LLM 行为解释器（自然语言训练反馈）
- Transformer-Mamba 长序列行为分析
- RL 评分优化（个性化）

## 5. 阶段升级决策流程

> 来自 AGENTS.md §5.3

1. 阶段完成 → 验证报告归档 `reports/phase-{N}-validation.md`
2. 用户判断是否升级到下一阶段
3. 升级决策记录到 `dev-docs/decisions/`

**不设硬性精度阈值**，由用户判断是否升级。

## 6. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| 3-4 周做不完 | 40% | 高 | 砍幼犬选育，仅交付科目测评 |
| TensorRT Windows 装不上 | 20% | 中 | 降级 ONNX Runtime GPU |
| 规则引擎准确率 < 80% | 50% | 中 | 触发 Phase 1.3 PoseC3D（条件触发） |
| 公开数据集下载受限 | 20% | 中 | 退化为优酷视频 + 自标 |
| 评分卡 YAML 设计不当 | 30% | 中 | 先做最简版本，基地反馈后迭代 |

## 7. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-27 | 新建，基于用户立项研讨 + 3 份调研报告，3-4 周 Phase 1 + 学术副产物轨道 |
| v1.1 | 2026-08-01 | 同步 Phase 3 实施进度（v2.4：3.1b ST-GCN+BC + 3.2b 多犬追踪 + 3.2c ReID + 3.3b 3D 配对已完成）；补全 Phase 0/1/2 验收日期 |
| v1.2 | 2026-08-02 | **sliver-vibe-coding 接管审计 + Phase 3 v2.8 状态同步**：①Phase 3 行更新为 v2.8（3.1b/c/d/e + 3.2b/c + 3.3b/c/d + 3.4b/c + 3.5c 部分 + 3.6a + 3.6b 部分完成）；②新增 Git 检查点 `2e6aef3` + 538 单元测试新鲜验证证据；③对应 phase-3.md v2.8 + AGENTS.md v1.20 + runtime.md v1.4 |
| v1.3 | 2026-08-02 | **3.6 RBAC 端到端验收通过**：Phase 3 行更新为 v2.9（3.6a/b/c 完成）+ 566 单元测试 + RBAC 端到端 25/25 通过（authentication 6 + role_permission 5 + base_isolation 2 + base_crud 5 + user_management 4 + change_password 3）；对应 phase-3.md v2.9 + AGENTS.md v1.21 + runtime.md v1.5 |
| v1.4 | 2026-08-02 | **3.7 训练历史对比端到端验收通过 + Phase 3 v3.0 状态同步**：①Phase 3 行更新为 v3.0（3.7a/b/c 完成：/api/scores/compare API + CompareView 前端 ECharts 折线+雷达图 + 单元测试 48/48 + 端到端 41/41 通过）；②631 单元测试新鲜验证 + onnxruntime 自然修复 + FastAPI 0.115.6 204 路由兼容性修复（4 个 204 路由 response_class=Response）；③对应 phase-3.md v3.0 + AGENTS.md v1.22 + runtime.md v1.6 |
| v1.5 | 2026-08-02 | **3.8a/b/c 系统集成收尾 + 项目守卫脚本补建**：①项目守卫脚本 `scripts/check_project_guardrails.py`（Phase 0 遗留补建，7 项检查 truth 文档存在性+ADR 完整性+阶段验收报告+AGENTS.md 版本+阶段状态一致性+修订历史完整性+阶段文档头部，bootstrap 模式 7 PASS/0 WARN/0 FAIL）；②3.8a 端到端测试 `scripts/phase3_8_e2e_test.py`（6 PASS + 3 SKIP 硬件/数据阻塞）；③3.8b 延迟验证 `scripts/eval_latency.py`（FCI-IGP 0.63x 通过）；④3.8c 部署文档 `dev-docs/deployment/phase3-features-deploy.md`；⑤对应 phase-3.md v3.1 + AGENTS.md v1.23 + runtime.md v1.6 |
| v1.6 | 2026-08-02 | **3.8d 用户手册更新完成**：①`docs/user-guide.md` 升级至 Phase 3 v0.3.0（13 章节，新增 FCI-IGP 7 维评分 + 22 行为映射 + IGP1/2/3 三阶段 + DQ 硬约束 + 5 级评级 + §6 训练对比章节 + §8.4 RBAC 5 角色 + §8.5 基地管理 + §9.6/9.7/9.8 三新工作流 + 12 新术语 + 7 新 API 端点）；②对应 phase-3.md v3.3 + AGENTS.md v1.24 |
| v1.7 | 2026-08-19 | **漂移修正（用户决策）**：①Phase 4 行更新（LLM 双阶段：实验室 Agnes API + 实用期本地模型，ADR 0011 v1.1；Transformer-Mamba 2026-08-19 用户授权纳入）；②学术副产物行 ⏳ → 🔄 整理中 + §3 轨道更新（主方向 A 已确认：可解释评分卡 + 端到端工作犬评估系统）；③对应 project-brief.md v2.1 + AGENTS.md v1.26 + phase-4.md + phase-1.md §1.8b + HANDOVER.md v1.3 |
| v1.8 | 2026-08-19 | **基座系统方向确认**：新增"数据解阻（P0）"阶段行——AimCLR 自监督预训练 + SMQ 无监督时序分割 + TCL 半监督自训练 + 主动学习 100-200 标注 → ST-GCN+BC ≥ 85%（ADR 0012）；学术副产物目标扩展至 Pattern Recognition；系统开发暂停，转向全面解决数据问题 |
