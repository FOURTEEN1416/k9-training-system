# ADR 0010: Phase 2 → Phase 3 升级决策

> 状态: ✅ 已确认（用户决策：升级 Phase 3，范围按 v2.0 收敛 + 本 ADR 调整）
> 日期: 2026-07-30
> Owner: 项目宪法 + 阶段计划 Truth
> 修改触发: Phase 3 启动 / 阶段范围变更
> 依据: [Phase 2 验收报告](../../reports/phase-2-validation.md) + [ADR 0007 Phase 2 启动](0007-phase-2-start.md) + [ADR 0008 1.2f 补强方案 v1.6](0008-1.2f-supplement-plan.md) + [ADR 0009 数据策略四路径](0009-data-strategy-four-paths.md) + [phase-2.md v2.6](../stages/phase-2.md)

## 1. 上下文

### 1.1 Phase 2 核心验收结论

Phase 2 核心已于 2026-07-30 验收通过，见 `reports/phase-2-validation.md`：

**已达出口条件**（§6.1-§6.8 全部通过，v2.0 收敛后 8 项）：
- ✅ §6.1 数据飞轮：DB migration + LS API 8 端点 + 微调流水线 + 模型版本管理 + 数据飞轮 API（4 端点全 200，19 videos / 1 active POSE mAP50=0.9239）
- ✅ §6.2 16 类行为：P1 8 类规则引擎扩展 + 18/18 单元测试 + 合成准确率 96.3%（P0 92.9% / P1 100.0%）+ 评分卡 16 行为映射
- ✅ §6.3 USPCA 5 维评分卡：准确度 0.30 + 延迟 0.20 + 保持 0.20 + 搜索效率 0.15 + 注意力 0.15，13 单元测试通过
- ✅ §6.4 端到端：`phase2_6_e2e_test.py` 9/9 全通过（USPCA 29.5s/90s + PDF 4035B）
- ✅ §6.5 延迟：USPCA 0.33x（29.5s/90s，优于 Phase 1 的 0.5x）
- ✅ §6.6 历史评分：`GET /api/scores/by-dog/1` → 200 + []
- ✅ §6.7 文档：deployment.md v0.2.0 + user-guide.md v0.2.0
- ✅ §6.8 1.2f 补强：双轨验证通过（轨道 A dog-pose val mAP50=92.2% + 部署一致性 0.19% + 轨道 B 合成行为 96.3%）

### 1.2 v2.0 范围收敛说明

ADR 0007 §2.3 原 Phase 2 出口条件中，**用户权限多角色验证**已通过 v2.0 收敛决策延后到 Phase 3：

| ADR 0007 §2.3 原出口条件 | 实际状态 | 备注 |
|--------------------------|---------|------|
| 16 类行为准确率 ≥ 85%（真实数据） | ✅ 合成 96.3% + dog-pose val mAP50 92.2% | 真实行为级标注延后 |
| USPCA 5 维评分卡验证 | ✅ 通过 | 13 单元测试 |
| 数据飞轮闭环 | ✅ 通过 | 4 端点 + 微调流水线 |
| **用户权限多角色验证** | ⚠️ **v2.0 延后 Phase 3** | 单犬单基地场景价值低，Phase 3 多犬多基地时更合理 |
| 1.2f 真实视频准确率 ≥ 80% | ✅ 双轨通过 | dog-pose val + 合成行为 |
| 端到端延迟 ≤ 1 min/min | ✅ USPCA 0.33x | 29.5s/90s |

**收敛理由**（来自 phase-2.md v2.0）：
1. Phase 2 单犬单基地场景，多租户价值低；Phase 3 多犬多基地时再做更合理
2. 2.5 训练历史过小，单独成子阶段过小，已合并到 2.6 系统集成
3. 2.0 数据补强降级为后台并行任务（1.2f 不阻塞主线，已双轨验证通过）

### 1.3 1.2f 数据源问题彻底解决

- AK 211 犬类视频因数据域不匹配（YOLO 检测率 0%-19%）放弃做行为级验证
- image.tar.gz 42GB 正式取消下载（同源野生动物图像无价值，用户确认）
- 改用 dog-pose val 1703 张同域数据做姿态级 mAP 验证（mAP50=92.2% + 部署一致性 0.19%）
- 详见 ADR 0008 v1.6 + `reports/phase-2.0c-1_2f-dogpose-val-map.md`

### 1.4 Phase 2 范围外但已完成的重要工作

- **ONNX pose task bug 修复**（`inference.py:150` 显式 `task='pose'`）：ONNX/Engine 模型无 task 元数据，ultralytics 默认回退 `detect` 导致 keypoints=None，检测率 0%。修复后 YouTube 5 视频检测率恢复 38.3%-80.5%
- **keypoint-MoSeq 无监督行为发现**（ADR 0009 Path 1）：226 clips / 43 syllables / 138347 帧，32 mapped + 11 待人工
- **弱监督自训练**（ADR 0009 Path 4）：8880 窗口 + RF 99.8% CV，模型 `data/weak_supervised/weak_supervised_rf.pkl`
- **Label Studio 标注环境**：v1.23 + sandbox 兼容启动脚本 + 5 YouTube 视频 + 20543 帧预标注（63.1% 检测率），待人工标注（后台并行）

## 2. 决策

### 2.1 确认升级 Phase 3

**确认 Phase 3 启动**。Phase 2 核心验收通过 + 1.2f 数据源问题彻底解决，满足升级要求。

### 2.2 Phase 3 范围（按 stage-plan.md §4 + 本 ADR 调整）

按 `dev-docs/stage-plan.md` §4 Phase 3 原规划 + v2.0 收敛延后项：

**核心交付**：
1. **ST-GCN+BC 行为识别**：22 类（P0 8 + P1 8 + P2 6 高级），替代规则引擎作为主算法
2. **多犬追踪 + ID 关联**：ByteTrack / BoT-SORT + ReID，支持多犬同时测评
3. **3D 姿态重建**：多视角融合（动物 3D 姿态现有方案调研）
4. **FCI-IGP 标准映射**：国际工作犬标准评分卡
5. **Jetson 边缘部署**：从 Windows 优先扩展到 Jetson Orin Nano

**v2.0 收敛延后项**：
6. **用户权限 + 多租户**：RBAC + 基地管理员角色（从 Phase 2 延后）
7. **训练历史对比可视化**：历史评分查询 + 对比图表（从 Phase 2 延后）

### 2.3 Phase 3 出口条件

- ST-GCN+BC 22 类行为准确率 ≥ 85%（真实数据）
- 多犬追踪 MOTA ≥ 70%（多犬场景）
- 3D 姿态重建 MPJPE ≤ 50mm（多视角）
- FCI-IGP 评分卡验证通过
- Jetson 边缘部署延迟 ≤ 0.5 min/min 视频
- 用户权限多角色验证通过
- 端到端延迟 ≤ 1 min/min 视频（维持）

### 2.4 Phase 3 时间约束

按 `stage-plan.md` §4：+8-12 周单人全职。子阶段分解见 `stages/phase-3.md`（待创建）。

### 2.5 Phase 3 启动调研前置

按 AGENTS.md §1.1「广泛调研优先」原则，Phase 3 启动前需完成以下调研：
1. **ST-GCN+BC 调研**：动作识别 SOTA + 行为分类算法
2. **多犬追踪调研**：ByteTrack / BoT-SORT / ReID for animals
3. **3D 姿态重建调研**：多视角融合 + 动物 3D 姿态
4. **FCI-IGP 标准调研**：评分规则映射
5. **Jetson 部署调研**：TensorRT for Jetson + 模型优化

**调研执行方式**：按 AGENTS.md §1.3，强制 github-search-strategy + browser-automation 流程，禁止 WebSearch。

## 3. 风险与缓解

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| ST-GCN+BC 训练数据不足 | 60% | 高 | 数据飞轮 + keypoint-MoSeq + 弱监督多路径补充 |
| 多犬追踪 ID switch 高 | 50% | 中 | ReID + 时序平滑 + 单犬场景降级 |
| 3D 姿态重建精度不达标 | 70% | 高 | 多视角融合 + 单目 3D 降级 + Phase 4 RL 优化 |
| FCI-IGP 标准映射不全 | 30% | 中 | 仅映射覆盖的维度，其余延后 |
| Jetson 边缘部署兼容性 | 40% | 中 | TensorRT 优先 + ONNX Runtime 降级 |
| Windows ↔ Jetson 双平台维护成本 | 50% | 中 | Docker 容器化 + 平台抽象层 |

## 4. 影响的 Truth 文档

| 文档 | 变更类型 | 说明 |
|------|---------|------|
| `stages/phase-3.md` | 新增 | Phase 3 阶段计划（本 ADR 同步交付，待调研后细化） |
| `stages/phase-2.md` | 更新 | §7 出口决策引用本 ADR |
| `stages/phase-1.md` | 更新 | §7 出口决策链引用本 ADR（间接，Phase 1 → Phase 2 → Phase 3） |
| `AGENTS.md` §7 | 更新 | Phase 2 状态改为 ✅ 完成，Phase 3 状态改为 ⏳ 启动中 |
| `AGENTS.md` §0 | 更新 | 当前阶段改为 Phase 3 启动中 |
| `stage-plan.md` | 更新 | 项目阶段总览同步 Phase 2 完成 + Phase 3 启动 |

## 5. 不可逆操作清单

Phase 3 涉及以下不可逆操作，需在执行前再次确认：

- ⏳ Phase 3 ST-GCN+BC 模型权重下载/训练
- ⏳ Phase 3 多犬追踪算法集成（影响 inference.py 接口）
- ⏳ Phase 3 3D 姿态重建（可能引入新数据格式）
- ⏳ Phase 3 FCI-IGP 评分卡 YAML（新增评分卡文件）
- ⏳ Phase 3 Jetson 部署环境配置
- ⏳ Phase 3 用户权限 + 多租户 DB migration（RBAC + 基地角色表）

## 6. 未解决问题

- ⏳ Phase 3 详细子阶段分解（待调研后写入 `stages/phase-3.md`）
- ⏳ ST-GCN+BC vs 规则引擎并存策略（迁移期双轨 vs 直接替换）
- ⏳ 3D 姿态重建数据源（多视角视频采集 vs 公开数据集）
- ⏳ Jetson 边缘部署优先级（Phase 3 内必达 vs Phase 4 按需）
- ⏳ 2.0b Label Studio 人工标注（后台并行，Phase 3 期间用户闲暇推进）

## 7. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-30 | 初始版本，确认 Phase 3 升级。Phase 2 核心 ✅ 通过（8/8 验收清单），v2.0 范围收敛说明（用户权限延后 Phase 3），1.2f 数据源问题彻底解决（AK 取消 + dog-pose val 双轨通过）。Phase 3 范围：ST-GCN+BC + 多犬 + 3D + FCI-IGP + Jetson + 用户权限 + 训练历史对比。时间约束 8-12 周 |
