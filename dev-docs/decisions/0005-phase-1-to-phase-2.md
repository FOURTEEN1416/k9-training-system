# ADR 0005: Phase 1 → Phase 2 升级决策

> 状态: ✅ 已确认（v1.2 修订: 用户决策完成，Case A + Case B 并行推进，Phase 2 已启动）
> 日期: 2026-07-28
> Owner: 项目宪法 + 阶段计划 Truth
> 修改触发: Phase 2 启动 / 阶段范围变更
> 依据: [Phase 1 验收报告](../../reports/phase-1-validation.md) + [ADR 0006 DogMo 替代方案](0006-dogmo-open-alternative.md) + [1.6d 验证报告](../../reports/phase-2-prereq-1.6d-validation.md) + [1.2f 验证报告](../../reports/phase-2-prereq-1.2f-validation.md) + [ADR 0007 Phase 2 启动](0007-phase-2-start.md) + [ADR 0008 1.2f 补强方案](0008-1.2f-supplement-plan.md)

## 1. 上下文

### 1.1 Phase 1 MVP 验收结论

Phase 1 MVP 已于 2026-07-28 验收通过（**带条件**），见 `reports/phase-1-validation.md`：

**已达出口条件**（§6.1-§6.7 全部通过）：
- ✅ §6.1 姿态检测：YOLO26-pose + Dog-Pose 24 关键点 best.pt 微调完成（Pose mAP50=0.9239）
- ✅ §6.2 TensorRT 加速：ONNX Runtime GPU 13.39 ms/frame（74.7 FPS）作为生产后端
- ✅ §6.3 规则引擎：P0 8 类行为合成数据准确率 92.9%（≥ 80% 阈值，跳过 Phase 1.3 PoseC3D）
- ✅ §6.4 PoseC3D：未触发（条件 ≥ 80% 满足）
- ✅ §6.5 评分引擎 + 报告 + 前后端：双场景 YAML 评分卡 + 热加载 + PDF + 9/9 集成测试通过
- ✅ §6.6 物体检测 + 选育信号：YOLO26 COCO + 3 维 9 信号提取
- ✅ §6.7 系统集成 + 端到端：7/7 端到端测试通过，延迟 obedience 0.36x / puppy 0.78x

### 1.2 带条件项（Phase 2 启动前置）

Phase 1 验收"带条件"指以下两项作为 **Phase 2 启动前置条件**（不阻塞 Phase 1 验收）：

1. **1.6d 真实序列验证**：DogMo 数据集需购买，已决策采用 InterPet4D（主）+ YouTube 玩球视频（补充）替代，见 ADR 0006
2. **1.2f 真实数据复核**：合成数据 92.9% 已达标，待 InterPet4D sit/down/stand/come 真实评估复核 Phase 1.3 跳过决策

### 1.3 学术副产物方向

Phase 1.8b 已选定学术方向：**跨物种行为识别基准**（基于 InterPet4D + Animal Kingdom），见 `reports/phase-1-validation.md` §11。

## 2. 决策

### 2.1 Phase 1 MVP 验收通过（带条件）

**确认 Phase 1 MVP 验收通过**。所有 §6.1-§6.7 出口条件已达成，§6.8 验收报告 + 学术方向 + 升级决策三项已完成。

### 2.2 Phase 2 启动条件（双前置）— v1.1 修订: 已达成

**不立即启动 Phase 2**。Phase 2 启动需同时满足：

| 前置条件 | 验证方法 | 完成标志 | 状态 |
|---------|---------|---------|------|
| 1.6d 真实序列验证 | InterPet4D kp_world → `puppy_signals.py` 9 信号合理性 + YouTube 玩球视频物体检测 | 信号合理性报告 + 物体检测 mAP ≥ 0.5 | ✅ 通过（2026-07-28，226/226 clips + 9/9 姿态指标变异；YouTube 物体检测延后 Phase 2 内推进） |
| 1.2f 真实数据复核 | InterPet4D 视频帧 → YOLO26-pose → 规则引擎 → sit/down/stand/come 准确率 | 准确率 ≥ 80% 确认跳过 PoseC3D，< 80% 触发 Phase 1.3 复核 | ✅ 条件通过（数据限制）（2026-07-28，InterPet4D v1 无视频/标签，采用三层降级验证：合成基线 92.9% + kp_world 管线 100% + 真实视频延后） |

**验证报告**：
- 1.6d: `reports/phase-2-prereq-1.6d-validation.md`
- 1.2f: `reports/phase-2-prereq-1.2f-validation.md`（v1.1）

**执行脚本**（Phase 1.8 同步交付）：
- `scripts/download_interpet4d.py`：InterPet4D HuggingFace 下载（10.7 GB）
- `scripts/validate_phase2_prereq.py`：1.6d + 1.2f 验证统一脚本（`--task 1.6d` / `--task 1.2f`）

**1.2f 条件通过说明**：
- InterPet4D v1 实际无视频文件 + 无行为标签（与 ADR 0006 v1.0 假设不符，见 ADR 0006 v1.1）
- 原设计（视频→YOLO26-pose→准确率）无法执行
- 采用三层降级验证：合成基线 92.9%（主要证据）+ kp_world 管线 100% 运行（辅助证据）+ 真实视频延后
- PoseC3D 跳过决策维持（合成 92.9% ≥ 80% 阈值）
- 真实视频准确率验证作为 Phase 2 内推进项，不阻塞 Phase 2 启动

### 2.3 Phase 2 范围预览（待 1.6d/1.2f 完成后细化）

按 `dev-docs/stage-plan.md` + `dev-docs/function-list.md` Phase 2 边界：

- **数据飞轮**：用户上传 → 标注 → 微调闭环
- **16 类 P1 行为**：在 P0 8 类基础上扩展（取物/前进/后退/转向/跳跃/搜索/咬合/释放）
- **USPCA 标准**：评分卡扩到 5 维（准确度/延迟/保持/搜索效率/注意力）
- **用户权限**：多训导员 + 基地管理员角色
- **1.6d/1.2f 真实验证**：作为 Phase 2 首批任务并行推进

### 2.4 升级决策路径

```
当前状态: Phase 1 ✅ 验收通过 + 1.6d/1.2f ✅ 前置条件达成
    ↓
[已执行] 1.6d 真实序列验证 → ✅ 通过（226/226 + 9/9 姿态指标变异）
[已执行] 1.2f 真实数据复核 → ✅ 条件通过（数据限制: 合成 92.9% + kp_world 管线 100%）
    ↓
[已执行] 用户决策（2026-07-28）: Case A + Case B 并行
    ↓
✅ Case A: 确认升级 Phase 2（ADR 0007）→ 真实视频补强作为 Phase 2 首批任务
✅ Case B: 1.2f 补强方案（ADR 0008）→ Animal Kingdom 申请 + YouTube 自标 + Label Studio
    ↓
[当前] Phase 2 已启动（详见 stages/phase-2.md）
```

**用户决策详情**（2026-07-28）：

用户确认 **Case A + Case B 并行推进**：
- **Case A**（升级决策）: 确认升级 Phase 2，数据飞轮 + 16 行为 + USPCA 作为核心目标，真实视频补强作为 Phase 2 首批任务（不阻塞启动）
- **Case B**（1.2f 补强）: 调研含视频+标签数据集（Animal Kingdom）+ YouTube 自标方案，作为 Phase 2.0a-b 子阶段并行推进

详见 [ADR 0007](0007-phase-2-start.md) + [ADR 0008](0008-1.2f-supplement-plan.md)。

## 3. 风险与缓解

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| InterPet4D 关键点分布与 YOLO26-pose 推理结果不一致 | 30% | 中 | 同时跑 YOLO26-pose 推理 + SMAL 拟合结果对比 |
| 1.2f 真实准确率 < 80% | 40% | 高 | 触发 Phase 1.3 PoseC3D 复核（mmaction2 安装 + 训练） |
| 1.6d 9 信号在 InterPet4D 上合理性差 | 20% | 中 | 修订信号阈值表 + 评分卡 YAML |
| YouTube 玩球视频物体检测精度不足 | 20% | 低 | COCO 预训练 + 阈值调优 |
| Phase 2 范围扩张失控 | 30% | 中 | 严格按 function-list.md Phase 2 边界 |

## 4. 影响的 Truth 文档

| 文档 | 变更类型 | 说明 |
|------|---------|------|
| `stages/phase-1.md` | 更新 | §6.8 标记完成 + §7 出口决策引用本 ADR + 修订历史 v2.5 |
| `stages/phase-2.md` | 新增（待创建） | Phase 2 阶段计划，1.6d/1.2f 完成后启动 |
| `AGENTS.md` §7 | 更新 | Phase 1 状态改为 ✅ 完成（带条件），Phase 2 状态改为 ⏳ 启动条件待满足 |
| `stage-plan.md` | 更新 | 项目阶段总览同步 Phase 1 完成 |

## 5. 不可逆操作清单

Phase 1 → Phase 2 升级涉及以下不可逆操作，需在执行前再次确认：

- ⏳ InterPet4D 数据集下载（10.7 GB，本地存储）
- ⏳ YouTube 玩球视频下载（3-5 段，每段 30-60 秒）
- ⏳ Phase 2 数据库 migration（用户表 / 角色表 / 标注表）
- ⏳ Phase 2 评分卡 YAML 扩展（5 维 + 16 行为）

## 6. 未解决问题

- ✅ 1.6d 真实序列验证（InterPet4D kp_world）— 通过（2026-07-28）
- ✅ 1.2f 真实数据复核（InterPet4D）— 条件通过（数据限制）（2026-07-28）
- ✅ 用户决策是否升级 Phase 2 — 已确认 Case A + Case B 并行（2026-07-28，见 ADR 0007 + ADR 0008）
- ⏳ YouTube 玩球视频物体检测验证（1.6d 补充，Phase 2.0c 内推进）
- ⏳ 1.2f 真实视频补强（Animal Kingdom 申请 + YouTube 自标，Phase 2.0a-b 内推进，见 ADR 0008）
- ⏳ Phase 1.3 PoseC3C 跳过决策维持（合成 92.9% + kp_world 管线 100% 支撑，待 1.2f 补强后复核）
- ✅ Phase 2 详细阶段计划（已交付 `stages/phase-2.md`，Phase 2 已启动）

## 7. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-28 | 初始版本，Phase 1 验收通过（带条件），Phase 2 启动条件明确为 1.6d/1.2f 真实验证 |
| v1.1 | 2026-07-28 | 1.6d/1.2f 验证完成: 1.6d ✅ 通过（226/226 + 9/9 姿态指标变异），1.2f ✅ 条件通过（数据限制: InterPet4D v1 无视频/标签，采用三层降级验证）。§2.2 状态更新，§2.4 升级路径更新，§6 未解决问题更新。Phase 2 启动条件达成，待用户决策 |
| v1.2 | 2026-07-28 | 用户决策完成: Case A + Case B 并行推进。§2.4 升级路径更新（用户决策已执行），§6 未解决问题更新（用户决策 ✅ 解决 + Phase 2 计划 ✅ 交付）。Phase 2 已启动（ADR 0007 + ADR 0008 + stages/phase-2.md 同步交付）。本 ADR 关闭 |
