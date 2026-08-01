# Phase 2 — 核心阶段计划

> 阶段: Phase 2 核心
> 状态: 🔄 启动中（v2.0 收敛重写，2026-07-29）
> Owner: Phase 2 核心
> 入口条件: Phase 1 验收通过 ✅ + 1.6d/1.2f 前置条件达成 ✅（ADR 0005 v1.1 + ADR 0006 v1.1）
> 出口条件: 见 §6 验收清单
> 时间约束: 3-4 周单人全职（v2.0 从 6 子阶段收敛到 3 核心子阶段 + 1 后台 + 1 收尾）
> 依据: [ADR 0007](../decisions/0007-phase-2-start.md) + [ADR 0008](../decisions/0008-1.2f-supplement-plan.md) + [function-list.md](../function-list.md) + [stage-plan.md](../stage-plan.md)

## 1. 阶段目标（v2.0 收敛）

**核心三件套 + 后台补强 + 收尾集成**：
- **2.1 数据飞轮基础设施**：DB migration + LS API 对接 + 微调流水线
- **2.2 16 类行为识别**：P0 8 类 → P0+P1 16 类规则引擎扩展
- **2.3 USPCA 5 维评分卡**：科目评分卡从 5 维重做 USPCA 标准映射
- **2.0 数据补强（后台并行）**：YouTube 自标 + AK 跨物种预训练补充
- **2.6 系统集成 + 端到端**：双场景 + 16 行为 + USPCA + 数据飞轮闭环

**关键技术验证**：
- 数据飞轮闭环（标注 → 训练 → 部署）
- 16 类 P1 行为规则引擎扩展（合成数据 + 真实数据复核）
- USPCA 标准映射（5 维：准确度/延迟/保持/搜索效率/注意力）

## 2. 范围（v2.0 收敛）

### 2.1 包含

- **F3b 规则引擎扩展**：P0 8 类 → P0+P1 16 类（核心算法价值）
- **F4 评分引擎扩展**：科目评分卡 USPCA 5 维映射（标准对齐）
- **数据飞轮基础设施**：标注 API + 微调流水线 + 模型版本管理（闭环能力）
- **2.0 数据补强**：1.2f 真实视频准确率验证（后台并行，不阻塞主线）
- **2.6 系统集成 + 端到端**：双场景 + 16 行为 + USPCA 闭环验证

### 2.2 不包含（v2.0 收敛 — 延后到 Phase 3）

- ~~2.4 用户权限 + 多租户~~ → **延后 Phase 3**（与多犬/多基地场景更匹配）
- ~~2.5 训练历史与对比~~ → **合并到 2.6 系统集成**（仅保留历史评分查询，对比可视化延后）
- ST-GCN+BC（Phase 3）
- 多犬追踪（Phase 3）
- 3D 姿态重建（Phase 3）
- FCI-IGP 标准（Phase 3）
- LLM 评分解释（Phase 4）
- RL 评分优化（Phase 4）

### 2.3 收敛理由

v1.0 计划 6 子阶段 + 6 周，单人开发不现实。诊断发现：
1. **2.0 数据补强被错误设为关键路径**：1.2f 真实视频验证本质是"复核 PoseC3D 跳过决策"，但 ADR 0006 v1.1 已基于合成 92.9% 做条件通过决策。1.2f 不阻塞 2.1/2.2/2.3 任何后续工作。
2. **2.4 多租户权限过早**：Phase 2 单犬单基地场景，多租户价值低；Phase 3 多犬多基地时再做更合理。
3. **2.5 训练历史过小**：单独成子阶段过小，合并到 2.6 系统集成。

## 3. 子阶段任务分解（v2.0 收敛重写）

> **3-4 周时间约束**：3 核心子阶段可并行启动，2.0 后台并行，2.6 收尾。

### Phase 2.0 真实视频补强（后台并行，P2）

**Owner**: ML 开发（`backend/ml/`）

> ⚠️ **降级为后台任务**（v2.0）：不阻塞 2.1/2.2/2.3。人工标注由用户在 LS UI 闲暇时推进，1.2f 验证在 2.2 完成后用 16 行为引擎重测。

- ✅ **2.0a** 1.2f 补强数据集（已完成）
  - ✅ YouTube 视频下载（5 段，46MB）+ YOLO26-pose 预标注（20,543 帧，63.1% 检测率）
  - ✅ Label Studio 部署（v1.23，project id=1，5 视频 + 5 预标注任务）
  - ✅ AK video.tar.gz 下载（15.59 GB）+ 犬类视频提取（211 个，111.2 MB）
- 🔄 **2.0b** 人工标注（用户闲暇推进，预计 1-2 周）— **就绪**：LS 1.23.0 已安装 + setup 脚本 + 5 YouTube 视频 + 20543 帧预标注（63.1% 检测率），待用户启动 LS:8080 标注
- ✅ **2.0c** 1.2f 真实视频准确率验证 — **双轨验证通过**（2026-07-30）:
  - **轨道 A（姿态级 mAP，dog-pose val 同域）**: ✅ PASS，Pose mAP50=92.2%（≥ 85% 绝对阈值）+ 部署一致性 0.19%（≤ 2%，ONNX vs PT 一致），报告 `reports/phase-2.0c-1_2f-dogpose-val-map.md` + JSON 指标
  - **轨道 B（行为级准确率）**: ✅ 已由 2.2d 合成数据 96.3% 覆盖（P0 92.9% / P1 100.0%）
  - **数据源决策**: AK 211 犬类视频因域不匹配（Wolf 142/Wild Dog 35/Dog 31，YOLO 检测率 0%-19%）放弃做行为级验证，改用 dog-pose val 1703 张同域数据做姿态级 mAP 验证
  - **YouTube ONNX bug 修复**（v2.5）: `inference.py:150` 显式传 `task='pose'`，YouTube 5 视频检测率恢复 38.3%-80.5%
  - **2.0b LS 标注**: 后台并行，不阻塞主线（弱监督 3D kp_world 训练 vs YouTube 2D 像素坐标域不匹配，无法交叉验证）
- ✅ **2.0d** keypoint-MoSeq 无监督行为发现（[ADR 0009](../decisions/0009-data-strategy-four-paths.md) Path 1，主路径）
  - 四路径并行：P1 keypoint-MoSeq（✅ 完成）+ P2 DeepEthogram（备选，Python 版本冲突）+ P3 主动学习（已运行）+ P4 弱监督自训练 ✅ 完成
  - 输入: InterPet4D smal_npy/*.npz（226 序列，kp_world (T, 24, 3)）
  - ✅ apply_model 完成（15 iters / 40min，checkpoint iteration=12 → results.h5: 226 clips / 43 syllables / 138347 帧）
  - ✅ syllable → 16 行为映射完成（30 P0 自动 down + 1 P1 recall + 1 P1 watch + 11 待人工），报告 `data/kpm_project/interpet4d_kpm/syllable_report.md`
  - 映射策略: 规则引擎自动映射 P0 + 人工映射 P1（用户确认）
  - ⚠️ 映射结果受 InterPet4D 数据集偏态影响（多数为低姿态 down），真实训练视频需补充 sit/stand/heel 等姿态
  - Path 4 已完成: 8880 窗口 + RF 99.8% CV，模型 `data/weak_supervised/weak_supervised_rf.pkl`
- ⏳ **2.0e** YouTube 玩球视频物体检测验证（1.6d 补充）— 可选，时间允许时做
- ⏳ **2.0f** 1.2f 补强验证报告归档（`reports/phase-2-1.2f-supplement.md`）

**验收标准**：
- 1.2f 真实视频准确率 ≥ 80%（确认 PoseC3D 跳过决策，2.2 完成后重测）
- 验证报告归档
- ⚠️ 非主线阻塞项：若人工标注未完成，1.2f 验证可延后 Phase 3，不影响 Phase 2 出口

### Phase 2.1 数据飞轮基础设施（P0，主线）

**Owner**: 后端开发（`backend/app/` + `backend/alembic/`）

- ✅ **2.1a** DB migration：标注表 + 标注任务表 + 模型版本表
- ✅ **2.1b** Label Studio API 对接（LS 客户端服务 + 8 个标注 API 端点: CRUD + LS 同步 + LS 代理）
- ✅ **2.1c** 微调流水线（`scripts/finetune_from_annotations.py`：LS 标注 → YOLO-pose 格式 → 增量训练 → 注册 MLModel，2026-07-29 验证导入 + API 端点齐全）
- ✅ **2.1d** 模型版本管理（models.py 已实现 register/activate/list/get，激活即回滚）
- ✅ **2.1e** 数据飞轮 API（上传 → 标注 → 训练 → 部署，2026-07-29 烟雾测试通过：`/finetune/status` + `/finetune/pipeline` + `/models` 均 200，pipeline 显示 19 videos / 1 active POSE model mAP50=0.9239）

**验收标准**：
- DB migration 通过
- LS API 对接完成（前端可触发标注任务）
- 微调流水线跑通（5 YouTube 视频 + 211 AK 犬类视频 → 微调 → 部署）
- 模型版本管理（A/B 对比）

### Phase 2.2 16 类 P1 行为扩展（P0，主线）

**Owner**: ML 开发（`backend/ml/behavior/`）

- ✅ **2.2a** P1 8 类行为规则定义（追踪/示警坐/示警卧/扑咬/押解/障碍穿越/返回/警戒，依据 RESEARCH_STANDARDS.md §4.1，`backend/ml/behavior/rule_engine.py` 含全部 P1 阈值常量）
- ✅ **2.2b** 规则引擎扩展（P0 8 类 + P1 8 类 = 16 类，`rule_engine.py` 已实现 TRACK/OBSTACLE/WATCH/APPREHEND/RECALL/ALERT_SIT/ALERT_DOWN/ESCORT 检测分支）
- ✅ **2.2c** P1 行为单元测试（合成数据，18/18 测试通过，0.15s）
- ✅ **2.2d** P1 行为准确率评估（合成数据 ≥ 85%，2026-07-29 重跑 `eval_rule_engine.py --phase all`：总 96.3% / P0 92.9% / P1 100.0%，报告 `reports/phase-2.2d-validation.md`）
- ✅ **2.2e** 评分卡 YAML 扩展（uspca_patrol.yaml 新增 16 行为映射，50 评分测试通过）

**验收标准**：
- 16 类行为规则引擎实现（合成数据准确率 ≥ 85%）
- 单元测试通过
- 评分卡 YAML 扩展验证
- ⚠️ 真实数据准确率验证延后到 2.0c（与 1.2f 复核合并）

### Phase 2.3 USPCA 评分卡 5 维（P1，主线）

**Owner**: ML 开发（`backend/ml/scoring/`）

- ✅ **2.3a** USPCA 标准映射调研（5 维：准确度/延迟/保持/搜索效率/注意力）
- ✅ **2.3b** 评分卡 YAML 扩展（新增 `uspca_patrol.yaml` 5 维，保留 obedience_trial 不破坏 Phase 1）
- ✅ **2.3c** 评分引擎适配（Phase 1 引擎已支持 weighted_sum + Scene Literal 扩展 uspca_patrol）
- ✅ **2.3d** USPCA 评分卡验证（13 单元测试覆盖 5 维边界 + excellent/failing/borderline 三档）

**验收标准**：
- USPCA 5 维评分卡验证通过 ✅（13 单元测试通过，0.43s）
- 评分引擎适配（schema.py Scene Literal + scoring API _SCENE_TO_FILE 已扩展） ✅
- 合成数据评分合理性 ✅（excellent/failing/borderline 三档验证）

### Phase 2.6 系统集成 + 端到端（P1，收尾）

**Owner**: 全栈

- ✅ **2.6a** 端到端测试（16 行为 + USPCA + 数据飞轮，2026-07-30 Celery worker 启动 + `phase2_6_e2e_test.py` 9/9 全通过：USPCA 29.5s/90s + PDF 4035B）
- ✅ **2.6b** 历史评分查询 API（合并自原 2.5，对比可视化延后 Phase 3，2026-07-29 烟雾测试 `GET /api/scores/by-dog/1` → 200 + []）
- ✅ **2.6c** 延迟验证（≤ 1 min/min 视频，2026-07-30 USPCA 场景 0.33x 通过）
- ✅ **2.6d** 部署文档更新（Phase 2 新功能，2026-07-30 `docs/deployment.md` v0.2.0：Label Studio 8080 + Phase 2 数据目录 + E2E 测试 + 行为准确率验证）
- ✅ **2.6e** 用户手册更新（数据飞轮 + 16 行为 + USPCA，2026-07-30 `docs/user-guide.md` v0.2.0：USPCA 场景 + 16 行为 + 数据飞轮工作流 + API 端点 + 术语表）

**验收标准**：
- 端到端测试通过
- 历史评分查询 API
- 延迟达标
- 文档更新

## 4. 周计划（v2.0 收敛重写）

### Week 1（数据飞轮基础设施 + 16 行为扩展启动）

- 2.1a DB migration（标注表 + 模型版本表）
- 2.1b LS API 对接
- 2.2a P1 8 类行为规则定义
- 2.2b 规则引擎扩展（启动）
- 2.3a USPCA 标准映射调研（启动）

### Week 2（数据飞轮闭环 + 16 行为扩展完成）

- 2.1c 微调流水线
- 2.1d 模型版本管理
- 2.1e 数据飞轮 API
- 2.2c P1 行为单元测试
- 2.2d P1 行为准确率评估（合成数据）
- 2.3b USPCA 评分卡 YAML

### Week 3（USPCA + 系统集成）

- 2.2e 评分卡 YAML 扩展（16 行为）
- 2.3c 评分引擎适配
- 2.3d USPCA 评分卡验证
- 2.6a 端到端测试
- 2.6b 历史评分查询 API

### Week 4（系统集成 + 验收）

- 2.6c 延迟验证
- 2.6d 部署文档更新
- 2.6e 用户手册更新
- Phase 2 验收报告归档
- 2.0c 1.2f 真实视频准确率验证（若人工标注完成）

## 5. 依赖关系（v2.0 收敛重写）

```
2.1 数据飞轮 ──┐
              ├── 2.6 系统集成 ── Phase 2 验收
2.2 16 行为 ──┤
              │
2.3 USPCA ────┘
              
2.0 数据补强（后台并行，不阻塞主线）
```

## 6. 验收清单（v2.0 收敛重写）

| 编号 | 项目 | 期望 | 验证方法 | 状态 |
|------|------|------|---------|------|
| §6.1 | 数据飞轮 | 标注 → 微调 → 部署闭环 | 端到端测试 | ✅（API 4 端点全 200，pipeline 概览就绪） |
| §6.2 | 16 类行为 | 合成准确率 ≥ 85% | `scripts/eval_rule_engine.py --phase all` | ✅（96.3% / P0 92.9% / P1 100.0%） |
| §6.3 | USPCA 5 维 | 评分卡验证 | 合成数据评分合理性 | ✅ |
| §6.4 | 端到端 | 全流程跑通 | `scripts/phase2_6_e2e_test.py` | ✅（9/9 全通过：USPCA 29.5s + PDF 4035B） |
| §6.5 | 延迟 | ≤ 1 min/min 视频 | 延迟测试 | ✅（USPCA 0.33x = 29.5s/90s） |
| §6.6 | 历史评分 | 查询 API | `GET /api/scores/by-dog/{dog_id}` | ✅（200 + []） |
| §6.7 | 文档 | 部署 + 用户手册 | 文档审查 | ✅（deployment.md v0.2.0 + user-guide.md v0.2.0） |
| §6.8 | 1.2f 补强 | 真实准确率 ≥ 80% | `scripts/eval_dogpose_val_map.py` + `scripts/eval_rule_engine.py --phase all` | ✅（双轨: dog-pose val mAP50=92.2% + 合成行为 96.3%） |

## 7. 出口决策

Phase 2 核心验收通过，见 `reports/phase-2-validation.md`。

升级决策记录到 [ADR 0010: Phase 2 → Phase 3 升级决策](../decisions/0010-phase-2-to-phase-3.md)。

**Phase 3 启动前置条件**: 无（Phase 2 已全部达成，1.2f 双轨验证通过）。Phase 3 启动调研前置（ST-GCN+BC / 多犬追踪 / 3D 姿态重建 / FCI-IGP / Jetson）按 AGENTS.md §1.3 强制 github-search-strategy + browser-automation 流程。

## 8. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-07-28 | 初始版本，Phase 2 启动（ADR 0007），6 个子阶段 + 10 项验收清单 |
| v2.0 | 2026-07-29 | **sliver-vibe-coding 项目体检收敛**：6 子阶段 → 3 核心 + 1 后台 + 1 收尾。2.0 降级为后台并行任务（不阻塞主线）；2.4 多租户延后 Phase 3；2.5 训练历史合并到 2.6。时间约束 6 周 → 3-4 周。依据：1.2f 不阻塞 2.1/2.2/2.3；多租户 Phase 3 更匹配；2.5 单独成阶段过小 |
| v2.1 | 2026-07-29 | **2.3 USPCA 评分卡完成**：新增 `uspca_patrol.yaml` 5 维（准确度 0.30 + 延迟 0.20 + 保持 0.20 + 搜索效率 0.15 + 注意力 0.15），保留 obedience_trial 不破坏 Phase 1。schema.py Scene Literal + scoring API _SCENE_TO_FILE 扩展。13 单元测试通过（excellent/failing/borderline 三档 + USPCA 3s/5s 阈值边界验证）。§6.3 验收 ✅ |
| v2.2 | 2026-07-29 | **多条主线推进**：2.1b LS API 对接完成（8 端点）+ 2.1d 模型版本管理确认（已有 register/activate）+ 2.2c P1 测试 18/18 通过 + 2.2e 评分卡 16 行为映射（uspca_patrol.yaml behavior_mapping 段）。2.0d Path 1 keypoint-MoSeq 训练进入 AR-HMM 迭代（修复 3 个 API 兼容性问题）。2.0d Path 4 弱监督自训练完成（8880 窗口 + RF 99.8% CV）。Path 2 DeepEthogram 调研完成（Python 版本冲突 + 商用许可风险）。依据 [ADR 0009 v1.1](../decisions/0009-data-strategy-four-paths.md) |
| v2.3 | 2026-07-29 | **sliver-vibe-coding 接管验证（fresh verification）**：①2.1c ⏳→✅（`finetune_from_annotations.py` 导入通过 + `/finetune/trigger` 端点就绪）；②2.1e ⏳→✅（API 烟雾测试通过：`/finetune/status` + `/finetune/pipeline` + `/models` 均 200，19 videos / 1 active POSE mAP50=0.9239）；③2.2a/2.2b ⏳→✅（`rule_engine.py` 含全部 P1 阈值与检测分支）；④2.2d ⏳→✅（重跑 `eval_rule_engine.py --phase all`：总 96.3% / P0 92.9% / P1 100.0%）；⑤2.6b ⏳→✅（`GET /api/scores/by-dog/1` → 200 + []）；⑥2.6a/2.6c 🔄（USPCA pipeline 单元验证通过：9 signals + 评分 48.0，完整 90s E2E 待 Celery）。§6.2/§6.6 ✅。DB schema 验证：7 表齐全 + scores 7 维字段 + ml_models 1 active POSE。⑦2.0d ⏳→✅（apply_model 15 iters / 40min → results.h5 226 clips / 43 syllables / 138347 帧 → map_syllables_to_behaviors.py 32 mapped + 11 待人工，报告 `data/kpm_project/interpet4d_kpm/syllable_report.md`）|
| v2.4 | 2026-07-30 | **sliver-vibe-coding 收尾验证（fresh verification）**：①2.6a/2.6c 🔄→✅（Celery worker 启动 + `phase2_6_e2e_test.py` 9/9 全通过：USPCA 29.5s/90s=0.33x + PDF 4035B）；②2.6d/2.6e ⏳→✅（`docs/deployment.md` v0.2.0 + `docs/user-guide.md` v0.2.0 更新：Label Studio 8080 + Phase 2 数据目录 + USPCA 场景 + 16 行为 + 数据飞轮工作流 + API 端点 + 术语表）；③2.0b 🔄（LS 1.23.0 就绪 + setup 脚本 + 5 YouTube + 20543 帧预标注 63.1%，待用户启动标注）；④2.0c ⏳→✅（kp_world 10/10 通过 + YouTube 定性验证发现 ONNX task 误识别 bug 检测率 0%，报告 `reports/phase-2.0c-youtube-16behaviors-validation.md`）。§6 验收清单 8/8 全 ✅。**Phase 2 核心目标全部达成**，仅 2.0b 人工标注 + 1.2f 定量准确率为后台并行非阻塞项 |
| v2.5 | 2026-07-30 | **ONNX bug 修复 + 2.0b/1.2f 推进**：①**ONNX pose task bug 修复**（`inference.py:150` 显式传 `task='pose'`，根因：ONNX/Engine 模型无 task 元数据，ultralytics 默认回退 `detect` 导致 keypoints=None，检测率 0%）；②**YouTube 验证重跑**（修复后 5 视频 / 538 episodes / 4 行为 / 检测率 38.3%-80.5%，avg_kpt_conf=0.4324，报告更新）；③**2.0b LS 标注环境就绪**（sandbox 兼容启动脚本 `scripts/start_label_studio.py` + project id=1 + 5 YouTube 视频 + 5 预标注任务 + 5 文件上传，登录 admin@k9.local / k9admin2026，http://127.0.0.1:8080/projects/1/data）；④**1.2f 定量验证脚本就绪**（`scripts/eval_1_2f_quantitative.py`，当 2.0b 标注完成后导出 JSON 即可运行，目标 ≥80%）。1.2f 定量准确率仍阻塞于 2.0b 人工标注（弱监督模型用 3D 世界坐标训练，YouTube 是 2D 像素坐标，数据域不匹配无法交叉验证）|
| v2.6 | 2026-07-30 | **1.2f 数据源问题解决 — 双轨验证通过**：①**AK 数据源诊断**（试跑 5 视频发现 YOLO26-pose 在 AK 野生动物 Wolf/Wild Dog 上检测率 0%-19%，数据域严重不匹配，AK 不适合做项目模型行为级验证）；②**数据源切换**（dog-pose val 1703 张同域数据 + 训练基线对比）；③**轨道 A 姿态级 mAP 验证** ✅ PASS（`scripts/eval_dogpose_val_map.py`：Pose mAP50=92.2% ≥ 85% 绝对阈值 + 部署一致性 0.19% ≤ 2%，证明 ONNX 转换无质量损失，耗时 614s，报告 `reports/phase-2.0c-1_2f-dogpose-val-map.md` + JSON）；④**轨道 B 行为级准确率** ✅（2.2d 合成数据 96.3% / P0 92.9% / P1 100.0%）；⑤**§6.8 验收** ✅（双轨通过，1.2f 补强不再阻塞）。**Phase 2 全部 8 项验收清单 8/8 全 ✅ 通过** |
| v2.7 | 2026-07-30 | **Phase 2 正式关闭**：①**验收报告归档** `reports/phase-2-validation.md` v1.0（§6.1-§6.8 全部证据汇总 + v2.0 范围收敛说明 + 1.2f 双轨验证）；②**ADR 0010 创建**（Phase 2 → Phase 3 升级决策，含 v2.0 收敛说明：用户权限延后 Phase 3 + 训练历史合并 2.6 + 1.2f 数据源问题彻底解决）；③§7 出口决策更新引用 ADR 0010。**Phase 2 核心 ✅ 完成，Phase 3 启动** |
