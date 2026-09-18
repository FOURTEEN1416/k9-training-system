# ADR 0014：K9 → PSD-Framework Phase 4.1 Migration Audit

## 状态: ✅ 已完成 (2026-08-24)

## 审计概要

本次审计由歆歆（sliver-vibe-coding 协调者）于 2026-08-24 完成，对象为** Phase 4.1 “基座系统产品化”** 从 k9-training-system 仓库向 psd-framework 独立仓库（ADR 0013）的资产迁移。

**核心原则（零代码修改）**：
- K9 仓全程只读：不修改任何 `.py` 文件；`.venv`/`__pycache__` 不同步
- 跨仓只允许文档指针，禁止 `import`
- 22类标签、关键点索引为单一真见于 `psd-framework/docs/assets-map.md` §1/§2
- 所有移植通过 `psd-framework/docs/assets-map.md` 显式登记后实施

## 决策 1：Phase 4.1 范围确认

**决策**：Phase 4.1 产品升级线留在 k9 仓；研究线 P0 数据解阻主线已移交 psd-framework（ADR 0013）。

- ** k9 仓保留**：LLM 行为解释器（ADR 0011 v1.1 双阶段）、Transformer-Mamba 基线、Mamba 产品化、RBAC 权限、训练历史对比——全部作为 Phase 4 产品升级继续推进。
- ** psd-framework 承接**：P0 数据解阻完整子阶段（P0.1—P0.6），方法论管线（AimCLR+SMQ+TCL+姚青 JIA 迁移+主动学习），论文实验素材（可解释评分卡 + 端到端工作犬评估系统 + 物理-语义解耦基座）。

## 决策 2：移植方式授权

**决策**：所有代码移植通过 `psd-framework/docs/assets-map.md` §3 显式移植纪律完成。k9 仓于移植瞬间锁定只读，移植产物在 psd-framework 独立维护。

| 移植类别 | k9 仓状态 | psd-framework 目标 | 移植方式 |
|---------|-----|-----|-----|
| ST-GCN+BC 主干 | ⏳ 只读（锁定） | `psd/models/stgcn_*.py` / `psd/training/train_stgcn_bc.py` | §3.1 复制+namespace适配 |
| 22类标签 / 24关键点 | ✅ 只读指针登记 | `psd/data/stgcn_labels.py`（docstring 引用） | §3.4 只读引用 |
| FCI-IGP 评分卡 YAML | ⏳ 待移植 | `psd/scoring/fci_igp.yaml` | §3.5 直接移植 |
| API 端点 | ⏳ 只读（产品线留 k9） | 文档指针仅（LLM 双阶段见 ADR 0011） | 无代码迁移 |

## 决策 3：依赖冲突决议

| 冲突 | 根因 | 解决 |
|------|------|-----|
| YOLO26-pose 在 AK 野生动物上的检测率 0%-19% | 图像域不匹配（ADR 0008 v1.6） | L1 物理表征层改用 AimCLR 预训练骨干 + SMQ 时序分割 |
| `image.tar.gz` 42GB 下载冲突 | 同源无价值 + Google Drive 配额 | 已取消下载；改用 APTv2 83K + InterPet4D 226 seq |
| `onnxruntime` 版本混合 | 环境残留 | 采用 `onnxruntime-gpu` 1.20.1 单一版本 |
| `passlib`↔`bcrypt` 不兼容 | RBAC 实现历史遗留 | 直接调用 passlib→bcrypt；或改用 bcrypt 单独实现 |

## 决策 4：工作量与优先级

**总估算**：~7 person-days，全部为文档/指针适配，零代码修改。

**优先级**：
1. 登录 22类行为标签 + 24关键点索引在 assets-map §1/§2——作为全仓唯一真相
2. 复制 ST-GCN+BC 核心模块至 psd models/training——仅 import 路径适配
3. 移植 FCI-IGP 评分卡 YAML——结构兼容，低风险
4. API route 注册适配——保留 FastAPI 结构，作产品线登记

## 决策 5：跨仓指针登记

**决策**：在 `psd-framework/docs/assets-map.md` 新增 Section 5 “K9→PSD Pointer Map (Audit v1.0, 2026-08-24)”，登记全部 7 项 K9 资产的移植指向与状态。仅作文档指针，不修改任何 `.py` 文件。

**登记痕迹**：
- `k9-training-system`: git commit 参照 `ADR 0014`；`dev-docs/stages/phase-4.md` 章节注释更新
- `psd-framework`: `docs/assets-map.md` Section 5 已更新（见本文件编辑记录）

## 相关文档

- `ADR 0013`: 研究仓库拆分（psd-framework 独立仓库）
- `ADR 0012`: 基座系统方向决策（物理-语义解耦 + 小样本自监督数据飞轮）
- `ADR 0011`: Phase 4 LLM 行为解释器（双阶段策略）
- `stage-plan.md`: 大阶段计划（P0 移交 + Phase 4 并行实施）
- `dev-docs/research/RESEARCH_FOUNDATION_SYSTEM_VISION.md`: 基座系统构想调研报告

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-24 | 创建：迁移审计报告，5项决策确认，资产×需求矩阵，7项依赖冲突决议，工作量估算，跨仓指针登记 |