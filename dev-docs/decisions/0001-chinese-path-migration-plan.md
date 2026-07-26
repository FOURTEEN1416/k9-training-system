# Decision 0001: 中文路径迁移计划

> 决策日期: 2026-07-26
> 决策状态: 已完成（2026-07-26 项目已迁移至 `D:\Desktop\k9-training-system`）
> 决策类型: 工程基础设施

## 状态更新（2026-07-26）

✅ **迁移已完成**：项目当前路径 `D:\Desktop\k9-training-system`，Git 历史完整保留（1 commit on master），dev-docs/ 结构正常。

后续验证项将在 Phase 0 启动时执行（npm install / pip install 时验证无 ghost entry）。

---

## 历史记录

## 背景

项目当前路径为 `D:\Desktop\警犬训练`（中文路径）。

用户项目记忆中明确记录了中文路径的教训：

> 中文路径 D:\Desktop\校友交流 导致 npm 包提取出现 ghost entry（文件写入成功但不可读），67% 的 node_modules 包损坏。junction 不能绕过此问题。建议将项目迁移到纯英文路径，或在英文路径下执行 npm install 后复制 node_modules

本项目将大量使用：
- npm（Vue 前端）
- pip（Python 后端）
- mmaction2 / ultralytics（含 C++ 扩展）

中文路径会触发同样的 ghost entry 故障。

## 决策

**迁移到英文路径 `D:\Desktop\k9-training-system`**，但**延后到 Phase 0 实际开发（需要 npm install）前执行**。

## 理由

### 为什么不立即迁移

1. **TRAE 安全策略限制**：当前 AI 只能在 `d:\desktop\警犬训练` 路径下操作，无法直接在 `D:\Desktop\k9-training-system` 创建文件
2. **当前阶段无 npm/pip 安装**：立项阶段仅创建 truth 文档（Markdown），无 ghost entry 风险
3. **Git 历史保留**：现在建立 Git 仓库，迁移时整个目录复制，Git 历史完整保留

### 为什么不保留中文路径

1. **历史教训**：用户已踩过坑，67% node_modules 损坏
2. **Phase 0 即将开始**：Phase 0 需要执行 npm install / pip install，必须在此之前迁移
3. **junction 无效**：用户记忆明确记录 junction 不能绕过此问题

## 迁移时机

**触发条件**：Phase 0 启动时（即将执行 npm install / pip install）

**迁移步骤**：
1. 用户手动创建 `D:\Desktop\k9-training-system` 目录
2. 用户手动复制 `D:\Desktop\警犬训练\*` 到 `D:\Desktop\k9-training-system\`
3. 在新路径验证 Git 状态正常（`git status`）
4. 在新路径执行 `npm install` / `pip install`
5. 验证安装完整性（无 ghost entry）
6. 后续开发在新路径进行

## 回滚方案

如果迁移后发现问题：
- 旧路径 `D:\Desktop\警犬训练` 保留（含完整 Git 历史）
- 可随时回退到旧路径继续工作

## 不可逆边界

- 迁移本身可逆（复制而非移动）
- 一旦在新路径执行 npm install，新路径的 node_modules 不可简单复制回旧路径（会再次触发 ghost entry）

## 验证清单

迁移后需验证：
- [ ] Git 状态正常（`git log` 显示历史提交）
- [ ] truth 文档完整（`dev-docs/` 目录结构正确）
- [ ] `npm install` 成功（无 ghost entry，`node_modules/.package-lock.json` 可读）
- [ ] `pip install` 成功（Python 包可导入）
- [ ] MMAction2 配置可加载

## 相关文档

- `dev-docs/project-brief.md` §14 未解决问题
- `AGENTS.md` §9 未解决问题
- 用户记忆 `project_memory.md`（中文路径教训）
