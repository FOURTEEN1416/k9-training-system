# 文档与验证证据索引

> 建立日期：2026-08-20
> 用途：区分当前 Truth、历史研究、工程验收和未完成验证，避免把旧报告当成当前执行计划。

## 当前 Truth

| 文件 | 作用 |
|------|------|
| `AGENTS.md` | 项目宪法与硬约束 |
| `dev-docs/project-brief.md` | 项目定位、研究范围与边界 |
| `dev-docs/stage-plan.md` | 阶段总览，P0 主线、Phase 4 并行 |
| `dev-docs/stages/data-unblocking.md` | 当前 P0 执行计划 |
| `dev-docs/decisions/0011-phase-4-llm-explainer-adr.md` | LLM 双阶段边界 |
| `dev-docs/decisions/0012-foundation-system-direction.md` | 当前基座方向决策 |
| `dev-docs/runtime.md` | 本机运行时基线 |

## 工程验收证据

| 证据 | 结论边界 |
|------|----------|
| `reports/phase-0-validation.md` | 基础设施验收 |
| `reports/phase-1-validation.md` | MVP 工程闭环，带条件 |
| `reports/phase-2-validation.md` | 规则/评分/USPCA 工程闭环，行为结果主要为合成或替代验证 |
| `reports/phase-3.8a-e2e-eval.json` | Phase 3 聚合；真实数据、Jetson、多犬 GT 有 SKIP |
| `reports/phase-3.8b-latency-eval.json` | FCI-IGP 0.63x，SHADOW 1.13x，Jetson 未就绪 |
| `reports/phase-4.2-mamba-baseline.md` | 合成数据 Mamba 85.61%，真实数据未验证 |

## 数据域口径

- 合成数据：证明代码和训练管线可以运行。
- 公开真实数据：证明跨域泛化能力。
- 真实 K9 数据：证明目标训练场景有效性；不是 P0 启动硬依赖。

## 历史/已替代路径

早期 Phase 1-2 报告中的 YouTube、旧 Label Studio 专用服务、DogMo 替代方案和 keypoint-MoSeq 路径保留为历史证据，不再作为当前执行入口。当前标注工具可使用 Label Studio，但必须接入 P0 主动学习协议。

逐视频 JSON、PDF 和其他生成报告统一归入历史实验产物，不单独建立新的 Truth。其文件级分类规则见 `reports/README.md`。
