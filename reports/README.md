# 验收与实验报告索引

> 本目录的报告是历史验收证据、实验结果和生成产物，不是当前项目计划。报告中的时间戳、测量值和当时决策应保留；当前解释服从 `dev-docs/` Truth 和 `reports/documentation-conflict-register.md`。

## 证据分类

| 类别 | 文件模式 | 使用规则 |
|------|----------|----------|
| 阶段验收 | `phase-{0,1,2}-validation.*` | 保留历史出口证据；注意合成/替代验证边界 |
| Phase 3 工程证据 | `phase-3.*` | 证明工程能力；真实数据、Jetson、多犬 GT 的 SKIP 必须保留 |
| Phase 4 实验 | `phase-4.*` | 并行实验记录；合成结果不得当作真实泛化 |
| 逐视频追踪 | `tracking_*.json` | 实验产物，不是全局模型指标或当前 Truth |
| PDF/生成报告 | `*.pdf` | 输出样本，仅证明生成链路，不证明模型准确率 |

## 当前解释规则

1. 合成数据只证明管线可运行。
2. 公开真实数据用于跨域泛化判断。
3. 真实 K9 数据用于目标场景判断，且不是 P0 启动硬依赖。
4. Phase 3 工程关闭不等于所有后验科学/硬件验证完成。
5. 已废弃的 YouTube/旧 Label Studio 专用流程只作为历史证据，不得从报告反推当前执行计划。

详细映射见 [`documentation-evidence-index.md`](documentation-evidence-index.md)；冲突见 [`documentation-conflict-register.md`](documentation-conflict-register.md)。
