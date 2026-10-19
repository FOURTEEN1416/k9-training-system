# P0 生产底座收尾验证报告（2026-10-19）

> 依据: `dev-docs/production-readiness-gap-plan.md` P0 段（P0-1 安全基线 / P0-2 一键启动 / P0-3 评测协议 / P0-4 权重盘点 / P0-5 报告溯源 / P0-6 产品最小闭环）
> 性质: 新鲜验证证据（当次运行）；全部证据可复核
> 执行窗口: 本窗（默默指令「开始进行编排」，用户主权推动）

---

## 1. 完成声明四要素

- **验证证据**：本报告 §2 逐条列出命令 + 实测输出 + 提交哈希
- **边界检查**：仅触碰白名单内文件；`.env`/密钥零入库（.env 在 .gitignore；凭证文件随用随删）；数据库仅 ALTER USER 一次（轮换口令），未删改任何业务表
- **已知限制**：§3
- **置信度**：P0-1/2/4/5 = 高（命令级实锤）；P0-6 = 高（API 冒烟 + 浏览器端到端）；P0-3 = 高（文档+索引落盘，生效仍待用户裁决）

## 2. 逐项验收

| 项 | 验收线（gap-plan） | 实测证据 | 判定 |
|----|------|------|------|
| P0-1 密钥轮换 | production 启动过校验；.env 不含 .env | `ALTER USER k9system` 随机口令轮换后用新口令 psycopg2 重连成功；uvicorn 起 8000 无 ValidationError；`git check-ignore .env` = .gitignore:80 | ✅ |
| P0-2 一键启动+health | 干净步骤起全服务；停 Redis 指名故障 | `scripts/start_all.ps1`（预检/Redis 拉起/PG 双检/Alembic/Celery/FastAPI/前端 + 组件账本）；`/health` 实测 `{"status":"ok",checks:{database:ok,redis:ok,models:ok},deploy_mode:"shadow"}`；测试 4 绿含「停 Redis → degraded + 指名 redis」 | ✅ |
| P0-3 协议冻结 | 预注册文档入库 | ADR 0015 `dev-docs/decisions/0015-k9-bench-v0-protocol.md`（切分双隔离/指标/70%/85% 门槛/报告格式/反验收条款/§2 五项用户待定项显式标注）；README ADR 索引补登 | ✅（生效待用户裁决） |
| P0-4 权重盘点+rule_only 默认 | 模型缺失不崩溃；默认 mode 文档代码一致 | `scripts/inventory_models.py` 实测 5 组全可加载（pose/stgcn_bc/mamba/mamba_bc/motionbert，JSON 报告归档）；`behavior_deploy_mode` 未配置时 production→rule_only、dev/test→shadow（5 测试绿）；tasks.py 路由 getattr 兜底已核 | ✅ |
| P0-5 报告溯源 | 任一 PDF 可回答「分从哪来、用哪个模型、是否需复核」 | PDF 头部三字段（识别模式/模型版本/系统版本）+ 置信度摘要 + 复核橙色横幅 + 「评分溯源」附录（维度→命中规则→评分卡版本）；`decide_review` 纯函数（rule_only 恒复核/空 episode/均值<0.60/最低<0.30）；worker 日志带 mode+review；10 测试绿 | ✅ |
| P0-6 登录+Handler 闭环 | HANDLER/ADMIN 全流程手测过；未登录/错密/失效可读 | ADMIN 账号（id=64）造号后：正确登录 200 / 错密统一 401「邮箱或密码错误」/ 无 token me 401 / 带 token me 200；浏览器：访问 /upload 被守卫重定向登录页 → 登录 → 落 /upload，页头显示 `v0.1.0 · development · 识别模式: shadow` + 用户名 + admin 角色 tag（截图留档）；导航按角色过滤 | ✅ |

**回归**：pytest 644 passed + 2 skipped + 0 failed（P0 前基线 631，新增 13 测试，修复 1 存量漂移）；前端 `vue-tsc --noEmit` 0 错。

## 3. 已知限制与边界

1. **P0-3 生效依赖用户裁决**：§2 五项（视频来源/保密级别/首发科目/规模/目标机）未裁决前，协议处于「预注册」状态——评测链路代码就绪，但没有真实数据可评。
2. **deploy_mode=shadow 是 dev 语义**：当前 .env `APP_ENV=development`；上产线设 `APP_ENV=production` 后默认自动切 `rule_only`（也可显式 `BEHAVIOR_DEPLOY_MODE` 覆盖）。
3. **种子账号**：冒烟用 ADMIN（admin@k9.local）已留在 DB（首次登录建议改密）；临时凭证文件已删除，未入库。
4. **浏览器冒烟中的两度 401**：系种子凭证文件含行尾换行/特殊字符被 shell 展开截断所致（`token_urlsafe` 的 `^M$`），非产品缺陷——改用精确凭据后一次通过。
5. **Redis 为手工拉起**（`redis-server --port 6379`）：一键脚本可自动拉起，但 Windows 服务常驻仍建议注册服务（deployment.md §4.0 已注明边界）。

## 4. 提交链（本窗，未推送）

```
af7f1b3 docs(plan): 收编生产可用性差距整改方案 v1.0
80e6491 feat(p0): P0-2 一键启动编排 + /health 聚合
b0bff3a feat(p0): P0-4 模型盘点 + deploy_mode 按环境推断
b00c506 feat(p0): P0-5 报告溯源三字段 + 复核标注 + 溯源附录
7891d65 docs(p0): P0-3 K9-Bench-v0 协议预注册（ADR 0015）
77bc3ef feat(p0): P0-6 前端登录 + Handler 闭环 + 版本/模式页头
```

`main...origin/main [ahead 7]`（含前置 `92f106f` 安全默认值），**未推送**——推送到 public GitHub 属跨工具动作，待用户拍板。

## 5. P1 启动前置（用户裁决项）

ADR 0015 §2 / gap-plan §12 五问：①真实视频来源与可标注时长 ②保密级别（可否外包/必须全离线）③首发科目范围 ④部署目标机 ⑤识别过线前的产品定位（正式生产工具 vs 试点辅助工具）。
