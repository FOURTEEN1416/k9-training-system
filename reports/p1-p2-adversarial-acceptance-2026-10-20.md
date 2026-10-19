# P1/P2 纯工程三线交付 · 对抗验收报告（2026-10-20）

> 验收对象：工作流 `P1/P2 纯工程三线推进` 的三线交付（数据飞轮 P1-2 / 运维延迟 P1-6 / 稳健性 P2-3·P2-5·ADR0016）
> 验收方法：`adversarial-acceptance` 技能五步（病灶复现 → 三失败场景 → 修 A 破 B → 证据表 → 未验证项）
> 立场：**验收人不是证明人**——不复跑作者已绿的测试自证，专门找写错的地方
> 执行人：主代理（子代理对抗验收因 provider 故障——单个 ask 连续 server_error 7h08m 零产出——已从工作流裁撤）

---

## 1. 逐项检查结果（证据表）

| # | 步骤 | 预期 | 实际 | 证据 |
|---|------|------|------|------|
| 1 | 泄漏口令病灶复现 | psql 报认证失败 | **拒绝成功** | `psql ... 致命错误: 用户 "k9system" Password 认证失败` |
| 2 | 新口令正面回归 | SELECT 返回 | `NEW_PASSWORD_WORKS` | psql 实测 |
| 3 | live 故障注入：停 Redis | /health 指名 | **真过** | `checks.redis={status:down,detail:TimeoutError}`，overall=degraded |
| 4 | 迁移链完整性 | head 单一线性 | **正确** | `alembic heads` = e5f6a7b8c9d0（当时）；链 c3d4e5f6a7b8→d4e5f6a7b8c9→… |
| 5 | 库枚举 vs ORM 口径对账 | 一致 | **不一致（发现#1）** | ORM 期望 `['PENDING',...]`；`pg_enum` 实测 `pending,in_progress,...` |
| 6 | ORM 真写 annotation_tasks | 成功 | **修复前 DataError / 修复后 OK** | `invalid input value for enum ...: "PENDING"` → 迁移后 `ORM 写入: OK (video_id=29)` |
| 7 | legacy 导出脚本字面量 | 大写成员名 | **小写（发现#2）** | `finetune_from_annotations.py:79 'completed'` / `:101 'keypoint'` |
| 8 | dogs 端点基地隔离 | 有校验 | **全线无（发现#3）** | `dogs.py` 无任何 `check_dog_access` 调用；渗透实测 PUT/DELETE 他基地犬返回 200/204 |
| 9 | 渗透脚本自证能力 | 能独立发现 | **名副其实** | 163 用例（A 53 / B 53 / C 57），首轮 28 条含 4 CRITICAL；前置不过时拒绝出安全结论（exit=2） |
| 10 | ADR 0016 是否「不降级」 | 明确否决降级 | **符合（用户硬要求）** | `0016:15` 「产品定位【不降级】」；`:18` 书面否决审计型默认路径；`:29` 不自设判定日期 |
| 11 | start_all.ps1 迁移步骤 | 可执行 | **不可执行（发现#5）** | `python -m alembic` → `No module named alembic.__main__`（cwd=backend 使 import 命中本仓 alembic/ 目录） |
| 12 | 畸形 JWT | 401 | **500（发现#6）** | `/api/auth/me` + 缺 sub → `KeyError: 'sub'`；sub 非数字 → `ValueError` |
| 13 | /health 信息面 | 不泄露内网 | **回显 redis URL（发现#7）** | 渗透 INFO `OBS-health-redis-url` |
| 14 | annotation review 是否缺鉴权 | 有权限层 | **有（假阳性）** | `annotations.py` `ReviewForbidden`→403 路径；`finetune` export/trigger 均挂 `AdminOrManager` |
| 15 | 全量回归 | 绿 | **742 passed / 2 skipped / 0 failed** | pytest backend/tests（--ignore=integration） |
| 16 | 前端类型检查 | 0 错 | **0 错** | `vue-tsc --noEmit` exit=0 |
| 17 | 渗透复跑（决定性验证） | 越权面收敛 | **28 → 3 条，HIGH/CRITICAL 清零** | `cases=163 findings={INFO:2, MEDIUM:1} exit=0` |

---

## 2. 发现清单（现象 → 根因 → 当场修复 → 回归证据）

### 发现#1（真缺陷，HIGH，已修）——标注枚举大小写混存
- **现象**：`GET /api/annotations/tasks/{id}` 500，日志 `LookupError: 'completed' is not among the defined enum values`
- **根因**：ORM 用 SQLAlchemy 默认**成员名**（`PENDING`）存储；历史迁移建的是**小写标签**（`pending`）。工作流的迁移 `e5f6a7b8c9d0` 只 `ADD VALUE` 补大写标签——但 **PG 无法删除枚举标签**，小写仍在类型内，任何 raw-SQL 路径写小写都静默成功、ORM 一读即炸
- **修复**：新增迁移 `f6a7b8c9d0e1` 重建三个枚举类型为**纯成员名**（新建→USING 转换→删旧→改名，`annotation_source` 用 CASE 特判 `prelabel→MODEL_PRELABEL`）
- **回归**：`SELECT 'completed'::annotation_task_status` → `错误: 输入值无效`（类型层面根除）；ORM 写入成功

### 发现#2（真缺陷，MEDIUM，已修）——legacy 导出脚本被枚举规范化打破
- **现象**：`scripts/finetune_from_annotations.py` 枚举规范化后将查不到任何行（导出为空）
- **根因**：脚本用 raw SQL 小写字面量过滤（`:79 'completed'` / `:101 'keypoint'`）；工作流已在自己的迁移 docstring 里标为「发布阻断项」，但因该文件不在其白名单而**未修**
- **修复**：两处字面量改大写成员名
- **回归**：随发现#1 的类型净化一并生效（小写已不可能被写入）

### 发现#3（真缺陷，CRITICAL/HIGH，已修）——资源级基地隔离系统性缺失
- **现象**：渗透实测跨基地 MANAGER 可 `PUT`/`DELETE` 他人基地犬档案（200/204）、跨基地可读犬/视频/评分/标注（HIGH 多条）
- **根因**：`dogs.py` / `videos.py` / `scores.py` / `annotations.py` **从未调用**仓库既有的 `check_dog_access` / `require_dog_access`（Phase 3.6 建好但没接线）。此前"基地隔离 25/25 通过"只覆盖 bases API 与 deps 单测
- **修复**：新增单一真源 `deps.visible_dog_id_select`（ADMIN 全局 / HANDLER 自有+关联 / MANAGER·RESEARCHER·VIEWER 本基地+未过期临时授权 / 其他**空集 deny-by-default**）与 `check_video_access`；dogs 列表改走同一口径（消除重复实现）；四张资源的 list/get/write 端点全部接线；`annotations` 任务创建补角色守卫（VIEWER 只读）
- **回归**：渗透复跑 **10 条跨基地用例全部消失**（含 4 CRITICAL），HIGH 31→0

### 发现#4（假阳性，已澄清）——annotations/finetune「无鉴权」
- **初判**：grep 到 `annotations.py`/`finetune.py` 端点只用 `CurrentHandler`，疑 VIEWER 可审标注、可触发微调
- **核实（技艺 9 权威源码核实）**：`finetune` 的 export/trigger 均挂 `AdminOrManager`；`review_episode` 内部有 `ReviewForbidden`→403 归属校验
- **结论**：**不是缺陷**，如实撤回，未做无谓改动

### 发现#5（真缺陷，MEDIUM，已修）——start_all.ps1 迁移步骤不可执行（我 P0-2 的遗留）
- **根因**：脚本 `Push-Location backend` 后跑 `python -m alembic`；`alembic.ini` 在仓库根，且 cwd=backend 时 `import alembic` 命中本仓 `backend/alembic/` 迁移目录而被遮蔽
- **修复**：改在仓库根调用
- **回归**：`alembic current/heads/upgrade` 实测通过

### 发现#6（真缺陷，HIGH，已修）——畸形 JWT claim 冒泡 500
- **现象**：`/api/auth/me` 携缺 `sub` 或 `sub` 非数字的 JWT → 500（预期 401）
- **根因**：`TokenPayload.from_dict` 直接 `int(d["sub"])`，KeyError/ValueError 未捕获
- **修复**：捕获 `KeyError/TypeError/ValueError` → `InvalidTokenError`（401）
- **回归**：`test_rbac.py` 新增 6 例畸形 payload 参数化测试，全绿

### 发现#7（真缺陷，LOW，已修）——/health 回显 redis 连接串
- **修复**：detail 去掉 URL（避免向未认证调用方泄露内网地址）

---

## 3. 复验结论（决定性证据）

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 渗透发现总数 | 28（基线）→ 35（首轮复跑） | **3** |
| CRITICAL | 4 | **0** |
| HIGH | 31 | **0** |
| 剩余 | — | INFO×2（openapi 公开、role claim 以 DB 为准——设计如此）+ MEDIUM×1（审计日志未实现，属 P2-3 范围，已知缺口） |
| 后端回归 | — | 742 passed / 0 failed |
| 前端类型 | — | 0 错 |

---

## 4. 未验证项（如实声明）

1. **审计日志**（渗透 MEDIUM `OBS-audit-log-missing`）：未实现，属 P2-3 后续范围，本次未做。
2. **多机/网络隔离场景**：渗透仅对本机 127.0.0.1:8000 执行，未验证非 loopback 部署下的隔离。
3. **前端无单元测试体系**：vitest 0 文件；前端正确性靠 type-check + 人工冒烟（本次未做浏览器复验）。
4. **P1-2 端到端业务闭环**（标注审核→导出→微调→注册）：新增 92 个单测为 **mock 级**，真库链路只验证到「枚举/ORM 写入可用」，未跑通完整训练闭环（需 GPU 与时长）。
5. **真实 K9 数据**：P1 科学验收仍待 ADR 0015 §2 五项用户裁决（未变）。
6. **迁移在真实数据量下的开销**：本次库内 annotation 表为空，`ALTER TYPE` 表重写为秒级；线上需先备份评估。
