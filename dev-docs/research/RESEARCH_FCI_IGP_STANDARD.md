# RESEARCH: FCI-IGP 国际工作犬标准调研

> 调研日期: 2026-07-30
> 调研人: AI Agent (general_purpose_task subagent)
> 阶段: Phase 3.4a
> 依据: AGENTS.md §1.3 (github-search-strategy + browser-automation)
> 一手来源: FCI 官方 2025 规则 PDF (82 页, 2.4MB, fci.be/medias/UTI-REG-IGP-en-2025-22401.pdf)

---

## 0. 调研方法与证据链（AGENTS.md §1.3 合规声明）

本调研严格遵循 AGENTS.md §1.3「调研搜索强制 GitHub-First（零容忍硬规则）」，**未使用 WebSearch**。

### 0.1 调研流程
1. **github-search-strategy skill (Directory-First)**：
   - 扫描 `sindresorhus/awesome` 元目录 README（79,604 字符），检索 dog/canine/working dog/dog sport/schutzhund/igp/ipo/k9 等关键词 → **元目录无 dog/working-dog 相关条目**（awesome-working-dog / awesome-dog-sports 目录不存在）
   - GitHub Search API 查询：`FCI-IGP`（0 仓库）、`schutzhund IPO rules`（0 仓库）、`dog behavior recognition`（7 仓库）
   - 验证 top 候选 README：`rjdbcm/BEAGLES`（4★，犬行为标注工具）、`Ekta729/Dog-Behavior-Recognition`（3★）
   - **结论**：GitHub 无专门 FCI-IGP 规则仓库或开源评分实现；dog behavior recognition 仓库规模小（≤4★），仅与本项目 ML 侧间接相关，非 IGP 标准源
2. **browser-automation skill (Playwright + requests)**：
   - 直连下载 FCI 官方 2025 IGP 规则 PDF（绕过本地 Clash 代理 7897 未运行；fci.be SSL 证书过期用 `verify=False` 跳过）
   - **PDF 规格**：`UTI-REG-IGP-en-2025-22401.pdf`，82 页，2,434,431 bytes，HTTP 200，content-type=application/pdf
   - 用 `pymupdf` (fitz) 解析全文 270,990 字符
   - FCI General Committee 2024-09-03 批准，2025-01-01 生效

### 0.2 一手源可信度
| 源 | 类型 | 可信度 | 用途 |
|----|------|--------|------|
| fci.be 官方 2025 IGP PDF | 一手官方文档 | ⭐⭐⭐⭐⭐ | 全部评分规则/分值/扣分项的权威依据 |
| GitHub awesome 元目录 | Directory-First | ⭐⭐⭐⭐ | 确认无专门 IGP 开源生态 |
| 项目 `RESEARCH_STANDARDS.md` v1.1 | 项目既有调研 | ⭐⭐⭐⭐ | 交叉验证（与官方 PDF 一致） |

---

## 1. FCI-IGP 标准概述

### 1.1 标准身份

| 属性 | 内容 |
|------|------|
| 全称 | Fédération Cynologique Internationale - International Utility Dog Regulations (FCI-IGP) |
| 德文原名 | Internationale Gebrauchshund Prüfungsordnung（IGP 即源自 Gebrauchshund Prüfung） |
| 管理机构 | FCI（世界犬业联盟，总部比利时 Thuin），Utility Dogs Commission |
| 现行版本 | **2025 版**（2024-09-03 General Committee 批准，2025-01-01 生效） |
| 官方语言 | 英语（争议时以英文版为准） |
| 适用范围 | 所有 FCI 成员国和合同伙伴的国际赛事 |
| 历史名称 | IPO（Internationale Prüfungs-Ordnung），2019 年更名为 IGP |

### 1.2 核心结构：三阶段 × 三等级

FCI-IGP 由 **3 个阶段（Phase A/B/C）** 组成，每个阶段满分 100 分，总分 300 分。每个阶段必须独立达到 70%（Satisfactory）才算通过。

| 阶段 | 德文 | 英文 | 满分 | 及格线 | 核心 |
|------|------|------|------|--------|------|
| Phase A | Fährtenarbeit | Tracking（追踪） | 100 | 70 | 鼻子工作、迹线跟随、物品指示 |
| Phase B | Unterordnung | Obedience（服从） | 100 | 70 | 随行、坐/卧/立、衔取、跳跃、前进 |
| Phase C | Schutzdienst | Protection（护卫） | 100 | 70 | 搜索、吠叫警戒、扑咬、押解、放口 |

### 1.3 三个等级（IGP 1/2/3）

等级必须**顺序通过**（1→2→3），每级每阶段至少 Satisfactory(70%)：

| 等级 | 定位 | 难度递增点 |
|------|------|-----------|
| **IGP 1** | 入门级 | 自有迹线、基本服从、基础护卫 |
| **IGP 2** | 中级 | 他人迹线、增加 Stand 科目与 Back Transport |
| **IGP 3** | 高级（最高） | 他人长迹线 600 步、跑动中科目、双攻击、Back Transport 攻击 |

### 1.4 评分等级体系（官方 Point Table）

**单科目评分（按满分百分比）**：

| 评级 (Rating) | 占满分比例 |
|--------------|-----------|
| Excellent（优秀） | 96 – 100% |
| Very Good（很好） | 90 – 95.5% |
| Good（良好） | 80 – 89.5% |
| Satisfactory（合格） | 70 – 79.5% |
| Insufficient（不合格） | 0 – 69.5% |

**IGP 1-3 总分评级（300 分制）**：

| 评级 | 总分区间 |
|------|---------|
| Excellent | 286 – 300 |
| Very Good | 270 – 285 |
| Good | 240 – 269 |
| Satisfactory | 210 – 239 |
| Insufficient | 0 – 209 |

**关键规则**：
- 每阶段（A/B/C）必须独立 ≥ 70%（Satisfactory），否则整段不合格
- 单项可给部分分（partial points），但阶段总分仅给整数分
- 扣分精度：**0.5 分**（比中国 GA/T 整数分更细）

### 1.5 基本评估要求（所有等级通用）

FCI-IGP 评估的核心维度（PDF §"Basic requirements"）：
1. **表现力/自信 (Expressive behaviour/self-confidence)**：面部表情、耳尾姿态、肌肉紧张度、呼吸、流涎、焦躁行为
2. **愉悦主动的工作 (Joyful, motivated work)**：工作意愿是最高优先级
3. **专注 (Concentration)**：全程专注训导员，无异常体态/头部姿态
4. **即时响应 (Immediate response)**：直接接受口令，无迟疑执行

> ⚠️ **重要**：FCI-IGP 明确将「恐惧/压力/焦躁」行为列为扣分项，这与本项目 7 维评分中的「注意力」「胆量」维度高度契合——机器视觉可通过姿态紧绷度、呼吸频率、异常运动检测这些主观指标。

---

## 2. IGP 科目细分

### 2.1 追踪科目（Phase A - Fährtenarbeit）— 100 分

#### 2.1.1 等级参数对比

| 参数 | IGP 1 | IGP 2 | IGP 3 |
|------|-------|-------|-------|
| 迹线类型 | 自有迹线（训导员自己走） | 他人迹线 | 他人迹线 |
| 牵引绳长 | 5 米 | 10 米 | 10 米 |
| 迹线长度 | ≥ 300 步 | ≥ 400 步 | ≥ 600 步 |
| 迹线段数 | 3 段 | 3 段 | 5 段 |
| 转弯数 | 2 个（约 90°） | 2 个（约 90°） | 4 个（约 90°） |
| 物品数 | 3 个自有物品 | 3 个他人物品 | 3 个他人物品 |
| 物品分值 | 3 × 7 = 21 分 | 3 × 7 = 21 分 | 3 × 7 = 21 分 |
| 迹线老化时间 | ≥ 20 分钟 | 30 分钟 | 60 分钟 |
| 工作时限 | ≤ 15 分钟 | ≤ 15 分钟 | ≤ 20 分钟 |
| 物品尺寸 | 10×2-3×0.5-1 cm | 同左 | 同左 |

#### 2.1.2 分值构成
- **追踪工作（迹线跟随）**：79 分
- **物品指示**：3 × 7 = 21 分
- **合计**：100 分

#### 2.1.3 评估标准（官方原文提炼）
- 深鼻、恒定强度、转弯前后一致速度
- 训练水平（焦躁/压力/回避行为 = 不良）
- 训导员与犬的协作
- 困难因素：植被/土壤/地形/风向/野生动物/天气

#### 2.1.4 扣分项（明确数值）

| 违规 | 扣分 |
|------|------|
| 起跑旗前给口令 | -1 分 |
| 误指示物品，训导员在绳端给额外口令 | -2 分 |
| 误指示物品，训导员走近犬给口令 | -4 分 |
| 训导员指示方向帮助 | -2 分 |
| 排尿/排便 | **-8 分** |
| 游荡/牵引帮助/持续鼓励 | 相应扣分 |

#### 2.1.5 自动化可行性：⭐⭐⭐（需 GPS + 姿态融合）
- 可视觉判定：低头嗅闻姿态、物品指示（坐/卧）、迹线跟随路径
- 不可视觉：气味辨别（需传感器）

### 2.2 服从科目（Phase B - Unterordnung）— 100 分

#### 2.2.1 分值表（官方 Scheme of obedience exercises FCI-IGP）

| # | 科目 | IGP 1 | IGP 2 | IGP 3 |
|---|------|-------|-------|-------|
| 1 | 自由随行 (Free heeling) | 15 | 15 | 15 |
| 2 | 行进中坐 (Sit in motion) | 10 | 10 | 10 |
| 3 | 行进中卧 + 召回 (Down with recall) | 10 | 10（常步） | 10（跑步） |
| 4 | 行进中立 (Stand exercise) | — | 10（常步，接回） | 10（跑步，召回） |
| 5 | 平地衔取 (Retrieve on flat) | 15 | 10 | 10 |
| 6 | 跨栏衔取 (Retrieve/jump over 1m hurdle) | 15（两次跳，无衔取） | 15（两次跳+衔取） | 15（两次跳+衔取） |
| 7 | 攀墙衔取 (Retrieve/climb-jump over scaling wall) | 15（1 次，160cm，无衔取） | 10（1 次，160cm，无衔取） | 10（攀跳+衔取，160cm） |
| 8 | 前进并卧 (Send out with down) | 10 | 10 | 10 |
| 9 | 干扰下卧 (Down under distraction) | 10 | 10 | 10 |
| | **合计** | **100** | **100** | **100** |

#### 2.2.2 关键科目细则

**自由随行 (Free heeling) — 15 分**：
- 犬在训导员左侧，肩胛对齐膝盖，单口令"Heel"
- 含正常/跑步/慢步三种步态，左转弯，停步自动坐
- 穿过至少 4 人移动人群，绕左+绕右+停一次
- **枪声测试**：第一直线发 2 枪（6mm，间隔 5 秒，距离≥15 步），枪怯 = **取消资格（DQ）+ 已得分清零**

**行进中坐 (Sit in motion) — 10 分**：
- 1st part（50%）：起步、执行"坐"
- 2nd part（50%）：离开犬、返回、基础位置
- 建 10-15 步，犬立即坐，训导员不改变步态/不回头
- 若犬站/卧：扣该科目 50% 分

**行进中卧+召回 (Down with recall) — 10 分**：
- IGP3 特殊：常步 10-15 步后再跑 10-15 步才下达"卧"
- 犬立即卧，训导员走≥30 步转身，法官指令召回
- 犬立即愉悦召回，近坐前方，3 秒后回基础位置
- 若犬坐/站：强制扣 50%；二次额外口令不召回 = Insufficient 0 分

**干扰下卧 (Down under distraction) — 10 分**：
- 另一犬作业时，本犬卧在不分心位置
- 距离递增：IGP1 10m 可见侧身 / IGP2 20m 背对 / IGP3 30m 不可见
- 离开卧位>3m：扣 50% + 其他错误；3m 内坐/站：扣 50%
- IGP1 第 3 项后、IGP2 第 4 项后、IGP3 第 5 项后才给部分分

**前进并卧 (Send out with down) — 10 分**：
- 犬直线前进，远距离下达"卧"
- 不能 3 口令停止 = 0 分
- 破卧（被召回时）：50% 距离内可停 = 扣至 -5 分；>50% 距离 = 0 分

**衔取 (Retrieve)**：
- 哑铃重量 IGP1：650 克
- 第三口令不吐哑铃 = **取消资格（不服从）**

#### 2.2.3 评估原则（主次项区分）
- **主要项 (Primary)**：科目核心执行（如坐的速度、卧的直接性）—— 权重高
- **次要项 (Secondary)**：基础位置、起步结束 —— 权重低
- 次要项严重错误可导致整科 Insufficient 或 DQ

#### 2.2.4 自动化可行性：⭐⭐⭐⭐⭐（最高，~70%）
- 全部 9 科可视觉判定：姿态（坐/卧/立）、位置（随行偏移）、衔取（口含物品）、跳跃轨迹、距离保持

### 2.3 护卫科目（Phase C - Schutzdienst）— 100 分

#### 2.3.1 分值表（官方 Examination levels FCI-IGP-1 to IGP-3）

| # | 科目 | IGP 1 | IGP 2 | IGP 3 |
|---|------|-------|-------|-------|
| 1 | 搜索助训员 (Search for the helper) | 5 | 5 | 10 |
| 2 | 看守与吠叫 (Hold and Bark) | 15 | 15 | 15 |
| 3 | 阻止逃跑 (Prevention of attempted escape) | 20 | 15 | 10 |
| 4 | 防守攻击（看守后）(Defense against attack from guarding) | 30 | 20 | 15 |
| 5 | 押解 (Back Transport) | — | 5 | 5 |
| 6 | 押解中攻击 (Attack on dog out of back transport) | — | — | 15 |
| 7 | 运动中攻击 (Attack on dog out of motion) | 30 | 20 | 15 |
| 8 | 防守攻击（看守后，第二次）(Defense against attack from guarding) | — | 20 | 15 |
| | **合计** | **100** | **100** | **100** |

> 注：IGP1 仅 5 项（无押解相关），IGP2 增至 7 项，IGP3 达 8 项（最全）。

#### 2.3.2 关键科目细则

**搜索助训员 (Search for the helper)**：
- 掩体数：IGP1 = 2 个 / IGP2 = 4 个 / IGP3 = 6 个
- 犬快速直接奔向掩体，紧密绕行，召回后送下一掩体
- 训导员走中线，不得离开
- 犬回基础位中断搜索 = 0 分；二次回基础位 = 终止护卫
- 3 次口令未找到 = 终止护卫

**看守与吠叫 (Hold and Bark) — 15 分**：
- 分值拆分：**看守 10 分 + 吠叫 5 分**
- 犬自信、主动、专注、强势地对助训员持续吠叫约 **20 秒**
- 犬不得碰/撞助训员；咬袖不主动放 = Insufficient -14 分
- 离开助训员 = 终止；咬袖不放口或不回基础位 = **DQ**

**阻止逃跑 (Prevention of attempted escape)**：
- 助训员逃跑 20 步，犬必须扑咬阻止
- 评估：专注、速度、正确扑咬、放口

**放口 (Out/Release)** — 护卫核心控制指标：
- 犬必须主动放口（释放袖套）
- 不放口 = 取消资格（DQ）
- 这是机器视觉可判定的关键控制行为（口部脱离袖套）

**押解 (Back Transport)**：
- IGP2/3 才有，犬在助训员后方跟随押解
- IGP3 增加押解中对犬的攻击

#### 2.3.3 护卫扣分评级示例

| 行为 | 评级/扣分 |
|------|----------|
| 弱/不持续/不强势/不专注吠叫 | Satisfactory → Insufficient |
| 不吠但主动看守 | Insufficient |
| 推撞助训员 | 相应扣分至 Insufficient |
| 咬袖且需走近+口令才放 | Insufficient -14 分 |
| 离开助训员（训导员离中线后） | Insufficient -14 |
| 不放口/不回基础位 | **DQ** |

#### 2.3.4 自动化可行性：⭐⭐（~20%，最难）
- 可视觉判定：搜索路径、吠叫（需音频辅助）、看守姿态、押解位置
- 不可视觉：咬合力、扑咬力度、勇气主观评估

---

## 3. 评分规则总结

### 3.1 总分与及格

| 维度 | 规则 |
|------|------|
| 总分 | 300 分（A 100 + B 100 + C 100） |
| 阶段及格 | 每阶段 ≥ 70 分（Satisfactory 70%） |
| 总评及格 | ≥ 210 分（Satisfactory） |
| 最高评级 | Excellent（286-300） |

### 3.2 取消资格（Disqualification, DQ）情形
1. **枪怯 (Gun-shy)**：枪声测试中犬表现怯枪 → DQ + 已得分清零
2. **不放口 (No release)**：护卫中咬袖不放 → DQ
3. **不服从 (Disobedience)**：如衔取第三口令不吐哑铃 → DQ
4. **离开助训员**：特定条件下离开 → DQ

### 3.3 终止 (Termination) 情形
- 追踪：超时、离开迹线
- 护卫：3 次口令未找到助训员、二次回基础位

### 3.4 扣分精度
- **0.5 分**为最小扣分单位（区别于中国 GA/T 的整数分）
- 单项可给部分分；阶段总分取整

### 3.5 主次项评估原则
FCI-IGP 独创的评估方法：
- **主要项 (Primary)**：科目核心（如坐的速度）—— 权重高
- **次要项 (Secondary)**：基础位置、起步 —— 权重低
- 次要项严重错误可拉低整科至 Insufficient

> 💡 **对项目的启示**：本项目 7 维评分可借鉴此主次项原则——为每个 IGP 科目定义 primary 信号（如坐的速度→延迟）和 secondary 信号（如基础位置→准确度），primary 权重高。

---

## 4. 与项目 7 维评分映射

项目 7 维：准确度 / 延迟 / 保持 / 搜索效率 / 注意力 / 胆量 / 步态

### 4.1 映射矩阵

| 项目维度 | FCI-IGP 对应评估点 | 对应阶段 | 权重建议 | 信号来源 |
|---------|-------------------|---------|---------|---------|
| **准确度 (accuracy)** | 科目执行正确性、位置精度、物品指示、随行位置 | A/B/C 全 | 0.25 | action_correct/action_count、姿态偏差、物品指示命中率 |
| **延迟 (latency)** | 口令响应即时性（"immediately, without hesitation"）、执行速度 | B 主 | 0.15 | command_to_action_latency（FCI 强调即时响应） |
| **保持 (duration)** | 干扰下卧、看守吠叫 20 秒、姿态保持、卧不破 | A/B/C | 0.15 | action_duration（干扰下卧 IGP3 达 30m 不可见） |
| **搜索效率 (search_efficiency)** | 迹线跟随（鼻工作）、掩体搜索覆盖、搜索速度 | A/C | 0.15 | search_coverage、search_speed、track_follow_accuracy |
| **注意力 (attention)** | 随行专注、枪声测试、对训导员/助训员专注、无分心 | B/C | 0.10 | focus_ratio、gunshot_reaction（FCI 枪怯=DQ） |
| **胆量 (courage)** | 护卫自信、防守攻击、对助训员强势、无恐惧/压力表现 | C 主 | 0.10 | courage_score（扑咬速度、看守强势度、无回避） |
| **步态 (gait)** | 随行步态、运动步态、跳跃/攀墙技术、步态变换 | B 主 | 0.10 | gait_symmetry、jump_trajectory、pace_change_smoothness |

### 4.2 维度新增/强化的理由

相比项目 Phase 2 的 USPCA 5 维（准确度/延迟/保持/搜索效率/注意力），FCI-IGP 需要**新增 2 维**：

1. **胆量 (courage)**：FCI-IGP Phase C 护卫科目的核心评估点（"confident, active, attentive, dominant"），官方明确将"恐惧/压力/回避"列为扣分项。USPCA 也有扑咬但未独立成维。
2. **步态 (gait)**：FCI-IGP Phase B 随行含三种步态变换（正常/跑/慢）+ 跳跃 + 攀墙，步态质量是独立评估点。

### 4.3 IGP 特有强约束（映射到维度阈值）

| IGP 规则 | 项目维度 | 阈值建议 |
|---------|---------|---------|
| 枪怯 = DQ | attention | gunshot_reaction=fail → 整段评分强制 fail（DQ 模拟） |
| 不放口 = DQ | accuracy/duration | release_command_fail → 强制 fail |
| 衔取 3 口令不吐 = DQ | accuracy | retrieve_release_fail → 强制 fail |
| 排尿/排便 = -8 | attention | excretion_detected → duration/attention 重扣 |
| 离开卧位>3m = -50% | duration | down_break_distance > 3m → duration 扣 50% |

---

## 5. 与项目 22 行为映射

项目 22 行为：P0 基础 8 / P1 训练 8 / P2 高级 6

### 5.1 P0 基础行为（8）→ IGP 科目

| 项目行为 | IGP 科目 | IGP 分值 | 映射维度 | 自动化 |
|---------|---------|---------|---------|--------|
| sit（坐） | Sit in motion（行进中坐） | 10 | accuracy/latency | ⭐⭐⭐⭐⭐ |
| down（卧） | Down with recall + Down under distraction | 10+10 | accuracy/duration | ⭐⭐⭐⭐⭐ |
| stand（立） | Stand exercise（行进中立） | 10 (IGP2/3) | accuracy/latency | ⭐⭐⭐⭐⭐ |
| heel（随行） | Free heeling | 15 | accuracy/attention/gait | ⭐⭐⭐⭐ |
| sit_up（坐立） | 基础位置相关（非独立科目） | — | accuracy | ⭐⭐⭐⭐ |
| stay（停留） | Down under distraction | 10 | duration/attention | ⭐⭐⭐⭐⭐ |
| bark（叫） | Hold and Bark（吠叫部分） | 5 (of 15) | accuracy/attention | ⭐⭐⭐⭐（需音频） |
| bite（咬） | Protection grip（扑咬/防守） | 10-30 | accuracy/courage | ⭐⭐⭐ |

### 5.2 P1 训练专项行为（8）→ IGP 科目

| 项目行为 | IGP 科目 | IGP 分值 | 映射维度 | 自动化 |
|---------|---------|---------|---------|--------|
| track（追踪） | Phase A Tracking（迹线跟随） | 79 | search_efficiency/attention | ⭐⭐⭐ |
| alert_sit（示警坐） | Article indication（物品指示-坐） | 3×7=21 | accuracy/latency | ⭐⭐⭐⭐⭐ |
| alert_down（示警卧） | Article indication（物品指示-卧） | 3×7=21 | accuracy/duration | ⭐⭐⭐⭐⭐ |
| apprehend（扑咬） | Attack on dog out of motion + Defense | 15-30 | accuracy/courage | ⭐⭐ |
| escort（押解） | Back Transport（押解） | 5 (IGP2/3) | attention/duration | ⭐⭐⭐ |
| obstacle（障碍穿越） | Hurdle（跨栏）+ Scaling wall（攀墙） | 15+10~15 | accuracy/gait | ⭐⭐⭐⭐ |
| recall（返回） | Down with recall + Stand recall | 10 | accuracy/latency | ⭐⭐⭐⭐⭐ |
| watch（警戒） | Hold and Bark（看守部分）+ Guarding | 10 (of 15) | attention/duration | ⭐⭐⭐⭐ |

### 5.3 P2 高级行为（6）→ IGP 科目

| 项目行为 | IGP 对应 | 映射维度 | 说明 |
|---------|---------|---------|------|
| food_drive（食物欲望） | 通用评估"motivated work" | courage | FCI 未独立科目，但"工作意愿"是评估通则 |
| courage（胆量测试） | Phase C 全阶段（confident/dominant） | courage | **FCI-IGP 核心评估点**，尤其 IGP3 双攻击 |
| attention（注意力测试） | 随行专注 + 枪声测试 | attention | 枪声测试是 FCI 独有硬约束 |
| search_eff（搜索效率） | 追踪效率 + 掩体搜索 | search_efficiency | 覆盖率+速度+鼻工作强度 |
| impulse_ctrl（冲动控制） | 干扰下卧 + 放口 + 前进卧 | duration/accuracy | 放口(Out)是 IGP 核心控制指标 |
| gait（步态分析） | 随行步态变换 + 跳跃技术 | gait | 三种步态+跳跃/攀墙 |

### 5.4 行为映射覆盖率分析

| 项目行为集 | 总数 | IGP 直接对应 | 部分对应 | 覆盖率 |
|-----------|------|-------------|---------|--------|
| P0 基础 | 8 | 7 | 1（sit_up） | 100% |
| P1 训练 | 8 | 8 | 0 | 100% |
| P2 高级 | 6 | 4 | 2（food_drive/gait 部分） | 100% |
| **合计** | **22** | **19** | **3** | **100%** |

> ✅ **结论**：项目 22 行为与 FCI-IGP 三阶段科目**完全覆盖**，无需新增行为类别。IGP 的优势在于提供了**三等级难度梯度**（IGP1/2/3），可复用现有行为，仅调整距离/时长/复杂度参数。

---

## 6. IGP 评分卡 YAML 设计建议

参考 `backend/ml/scoring/configs/uspca_patrol.yaml` 结构，设计 `fci_igp.yaml`。

### 6.1 设计要点
1. **7 维评分**（比 USPCA 5 维多胆量+步态）
2. **3 等级 × 3 阶段**结构（IGP1/2/3 × Phase A/B/C）
3. **DQ 硬约束**模拟（枪怯/不放口/不服从 → 强制 fail）
4. **主次项**评估（primary/secondary 信号权重）
5. **0.5 分扣分精度**

### 6.2 YAML 骨架建议

```yaml
# FCI-IGP 国际工作犬评分卡（7 维）
# Owner: ML 开发（见 AGENTS.md §2.2）
# Phase: 3.4a
# 依据: dev-docs/research/RESEARCH_FCI_IGP_STANDARD.md
# 标准来源: FCI International Utility Dog Regulations 2025 (fci.be)
# 一手源: UTI-REG-IGP-en-2025-22401.pdf (82 页, 2025-01-01 生效)
#
# FCI-IGP 分值结构（参考用，本评分卡归一化为 0-100 分）:
#   Phase A 追踪 (Tracking)    100 pts  可自动化 40%  → 搜索效率 + 注意力
#   Phase B 服从 (Obedience)   100 pts  可自动化 70%  → 准确度 + 延迟 + 保持 + 步态
#   Phase C 护卫 (Protection)  100 pts  可自动化 20%  → 胆量 + 注意力 + 准确度
#   总分 300 pts  及格 210 (70%/阶段) → 本卡 pass=70（对齐 FCI Satisfactory 70%）
#
# 7 维评分：准确度（0.25）+ 延迟（0.15）+ 保持（0.15）+ 搜索效率（0.15）+
#           注意力（0.10）+ 胆量（0.10）+ 步态（0.10）
#
# FCI-IGP 扣分/DQ 规则映射（RESEARCH_FCI_IGP_STANDARD.md §3）:
#   - 枪怯 (gun-shy): DQ → attention gunfire_fail → 强制 fail
#   - 不放口 (no release): DQ → accuracy release_fail → 强制 fail
#   - 衔取 3 口令不吐: DQ → accuracy retrieve_fail → 强制 fail
#   - 排尿/排便: -8 分 → attention excretion → 重扣
#   - 离开卧位>3m: -50% → duration down_break_distance>3m → 扣 50%
#   - 误指示物品: -2~-4 分 → accuracy false_alert
#
# 评级映射（300 分制 → 100 分制）:
#   Excellent   286-300 (96%+)  → 本卡 90-100
#   Very Good   270-285 (90%+)  → 本卡 80-89
#   Good        240-269 (80%+)  → 本卡 70-79
#   Satisfactory 210-239 (70%+) → 本卡 60-69（及格线）
#   Insufficient 0-209 (<70%)   → 本卡 0-59

scoring_engine:
  name: "FCI-IGP 国际工作犬评分卡"
  version: "1.0.0"
  scene: "fci_igp"
  description: |
    FCI-IGP 国际工作犬 7 维评分（对齐 FCI 2025 国际工作犬规则）：
    准确度（0.25）+ 延迟（0.15）+ 保持（0.15）+ 搜索效率（0.15）+
    注意力（0.10）+ 胆量（0.10）+ 步态（0.10）
    依据: RESEARCH_FCI_IGP_STANDARD.md
    及格线: 70 分（对齐 FCI Satisfactory 70%）

  # IGP 等级（影响距离/时长/复杂度参数）
  igp_level: "IGP1"  # IGP1 | IGP2 | IGP3

  # DQ 硬约束（任一触发 → 整段评分强制 fail）
  disqualifications:
    - id: gunfire_fail
      condition: "gunshot_reaction == 'shy'"
      action: "force_fail"
      label: "枪怯（FCI DQ）"
    - id: release_fail
      condition: "release_command_count >= 2 and not sleeve_released"
      action: "force_fail"
      label: "不放口（FCI DQ）"
    - id: retrieve_fail
      condition: "retrieve_release_command_count >= 3 and not dumbbell_released"
      action: "force_fail"
      label: "衔取不吐（FCI DQ）"

  dimensions:
    # === 准确度（0.25，贯穿 A/B/C，最高权重）===
    # FCI: 科目执行正确性、位置精度、物品指示、主次项原则
    - id: accuracy
      name: "准确度"
      weight: 0.25
      rules:
        - id: accuracy_excellent
          condition: "action_correct / action_count > 0.96"  # 对齐 FCI Excellent 96%
          score: 95
          label: "优秀"
        - id: accuracy_good
          condition: "action_correct / action_count > 0.80"  # 对齐 FCI Good 80%
          score: 75
          label: "良好"
        - id: accuracy_satisfactory
          condition: "action_correct / action_count > 0.70"  # 对齐 FCI Satisfactory 70%
          score: 65
          label: "合格"
        - id: accuracy_insufficient
          condition: "action_correct / action_count <= 0.70"  # 对齐 FCI Insufficient <70%
          score: 30
          label: "不合格"
        - id: accuracy_default
          condition: "True"
          score: 40
          label: "待评估"

    # === 延迟（0.15，Phase B 核心，FCI 强调"immediately, without hesitation"）===
    - id: latency
      name: "延迟"
      weight: 0.15
      rules:
        - id: latency_excellent
          condition: "command_to_action_latency < 0.5"
          score: 95
          label: "优秀"
        - id: latency_good
          condition: "command_to_action_latency < 1.5"
          score: 75
          label: "良好"
        - id: latency_satisfactory
          condition: "command_to_action_latency < 3.0"  # 对齐 USPCA 3 秒及格线
          score: 65
          label: "合格"
        - id: latency_insufficient
          condition: "command_to_action_latency >= 3.0"
          score: 30
          label: "不合格"
        - id: latency_default
          condition: "True"
          score: 40
          label: "待评估"

    # === 保持（0.15，干扰下卧/看守吠叫/姿态保持）===
    # FCI: 干扰下卧 IGP3 达 30m 不可见；看守吠叫约 20 秒
    - id: duration
      name: "保持"
      weight: 0.15
      rules:
        - id: duration_excellent
          condition: "action_duration > 30.0"
          score: 95
          label: "优秀"
        - id: duration_good
          condition: "action_duration > 10.0"
          score: 75
          label: "良好"
        - id: duration_satisfactory
          condition: "action_duration > 5.0"
          score: 65
          label: "合格"
        - id: duration_insufficient
          condition: "action_duration <= 5.0"
          score: 30
          label: "不合格"
        - id: duration_down_break
          # FCI: 离开卧位>3m 扣 50%
          condition: "down_break_distance > 3.0"
          score: 32
          label: "破卧（FCI -50%）"
        - id: duration_default
          condition: "True"
          score: 40
          label: "待评估"

    # === 搜索效率（0.15，Phase A 追踪 + Phase C 掩体搜索）===
    - id: search_efficiency
      name: "搜索效率"
      weight: 0.15
      rules:
        - id: search_excellent
          condition: "search_coverage > 0.85 and search_speed > 0.5 and target_found"
          score: 95
          label: "优秀"
        - id: search_good
          condition: "search_coverage > 0.60 and search_speed > 0.3"
          score: 75
          label: "良好"
        - id: search_satisfactory
          condition: "search_coverage > 0.40 or target_found"
          score: 65
          label: "合格"
        - id: search_insufficient
          condition: "search_coverage <= 0.40 and not target_found"
          score: 30
          label: "不合格"
        - id: search_default
          condition: "True"
          score: 40
          label: "待评估"

    # === 注意力（0.10，随行专注 + 枪声测试 + 无分心）===
    # FCI: 枪怯 = DQ（见 disqualifications）
    - id: attention
      name: "注意力"
      weight: 0.10
      rules:
        - id: attention_high
          condition: "focus_ratio > 0.80 and unnecessary_movement_count < 2"
          score: 95
          label: "高"
        - id: attention_medium
          condition: "focus_ratio > 0.50"
          score: 70
          label: "中"
        - id: attention_low
          condition: "focus_ratio <= 0.50"
          score: 30
          label: "低"
        - id: attention_excretion
          # FCI: 排尿/排便 -8 分（追踪阶段）
          condition: "excretion_detected"
          score: 20
          label: "排泄（FCI -8）"
        - id: attention_default
          condition: "True"
          score: 40
          label: "待评估"

    # === 胆量（0.10，Phase C 护卫核心，FCI 独有强调）===
    # FCI: "confident, active, attentive, dominant"；恐惧/压力/回避 = 扣分
    - id: courage
      name: "胆量"
      weight: 0.10
      rules:
        - id: courage_excellent
          condition: "courage_score > 0.90 and not avoidance_detected"
          score: 95
          label: "优秀"
        - id: courage_good
          condition: "courage_score > 0.70 and not avoidance_detected"
          score: 75
          label: "良好"
        - id: courage_satisfactory
          condition: "courage_score > 0.50"
          score: 65
          label: "合格"
        - id: courage_insufficient
          condition: "courage_score <= 0.50 or avoidance_detected"
          score: 30
          label: "不合格"
        - id: courage_default
          condition: "True"
          score: 40
          label: "待评估"

    # === 步态（0.10，随行步态变换 + 跳跃/攀墙技术）===
    # FCI: 三种步态（正常/跑/慢）+ 跨栏 + 攀墙 160cm
    - id: gait
      name: "步态"
      weight: 0.10
      rules:
        - id: gait_excellent
          condition: "gait_symmetry > 0.90 and pace_change_smoothness > 0.85"
          score: 95
          label: "优秀"
        - id: gait_good
          condition: "gait_symmetry > 0.70"
          score: 75
          label: "良好"
        - id: gait_satisfactory
          condition: "gait_symmetry > 0.50"
          score: 65
          label: "合格"
        - id: gait_insufficient
          condition: "gait_symmetry <= 0.50"
          score: 30
          label: "不合格"
        - id: gait_default
          condition: "True"
          score: 40
          label: "待评估"

  thresholds:
    pass: 70        # FCI Satisfactory 70%
    borderline: 60  # 基本合格（需补测）
    fail: 0         # 不合格

  aggregation: "weighted_sum"

  # === 22 行为映射（按 IGP 三阶段分组）===
  behavior_mapping:
    # --- Phase A 追踪阶段 ---
    track:
      dimensions: [search_efficiency, attention]
      weight: 1.5
      label: "追踪"
      igp_phase: "A"
    alert_sit:
      dimensions: [accuracy, latency]
      weight: 1.2
      label: "物品指示-坐"
      igp_phase: "A"
    alert_down:
      dimensions: [accuracy, duration]
      weight: 1.2
      label: "物品指示-卧"
      igp_phase: "A"

    # --- Phase B 服从阶段 ---
    sit:
      dimensions: [accuracy, latency]
      weight: 1.0
      label: "坐"
      igp_phase: "B"
    down:
      dimensions: [accuracy, duration]
      weight: 1.0
      label: "卧"
      igp_phase: "B"
    stand:
      dimensions: [accuracy, latency]
      weight: 1.0
      label: "立"
      igp_phase: "B"
    heel:
      dimensions: [accuracy, attention, gait]
      weight: 1.3
      label: "随行"
      igp_phase: "B"
    stay:
      dimensions: [duration, attention]
      weight: 1.2
      label: "干扰下卧"
      igp_phase: "B"
    recall:
      dimensions: [accuracy, latency]
      weight: 1.0
      label: "召回"
      igp_phase: "B"
    obstacle:
      dimensions: [accuracy, gait]
      weight: 1.0
      label: "跨栏/攀墙"
      igp_phase: "B"
    # sit_up: 基础位置相关，非独立 IGP 科目
    sit_up:
      dimensions: [accuracy]
      weight: 0.5
      label: "坐立（基础位置）"
      igp_phase: "B"

    # --- Phase C 护卫阶段 ---
    bark:
      dimensions: [accuracy, attention]
      weight: 1.0
      label: "看守吠叫"
      igp_phase: "C"
    watch:
      dimensions: [attention, duration]
      weight: 1.0
      label: "看守"
      igp_phase: "C"
    bite:
      dimensions: [accuracy, courage]
      weight: 1.3
      label: "扑咬"
      igp_phase: "C"
    apprehend:
      dimensions: [accuracy, courage]
      weight: 1.5
      label: "运动中攻击"
      igp_phase: "C"
    escort:
      dimensions: [attention, duration]
      weight: 1.0
      label: "押解"
      igp_phase: "C"

    # --- P2 高级（跨阶段评估）---
    food_drive:
      dimensions: [courage]
      weight: 0.8
      label: "工作意愿"
      igp_phase: "ALL"
    courage:
      dimensions: [courage]
      weight: 1.5
      label: "胆量"
      igp_phase: "C"
    attention:
      dimensions: [attention]
      weight: 1.0
      label: "注意力"
      igp_phase: "ALL"
    search_eff:
      dimensions: [search_efficiency]
      weight: 1.2
      label: "搜索效率"
      igp_phase: "A"
    impulse_ctrl:
      dimensions: [duration, accuracy]
      weight: 1.0
      label: "冲动控制（放口）"
      igp_phase: "C"
    gait:
      dimensions: [gait]
      weight: 1.0
      label: "步态"
      igp_phase: "B"
```

### 6.3 信号字段补充建议

FCI-IGP 相比 USPCA 需新增的信号字段（`rule_engine.py` signals）：

```python
# FCI-IGP 新增信号
gunshot_reaction: str        # "indifferent" | "shy" | "unclear"  枪声反应（DQ 判定）
sleeve_released: bool        # 是否放口（护卫 DQ 判定）
release_command_count: int   # 放口口令次数
dumbbell_released: bool      # 衔取是否吐哑铃（DQ 判定）
retrieve_release_command_count: int  # 衔取吐出口令次数
down_break_distance: float   # 破卧距离（米，FCI 3m 阈值）
excretion_detected: bool     # 是否排尿/排便（FCI -8）
courage_score: float         # 胆量评分 0-1（扑咬速度+看守强势度-回避）
avoidance_detected: bool     # 是否检测到回避行为
gait_symmetry: float         # 步态对称性 0-1
pace_change_smoothness: float  # 步态变换平滑度 0-1
igp_phase: str               # "A" | "B" | "C"  当前阶段
igp_level: str               # "IGP1" | "IGP2" | "IGP3"  等级
```

---

## 7. 与 USPCA 对比

### 7.1 结构对比

| 维度 | USPCA PDI | FCI-IGP |
|------|-----------|---------|
| 管理机构 | USPCA（美国警犬协会） | FCI（世界犬业联盟） |
| 总分制 | 700 分（服从120+搜索140+扑咬340，及格448=70%） | 300 分（追踪100+服从100+护卫100，及格210=70%） |
| 阶段及格 | 总分及格（无单阶段硬约束） | **每阶段必须独立 ≥70%**（更严格） |
| 评级精度 | 0.5 分 | 0.5 分 |
| 评级体系 | Pass/Fail（70%） | 5 级（Excellent/Very Good/Good/Satisfactory/Insufficient） |
| 追踪科目 | 无独立追踪（搜索含建筑物/车辆） | **独立 Phase A**（鼻工作，IGP3 达 600 步/60 分钟） |
| 服从科目 | 7 项（随行/坐/卧/立/召回/远距离/跳跃） | 9 项（含攀墙、前进卧、干扰下卧） |
| 护卫/扑咬 | 340 分（占比 49%，最重） | 100 分（占比 33%，均衡） |
| 标准化程度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐（最详尽，82 页规则） |
| 实战导向 | ⭐⭐⭐⭐⭐（警犬实战） | ⭐⭐⭐（运动竞技导向） |
| 国际认可 | 美国法院系统 | 全球 FCI 成员国 |

### 7.2 评分维度对比

| 项目维度 | USPCA 5 维 | FCI-IGP 7 维 | 差异 |
|---------|-----------|-------------|------|
| 准确度 | ✅ 0.30 | ✅ 0.25 | FCI 权重略降（因维度增多） |
| 延迟 | ✅ 0.20 | ✅ 0.15 | FCI 权重略降 |
| 保持 | ✅ 0.20 | ✅ 0.15 | FCI 权重略降 |
| 搜索效率 | ✅ 0.15 | ✅ 0.15 | 一致 |
| 注意力 | ✅ 0.15 | ✅ 0.10 | FCI 权重略降 |
| 胆量 | ❌ | ✅ 0.10 | **FCI 新增**（护卫核心） |
| 步态 | ❌ | ✅ 0.10 | **FCI 新增**（服从步态+跳跃） |

### 7.3 DQ 硬约束对比

| DQ 情形 | USPCA | FCI-IGP |
|---------|-------|---------|
| 枪怯 | 无明确 DQ | **DQ + 清零** |
| 不放口 | 扣分 | **DQ** |
| 衔取不吐 | 无 | **DQ** |
| 阶段不及格 | 总分补 | **整段不合格** |

### 7.4 共同点
1. 70% 及格线一致
2. 0.5 分扣分精度一致
3. 坐/卧/立/随行/召回/扑咬为共同科目
4. 均要求枪声测试（USPCA 也有，但 FCI 更严格=DQ）
5. 均有搜索科目（USPCA 建筑物/车辆 vs FCI 追踪+掩体）

### 7.5 自动化可行性对比

| 阶段 | USPCA 可自动化 | FCI-IGP 可自动化 |
|------|---------------|-----------------|
| 服从 | 60% | **70%**（FCI 更标准化，9 项全可视觉判定） |
| 搜索 | 70% | 40%（FCI 追踪需 GPS+姿态融合） |
| 扑咬/护卫 | 25% | 20%（均需力传感器/主观勇气评估） |
| **总体** | ~42% | ~50%（FCI 服从占比高且可自动化高） |

---

## 8. 推荐方案

### 8.1 核心结论

1. **FCI-IGP 官方 2025 PDF 已获取并完整解析（82 页）**，是本项目 Phase 3 FCI-IGP 评分卡实现的权威一手依据，无需依赖二手解读。
2. **项目 22 行为与 FCI-IGP 三阶段科目 100% 覆盖**，无需新增行为类别，仅需调整距离/时长/复杂度参数以支持 IGP1/2/3 三等级。
3. **需从 USPCA 5 维扩展到 7 维**（新增胆量 courage + 步态 gait），这是 FCI-IGP Phase C 护卫和 Phase B 服从步态的硬需求。
4. **FCI-IGP 比 USPCA 更标准化**（82 页详尽规则 vs USPCA 分散评分表），自动化友好度更高（服从 70% vs 60%）。

### 8.2 实施建议（Phase 3.4a → 3.4b）

| 步骤 | 内容 | 优先级 |
|------|------|--------|
| 1 | 创建 `backend/ml/scoring/configs/fci_igp.yaml`（按 §6 骨架） | 高 |
| 2 | `rule_engine.py` 新增 FCI-IGP 信号字段（§6.3） | 高 |
| 3 | 评分引擎支持 DQ 硬约束（force_fail 逻辑） | 高 |
| 4 | 评分引擎支持 7 维聚合（胆量+步态新维度） | 高 |
| 5 | 评分引擎支持 IGP 等级参数切换（IGP1/2/3 距离/时长） | 中 |
| 6 | 评分引擎支持阶段独立及格判定（A/B/C 各 ≥70%） | 中 |
| 7 | 端到端验证：用 IGP 服从视频验证 7 维评分 | 中 |

### 8.3 风险与限制

| 风险 | 影响 | 缓解 |
|------|------|------|
| 胆量/步态维度视觉信号提取难度 | 评分精度 | Phase 3 先用代理信号（扑咬速度代理胆量；步态对称性代理步态） |
| 护卫科目 DQ 判定（不放口）依赖清晰口部检测 | DQ 准确率 | 需高质量近距视频；Phase 3 仅服从阶段全自动化，护卫阶段半自动 |
| 追踪科目需 GPS+姿态融合 | 追踪自动化率 40% | Phase 3 追踪阶段保留人工评分接口 |
| fci.be SSL 证书过期 | 后续 PDF 再获取 | 已下载本地存档；建议项目内备份 PDF |

### 8.4 与 Phase 2 USPCA 的复用

| 复用项 | 来源 | 复用方式 |
|--------|------|---------|
| 评分引擎框架 | `uspca_patrol.yaml` | 复用 dimensions/rules/thresholds 结构 |
| 5 维评分逻辑 | `rule_engine.py` | 准确度/延迟/保持/搜索效率/注意力 直接复用 |
| 16 行为映射 | USPCA behavior_mapping | 扩展为 22 行为 + igp_phase 字段 |
| 扣分阈值 | USPCA 3 秒/5 秒 | FCI 对齐（部分复用） |

### 8.5 不建议的事项
- ❌ 不建议新增行为类别（22 行为已 100% 覆盖 IGP）
- ❌ 不建议为 IGP 单独重建评分引擎（应复用 USPCA 框架扩展）
- ❌ 不建议 Phase 3 全自动化护卫科目（DQ 判定需人工兜底）
- ❌ 不建议依赖 GitHub 上的 dog behavior recognition 小仓库（≤4★，质量不足）

---

## 9. 参考

### 9.1 一手源（本次调研直接获取）
1. **FCI 官方 2025 IGP 规则 PDF**（权威源，82 页）
   - URL: https://fci.be/medias/UTI-REG-IGP-en-2025-22401.pdf
   - 大小: 2,434,431 bytes (2.4 MB)
   - 批准: FCI General Committee 2024-09-03
   - 生效: 2025-01-01
   - 获取方式: requests + verify=False（fci.be SSL 证书过期）+ pymupdf 解析
   - 解析全文: 270,990 字符

### 9.2 GitHub Directory-First 调研结果
2. **sindresorhus/awesome** 元目录
   - URL: https://raw.githubusercontent.com/sindresorhus/awesome/main/readme.md
   - 扫描结果: 79,604 字符，**无 dog/working-dog/canine 相关条目**（awesome-working-dog / awesome-dog-sports 目录不存在）
3. **GitHub Search API 查询**:
   - `FCI-IGP`: 0 仓库（无专门 FCI-IGP 规则开源实现）
   - `schutzhund IPO rules`: 0 仓库
   - `dog behavior recognition`: 7 仓库，top 候选:
     - `rjdbcm/BEAGLES`（4★，2023-02，BEhavioral Annotation and Gesture Learning Suite，犬行为标注工具，README 已验证）
     - `Ekta729/Dog-Behavior-Recognition`（3★，2021-11，犬行为识别，README 已验证）
   - 结论: GitHub 无 FCI-IGP 标准的开源实现或评分系统，dog behavior recognition 仓库规模小（≤4★），仅与项目 ML 侧间接相关

### 9.3 项目内既有调研（交叉验证）
4. `dev-docs/research/RESEARCH_STANDARDS.md` v1.1 §2.3 FCI-IGP 详解（与官方 PDF 一致）
5. `dev-docs/research/工作犬标准自动化深度调研报告.md` §5 FCI-IGP 标准详解（与官方 PDF 一致）
6. `backend/ml/scoring/configs/uspca_patrol.yaml`（YAML 结构参考）

### 9.4 调研工具与合规
- **github-search-strategy skill**: Directory-First 流程（sindresorhus/awesome → 搜索 → 验证）
- **browser-automation skill**: Playwright（已装）+ requests（已装）+ pymupdf（已装）
- **AGENTS.md §1.3 合规**: 全程未使用 WebSearch；GitHub 调研 + FCI 官方 PDF 直接抓取
- 临时脚本: `scripts/_research_fci_igp.py`（调研后按 AGENTS.md §1.3「清理临时脚本」要求清理）
- 中间产物: `_fci_research_raw.json`、`_fci_pdf_text.txt`（调研后清理）
