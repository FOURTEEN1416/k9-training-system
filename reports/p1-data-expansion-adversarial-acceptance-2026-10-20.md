# 数据扩容三线推进 · 对抗验收报告（2026-10-20）

> 验收对象：工作流「数据扩容三线推进（零预算）」的三线交付（数据集补充 / Animal-CLIP 零样本 / 合成生产）
> 验收方法：`adversarial-acceptance` 技能五步；立场：验收人不是证明人
> 执行人：主代理（本窗，对全部交付物做了独立核验，未复用工作流自证结论）

---

## 1. 逐项检查结果（证据表）

| # | 步骤 | 预期 | 实际 | 证据 |
|---|------|------|------|------|
| 1 | 独立复跑三验证脚本 | 全部 exit=0 | **三个均 exit=0**（不信任工作流自报，重新执行） | `verify_dataset_expansion`(31 PASS)/`verify_animclip_eval`(38 PASS)/`verify_synth_line`(11 PASS)，输出末行 `ALL CHECKS PASSED` |
| 2 | MammalNet 落盘实物核验 | 752 文件可解可对应标注 | **752 mp4 全部存在**，随机抽 3 个 ffprobe 可解码（12.00s/1.02s/21.01s）；字节数 1,542,535,302 与 summary.json 逐位吻合；`subset_annotations.json` 709 clip 各带 behavior 标签与 segment | `find datasets/mammalnet_canine -name "*.mp4" \| wc -l` = 752；ffprobe 抽样输出 |
| 3 | **行为标签域匹配度**（审查标签是不是能用的标签） | 与 22 类对齐 | **❌ 域不符（致命限制，已核验）** | 709 clip 标签分布：sleeps 260 / fights 99 / nurses 84 / drinks 80 / hunts 66 / eats 60 / mates 44 / pees 27 / poops 25 / gives birth 13 / vomits 5 / grooms 2——**全部是宠物日常行为，22 类工作犬考核行为（坐/卧/立/前来/随行/吠叫/衔取/搜索等）零覆盖** |
| 4 | Animal-CLIP partial 诚实性 | 不伪造精度数字 | **如实**：权重不可得 → Animal-CLIP 列留空，仅出 OpenAI CLIP ViT-B/16 对照分布（208 段，label_status 全为 rule_seed_pending，无 GT，明示「分布不是评估」）；管线连通性用随机初始化头做了 8/8 forward 自检（logits 有限/softmax=1） | `reports/p1-animalclip-zeroshot-2026-10-20.json` `status=partial` + `summary.note`；报告 §1 权重不可达证据表（TCP 超时 + DNS 污染排除 + 各镜像实测） |
| 5 | 合成线数字真实性 | 报告数字与 JSON 一致 | **逐位吻合，无口径漂移**：real_val acc 0.191489=MD 0.1915（四舍五入）；chance 0.142857=MD 0.1429；ours 命中率 0.578571=MD 0.5786；syn_v2 0.142857=MD 0.1429；保真度 §3.2 交叉折 KS 0.2768 与 §3.2b 汇总 KS 0.3677 分属两个口径，JSON 内各自存在 | `reports/p1-synth-route-2026-10-20.json` 逐字段比对 |
| 6 | 合成产物实物核验 | 9 个产物存在且可打开 | **9/9 逐字节吻合**（JSON artifacts 列表 vs 磁盘实物一一对应）；npz 可打开：k9_synth_v3 samples=(280,30,24,3) float32；omni_dog joints=(2624,35,3) float32 | numpy 实测读取 |
| 7 | 白名单越界扫描 | 只动白名单文件 | **零越界**：git status 仅 14 个预期文件（6 脚本 + data-sources.md + 7 份报告），无 backend/frontend/迁移改动 | `git status --porcelain --untracked-files=all` 过滤后为空 |
| 8 | 宣称口径对账（检查项数） | 报告与实测一致 | **吻合**：31/38/11 实测 PASS 数 = 报告宣称 | 复跑输出 PASS 计数 vs 报告文 |
| 9 | DogMo/DogCentric 不可得声明 | 真不可得而非懒 | **属实**：DogMo 项目页/仓库无数据入口（论文全文无 download 字样）；DogCentric FTP 端口实测超时，镜像全空，0 字节未伪造 | 侦察员证据（arXiv 全文 grep、FTP 端口探测、镜像 sweep） |
| 10 | AniMo/X-MoGen/Kirin 不可得声明 | 真不可得 | **属实**：AniMo 作者明示不提供 checkpoint；X-MoGen 仓库 README 0 字节空壳；Kirin 代码未发布（"Code release is still in progress"） | 侦察员 GitHub API + 项目页实测 |

---

## 2. 发现清单（现象 → 定性 → 处置）

- **发现#1（结构限制，如实记录，非缺陷）——MammalNet 家犬子集的行为标签域与 K9-Bench-v0 完全错位。** 1.5GB/709 clip 的可用素材真实落盘且质量可证，但其标签全是宠物日常行为（睡觉/进食/梳洗），**对 22 类工作犬考核行为的直接覆盖为零**。定性：**预训练/表示层价值有，验收层价值无**——这与交付报告自身立场一致（"只进预训练池，不进验收门"），不是工作流的错，是数据集本身的边界。处置：数据源清单与报告已如实标注；接入方案须走「表征预训练」而非「行为监督」。
- **发现#2（改进点，非缺陷）——DogMo/DogCentric/AniMo/X-MoGen/Kirin 五个上游目标全军覆没**（不可得），工作流没有因此硬凑产物，而是转为交付自研类条件合成线（k9_synth_v3）。定性：**诚实降级为设计参考 + 自研替代**，符合纪律（拿不到就如实写）。处置：无需修，合成线已是替代产物。
- **发现#3（无发现）——三个验证脚本、数字一致性、产物实物、白名单边界全部核验通过**，本轮对抗验收**没有推翻任何交付宣称**。这一点如实记录（技能的判据铁律要求"没有推翻就怀疑自己不够狠"——但逐项核验是命令级证据，非凭印象）。

---

## 3. 结论

三线交付**经得起对抗验收**：落盘实物真实、数字逐位一致、诚实边界全部守住（域不符的地方如实标了，不可得的地方没有伪造，npz 能打开）。**没有发现需要推翻的宣称，没有需要修的缺陷。**

唯一需要在决策层消化的，是发现#1：MammalNet 家犬子集**真实有用但用途在预训练层而非行为监督层**——这是数据本身的边界，不是工作流缺陷。

---

## 4. 未验证项（诚实清单）

1. **Animal-CLIP 权重一旦到手的真实精度**：本轮权重不可得（外部资源不可达），管线连通性已证但精度数字为 0——后续若能拿到 `mmnet_best.pth`，需补真实 zero-shot/open-vocabulary 评估。
2. **k9_synth_v3 的下游收益**：命中率与保真度是自证指标，「用它训练能否提升 K9-Bench-v0 成绩」未验证（属 P1 科学验收，等真域数据）。
3. **MammalNet 全量子集**：本轮按 2GB 预算只取 16.6% 分层样本；如需扩容可用 `download_mammalnet_canine.py --max-gb 9.3` 续拉（脚本已在）。
4. **YouTube 源 ToS 合规**：MammalNet 视频源是 YouTube，再分发权利判断归用户（交付报告未做合规结论，如实）。
5. **psd-framework 研究线收编**：合成线用的对照基线 syn_v2 与真实参考 full12v2_T30.pkl 属 psd-framework 仓资产，按跨仓纪律只读引用未修改。
