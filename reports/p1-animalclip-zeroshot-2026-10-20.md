# P1 · Animal-CLIP 零样本行为识别基线（K9 工作犬）

> 日期：2026-10-20 ｜ 执行：算法实施员 ｜ 状态：**partial（权重不可得，Animal-CLIP 数字留空）**
> 机器：RTX 5060 Laptop 8 GB / torch 2.11.0+cu128 / Python 3.12.12
> 数据：`psd-framework/runs/data_campaign/video_gap_w42` 的 208 段片段（`duration_s` 合计 **5259.4 s**，
> 单段 3.03–30.0 s，共 152,106 源帧）

---

## 0. 一句话结论

**Animal-CLIP 的专用权重在本机拿不到**（唯一托管方 Google Drive 在 TCP 层不可达，全网红无镜像），
所以本报告**不提供任何 Animal-CLIP 精度数字**。作为可交付的替代，本轮完成了三件事：

1. 完整重建并**实测跑通** Animal-CLIP 的推理管线（读 mp4 → 等距抽 8 帧 → 上游同款预处理 →
   `XCLIP.forward(image, text, animal_text, animal_pred, edges)` → softmax），用
   **随机初始化的 Animal-CLIP 头**在 8 段真实视频上做了连通性自检（`pipeline_self_test.ok=true`）。
2. 用同一批 208 段、同一套标签，跑了 **OpenAI CLIP ViT-B/16 对照基线**，给出
   top-1 分布与置信度直方图 —— 明确标注为 **分布，不是评估**（无 GT）。
3. 把上游仓库的 5 处致命硬伤、依赖不自洽、权重格式不匹配全部定位到行号，写进 §5 偏差清单。

**208 段片段不是 DogCentric**，且 `label_status` 全部为 `rule_seed_pending`（无行为标签），
因此**不存在可算的 acc / F1 / mAP**。任何把它称作"准确率"的说法都是错的。

---

## 1. 权重为何拿不到（当次实测证据）

| 目标 | 实测结果 | 证据 |
|---|---|---|
| `drive.google.com:443` | 超时 21 s（TCP 不可达） | 见 JSON `run.network_probe` |
| `drive.usercontent.google.com:443` | 超时 21 s | 同上 |
| `docs.google.com:443` | 超时 21 s | 同上 |
| `huggingface.co:443` | 超时 22 s | 同上 |
| `hf-mirror.com:443` | **可达**（129 ms），但 API 搜 `Animal-CLIP` / `animalclip` / `PRIS-CV` / `MammalNet` 全返回 `[]` | `curl https://hf-mirror.com/api/models?search=...` |
| DNS 污染 | `nslookup drive.google.com` 返回 `157.240.20.8`（Facebook 段）；用 DoH 取到真 IP `142.250.199.206` 后 `--resolve` 仍超时 → **IP 级封锁，不是 DNS 问题** | 见 §7 复现命令 |
| 直连 Google 其他 IP | `142.250.4.101` / `142.251.10.101` / `172.217.2.46` / `216.58.200.46` 的 443 全部 BLOCKED | 同上 |
| 本地代理端口 | 7890/7897/10809/10808/1080/8118/8889/33210/17890/7891 全部无监听 | 同上 |
| 公共中继（allorigins / codetabs） | 均返回 `error code: 522`（Cloudflare 连不上源站） | 同上 |
| ModelScope / Gitee / GitHub Releases | ModelScope 404、Gitee 搜索 `[]`、GitHub Releases `[]` | 同上 |
| 本机全盘 | `find` 无任何 `mmnet_best.pth` / `AK_best.pth` / `LoTE_best.pth` | 同上 |

权重清单本身已确认（firecrawl 抓取 Drive 文件夹页面，HTTP 200）：

| 文件 | 大小 | 日期 |
|---|---|---|
| `mmnet_best.pth` | 1.92 GB | 2025-08-11 |
| `AK_best.pth` | 1.85 GB | 2025-08-11 |
| `LoTE_best.pth` | 1.85 GB | 2025-08-11 |
| 3 个 `*_log.txt` | 7–10 KB | 2025-08-11 |

> 作者侧也未补齐：GitHub issue #1（中文，"关于文件的问题"）、#2（"About external prompt files"）
> 截至本轮仍为 **open**，都在向作者索要同一个文件夹里的配套文件。

**这不是实现失败，是外部资源不可达。** 补齐方式见 §6。

---

## 2. 已跑通的部分：管线连通性自检

为了证明"权重一到就能跑"，本轮用 **OpenAI CLIP ViT-B/16 骨干 + 随机初始化的 Animal-CLIP 新增头**
（`prompts_generator1/2`、`graph`、`mit`、`prompts_visual_ln/proj`）在 8 段真实片段上跑完整
`XCLIP.forward`。结果（JSON `pipeline_self_test`）：

| 检查项 | 结果 |
|---|---|
| `all_forwards_ran` | ✅ true（8/8） |
| `all_logits_finite` | ✅ true |
| `softmax_sums_to_one` | ✅ true（误差 < 1e-4） |
| 输出形状（MammalNet 12 类） | `[1, 12]` |
| 输出形状（DogCentric 10 类） | `[1, 10]` |
| `video_feature_norm` | 1.0（已 L2 归一化，符合 `model_builder.py:328`） |
| 模型总参数 | 207.8 M |

> ⚠️ 随机头的 top-1 是**噪声**，只用于验证张量形状/设备/dtype/边表连通性，
> **不得**写进论文，也不得与任何基线比较。

这条自检同时排掉了三个已知坑：

- `use_cache=False` 必须开：`model_builder.py:229` 的 `cache_text` 只对 `140/12/21/14` 四种类别数
  做白名单，其它类别数会走 `view(n/50,50,512)` 直接 `RuntimeError`。
- `T` 必须为 8：`models/mit.py:41` 的 `positional_embedding` 形状是 `(1,T,512)`，写死。
- `torch.jit.load` 在 Windows 上对**非 ASCII 绝对路径**会 `fopen` 失败（`errno 2`），
  必须传文件对象 —— 本仓路径含中文（`D:\Desktop\PR论文\...`），踩到并已修。

---

## 3. 可用的部分：OpenAI CLIP ViT-B/16 对照基线（208/208）

**这是分布，不是评估。** 208 段无 GT，只能回答"模型把画面分到了哪些类别、有多自信"。

运行配置：

| 项 | 值 |
|---|---|
| 骨干 | OpenAI CLIP ViT-B/16，350,837,078 B（SHA-256 见 JSON `run.backbone.sha256`） |
| 来源 | `openaipublic.azureedge.net`（本机可直连，HTTP 200） |
| 预处理 | CLIP 原生：Resize 短边 224 → CenterCrop 224 → mean `[122.77,116.75,104.09]` / std `[68.50,66.63,70.32]` |
| 视频表征 | 8 帧逐帧 `encode_image` → L2 归一化 → 均值 → 再归一化 |
| 文本提示 | `a video of a dog {label}` |
| 样本 | 208/208，解码错误 0，推理错误 0 |
| 耗时 | 78.7 s（0.38 s/片段，含解码） |
| 解码 | 152,106 帧 → 每段等距抽 8 帧，回退模式 0 次 |

### 3.1 DogCentric 10 类标签空间（开放词表迁移）

对应 JSON `summary.clip_baseline.dogcentric.top1_share`：

| 占比 | 类别 |
|---|---|
| 34.13 % | walk |
| 23.08 % | ball play |
| 20.19 % | pet |
| 13.94 % | turn-left |
| 4.81 % | sniff |
| 2.88 % | feed |
| 0.48 % | turn-right |
| 0.48 % | shake |
| 0.00 % | car |
| 0.00 % | drink |

- 观测到 8/10 类；top-1 置信度 mean **0.3498** / median 0.3178 / min 0.1513 / max 0.7889。
- 置信度直方图（0.05 一档，共 208）：
  `0.10–0.15: 10 | 0.15–0.20: 35 | 0.20–0.25: 39 | 0.25–0.30: 39 | 0.30–0.35: 20 |`
  `0.35–0.40: 30 | 0.40–0.45: 9 | 0.45–0.50: 12 | 0.50–0.55: 4 | 0.55–0.60: 3 |`
  `0.60–0.65: 3 | 0.65–0.70: 2 | 0.70–0.75: 2`
- **解读**：中位数 0.32 相对于 10 类的均匀先验（0.10）只高出约 3 倍，属于"弱偏好"而非"明确判别"。
  这与"CLIP 是图像模型、视频段是第三人称训练素材、DogCentric 是犬背第一视角"三重域差一致。

### 3.2 MammalNet 12 类原生标签空间（对上 Animal-CLIP 的标签语义）

对应 JSON `summary.clip_baseline.native.top1_share`：

| 占比 | 类别 |
|---|---|
| 52.40 % | hunts other animals |
| 19.71 % | grooms/cleans itself or other animal |
| 12.98 % | pees |
| 6.25 % | mates with other animals |
| 5.77 % | fights against other animals |
| 0.96 % | drinks water |
| 0.96 % | nurses or breastfeeds its baby |
| 0.48 % | vomits |
| 0.48 % | eats food |
| 0.00 % | sleeps |
| 0.00 % | poops |
| 0.00 % | gives birth to a baby |

- 观测到 9/12 类；置信度 mean **0.3954** / median 0.3519 / min 0.1329 / max 0.9041。
- **解读**：`hunts other animals` 占一半，是典型的"训练/敏捷场景里犬只快速移动 + 前景小 + 背景杂乱"
  被 CLIP 读成捕猎姿态。这恰好说明 MammalNet 的 12 类**不覆盖** K9 训练科目（坐、卧、随行、
  搜索、扑咬、越障），标签体系本身就不对齐 —— 也是为什么必须拿到 DogCentric 才能做真正的评估。

### 3.3 与 DogCentric 的标签映射（显式审计表）

| DogCentric | MammalNet 索引 | 说明 |
|---|---|---|
| drink | 0 `drinks water` | 精确 |
| feed | 2 `eats food` | 精确 |
| shake | 6 `grooms/cleans itself...` | 近似（并非"甩身"） |
| ball play / car / turn-left / turn-right / pet / sniff / walk | — | MammalNet 无对应类 |

→ 10 类里只有 2 类精确对齐、1 类近似；**这直接排除了"用 MammalNet 12 类头去评估 DogCentric"的可行性**。

---

## 4. 实施内容（可复跑）

| 文件 | 作用 |
|---|---|
| `external/animal-clip/` | `git clone --depth 1`，HEAD `4cff5a8b36395b8a5473b4fe7763fdb6486c1d65`（2025-08-11），MIT |
| `external/animal-clip/_k9_stubs.py` | **新增**。以 `sys.modules` 注入方式提供 8 个装的上的依赖替身，不改上游一行 |
| `external/animal-clip/weights/ViT-B-16.pt` | OpenAI CLIP 骨干，350,837,078 B |
| `external/animal-clip/_vendor/` | `pip --target` 隔离安装的 `yacs 0.1.8` / `ftfy 6.3.1` / `wcwidth 0.9.2`（不污染主 venv） |
| `scripts/eval_animalclip_zeroshot.py` | **新增**。自研推理脚本，不调用 `main.py` / `val.py` |
| `scripts/verify_animclip_eval.py` | **新增**。复验脚本，exit 0 = 产物齐全且自洽 |
| `reports/p1-animalclip-zeroshot-2026-10-20.json` | 逐片段全量结果（含 logits/probs/decode/自检） |
| `reports/p1-animalclip-zeroshot-2026-10-20.md` | 本报告 |

### 复跑命令

```bash
# 推理（208 段，约 80 s）
"$VENV/python.exe" scripts/eval_animalclip_zeroshot.py \
  --manifest "D:/Desktop/PR论文/psd-framework/runs/data_campaign/video_gap_w42/manifest.jsonl" \
  --fragments-dir "D:/Desktop/PR论文/psd-framework/runs/data_campaign/video_gap_w42/fragments" \
  --out-json reports/p1-animalclip-zeroshot-2026-10-20.json \
  --self-test 8

# 复验（必须 exit 0）
"$VENV/python.exe" scripts/verify_animclip_eval.py
```

---

## 5. 上游仓库偏差清单（**未改动上游文件**，全部在自研脚本里绕开）

| # | 位置 | 缺陷 | 本机实测 | 绕法 |
|---|---|---|---|---|
| 1 | `main.py:231` | `init_process_group(backend='nccl')` | Windows torch 无 NCCL：`RuntimeError: Distributed package doesn't have NCCL built in` | 不走 `main.py` |
| 2 | `val.py:153/156` | `nu` / `tot` 从未初始化，`noi=True` 默认路径必抛 | AST 抽真实函数体执行 → `UnboundLocalError: cannot access local variable 'nu'` | 不走 `val.py` |
| 3 | `val.py:60` | `MultilabelAccuracy(num_labels=140)` 写死 | 12 类/10 类时形状不匹配 | 不 import `val.py` |
| 4 | `datasets/tools.py:117` | 模块级 `open('/mnt/sdb/.../action_ak.txt')` | 全仓 **15 处** `/mnt/sdb` 硬编码（`grep -rn "mnt/sdb" --include=*.py .`） | 不 import `datasets.*` |
| 5 | `datasets/tools.py:104` | import 时在 CWD 写 `description_ak.csv`（写盘副作用） | 同上 | 同上 |
| 6 | `models/model_builder.py:229` | `cache_text` 白名单只有 140/12/21/14 | 10 类 → `view(0,50,512)` → `RuntimeError` | `use_cache=False` |
| 7 | `models/mit.py:41` | `positional_embedding` 形状 `(1,T,512)` 写死 | T=4/T=16 均崩，仅 T=8 可用 | 固定 `T=8` |
| 8 | `requirements.txt` | 未列 `open_clip` / `transformers` / `torch_geometric` / `detectron2` / `fvcore` / `iopath` / `torchstat`，但代码顶层 import | 与 `val.py:43`、`model_builder.py:47,50`、`head_helper.py:8`、`build.py:15`、`pipeline.py:7/26` 不自洽 | `_k9_stubs.py` |
| 9 | `requirements.txt:32` | `mmcv-full==1.6.1` | PyPI 仅 sdist、无 Windows wheel、无 cp312；本机 `mmcv 2.2.0` 实为 mmcv-lite，无 `mmcv.utils`/`mmcv.parallel` | 只用 XCLIP 路径，不 import `datasets.*` |
| 10 | `requirements.txt`（apex） | apex 不在 PyPI | `pip index versions apex` → `No matching distribution` | `_k9_stubs.py`；且 `main.py` 仅在 `OPT_LEVEL!='O0'` 时用 |
| 11 | `val.py:43` / `val.py`（`coop`） | `import open_clip` / `import coop` 均为**死 import** | `grep` 无任何 `open_clip.` / `coop.` 调用 | 不 import `val.py` |
| 12 | `datasets/pipeline.py:1955/1977/1981/2061` | `astype(np.int)` | numpy 2.5.1 已删 `np.int` → `AttributeError` | 自写抽帧，不用该管线 |
| 13 | `utils/tools.py:254` vs `models/model_builder.py:337` | checkpoint 存 `{'model', 'optimizer', ...}` 包装 dict，`build_model` 直接当 raw state_dict 用 | 包装 dict 喂 `build_model` → `KeyError 'visual.conv1.weight'` | `load_animalclip_checkpoint` 里取 `ckpt['model']` |
| 14 | `torch.load`（torch 2.11 默认 `weights_only=True`） | 拒绝含 yacs `CfgNode` 的 pickle | `UnpicklingError` | 显式 `weights_only=False`（**意味着信任该 pickle**，需知情） |
| 15 | 无 LICENSE 冲突 | 仓库 MIT（`LICENSE` 首行 `MIT License`），论文正文版权归 Springer | — | 引用按学术规范 |

---

## 6. 如何补齐（唯一阻塞项）

**只需要一样东西：`mmnet_best.pth`（1.92 GB）。** 拿到后：

```bash
# 1) 放到指定位置（文件名必须一致）
external/animal-clip/weights/mmnet_best.pth

# 2) 重跑（脚本会自动检测到权重并切到 Animal-CLIP 推理）
"$VENV/python.exe" scripts/eval_animalclip_zeroshot.py \
  --manifest ... --fragments-dir ... \
  --out-json reports/p1-animalclip-zeroshot-2026-10-20.json --self-test 8

# 3) 复验
"$VENV/python.exe" scripts/verify_animclip_eval.py
```

脚本会做三件自检并写进 JSON：`run.weights.status`、`run.state_dict_report`
（missing/unexpected keys）、`run.checkpoint_meta`。**若 `missing_count > 0`，状态会降级为
`partial` 并写明原因**，不会静默当作可用权重。

有代理的通道可用：

```bash
gdown --folder 1iNMta_pFjhHLNK3FRZLUigSt3ya7i8sU -O external/animal-clip/weights/
# 或单文件（需先从文件夹页拿到 file id）
gdown <FILE_ID> -O external/animal-clip/weights/mmnet_best.pth
```

预期效果（供决策参考，非承诺）：加载后 FP32 显存约 0.83 GB，B=1 推理峰值约 1.2 GB，
8 GB 卡有余量；208 段外推约 8 分钟（解码 0.44 s + 推理 1.88 s 量级）。

---

## 7. 复现本轮网络探测

```bash
# DNS 污染
nslookup drive.google.com                 # -> 157.240.20.8（Facebook 段）

# DoH 取真 IP 后仍不通 => IP 级封锁
curl -sS "https://dns.alidns.com/resolve?name=drive.google.com&type=A"
# -> 142.250.199.206
curl -sS --max-time 20 --resolve drive.google.com:443:142.250.199.206 \
     -o /dev/null -w "%{http_code}\n" "https://drive.google.com/drive/folders/1iNMta_pFjhHLNK3FRZLUigSt3ya7i8sU"
# -> 000（超时）

# 无镜像
curl -sS "https://hf-mirror.com/api/models?search=Animal-CLIP&limit=50"   # -> []
curl -sS "https://hf-mirror.com/api/models?author=PRIS-CV&limit=50"       # -> []
curl -sS "https://gitee.com/api/v5/search/repositories?q=Animal-CLIP"     # -> []
curl -sS "https://api.github.com/repos/PRIS-CV/Animal-CLIP/releases"      # -> []
```

> 全部命令已清空 `http_proxy/https_proxy/all_proxy` 直连执行（`env -u ... curl`）。

---

## 8. 诚实边界

**做了的**
- 完整重建 Animal-CLIP 推理管线并在 8 段真实视频上跑通（随机头，仅连通性）。
- 208/208 段 OpenAI CLIP 基线推理，逐片段 logits/probs 落盘，0 错误。
- 上游 15 项偏差全部定位到文件:行号，全部在自研脚本里绕开，**未修改上游任何文件**。
- 依赖隔离：`pip --target` 到 `external/animal-clip/_vendor`，主 venv 未装任何新包。

**没做的 / 做不到的**
- ❌ **没有 Animal-CLIP 的任何推理数字**（权重缺失，绝不伪造）。
- ❌ **没有任何 acc / F1 / mAP**（208 段无行为标签，`label_status=rule_seed_pending`）。
- ❌ **没有 DogCentric**：本轮独立实测（非沿用侦察结论）——
  `curl ftp://robotics-ftp.ait.kyushu-u.ac.jp/dogcentric/` → `curl: (28) Failed to connect to
  robotics-ftp.ait.kyushu-u.ac.jp port 21 after 21044 ms`；
  `curl http://robotics-ftp.ait.kyushu-u.ac.jp/dogcentric/` → `000`（20 s 超时）；
  `find "D:/Desktop/PR论文" -iname "*dogcentric*"` → 空。
  208 段是 w42v 补采池（YouTube 122 / Bilibili 86），**不是** DogCentric（官方 209 段 / 4 只犬 /
  10 类 / 320×240@48fps），不可混称。
- ❌ 没跑训练/微调（预算是推理基线；且全参数微调 B=1 已超 8 GB）。
- ⚠️ 「zero-shot」措辞：Animal-CLIP 的三个 checkpoint 是**数据集专用微调权重**
  （MammalNet 12 类 / AK 140 类 / LoTE 21 类），其标签集与动物-动作关系图在训练时固定。
  架构与类别数解耦（GCN 权重 `(512,256)`/`(256,512)` 与类别数无关），所以"换新标签直接推理"
  工程上可行，但**这属于 open-vocabulary 迁移，不是论文声称的 zero-shot 设置**。
- ⚠️ `torch.load(weights_only=False)` 会执行 pickle，仅在信任该权重来源时使用。

**论文表述建议**（避免事实性错误）

> "Animal-CLIP 以 OpenAI CLIP ViT-B/16 为初始化骨干，在 MammalNet / Animal Kingdom / LoTE-Animal
> 三个数据集上**分别微调**；本文使用其 MammalNet 版本（`mmnet_best.pth`）作为对比基线。"

不要把 MammalNet 写成"预训练语料"，也不要把三个 checkpoint 混用、或把微调权重称作 zero-shot。
