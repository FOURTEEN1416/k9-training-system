# P1 数据集补充 — MammalNet / DogMo 实施报告（2026-10-20）

> 结论先行：**MammalNet 家犬子集已真实落盘并逐文件校验通过**（752 文件 / 709 clip / 1.543 GB / 6.71 h，8 品种 × 3 split 全覆盖）；
> **DogMo 无公开数据集渠道**，只有官网 209 MB 预览素材可用，已落盘但明确标注「不可作训练/评测集」；
> **DogCentric 传输层不可达**，0 字节。全部结论均有当次命令输出与 `scripts/verify_dataset_expansion.py` 的 31 项检查（exit 0）支撑。

---

## 1. 执行摘要

| 目标 | 侦察判定 | 本轮实际 | 落盘 | 复验 |
|------|----------|----------|------|------|
| MammalNet 家犬子集 | available | ✅ 1.543 GB / 752 文件 / 709 clip | `D:\Desktop\PR论文\datasets\mammalnet_canine` | 19/19 项 PASS |
| DogMo | available=false | ⚠️ 仅 227 MB 预览素材 | `D:\Desktop\PR论文\datasets\dogmo_preview` | 12/12 项 PASS |
| DogCentric | available=false | ⛔ 未获取（FTP 超时） | — | 0 字节，未伪造 |

单条复验命令：

```bash
D:/Desktop/PR论文/k9-training-system/.venv/Scripts/python.exe scripts/verify_dataset_expansion.py
# 实测输出末行：ALL CHECKS PASSED -> ...\reports\p1-dataset-expansion-verify.json
# 退出码：EXIT=0（31 checks / 23 s）
```

---

## 2. MammalNet：从「有链接」到「有文件」

### 2.1 侦察结论的修正

侦察报告给的是 `trimmed_videos.tar.gz`（178 GB）。本轮实测补充了三条**必须先知道**的事实：

1. **README 的包名不存在**：`trimmed_videos.tar.gz` → 403；桶内实际是 `trimmed_video.tar.gz`
   （158,954,389,396 B，HF 镜像 `linxxx3/MammalNet` 同源）。所以「照 README 走」必然失败。
2. **S3 单流 ≈20 KB/s**，必须高并发。本轮分级实测（同一对象 `trimmed_videos/--Ap3c8yiMk.mp4`，56,616,812 B）：

   | 并发 | 分块 | 吞吐 | 折算 1.85 GB |
   |------|------|------|--------------|
   | 1 | 4 MB | 19.9 KB/s | ~27 h |
   | 32 | 1 MB | 242 KB/s | ~2.1 h |
   | 128 | 256 KB | 1,173 KB/s | ~26 min |
   | 256 | 128 KB | 2,609 KB/s | ~12 min |
   | **768（跨文件）** | 128 KB | **1,932 KB/s** | ~16 min |

   注意 512 路反而回落到 2,246 KB/s（有 1 个 chunk 失败），说明并发不是越高越好。
3. **`duration` 字段不是片段时长**。实测 8 个样本：`bB5IyPzqdEo` 标注 26 s / 实际 15.0 s；
   `kb3LWKo5S98` 标注 28 s / 实际 27.0 s；`U27OW-FGr10` 标注 119 s / 实际 119.0 s。
   抽样比对 2,000 条，**1,190 条**的 behavior segment 末端超过标注 duration ——
   该字段是源 YouTube 视频时长。因此 manifest 里另存 `clip.duration_s`（ffprobe 实测）。

### 2.2 抽样策略（预算 ≤2 GB）

- 家犬判定：`genus == "canis"` 且 `keyword ∉ {wolf, wolfdog}` → 1,129 clip。
- 分层：`(split, breed)` 共 24 层；每层按该层占总字节的比例分配配额，层内按文件大小升序填充。
- 结果：**752 文件 / 709 clip / 1,542,535,302 B / 24,146.9 s（6.71 h）**，占全量子集 16.6 %。
- 每层都有样本，最小层 `val/foxhound` 1 文件 4.0 MB。

| split | 文件 | clip | 字节 | 实测时长 |
|-------|------|------|------|----------|
| train | 548 | 512 | 1,087,924,451 B (1.088 GB) | 17,356.2 s (4.82 h) |
| val | 55 | 54 | 127,302,622 B (127 MB) | 1,815.8 s (0.50 h) |
| test | 149 | 143 | 327,308,229 B (327 MB) | 4,974.9 s (1.38 h) |
| **合计** | **752** | **709** | **1,542,535,302 B** | **24,146.9 s** |

| 品种 | clip 数 | 品种 | clip 数 |
|------|---------|------|---------|
| bulldog | 253 | husky dog | 31 |
| terrier | 241 | shepherd dog | 27 |
| malamute | 120 | eskimo dog | 11 |
| hound | 66 | foxhound | 3 |

行为分布（clip 级，709 clip 全都有 ≥1 标签，12 类）：
sleeps 260 / fights 99 / nurses 84 / drinks water 80 / hunts 66 / eats food 60 /
mates 44 / pees 27 / poops 25 / gives birth 13 / vomits 5 / grooms 2。

### 2.3 完整性

| 检查 | 方法 | 结果 |
|------|------|------|
| 文件存在 | manifest vs 磁盘 | 752/752 |
| 字节一致 | `st_size` vs manifest | 0 mismatch |
| 内容一致 | 逐文件 sha256 复算 | 0 bad |
| 传输正确 | S3 ETag（单段上传 = MD5）逐文件比对 | 748 个单段对象 748/748 命中；另 4 个 multipart 对象（ETag 带 `-`）改由 sha256/ffprobe 校验 |
| 可解码 | ffprobe 全量 | 752/752 可解码 |
| 预算 | 目录字节 | 1.543 GB ≤ 2.000 GB |
| 品种/split 覆盖 | set 比较 | 8/8 品种、3/3 split |
| 无野生犬科 | keyword 过滤 | wolf/wolfdog = 0 |
| 片段元数据 | ffprobe 补齐 | 0 缺失 |

### 2.4 与全量子集的差距

- 全量家犬：**1,129 clip / 1,198 文件 / 9.293 GB**（逐文件字节求和）；
  时长口径有两套且都不等于片段实际时长：annotation `duration` 求和 28.43 h、behavior segment 跨度 23.19 h。
- 差距 7.75 GB，恢复命令：`python scripts/download_mammalnet_canine.py --dataset mammalnet --max-gb 9.3`
  （按本轮实测 ~4.6 MB/s 下载吞吐，约 28 min）。
- 不建议整包（159 GB / 178 GB full_video 392 GB）：D: 仅剩 96 GB，且 2 GB/数据集预算明确禁止。

---

## 3. DogMo：没有数据集，只有素材

### 3.1 侦察结论的复核

| 渠道 | 实测 |
|------|------|
| 论文 arXiv 2510.24117 | 无 data-availability 声明 |
| 项目页 `pie-lab.cn/DogMo/` | "download" 出现 0 次，无请求表单 |
| GitHub `BIT-PIE/DogMo` | 单分支 `Page`，仅 index.html + static，0 code / 0 data / 0 release |
| "Code" 按钮 | 指向页面自身 |
| HF `ahdjasdja/DogMo` | 无关仓库（imagefolder zip） |

**结论：DogMo 不是"暂时下载不到"，而是"根本没有公开发布"**。本轮未尝试伪造 SMAL 标注或把预览当训练集。

### 3.2 已落盘的预览素材

| 组 | 数量 | 字节 | 说明 |
|----|------|------|------|
| `dataset/` | 20 | 44.1 MB | 5 只狗 × 4 路（origin/depth/mask/barc），640x360 barc、1920x1080 其余，15 fps，11.47 s |
| `dataset_pre/` | 20 | 143.4 MB | 同 5 只狗 × 4 路，原始分辨率（origin 最大 34.2 MB） |
| `results/` | 7 | 21.3 MB | 单/多视角 RGB / RGBD 结果片段 |
| `glbs/` | 2 | 18.2 MB | pose 14,968,128 B + motion 3,273,204 B |
| **合计** | **49** | **227,078,945 B** | 视频 47/47 可解码，总 427.1 s |

### 3.3 两个只有实测才会发现的坑

1. **GLB 的 Range 是压缩实体**：HEAD 报 `Content-Range: bytes 0-0/6772757`，
   但普通 GET 返回 **14,968,128 B** 且 magic 为 `glTF`。
   照 HEAD 长度分块会写出 12 字节全零的坏文件（本轮第一遍就是这样）。
   脚本现在对 `.glb` 走**普通 GET 整文件**，并用 `glTF` magic 校验（verifier 的 `dogmo/glb_magic`）。
2. **dataset 与 dataset_pre 是同一内容的两个码率**：同 `avid` 的 origin，dataset 版 7,999,053 B、
   dataset_pre 版 29,765,983 B，时长都是 11.4667 s。下游用哪套必须显式选。

### 3.4 使用边界

- 49 个素材 **不能** 作为训练集、benchmark 或"DogMo 复现"证据。论文语料是
  1.2k clips / 10 dogs / 11 actions / >220 min / >1M frames / 5 路同步 ORBBEC Astra 2。
- 论文只承诺 per-frame binary masks + 2D keypoints + per-clip action label；**D-SMAL mesh/参数是否发布未声明**。
- 深度是 H.264 1920x1080 MP4（与 RGB 同分辨率），很可能是伪彩色而非 16-bit 米制深度 —— 未验证，不得据此做米制 3D。
- 若要全量：需联系通讯作者（Wei Liang, BIT）拿书面条款；在此之前**不排 GPU 训练时间**。

---

## 4. DogCentric：如实记为 blocked

- `curl ftp://robotics-ftp.ait.kyushu-u.ac.jp/dogcentric/` → **15 s 超时，exit 124**（本轮复测）。
- 镜像检索全部为空：Kaggle `[]`、HF `[]`、OpenAIRE 0、Wayback CDX `[]`。
- 本轮**未下载任何 clip**，不填任何推测大小。

若要推进：联系维护者（Yumi Iwashita / M. S. Ryoo）索取可用镜像；不要继续重试 FTP。

---

## 5. 产出物与复验

### 5.1 新增文件（本线白名单内）

| 路径 | 用途 |
|------|------|
| `scripts/download_mammalnet_canine.py` | 下载器：S3 索引 → 分层抽样 → 跨文件并发分块 → manifest；DogMo 预览抓取 |
| `scripts/verify_dataset_expansion.py` | 校验器：31 项硬检查，exit 0/1 |
| `docs/data-sources.md` | 数据源登记（来源/许可/坑/恢复指引） |
| `reports/p1-dataset-expansion-2026-10-20.md` | 本文件 |
| `reports/p1-dataset-expansion-2026-10-20.json` | 机器可读版本 |
| `reports/p1-dataset-expansion-verify.json` | verifier 逐项输出 |

### 5.2 数据集（非 git，`D:\Desktop\PR论文\datasets\`）

```
datasets/
├── mammalnet_canine/
│   ├── annotations/          16 文件（detection_annotations.json, taxonomy.txt, 4 套 composition）
│   ├── videos/{train,val,test}/   752 个 mp4
│   ├── manifest.json         每文件 s3_key/video_id/part/split/breed/行为区间/md5/sha256/ffprobe
│   ├── subset_annotations.json
│   ├── summary.json
│   ├── recovery.md
│   └── _cache/               annotation.tar + S3 索引（2.3 MB，可删）
└── dogmo_preview/
    ├── preview/{dataset,dataset_pre,results}/   47 mp4
    ├── glbs/{pose,motion}.glb
    ├── manifest.json
    └── summary.json
```

### 5.3 复验命令与实测输出

```bash
$ python scripts/verify_dataset_expansion.py | tail -3
[PASS] dogmo/preview_kinds: {'glb': 2, 'dataset': 20, 'dataset_pre': 20, 'results': 7}
[PASS] dogmo/preview_only: 49 assets (official corpus is 1.2k clips)

ALL CHECKS PASSED -> D:\Desktop\PR论文\k9-training-system\reports\p1-dataset-expansion-verify.json
EXIT=0
```

---

## 6. 已知限制（不做美化）

1. **MammalNet 只取了 16.6 %**：受 2 GB 预算约束，是分层抽样而非完整子集；报告里所有「小时数」都是**抽样内**实测。
2. **`duration` 口径已修正但上游仍混乱**：annotation 的 `duration` 是源视频时长；本 manifest 用 ffprobe 另存 `clip.duration_s`，两者不可混用。
3. **行为标签是 clip 级区间，不是帧级**：最多 4 段/ clip，均值 1.06 段；不能直接当帧级行为检测 GT。
4. **视频是 YouTube 二次分发**：作者声明不分发视频；本项目仅研究使用，**不得再分发**，也不上传任何云。
5. **DogMo 只有 227 MB 预览**：任何"DogMo 复现/评测"结论都无效；本轮不产出此类结论。
6. **DogCentric 为 0 字节**：未获取即未获取，没有用替代品冒充。
7. **S3 吞吐波动大**：本轮 4.6 MB/s，压测时 1.9 MB/s；`--max-gb 9.3` 的耗时估计（~28 min）误差可能 2× 以上。
8. **未跑全量 pytest**：按分工只跑了本线定向验证脚本与 `backend.app.main` 可 import 检查（`IMPORT_OK`）。

---

## 7. 下一步建议（不在本线实施）

1. 若要做犬种/行为预训练，先扩到 9.3 GB 全量家犬（28 min，磁盘允许），再考虑 Canidae 1,857 clip / 17.08 GB。
2. 行为标签需先转成帧级区间，才能接 ST-GCN+BC；当前 clip 级不能直接训练。
3. DogMo 若要正用，先发信拿条款；在拿到前把 `dogmo_preview` 仅用于定性对比图。
4. DogCentric 建议从数据计划中移除，除非维护者回信给出可用镜像。

---

## 附：本轮关键命令与输出

| 命令 | 关键输出 |
|------|----------|
| `python scripts/download_mammalnet_canine.py --dataset mammalnet --max-gb 1.6` | `S3 索引完成: 20034 objects, 130s` → `选取 752 文件 1.543 GB` → `完成 752 文件 / 1542.5 MB，339s` → `{"mammalnet": "ok"}` |
| `python scripts/download_mammalnet_canine.py --dataset dogmo` | `可下载 49 个 / 208.8 MB` → `OK static/glbs/pose.glb (15.0 MB)` → `{"dogmo": "ok"}` |
| `python scripts/verify_dataset_expansion.py` | 31 × PASS，`ALL CHECKS PASSED`，`EXIT=0` |
| `python -c "import backend.app.main"` | `IMPORT_OK` |
| `curl ... S3 32 路并发` | `242 KB/s`（单流 19.9 KB/s） |
| `curl ftp://robotics-ftp.ait.kyushu-u.ac.jp/dogcentric/` | 15 s 超时，exit 124 |
