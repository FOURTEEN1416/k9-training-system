# 数据源登记表（P1 数据补充线）

> 版本：v1.0（2026-10-20）
> Owner：数据实施员
> 适用范围：K9 Training System 的**外部公开犬类视频数据**登记、获取方式、许可与可用性状态。
> 真源规则：本文件只记录**当次实测**的事实（命令 + 输出）；未实测的一律标注「未验证」。

---

## 0. 一句话结论

| 数据集 | 状态 | 落盘位置 | 规模 |
|--------|------|----------|------|
| **MammalNet**（家犬子集） | ✅ 可用（受 2 GB 预算抽样） | `D:\Desktop\PR论文\datasets\mammalnet_canine` | 752 文件 / 709 clip / 1.543 GB / 6.71 h |
| **DogMo** | ⚠️ 仅预览素材（无公开数据集） | `D:\Desktop\PR论文\datasets\dogmo_preview` | 49 素材 / 227 MB |
| **DogCentric** | ⛔ 阻塞（传输层不可达） | — | 0 |

复验命令：

```bash
D:/Desktop/PR论文/k9-training-system/.venv/Scripts/python.exe scripts/verify_dataset_expansion.py
# exit 0 = 全部 31 项检查通过（2026-10-20 实测 ALL CHECKS PASSED）
```

---

## 1. MammalNet（CVPR 2023, Vision-CAIR）

### 1.1 来源

| 项 | 值 |
|----|-----|
| 仓库 | https://github.com/Vision-CAIR/MammalNet |
| 论文 | Chen et al., *MammalNet: A Large-scale Video Benchmark for Mammal Recognition and Behavior Understanding*, CVPR 2023, pp. 13052-13061 |
| 标注包 | `https://mammalnet.s3.amazonaws.com/annotation.tar`（**1,512,456 B**，实测 200） |
| 视频 | `https://mammalnet.s3.amazonaws.com/trimmed_videos/<youtube_id>[_n].mp4`（逐文件对象） |
| HF 镜像 | `https://hf-mirror.com/datasets/linxxx3/MammalNet` → `trimmed_video.tar.gz`（158,954,389,396 B） |
| 许可 | 标注 CC BY；**视频源自 YouTube，作者声明只提供标注，不得再分发** |

### 1.2 实测到的坑

1. **README 写的 `trimmed_videos.tar.gz` 已 403**。桶里实际对象名是 `trimmed_video.tar.gz`（单数）。
2. **S3 单流极慢**：`curl -r 0-4000000` 60 s 只拿到 1,176,386 B（≈20 KB/s）。
   32 路并发 ≈242 KB/s，128 路 ≈1.2 MB/s，**768 路跨文件分块 ≈1.9 MB/s**（本脚本默认 512 线程 × 128 KB，
   实测下载阶段 1,542.5 MB / 339 s ≈ 4.6 MB/s）。
3. **annotation 的 `duration` 是源 YouTube 视频时长，不是裁剪片段时长**。例：
   `bB5IyPzqdEo` 标注 `duration=26`，实际片段 `ffprobe` 为 15.0 s。
   因此 manifest 额外用 ffprobe 记录真实 `clip.duration_s`（本项目 752 个文件全部补齐）。
4. **视频 ID 可能自带 `_数字` 后缀**（如 `t387Sj7u_60`），不能用 `re.sub(r"_\d+$")` 剥基名；
   脚本改为「最长 annotation ID 前缀匹配」。
5. **标注元数据自相矛盾**：`taxonomy.txt` 实测 173 行 / 71 个 family / 19 个 order，
   而 README 写 69 families / 17 orders。对 Canidae 过滤无影响（canidae 8 行齐全）。

### 1.3 家犬子集口径

- 过滤：`taxnomy[].genus == "canis"` 且 `keyword not in {wolf, wolfdog}`。
- 全量家犬子集（按 S3 索引逐文件字节求和，**实测**）：
  **1,129 clip / 1,198 文件 / 9.293 GB / 28.43 h**（annotation duration 口径）；
  若按行为 segment 跨度算为 **23.19 h**。两者都不是「裁剪片段实际时长」。
- 品种分布（clip 数）：bulldog 384、terrier 357、malamute 177、hound 109、husky dog 43、
  shepherd dog 36、eskimo dog 17、foxhound 6。
- 行为标签 12 类（clip 级）：sleeps / fights / nurses / drinks water / hunts / eats food /
  mates / pees / poops / gives birth / vomits / grooms。

### 1.4 本次落盘（预算受限抽样）

预算 `≤2 GB/dataset`，实际取 1.6 GB 目标、按 `(split, breed)` 分层按字节配额轮转：

| 项 | 实测值 |
|----|--------|
| 文件 / clip | **752 / 709**（91 个文件是 `_1`/`_2` 分段） |
| 字节 | **1,542,535,302 B = 1.543 GB** |
| 片段实际时长 | **6.71 h**（ffprobe 求和 24,146.9 s），均值 32.1 s/clip |
| split | train 548 文件 / 512 clip / 4.82 h；val 55 / 54 / 0.50 h；test 149 / 143 / 1.38 h |
| 品种覆盖 | 8/8（bulldog 253、terrier 241、malamute 120、hound 66、husky 31、shepherd 27、eskimo 11、foxhound 3） |
| 分辨率 | 58 种，主流 1280x720 (202)、406x720 (176)、480x360 (114) |
| 完整性 | 752/752 sha256 复算通过、752/752 ffprobe 可解码、748 个单段对象的 MD5 与 S3 ETag 全部一致（另 4 个为 multipart，ETag 不可直接比对） |
| 占全量子集 | 16.6 % 字节 |

### 1.5 恢复完整子集（9.293 GB）

```bash
# 逐文件对象（推荐；脚本已内置分层+并发，改 --max-gb 即可）
python scripts/download_mammalnet_canine.py --dataset mammalnet --max-gb 9.3

# 或 HF 镜像整包流式选择性解包（159 GB 单遍，hf-mirror 实测 ~17.8 MB/s → 约 2.5 h）
export HF_ENDPOINT=https://hf-mirror.com
python scripts/download_mammalnet_canine.py --dataset mammalnet --max-gb 9.3  # S3 优先
```

详见 `D:\Desktop\PR论文\datasets\mammalnet_canine\recovery.md`。

---

## 2. DogMo（arXiv 2510.24117，BIT）

### 2.1 状态：**没有公开数据集**

- 论文、项目页（`https://pie-lab.cn/DogMo/`）均**无**下载链接、请求表单或 data-availability 声明；
- GitHub `BIT-PIE/DogMo` 只有 `index.html` + 静态资源，**0 code / 0 data / 0 release**；
- 页面上的 "Code" 按钮指回页面本身；"open-source code" 指向页面模板仓库 `silverster98/project-page-template`；
- 全文检索 "publicly available" / "will be released" / "upon request" / zenodo / huggingface：**0 命中**。

### 2.2 可用的只有官网预览素材（已落盘）

| 组 | 文件数 | 字节 |
|----|--------|------|
| `static/videos/dataset/`（压缩预览，各有 origin/depth/mask/barc 四路） | 20 | 44,118,104 B (44.1 MB) |
| `static/videos/dataset_pre/`（原始分辨率预览，同上四路） | 20 | 143,378,597 B (143.4 MB) |
| `static/videos/results/`（方法结果片段） | 7 | 21,340,912 B (21.3 MB) |
| `static/glbs/`（pose / motion） | 2 | 3,273,204 + 14,968,128 B |
| **合计** | **49** | **227,078,945 B (227 MB)** |

- 视频 47/47 ffprobe 可解码，总时长 427.1 s；分辨率含 1920x1080 与 3840x1088 / 4800x1088。
- **坑**：`static/glbs/*.glb` 在 HEAD / Range 请求下报的 `Content-Range: bytes 0-0/6772757` 是**压缩实体**长度，
  按该长度分块会写出 0 字节头的坏文件；必须走**普通 GET** 整文件下载（实测 14,968,128 B，magic `glTF`）。
  脚本已按扩展名分流。

### 2.3 使用边界

⚠️ 这 **209 MB 预览片段不能当训练集或 benchmark**。论文声明的语料是
1.2k clips / 10 dogs / 11 actions / >220 min / >1M frames / 5 路同步 ORBBEC Astra 2。

如确需 DogMo 全量，只能联系通讯作者（Wei Liang, BIT）索取数据集与 D-SMAL 标注授权，
在拿到书面条款前**不应为 DogMo 数据预留 GPU 训练时间**。

---

## 3. DogCentric（Kyushu Univ.）

### 3.1 状态：⛔ 阻塞（传输层）

- 页面 `http://robotics.ait.kyushu-u.ac.jp/~yumi/db/first_dog.html` HTTP 200（HTTPS 版 500）；
- 页面 `:81-83` 声明视频只在 `ftp://robotics-ftp.ait.kyushu-u.ac.jp` 的 `dogcentric` 目录；
- 该 FTP 主机 21/22/80/443/990/2121/8021 全部超时；`curl ftp://.../dogcentric/` 15 s 超时（exit 124，本次复测）；
- 镜像检索：Kaggle `[]`、HF `[]`、OpenAIRE 0 条、Wayback CDX `[]`；
- GitHub 只有解包脚本（`kiyoon/dogcentric_to_dir`），无数据。

### 3.2 结论

**本数据集未获得任何 clip，0 字节落盘**。若必须使用，只能联系维护者（Yumi Iwashita / M. S. Ryoo）索取可用镜像；
不要反复重试 FTP。

---

## 4. 复现与校验

```bash
# 下载（幂等：已有文件按 size 命中直接复用，不重下）
python scripts/download_mammalnet_canine.py --dataset all
python scripts/download_mammalnet_canine.py --dataset mammalnet --max-gb 1.6
python scripts/download_mammalnet_canine.py --dataset dogmo

# 校验（exit 0 = 31 项全过）
python scripts/verify_dataset_expansion.py
```

产物清单：

| 路径 | 内容 |
|------|------|
| `datasets/mammalnet_canine/annotations/` | annotation.tar 解包（16 文件，含 detection_annotations.json / taxonomy.txt / 4 套 composition 划分） |
| `datasets/mammalnet_canine/videos/{train,val,test}/*.mp4` | 752 个片段 |
| `datasets/mammalnet_canine/manifest.json` | 每文件：s3_key / video_id / part / split / breed / 源时长 / 行为区间 / bytes / md5 / sha256 / ffprobe clip 信息 |
| `datasets/mammalnet_canine/subset_annotations.json` | 抽样 clip 的原始 annotation |
| `datasets/mammalnet_canine/summary.json` | 抽样统计与字节配额覆盖 |
| `datasets/mammalnet_canine/recovery.md` | 全量子集恢复指引 |
| `datasets/dogmo_preview/manifest.json` | 49 素材：url / bytes / md5 / sha256 / kind |
| `datasets/dogmo_preview/summary.json` | 下载统计 + 「不可作为训练集」声明 |
| `reports/p1-dataset-expansion-verify.json` | 校验逐项结果（31 checks） |

---

## 5. 许可与合规

| 数据集 | 标注许可 | 视频许可 | 约束 |
|--------|----------|----------|------|
| MammalNet | CC BY（论文/仓库） | YouTube 源，作者声明不分发视频 | 研究引用；**不得再分发视频** |
| DogMo | 未声明（页面 CC BY-SA 4.0 只覆盖站点） | 未声明 | 预览素材仅定性参考 |
| DogCentric | 页面仅写 "please cite" | 未声明 | 未获得 |

所有数据仅用于本项目研究/验证，不进入产品分发链路，不上传任何云服务。

---

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-10-20 | 首版：MammalNet 家犬子集落盘（752 文件/1.543 GB）+ DogMo 预览（49 素材/227 MB）+ DogCentric 阻塞结论 |
