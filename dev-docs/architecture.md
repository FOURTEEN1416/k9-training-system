# Architecture — 项目架构 Truth

> Truth source: 基于 `technical-selection.md` 与 `research/` 调研文档
> 状态: ✅ 已确认（v1.1，2026-08-01 同步 Phase 2/3 实际结构）
> 日期: 2026-07-26（v1.0） / 2026-08-01（v1.1）

## 1. 架构概述

**架构风格**：单体应用 + 异步任务队列（Phase 0-3），微服务化留待 Phase 5

**核心原则**：
- 本地部署、无云依赖
- 异步推理（视频处理耗时长，不阻塞 API）
- 模块化分层（便于 Phase 4 引入 LLM/RL 等新模块）

## 2. 系统拓扑

```
┌─────────────────────────────────────────────────────────────┐
│                     用户浏览器（Vue 3）                       │
│  上传视频 │ 查看评分 │ 历史对比 │ 可视化标注 │ PDF 报告       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/WebSocket
┌──────────────────────────▼──────────────────────────────────┐
│                    FastAPI 后端（异步）                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 视频上传  │ │ 评分查询  │ │ 犬只管理  │ │ 模型管理  │       │
│  │ API      │ │ API      │ │ API      │ │ API      │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼───────┐  ┌───────▼───────┐  ┌──────▼────────┐
│   Celery      │  │ PostgreSQL 17  │  │   Redis       │
│  异步任务队列  │  │  + pgvector    │  │  Broker+缓存  │
│  ┌─────────┐  │  │  ┌──────────┐  │  └───────────────┘
│  │推理 Worker│ │  │  │业务数据   │  │
│  │YOLO26+   │  │  │  │关键点序列 │  │
│  │规则引擎+ │  │  │  │行为向量   │  │
│  │评分引擎  │  │  │  └──────────┘  │
│  └─────────┘  │  └────────────────┘
└───────────────┘
```

> **v1.1 修订**：① PostgreSQL 15 → 17（见 `runtime.md` §4.1）；② 推理 Worker 描述移除 PoseC3D（Phase 1.3 已跳过，见 ADR 0006 v1.1），改为"规则引擎"（Phase 1-2 主算法）+ ST-GCN+BC（Phase 3 主算法，规则引擎降级为备）。

## 3. 模块 Owner Map

### 3.1 前端模块（`frontend/`）

```
frontend/
├── src/
│   ├── views/           # 页面（实际文件名带 View 后缀）
│   │   ├── UploadView.vue   # 视频上传
│   │   ├── ReportView.vue   # 评分报告
│   │   ├── HistoryView.vue  # 历史对比
│   │   └── AdminView.vue    # 管理
│   ├── layouts/         # 布局
│   │   └── MainLayout.vue
│   ├── api/             # API 调用（http.ts + index.ts）
│   ├── stores/          # Pinia 状态（app.ts + dogs.ts）
│   ├── router/          # Vue Router
│   ├── App.vue
│   └── main.ts
└── vite.config.ts
```

> **v1.1 修订**：① 文件名 `Upload.vue` → `UploadView.vue` 等（与实际一致）；② 移除原 `components/VideoPlayer.vue` / `KeypointCanvas.vue` / `ScoreChart.vue` / `BehaviorTimeline.vue`——`frontend/src/components/` 目录**未建立**，UI 组件待 Phase 3+ 按需创建。

**Owner**：前端开发（Phase 0 建立骨架）

### 3.2 后端模块（`backend/`）

```
backend/
├── app/
│   ├── api/             # FastAPI 路由
│   │   ├── videos.py        # 视频上传/查询
│   │   ├── scores.py        # 评分查询
│   │   ├── scoring.py       # 评分触发
│   │   ├── dogs.py          # 犬只管理
│   │   ├── models.py        # 模型管理
│   │   ├── annotations.py   # 标注管理（Phase 2.1 数据飞轮）
│   │   ├── finetune.py      # 微调触发（Phase 2.1）
│   │   └── health.py        # 健康检查
│   ├── core/            # 核心配置
│   │   ├── config.py        # 配置
│   │   ├── database.py      # 异步 DB 连接
│   │   └── database_sync.py # 同步 DB 连接（Alembic 用）
│   ├── models/          # SQLAlchemy 模型（11 个）
│   │   ├── base.py          # Base + TimestampMixin
│   │   ├── dog.py / handler.py / video.py / training_session.py
│   │   ├── keypoint.py / behavior.py / behavior_vector.py
│   │   ├── score.py / ml_model.py
│   │   └── annotation.py    # Phase 2.1 标注数据
│   ├── schemas/         # Pydantic schema
│   │   └── common.py
│   ├── services/        # 业务逻辑
│   │   ├── label_studio.py  # LS 对接（Phase 2.1）
│   │   └── report.py        # PDF 报告
│   └── main.py          # FastAPI 入口
├── workers/             # Celery 任务（统一入口）
│   ├── celery_app.py        # Celery 应用
│   └── tasks.py             # 任务定义（含推理/评分/微调）
├── ml/                  # 机器学习模块
│   ├── pose/            # 姿态检测
│   │   ├── inference.py         # YOLO26-pose 推理
│   │   ├── train.py             # 训练
│   │   ├── export_engine.py     # ONNX/TensorRT 导出
│   │   ├── camera_projection.py # 3D→2D 相机投影（Phase 3.3）
│   │   ├── interpet4d_loader.py # InterPet4D 数据加载（Phase 3.3）
│   │   ├── lifting_pairing.py   # 2D-3D 配对（Phase 3.3b）
│   │   ├── download_dataset.py
│   │   └── motionbert/          # MotionBERT 2D→3D lifting（Phase 3.3c）
│   ├── behavior/        # 行为识别
│   │   ├── rule_engine.py       # 规则引擎（Phase 1-2 主算法）
│   │   ├── constants.py         # 16 类行为阈值常量
│   │   ├── object_detector.py   # 物体检测（Phase 1.5 选育）
│   │   ├── puppy_signals.py     # 幼犬信号（Phase 1.6）
│   │   └── stgcn_bc/            # ST-GCN+BC（Phase 3.1，目录非单文件）
│   ├── tracking/        # 多犬追踪（Phase 3.2）
│   │   ├── multi_dog_tracker.py # BoxMOT OccluBoost 追踪
│   │   ├── reid_extractor.py    # ReID 特征提取（Phase 3.2c）
│   │   ├── id_switch_monitor.py # ID switch 监控
│   │   ├── reid_finetune_dataset.py
│   │   └── types.py
│   └── scoring/         # 评分
│       ├── engine.py           # 评分引擎主入口
│       ├── conditions.py       # 命中条件
│       ├── schema.py           # Scene Literal 映射
│       └── configs/            # YAML 评分卡
│           ├── obedience_trial.yaml   # Phase 1 科目 5 维
│           ├── puppy_selection.yaml   # Phase 1 选育 3 维
│           ├── uspca_patrol.yaml      # Phase 2.3 USPCA 5 维
│           ├── working_dog_trial.yaml
│           └── (FCI-IGP 7 维待 Phase 3.4 添加)
├── alembic/             # 数据库迁移
│   └── versions/
│       ├── 5d7105cb28a9_phase0_initial_schema.py
│       ├── a1b2c3d4e5f6_phase1_4d_video_scene_report.py
│       └── b2c3d4e5f6a7_phase2_1a_annotation_tables.py
└── requirements.txt
```

> **v1.1 修订**：① `workers/` 实际只有 `celery_app.py` + `tasks.py`（原列 `inference.py/pose_detection.py/behavior_classification.py/scoring.py` 均不存在，统一在 `tasks.py`）；② `ml/pose/` 实际文件全部更名（原 `yolo26_pose.py/keypoint_utils.py` 不存在）；③ `ml/behavior/posec3d.py` 移除（ADR 0006 跳过）；④ `ml/scoring/` 实际为 `engine.py/conditions.py/schema.py/configs/`（原 `ga_t.py/uspca.py/fci_igp.py` 不存在）；⑤ `ml/inference/tensorrt.py` 移除（实际无此目录，TRT 集成在 `ml/pose/export_engine.py`）；⑥ 新增 `ml/tracking/` 模块（Phase 3.2）；⑦ 补全 `app/api/` 实际端点（annotations/finetune/health/scoring）；⑧ 补全 alembic 实际迁移文件。

**Owner**：后端开发（Phase 0 建立骨架）

### 3.3 数据层（PostgreSQL）

**核心表**（详见 `research/RESEARCH_IMPLEMENTATION.md` §六 + `backend/app/models/`）：
- `dogs` — 犬只档案
- `videos` — 训练视频
- `keypoints` — 24 关键点序列
- `behaviors` — 行为识别结果
- `scores` — 评分结果（7 维：准确度/延迟/保持/搜索效率/注意力/胆量/步态，Phase 3.4 扩展）
- `ml_models` — 模型版本
- `behavior_vectors` — 行为向量（pgvector）
- `handlers` — 训导员
- `training_sessions` — 训练会话
- `annotation_tasks` — 标注任务（Phase 2.1）
- `annotations` — 标注数据（Phase 2.1，DB 列名 `annotation_source`）

**Owner**：数据库设计（Phase 0 建立 schema）

### 3.4 数据目录（`data/`）

```
data/
├── uploads/            # 上传的视频（.gitignore）
├── generated/          # 生成的标注视频（.gitignore）
├── models_weights/     # 模型权重（.gitignore）
└── datasets/           # 数据集（.gitignore）
```

**Owner**：系统管理员

## 4. 进程与通信

### 4.1 进程拓扑

| 进程 | 作用 | 启动命令 |
|------|------|---------------------|
| FastAPI | API 服务 | `uvicorn backend.app.main:app` |
| Celery Worker | 异步推理 | `celery -A backend.workers.celery_app worker -l info` |
| PostgreSQL | 数据库 | 本地服务（端口 5433） |
| Redis | Broker+缓存 | 本地服务（端口 6379） |

> **v1.1 修订**：启动命令补 `backend.` 前缀（原 `uvicorn app.main:app` / `celery -A workers worker` 在项目根目录运行会找不到模块）；Celery 应用入口为 `backend.workers.celery_app`（非 `workers`）；PostgreSQL 端口 5433（避免与旧 PG15 的 5432 冲突，见 `runtime.md` §4.1）。

### 4.2 通信协议

- **前端 ↔ 后端**：HTTP REST + WebSocket（任务进度）
- **后端 ↔ Celery**：Redis Broker
- **Celery ↔ DB**：SQLAlchemy async
- **Celery ↔ ML**：进程内调用（PyTorch）

## 5. 数据流

### 5.1 视频推理数据流

```
1. 用户上传视频
   → POST /api/videos/upload
   → 存储到 data/uploads/
   → 创建 videos 记录
   → 触发 Celery 任务（workers/tasks.py::ingest_video）

2. Celery 推理任务
   → ml/pose/inference.py 检测 24 关键点
   → 存储关键点序列到 keypoints 表
   → ml/behavior/rule_engine.py 规则识别 16 类行为（Phase 2: P0 8 + P1 8）
   → 存储行为结果到 behaviors 表
   → ml/scoring/engine.py 计算 5 维评分（USPCA: 准确度/延迟/保持/搜索效率/注意力）
   → 存储评分到 scores 表
   → 更新 videos 状态为 completed

3. 用户查询结果
   → GET /api/scores/{video_id}
   → 返回评分 + 行为时间轴
   → GET /api/videos/{video_id}/report.pdf
   → 生成 PDF 报告
```

### 5.2 模型管理数据流

```
1. 管理员上传新模型
   → POST /api/models/upload
   → 存储权重到 data/models_weights/
   → 创建 models 记录

2. 切换模型版本
   → PATCH /api/models/{id}/activate
   → 更新 models.is_active
   → 下次推理使用新模型
```

## 6. 验证规则

### 6.1 API 契约验证

- 所有 API 使用 Pydantic schema 验证输入输出
- 错误响应统一格式：`{"error": {"code": "...", "message": "..."}}`
- 异步任务返回 task_id，通过 WebSocket 推送进度

### 6.2 数据验证

- 关键点：24 个，每个 (x, y, confidence)
- 行为：类别 ∈ 22 种枚举，置信度 ∈ [0, 1]
- 评分：7 维度，每维 ∈ [0, 100]

## 7. 安全边界

> 详见 `security-boundary.md`（Phase 0 创建）

**Phase 1 安全要点**：
- 本地部署，无外部网络暴露
- 文件上传：白名单格式 + 大小限制
- SQL 注入：SQLAlchemy 参数化查询
- 路径遍历：上传文件名清洗

## 8. 架构决策记录

### ADR-001: 单体应用 + 异步队列（而非微服务）

**决策**：Phase 0-3 采用单体应用 + Celery 异步队列
**理由**：
- 本地部署、用户规模小
- 微服务化增加运维复杂度
- Phase 5 商业化时再评估拆分

### ADR-002: PostgreSQL + pgvector（而非 MySQL + 外部向量库）

**决策**：使用 PostgreSQL + pgvector 同时处理关系数据和向量数据
**理由**：
- 单数据库简化部署
- pgvector 性能足够（百万级向量）
- 避免 Milvus/Pinecone 等额外服务

### ADR-003: 原生 Python 部署（而非 Docker）

**决策**：Phase 0-2 使用原生 Python venv 部署
**理由**：
- Windows 优先、用户环境熟悉
- Docker Desktop for Windows 体积大、用户不熟悉
- Phase 3+ Jetson 部署时再用容器

## 9. 未验证项

- ⏳ MMAction2 在 Windows 原生部署的兼容性（Phase 0 验证）
- ⏳ Celery 在 Windows 的稳定性（Phase 0 验证，可能需 celery-windows 补丁）
- ⏳ pgvector 在百万级行为向量的查询性能（Phase 2 验证）
- ⏳ TensorRT 在 Windows GPU 的安装复杂度（Phase 0 验证）

## 10. 架构演进路径

| 阶段 | 架构变化 |
|------|---------|
| Phase 0-1 | 单体 + Celery + PostgreSQL |
| Phase 2 | 加数据飞轮管道 |
| Phase 3 | 加多摄像头同步 + 3D 重建模块 |
| Phase 4 | 加 LLM 服务 / RL 训练管道 |
| Phase 5 | 评估微服务化 + 多基地联邦 |
