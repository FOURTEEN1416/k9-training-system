# Architecture — 项目架构 Truth

> Truth source: 基于 `technical-selection.md` 与 `research/` 调研文档
> 状态: ✅ 已确认（Phase 0-1 架构）
> 日期: 2026-07-26

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
│   Celery      │  │ PostgreSQL 15  │  │   Redis       │
│  异步任务队列  │  │  + pgvector    │  │  Broker+缓存  │
│  ┌─────────┐  │  │  ┌──────────┐  │  └───────────────┘
│  │推理 Worker│ │  │  │业务数据   │  │
│  │YOLO26+   │  │  │  │关键点序列 │  │
│  │PoseC3D+  │  │  │  │行为向量   │  │
│  │评分引擎  │  │  │  └──────────┘  │
│  └─────────┘  │  └────────────────┘
└───────────────┘
```

## 3. 模块 Owner Map

### 3.1 前端模块（`frontend/`）

```
frontend/
├── src/
│   ├── views/           # 页面
│   │   ├── Upload.vue       # 视频上传
│   │   ├── Report.vue       # 评分报告
│   │   ├── History.vue      # 历史对比
│   │   └── Admin.vue        # 管理
│   ├── components/      # 组件
│   │   ├── VideoPlayer.vue  # 视频播放器+标注
│   │   ├── KeypointCanvas.vue # 24 关键点 Canvas
│   │   ├── ScoreChart.vue   # ECharts 评分图
│   │   └── BehaviorTimeline.vue # 行为时间轴
│   ├── api/             # API 调用
│   ├── stores/          # Pinia 状态
│   └── router/          # Vue Router
└── vite.config.ts
```

**Owner**：前端开发（Phase 0 建立骨架）

### 3.2 后端模块（`backend/`）

```
backend/
├── app/
│   ├── api/             # FastAPI 路由
│   │   ├── videos.py        # 视频上传/查询
│   │   ├── scores.py        # 评分查询
│   │   ├── dogs.py          # 犬只管理
│   │   └── models.py        # 模型管理
│   ├── core/            # 核心配置
│   │   ├── config.py        # 配置
│   │   ├── security.py      # 安全
│   │   └── database.py      # DB 连接
│   ├── models/          # SQLAlchemy 模型
│   ├── schemas/         # Pydantic schema
│   ├── services/        # 业务逻辑
│   └── main.py          # FastAPI 入口
├── workers/             # Celery 任务
│   ├── inference.py         # 推理任务
│   ├── pose_detection.py    # YOLO26-pose
│   ├── behavior_classification.py # PoseC3D/ST-GCN
│   └── scoring.py           # 评分引擎
├── ml/                  # 机器学习模块
│   ├── pose/            # 姿态检测
│   │   ├── yolo26_pose.py
│   │   └── keypoint_utils.py
│   ├── behavior/        # 行为识别
│   │   ├── rule_engine.py   # 规则引擎
│   │   ├── posec3d.py       # PoseC3D
│   │   └── stgcn_bc.py      # ST-GCN+BC (Phase 3)
│   ├── scoring/         # 评分
│   │   ├── ga_t.py          # GA-T 标准
│   │   ├── uspca.py         # USPCA
│   │   └── fci_igp.py       # FCI-IGP
│   └── inference/       # 推理加速
│       └── tensorrt.py
├── alembic/             # 数据库迁移
└── requirements.txt
```

**Owner**：后端开发（Phase 0 建立骨架）

### 3.3 数据层（PostgreSQL）

**核心表**（详见 `research/RESEARCH_IMPLEMENTATION.md` §六）：
- `dogs` — 犬只档案
- `videos` — 训练视频
- `keypoints` — 24 关键点序列
- `behaviors` — 行为识别结果
- `scores` — 评分结果
- `models` — 模型版本
- `behavior_vectors` — 行为向量（pgvector）
- `handlers` — 训导员
- `training_sessions` — 训练会话

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

| 进程 | 作用 | 启动命令（Phase 0） |
|------|------|---------------------|
| FastAPI | API 服务 | `uvicorn app.main:app --reload` |
| Celery Worker | 异步推理 | `celery -A workers worker -l info` |
| PostgreSQL | 数据库 | 本地服务 |
| Redis | Broker+缓存 | 本地服务 |

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
   → 触发 Celery 任务 inference.ingest_video

2. Celery 推理任务
   → ml/pose/yolo26_pose.py 检测 24 关键点
   → 存储关键点序列到 keypoints 表
   → ml/behavior/rule_engine.py 规则识别 4 类基础动作
   → ml/behavior/posec3d.py 识别 8 类 P0 行为
   → 存储行为结果到 behaviors 表
   → ml/scoring/ga_t.py 计算 3 维评分
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
