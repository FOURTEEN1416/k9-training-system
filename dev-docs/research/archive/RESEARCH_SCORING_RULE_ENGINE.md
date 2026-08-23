# 调研：评分规则可配置架构

> Owner: 立项阶段调研
> 日期: 2026-07-27
> 依据: 用户立项研讨第 3 轮反馈（评分靠公开数据 + 系统动态调节规则）
> 状态: ✅ 完成

## 0. 调研目标

回答三个问题：
1. "动态调节规则"在工程上有哪些成熟实现方案？
2. 工作犬评分场景该选哪种？为什么？
3. 3-4 周内如何落地可配置评分引擎？

## 1. 成熟方案对比

### 1.1 Python 规则引擎生态

| 框架 | Stars | 许可 | 特性 | 适合本项目 |
|------|-------|------|------|-----------|
| [simple-rule-engine](https://pypi.org/project/simpleruleengine/) | 中 | MIT | 声明式 + 决策表 + 评分规则 + 链式规则 | ✅ 推荐 |
| [durable_rules](https://github.com/jruizariza/durable_rules) | 高 | MIT | 事件流 + CEP + 前向链推理 | ❌ 过重 |
| [experta](https://github.com/experta/experta) | 高 | LGPL | 专家系统 + RETE 算法 | ❌ 过重 |
| [business-rules](https://github.com/venmo/business-rules) | 中 | Apache | Python 变量 + 操作符 + YAML | ⚠️ 一般 |
| [JSONLogic](https://jsonlogic.com/) | 跨语言 | MIT | JSON 规则 + 跨语言 | ⚠️ 评分支持弱 |
| [JVS 规则引擎](https://segmentfault.com/a/1190000047223929) | 商业 | 商业 | 可视化拖拽 + SQL-like + Python | ❌ 商业 |

### 1.2 AI Skill 评分卡范式

来源：[阿里云 - AI Skill 构建十个层次](https://developer.aliyun.com:443/article/1746138)、[通用 Agent 技能 5 个开箱即用 Skill](https://developer.aliyun.com:443/article/1748850)

**YAML 评分卡结构**（行业事实标准）：
```yaml
scoring_engine:
  name: "<评分卡名称>"
  version: "<版本>"
  dimensions:
    - id: <维度ID>
      name: "<维度名称>"
      weight: <0-1 权重>
      rules:
        - condition: "<表达式>"
          score: <分数>
          label: "<标签>"
  thresholds:
    high: <高分阈值>
    medium: <中分阈值>
    low: <低分阈值>
```

**核心思想**（引自阿里云文章）：
> 当你在写第 3 个评分逻辑的硬编码 if-else 时，就该想想：能不能把规则抽到 YAML 里，让业务改配置而不是改代码？

### 1.3 信用评分卡工业实践

来源：[scorecardpy 全解析](https://blog.gitcode.com/49a7d6a33c421a55624ae88fcf93b920.html)

- **WOE 转换**：将原始值转为对目标的预测贡献度
- **IV 值**：变量预测能力体检
- **卡方分箱**：统计显著性 + 业务可解释性

**对本项目意义**：
- 工作犬评分无历史标签数据，无法用 WOE/IV
- 但分箱思想可用：把连续信号（速度/延迟）离散化为高/中/低

## 2. 工作犬评分场景分析

### 2.1 评分维度映射

**幼犬选育场景**（3 维，来自调研一）：
| 维度 | 权重 | 输入信号 | 评分规则示例 |
|------|------|---------|------------|
| 食物欲望 | 0.40 | approach_latency, approach_speed | latency<1s 且 speed>2m/s → 90 分 |
| 玩具欲望 | 0.40 | chase_latency, chase_speed, hold_duration | latency<0.5s 且 hold>3s → 90 分 |
| 胆量 | 0.20 | retreat_distance, freeze_duration, recovery_time | retreat<0.5m 且 recovery<2s → 90 分 |

**科目测评场景**（5 维，来自 PAT + 警犬标准）：
| 维度 | 权重 | 输入信号 | 评分规则示例 |
|------|------|---------|------------|
| 准确度 | 0.30 | action_correct, action_count | correct/total > 0.95 → 90 分 |
| 延迟 | 0.20 | command_to_action_latency | latency<0.5s → 90 分 |
| 保持 | 0.20 | action_duration | duration>30s → 90 分 |
| 注意力 | 0.15 | focus_ratio | focus>0.8 → 90 分 |
| 步态 | 0.15 | gait_score | gait>0.85 → 90 分 |

### 2.2 评分引擎需求

1. **YAML 配置化**：训导员可改阈值/权重，不改代码
2. **多维加权**：Σ(score_i × weight_i)
3. **场景隔离**：选育评分卡 vs 科目评分卡独立
4. **版本控制**：规则变更可追溯（git 管理 YAML）
5. **动态生效**：改配置不重启服务
6. **可解释**：每条评分给出命中规则 + 标签

## 3. 推荐方案：自研轻量评分引擎

### 3.1 为什么不直接用 simple-rule-engine？

**原因**：
1. simple-rule-engine 的 condition 表达式用 Python 语法，训导员难懂
2. 工作犬评分有特殊需求（场景隔离 + 信号字典 + 标签映射）
3. 自研约 200 行代码，3-4 周内可做，控制力更强
4. AGENTS.md §1.1 允许"必要时自研"

### 3.2 架构设计

```
backend/ml/scoring/
├── __init__.py
├── engine.py              # 评分引擎核心（~150 行）
├── schema.py              # Pydantic 数据模型（~80 行）
├── conditions.py          # 条件表达式解析（~100 行）
└── configs/               # YAML 评分卡
    ├── puppy_selection.yaml   # 幼犬选育评分卡
    └── obedience_trial.yaml   # 科目测评评分卡
```

### 3.3 核心 API

```python
from backend.ml.scoring import ScoringEngine, ScoringContext

# 加载评分卡（YAML）
engine = ScoringEngine.from_yaml("backend/ml/scoring/configs/puppy_selection.yaml")

# 构建上下文（来自姿态/行为识别输出）
ctx = ScoringContext(
    signals={
        "approach_latency": 0.8,    # 秒
        "approach_speed": 2.5,      # m/s
        "chase_latency": 0.4,
        "chase_speed": 4.0,
        "hold_duration": 4.2,
        "retreat_distance": 0.3,
        "freeze_duration": 0.0,
        "recovery_time": 1.5,
    },
    scene="puppy_selection",
)

# 评分
result = engine.evaluate(ctx)
print(result.total_score)        # 85.5
print(result.dimension_scores)   # {"food_drive": 90, "prey_drive": 85, "courage": 75}
print(result.labels)             # {"food_drive": "高", "prey_drive": "高", "courage": "中"}
print(result.hit_rules)          # 每维命中的规则详情
print(result.passed)             # True（≥70）
```

### 3.4 评分卡 YAML 完整示例

```yaml
# backend/ml/scoring/configs/puppy_selection.yaml
scoring_engine:
  name: "幼犬选育评分卡"
  version: "1.0.0"
  scene: "puppy_selection"
  dimensions:
    - id: food_drive
      name: "食物欲望"
      weight: 0.40
      rules:
        - id: food_high
          condition: "approach_latency < 1.0 and approach_speed > 2.0"
          score: 90
          label: "高"
        - id: food_medium
          condition: "approach_latency < 3.0 and approach_speed > 1.0"
          score: 60
          label: "中"
        - id: food_low
          condition: "approach_latency >= 5.0"
          score: 20
          label: "低"
        - id: food_default
          condition: "True"
          score: 40
          label: "中低"
    - id: prey_drive
      name: "猎物欲望"
      weight: 0.40
      rules:
        - id: prey_high
          condition: "chase_latency < 0.5 and hold_duration > 3.0"
          score: 90
          label: "高"
        - id: prey_medium
          condition: "chase_latency < 1.5 and hold_duration > 1.0"
          score: 60
          label: "中"
        - id: prey_low
          condition: "chase_latency >= 3.0"
          score: 20
          label: "低"
        - id: prey_default
          condition: "True"
          score: 40
          label: "中低"
    - id: courage
      name: "胆量"
      weight: 0.20
      rules:
        - id: courage_high
          condition: "retreat_distance < 0.5 and recovery_time < 2.0"
          score: 90
          label: "高"
        - id: courage_medium
          condition: "retreat_distance < 1.5 and recovery_time < 5.0"
          score: 60
          label: "中"
        - id: courage_low
          condition: "retreat_distance >= 2.0 or freeze_duration > 10.0"
          score: 20
          label: "低"
        - id: courage_default
          condition: "True"
          score: 40
          label: "中低"
  thresholds:
    pass: 70        # 合格
    borderline: 60  # 基本合格
    fail: 0         # 淘汰
  aggregation: "weighted_sum"  # weighted_sum | max | min
```

### 3.5 动态调节实现

**配置热加载**：
```python
# backend/ml/scoring/engine.py
class ScoringEngine:
    _instances: dict[str, "ScoringEngine"] = {}
    _mtime: dict[str, float] = {}

    @classmethod
    def get(cls, config_path: str) -> "ScoringEngine":
        """单例 + 热加载：YAML 修改后下次调用自动重载."""
        mtime = Path(config_path).stat().st_mtime
        if config_path not in cls._instances or cls._mtime[config_path] != mtime:
            cls._instances[config_path] = cls.from_yaml(config_path)
            cls._mtime[config_path] = mtime
        return cls._instances[config_path]
```

**API 端点**：
- `GET /api/scoring/configs` —— 列出所有评分卡
- `GET /api/scoring/configs/{scene}` —— 获取某场景评分卡
- `PUT /api/scoring/configs/{scene}` —— 更新评分卡（写 YAML）
- `POST /api/scoring/evaluate` —— 评分（body: signals + scene）

## 4. 与既有 truth 文档的差异

| 既有文档假设 | 调研结论 | 需更新 |
|------------|---------|--------|
| 7 维评分（准确度/延迟/保持/搜索效率/注意力/胆量/步态）| 选育 3 维 + 科目 5 维，分离 | ✅ 需重设 |
| 评分逻辑硬编码在代码里 | YAML 配置化 + 热加载 | ✅ 需重设 |
| 评分规则不可配置 | 训导员可改阈值/权重 | ✅ 需新增 |
| GA-T 3 维评分 | 拿不到 GA-T 原文，用公开数据自设 | ✅ 需更新 |

## 5. 学术副产物潜力

**方向**：可解释 AI 评分卡 + 工作犬行为识别

- 贡献 1：工作犬行为信号 → 评分规则的标准化框架
- 贡献 2：YAML 配置化评分引擎开源
- 贡献 3：在 DogMo 数据集上的评分基准

**目标会议**：ACM MM Application Track / AAAI Application Track / 动物行为学交叉会议

## 6. 引用

- [simple-rule-engine PyPI](https://pypi.org/project/simpleruleengine/)
- [业务规则频繁变动？可视化规则引擎 - SegmentFault](https://blog.segmentfault.com/a/1190000047223929)
- [AI Skill 构建十个层次 - 阿里云](https://developer.aliyun.com:443/article/1746138)
- [通用 Agent 技能 5 个开箱即用 Skill - 阿里云](https://developer.aliyun.com:443/article/1748850)
- [scorecardpy 全解析 - GitCode](https://blog.gitcode.com/49a7d6a33c421a55624ae88fcf93b920.html)
- [Python 规则使用全指南 - PingCode](https://docs.pingcode.com/insights/alg0obu3jal2qbsnxcntcurv)
- [规则层权重如何影响决策引擎输出结果 - CSDN](https://ask.csdn.net/questions/8591094)
