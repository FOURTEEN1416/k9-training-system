"""行为识别常量定义.

Owner: ML 开发
Phase: 1.2

24 关键点索引（与 data/dog-pose.yaml kpt_names 一致）:
    0-5:  左侧肢（前左爪/膝/肘 + 后左爪/膝/肘）
    6-11: 右侧肢（前右爪/膝/肘 + 后右爪/膝/肘）
    12-13: 尾（尾根/尾尖）
    14-15: 耳根（左/右）
    16-17: 头部中线（鼻/下巴）
    18-19: 耳尖（左/右）
    20-21: 眼（左/右）
    22: 鬐甲（withers，肩峰）
    23: 喉咙（throat）
"""
from __future__ import annotations

# ===== 24 关键点索引 =====

# 左侧肢
FRONT_LEFT_PAW = 0
FRONT_LEFT_KNEE = 1
FRONT_LEFT_ELBOW = 2
REAR_LEFT_PAW = 3
REAR_LEFT_KNEE = 4
REAR_LEFT_ELBOW = 5

# 右侧肢
FRONT_RIGHT_PAW = 6
FRONT_RIGHT_KNEE = 7
FRONT_RIGHT_ELBOW = 8
REAR_RIGHT_PAW = 9
REAR_RIGHT_KNEE = 10
REAR_RIGHT_ELBOW = 11

# 尾
TAIL_START = 12
TAIL_END = 13

# 耳根
LEFT_EAR_BASE = 14
RIGHT_EAR_BASE = 15

# 头部中线
NOSE = 16
CHIN = 17

# 耳尖
LEFT_EAR_TIP = 18
RIGHT_EAR_TIP = 19

# 眼
LEFT_EYE = 20
RIGHT_EYE = 21

# 躯干
WITHERS = 22  # 鬐甲（肩峰）
THROAT = 23

NUM_KEYPOINTS = 24

# 关键点分组（便于规则引擎引用）
FRONT_PAWS = (FRONT_LEFT_PAW, FRONT_RIGHT_PAW)
FRONT_KNEES = (FRONT_LEFT_KNEE, FRONT_RIGHT_KNEE)
FRONT_ELBOWS = (FRONT_LEFT_ELBOW, FRONT_RIGHT_ELBOW)
REAR_PAWS = (REAR_LEFT_PAW, REAR_RIGHT_PAW)
REAR_KNEES = (REAR_LEFT_KNEE, REAR_RIGHT_KNEE)
REAR_ELBOWS = (REAR_LEFT_ELBOW, REAR_RIGHT_ELBOW)
ALL_PAWS = FRONT_PAWS + REAR_PAWS
ALL_KNEES = FRONT_KNEES + REAR_KNEES

# 头部关键点
HEAD_POINTS = (NOSE, CHIN, LEFT_EAR_BASE, RIGHT_EAR_BASE,
               LEFT_EAR_TIP, RIGHT_EAR_TIP, LEFT_EYE, RIGHT_EYE)

# ===== 8 类行为类别（P0 基础） =====

# 行为类别常量
SIT = "sit"            # 坐
DOWN = "down"          # 卧
STAND = "stand"        # 立
HEEL = "heel"          # 随行
SIT_UP = "sit_up"      # 坐立
STAY = "stay"          # 停留
BARK = "bark"          # 叫
BITE = "bite"          # 咬（嘴部活跃动作）

P0_BEHAVIORS = (SIT, DOWN, STAND, HEEL, SIT_UP, STAY, BARK, BITE)
NUM_P0_BEHAVIORS = 8

# ===== P1 训练专项 8 类（Phase 2，依据 RESEARCH_STANDARDS.md §4.1） =====
# 注意：P1 的扑咬用 APPREHEND，避免与 P0 的 BITE（嘴部活跃）冲突

TRACK = "track"            # 追踪：鼻尖贴近地面 + 路径跟随
ALERT_SIT = "alert_sit"    # 示警坐：检出目标后坐姿示警
ALERT_DOWN = "alert_down"  # 示警卧：检出目标后卧姿示警
APPREHEND = "apprehend"    # 扑咬：高速接近 + 嘴部接触目标（USPCA Apprehension）
ESCORT = "escort"          # 押解：犬侧伴随 + 保持警觉
OBSTACLE = "obstacle"      # 障碍穿越：跳跃/攀爬姿态序列
RECALL = "recall"          # 返回：远离→朝向训导员快速移动
WATCH = "watch"            # 警戒：头部抬起 + 身体紧绷

P1_BEHAVIORS = (TRACK, ALERT_SIT, ALERT_DOWN, APPREHEND, ESCORT, OBSTACLE, RECALL, WATCH)
NUM_P1_BEHAVIORS = 8

# P0 + P1 = 16 类
ALL_BEHAVIORS = P0_BEHAVIORS + P1_BEHAVIORS
NUM_ALL_BEHAVIORS = NUM_P0_BEHAVIORS + NUM_P1_BEHAVIORS  # 16

# ===== P2 高级 6 类（Phase 3 新增，依据 FCI-IGP 国际工作犬标准） =====
# 来源: dev-docs/research/RESEARCH_FCI_IGP_STANDARD.md + RESEARCH_STGCN_BC.md §5.1
# 覆盖 FCI-IGP 三阶段（A 追踪 / B 服从 / C 护卫）的高级行为

GUARD = "guard"              # 守卫：警戒状态下的防御性姿态（IGP-C 护卫阶段）
RELEASE = "release"          # 放口：扑咬后释放目标（IGP-C DQ 硬约束: 不放口=取消资格）
RETRIEVE = "retrieve"        # 衔取：叼取物品并返回（IGP-B 服从阶段）
JUMP = "jump"                # 跳跃：跨越障碍（IGP-B 服从阶段，与 obstacle 区分: jump 专指高度跳跃）
SCALE = "scale"              # 攀登：攀越墙/坡（IGP-B 障碍专项）
SEARCH_BLIND = "search_blind"  # 搜索盲区：IGP-A 追踪阶段的高级搜索行为（定向 + 盲搜）

P2_BEHAVIORS = (GUARD, RELEASE, RETRIEVE, JUMP, SCALE, SEARCH_BLIND)
NUM_P2_BEHAVIORS = 6

# P0 + P1 + P2 = 22 类（Phase 3 ST-GCN+BC 完整标签集）
ALL_BEHAVIORS_22 = P0_BEHAVIORS + P1_BEHAVIORS + P2_BEHAVIORS
NUM_BEHAVIORS_22 = NUM_P0_BEHAVIORS + NUM_P1_BEHAVIORS + NUM_P2_BEHAVIORS  # 22

# 行为中文名称
BEHAVIOR_NAMES_CN = {
    # P0
    SIT: "坐",
    DOWN: "卧",
    STAND: "立",
    HEEL: "随行",
    SIT_UP: "坐立",
    STAY: "停留",
    BARK: "叫",
    BITE: "咬",
    # P1
    TRACK: "追踪",
    ALERT_SIT: "示警坐",
    ALERT_DOWN: "示警卧",
    APPREHEND: "扑咬",
    ESCORT: "押解",
    OBSTACLE: "障碍穿越",
    RECALL: "返回",
    WATCH: "警戒",
    # P2
    GUARD: "守卫",
    RELEASE: "放口",
    RETRIEVE: "衔取",
    JUMP: "跳跃",
    SCALE: "攀登",
    SEARCH_BLIND: "搜索盲区",
}

# 行为->科目映射（依据 RESEARCH_STANDARDS.md §4.1）
BEHAVIOR_SUBJECTS = {
    # P0 -> 服从/搜索
    SIT: "服从",
    DOWN: "服从",
    STAND: "服从",
    HEEL: "服从",
    SIT_UP: "服从",
    STAY: "服从",
    BARK: "服从/警戒",
    BITE: "服从",
    # P1 -> 训练专项
    TRACK: "追踪",
    ALERT_SIT: "搜毒/搜爆",
    ALERT_DOWN: "血迹搜索",
    APPREHEND: "搜捕",
    ESCORT: "搜捕",
    OBSTACLE: "服从/巡逻",
    RECALL: "服从",
    WATCH: "巡逻",
    # P2 -> FCI-IGP 国际工作犬标准
    GUARD: "IGP-C 护卫",
    RELEASE: "IGP-C 护卫",
    RETRIEVE: "IGP-B 服从",
    JUMP: "IGP-B 服从",
    SCALE: "IGP-B 服从",
    SEARCH_BLIND: "IGP-A 追踪",
}
