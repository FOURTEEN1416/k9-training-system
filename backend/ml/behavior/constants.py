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
BITE = "bite"          # 咬

P0_BEHAVIORS = (SIT, DOWN, STAND, HEEL, SIT_UP, STAY, BARK, BITE)
NUM_P0_BEHAVIORS = 8

# 行为中文名称
BEHAVIOR_NAMES_CN = {
    SIT: "坐",
    DOWN: "卧",
    STAND: "立",
    HEEL: "随行",
    SIT_UP: "坐立",
    STAY: "停留",
    BARK: "叫",
    BITE: "咬",
}
