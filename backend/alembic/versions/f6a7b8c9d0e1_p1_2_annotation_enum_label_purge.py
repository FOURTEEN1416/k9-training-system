"""p1_2_annotation_enum_label_purge

P1-2 修正补强：把标注枚举类型重建为**仅含 ORM 成员名**（大写），删除历史小写标签。

背景（2026-10-20 对抗验收发现）:
    前一迁移 ``e5f6a7b8c9d0`` 用 ``ALTER TYPE ... ADD VALUE`` 补了大写标签并把历史
    数据规范化，但 **PG 无法删除枚举标签**——小写标签 ``pending/completed`` 等仍然
    保留在类型里。后果:
        任何 raw-SQL 路径（legacy 脚本、渗透测试播种、psql 手写）写入小写值都会
        **静默成功**，而 ORM 读取同名成员时抛 ``LookupError``（ORM 期望成员名），
        表现为接口 500。对抗验收实测: 渗透脚本 ``INSERT ... status='completed'``
        → ``GET /api/annotations/tasks/{id}`` 返回 500。
    这使大小写混存成为一类"可被任意旁路触发"的隐性故障，故必须从类型层面消除。

本迁移:
    1. 新建仅含大写成员名的枚举类型 → 用 ``upper()``/CASE 转换现有列 → 删旧类型 → 重命名；
    2. 恢复列的 server default 为成员名；
    3. 同法处理 annotation_type 与 annotation_source（source 需 CASE 特判
       ``prelabel → MODEL_PRELABEL``，因该成员名与 value 不一致）。

前置: 逐表检查是否存在小写残留；本迁移用 ``upper()`` 一次性归并，混存行亦安全。
注意: 迁移需表重写（当前研发库 0 行，秒级）；线上须先备份。

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-10-20 00:40:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f6a7b8c9d0e1'
down_revision: Union[str, Sequence[str], None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- 1. annotation_tasks.status ---
    op.execute("ALTER TABLE annotation_tasks ALTER COLUMN status DROP DEFAULT")
    op.execute("CREATE TYPE annotation_task_status_v2 AS ENUM ('PENDING','IN_PROGRESS','COMPLETED','FAILED')")
    op.execute(
        "ALTER TABLE annotation_tasks ALTER COLUMN status TYPE annotation_task_status_v2 "
        "USING upper(status::text)::annotation_task_status_v2"
    )
    op.execute("DROP TYPE annotation_task_status")
    op.execute("ALTER TYPE annotation_task_status_v2 RENAME TO annotation_task_status")
    op.execute("ALTER TABLE annotation_tasks ALTER COLUMN status SET DEFAULT 'PENDING'")

    # --- 2. annotations.annotation_type ---
    op.execute("CREATE TYPE annotation_type_v2 AS ENUM ('KEYPOINT','BEHAVIOR','BBOX')")
    op.execute(
        "ALTER TABLE annotations ALTER COLUMN annotation_type TYPE annotation_type_v2 "
        "USING upper(annotation_type::text)::annotation_type_v2"
    )
    op.execute("DROP TYPE annotation_type")
    op.execute("ALTER TYPE annotation_type_v2 RENAME TO annotation_type")

    # --- 3. annotations.annotation_source（成员名与 value 不一致，CASE 特判）---
    op.execute("ALTER TABLE annotations ALTER COLUMN annotation_source DROP DEFAULT")
    op.execute("CREATE TYPE annotation_source_v2 AS ENUM ('HUMAN','MODEL_PRELABEL','IMPORTED')")
    op.execute(
        "ALTER TABLE annotations ALTER COLUMN annotation_source TYPE annotation_source_v2 "
        "USING (CASE lower(annotation_source::text) "
        "         WHEN 'prelabel' THEN 'MODEL_PRELABEL' "
        "         WHEN 'model_prelabel' THEN 'MODEL_PRELABEL' "
        "         ELSE upper(annotation_source::text) END)::annotation_source_v2"
    )
    op.execute("DROP TYPE annotation_source")
    op.execute("ALTER TYPE annotation_source_v2 RENAME TO annotation_source")
    op.execute("ALTER TABLE annotations ALTER COLUMN annotation_source SET DEFAULT 'HUMAN'")


def downgrade() -> None:
    """回到含小写标签的旧枚举（保留历史形态；数据转小写）。"""
    op.execute("ALTER TABLE annotation_tasks ALTER COLUMN status DROP DEFAULT")
    op.execute("CREATE TYPE annotation_task_status_v2 AS ENUM ('pending','in_progress','completed','failed','PENDING','IN_PROGRESS','COMPLETED','FAILED')")
    op.execute(
        "ALTER TABLE annotation_tasks ALTER COLUMN status TYPE annotation_task_status_v2 "
        "USING lower(status::text)::annotation_task_status_v2"
    )
    op.execute("DROP TYPE annotation_task_status")
    op.execute("ALTER TYPE annotation_task_status_v2 RENAME TO annotation_task_status")
    op.execute("ALTER TABLE annotation_tasks ALTER COLUMN status SET DEFAULT 'pending'")

    op.execute("CREATE TYPE annotation_type_v2 AS ENUM ('keypoint','behavior','bbox','KEYPOINT','BEHAVIOR','BBOX')")
    op.execute(
        "ALTER TABLE annotations ALTER COLUMN annotation_type TYPE annotation_type_v2 "
        "USING lower(annotation_type::text)::annotation_type_v2"
    )
    op.execute("DROP TYPE annotation_type")
    op.execute("ALTER TYPE annotation_type_v2 RENAME TO annotation_type")

    op.execute("ALTER TABLE annotations ALTER COLUMN annotation_source DROP DEFAULT")
    op.execute("CREATE TYPE annotation_source_v2 AS ENUM ('human','prelabel','imported','HUMAN','MODEL_PRELABEL','IMPORTED')")
    op.execute(
        "ALTER TABLE annotations ALTER COLUMN annotation_source TYPE annotation_source_v2 "
        "USING (CASE upper(annotation_source::text) "
        "         WHEN 'MODEL_PRELABEL' THEN 'prelabel' "
        "         ELSE lower(annotation_source::text) END)::annotation_source_v2"
    )
    op.execute("DROP TYPE annotation_source")
    op.execute("ALTER TYPE annotation_source_v2 RENAME TO annotation_source")
    op.execute("ALTER TABLE annotations ALTER COLUMN annotation_source SET DEFAULT 'human'")
