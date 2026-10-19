"""p1_2_flywheel_review_enums

P1-2 数据飞轮：标注枚举规范化（大小写）+ 服务端默认值修正。

背景:
    ORM ``Enum(AnnotationTaskStatus)`` 默认存 **成员名**（PENDING/IN_PROGRESS/
    COMPLETED/FAILED），而旧迁移 ``b2c3d4e5f6a7`` 创建的 enum 标签是小写
    （pending/in_progress/completed/failed），且 ``server_default='pending'`` /
    ``'human'``。结果:
        - ORM 读取小写历史行时抛 ``LookupError``（值不在成员名下）
        - 省略列插入时 server_default 又写回小写

本迁移:
    1. 显式枚举并 ADD VALUE（成员名大写）:
        annotation_task_status = PENDING / IN_PROGRESS / COMPLETED / FAILED
        annotation_type        = KEYPOINT / BEHAVIOR / BBOX
        annotation_source      = HUMAN / MODEL_PRELABEL / IMPORTED
       注意 ``AnnotationSource.MODEL_PRELABEL`` 是**成员名**，其 value 才是
       ``prelabel``；``upper('prelabel')`` 会得到不存在的 PRELABEL，
       因此 source 归一化用 CASE 特殊处理。
    2. 用 ``USING`` 表达式把历史小写数据转换成大写成员名。
    3. 把 server default 改为大写成员名（'PENDING' / 'HUMAN'）。
    4. 所有 ALTER TYPE ADD VALUE 放在 autocommit_block（PG 兼容）。

REQUIRES scripts/finetune_from_annotations.py update（行 79/101）:
    该 legacy 脚本过滤 ``t.status='completed'`` 与 ``annotation_type='keypoint'``；
    枚举规范化后它将查不到任何行（导出为空）。脚本不在 P1-2 白名单，
    列为**发布阻断项**，须由 scripts owner 同步修改。新链路
    （POST /api/finetune/export → Celery finetune.train_from_annotations）
    不依赖该脚本。

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-10-19 23:30:00

注意:
    - PG 不可达时仅能离线编译（alembic upgrade head --sql），
      不能证明真实 PG 行为；上线前须备份后 dry-run。
    - 未新增 annotation_tasks.video_id 唯一索引（历史可能多任务）。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TASK_STATUS = ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED')
ANNOTATION_TYPE = ('KEYPOINT', 'BEHAVIOR', 'BBOX')
ANNOTATION_SOURCE = ('HUMAN', 'MODEL_PRELABEL', 'IMPORTED')


def _add_enum_values(type_name: str, values: Sequence[str]) -> None:
    """ADD VALUE 必须在 autocommit_block 中执行（PG 限制）。"""
    with op.get_context().autocommit_block():
        for value in values:
            op.execute(
                sa.text(
                    f"ALTER TYPE {type_name} ADD VALUE IF NOT EXISTS '{value}'"
                )
            )


def upgrade() -> None:
    # 1. 显式枚举 + ADD VALUE（纯增量，幂等）
    _add_enum_values('annotation_task_status', TASK_STATUS)
    _add_enum_values('annotation_type', ANNOTATION_TYPE)
    _add_enum_values('annotation_source', ANNOTATION_SOURCE)

    # 2. 历史数据规范化（小写 value → 大写成员名）
    #    status: pending → PENDING / in_progress → IN_PROGRESS ...
    op.execute(
        """
        ALTER TABLE annotation_tasks
        ALTER COLUMN status TYPE annotation_task_status
        USING upper(status::text)::annotation_task_status
        """
    )
    op.execute(
        """
        ALTER TABLE annotations
        ALTER COLUMN annotation_type TYPE annotation_type
        USING upper(annotation_type::text)::annotation_type
        """
    )
    #    source: 成员名 MODEL_PRELABEL，value 是 prelabel；
    #    upper('prelabel') = PRELABEL（不存在），必须 CASE 特判。
    op.execute(
        """
        ALTER TABLE annotations
        ALTER COLUMN annotation_source TYPE annotation_source
        USING (
            CASE lower(annotation_source::text)
                WHEN 'prelabel' THEN 'MODEL_PRELABEL'
                WHEN 'model_prelabel' THEN 'MODEL_PRELABEL'
                ELSE upper(annotation_source::text)
            END
        )::annotation_source
        """
    )

    # 3. server default 改为大写成员名
    op.alter_column(
        'annotation_tasks',
        'status',
        existing_type=sa.Enum(name='annotation_task_status'),
        server_default=sa.text("'PENDING'"),
        existing_nullable=False,
    )
    op.alter_column(
        'annotations',
        'annotation_source',
        existing_type=sa.Enum(name='annotation_source'),
        server_default=sa.text("'HUMAN'"),
        existing_nullable=False,
    )


def downgrade() -> None:
    """回滚到小写 value（enum 标签无法删除，仅改数据与默认值）."""
    op.alter_column(
        'annotation_tasks',
        'status',
        existing_type=sa.Enum(name='annotation_task_status'),
        server_default=sa.text("'pending'"),
        existing_nullable=False,
    )
    op.alter_column(
        'annotations',
        'annotation_source',
        existing_type=sa.Enum(name='annotation_source'),
        server_default=sa.text("'human'"),
        existing_nullable=False,
    )

    op.execute(
        """
        ALTER TABLE annotation_tasks
        ALTER COLUMN status TYPE annotation_task_status
        USING lower(status::text)::annotation_task_status
        """
    )
    op.execute(
        """
        ALTER TABLE annotations
        ALTER COLUMN annotation_type TYPE annotation_type
        USING lower(annotation_type::text)::annotation_type
        """
    )
    op.execute(
        """
        ALTER TABLE annotations
        ALTER COLUMN annotation_source TYPE annotation_source
        USING (
            CASE lower(annotation_source::text)
                WHEN 'model_prelabel' THEN 'prelabel'
                ELSE lower(annotation_source::text)
            END
        )::annotation_source
        """
    )
