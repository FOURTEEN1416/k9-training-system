"""phase2_1a_annotation_tables

1. 创建 annotation_tasks 表（标注任务，关联 Label Studio）
2. 创建 annotations 表（逐帧标注数据：关键点/行为/检测框）
3. behavior_class 枚举新增 P1 8 类（TRACK/ALERT_SIT/ALERT_DOWN/APPREHEND/ESCORT/OBSTACLE/RECALL/WATCH）

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-29 22:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. annotation_tasks 表
    op.create_table(
        'annotation_tasks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=False),
        sa.Column('handler_id', sa.Integer(), nullable=True),
        sa.Column('ls_project_id', sa.Integer(), nullable=True, comment='Label Studio project ID'),
        sa.Column('ls_task_id', sa.Integer(), nullable=True, comment='Label Studio task ID'),
        sa.Column('status', sa.Enum('pending', 'in_progress', 'completed', 'failed', name='annotation_task_status'),
                  nullable=False, server_default='pending'),
        sa.Column('total_frames', sa.Integer(), nullable=True, comment='视频总帧数'),
        sa.Column('annotated_frames', sa.Integer(), nullable=False, server_default='0', comment='已标注帧数'),
        sa.Column('annotation_types', sa.JSON(), nullable=True, comment='标注类型列表'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['handler_id'], ['handlers.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_annotation_tasks_video_id', 'annotation_tasks', ['video_id'])
    op.create_index('ix_annotation_tasks_handler_id', 'annotation_tasks', ['handler_id'])
    op.create_index('ix_annotation_tasks_status', 'annotation_tasks', ['status'])

    # 2. annotations 表
    op.create_table(
        'annotations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('task_id', sa.Integer(), nullable=False),
        sa.Column('video_id', sa.Integer(), nullable=False),
        sa.Column('frame_idx', sa.Integer(), nullable=True, comment='帧索引（行为片段可为 NULL）'),
        sa.Column('annotation_type',
                  sa.Enum('keypoint', 'behavior', 'bbox', name='annotation_type'),
                  nullable=False),
        sa.Column('annotation_source',
                  sa.Enum('human', 'prelabel', 'imported', name='annotation_source'),
                  nullable=False, server_default='human'),
        sa.Column('data_json', sa.JSON(), nullable=False,
                  comment='标注数据（格式依 annotation_type 而定）'),
        sa.Column('confidence', sa.Float(), nullable=True, comment='模型预标注置信度'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['task_id'], ['annotation_tasks.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_annotations_task_id', 'annotations', ['task_id'])
    op.create_index('ix_annotations_video_id', 'annotations', ['video_id'])
    op.create_index('ix_annotations_frame_idx', 'annotations', ['frame_idx'])
    op.create_index('ix_annotations_annotation_type', 'annotations', ['annotation_type'])
    op.create_index('ix_annotations_annotation_source', 'annotations', ['annotation_source'])

    # 3. behavior_class 枚举新增 P1 8 类
    # PostgreSQL ALTER TYPE ADD VALUE IF NOT EXISTS 保证幂等
    op.execute("ALTER TYPE behavior_class ADD VALUE IF NOT EXISTS 'TRACK'")
    op.execute("ALTER TYPE behavior_class ADD VALUE IF NOT EXISTS 'ALERT_SIT'")
    op.execute("ALTER TYPE behavior_class ADD VALUE IF NOT EXISTS 'ALERT_DOWN'")
    op.execute("ALTER TYPE behavior_class ADD VALUE IF NOT EXISTS 'APPREHEND'")
    op.execute("ALTER TYPE behavior_class ADD VALUE IF NOT EXISTS 'ESCORT'")
    op.execute("ALTER TYPE behavior_class ADD VALUE IF NOT EXISTS 'OBSTACLE'")
    op.execute("ALTER TYPE behavior_class ADD VALUE IF NOT EXISTS 'RECALL'")
    op.execute("ALTER TYPE behavior_class ADD VALUE IF NOT EXISTS 'WATCH'")


def downgrade() -> None:
    # PostgreSQL 不支持从 enum 移除值，downgrade 不动 behavior_class
    op.drop_index('ix_annotations_annotation_source', table_name='annotations')
    op.drop_index('ix_annotations_annotation_type', table_name='annotations')
    op.drop_index('ix_annotations_frame_idx', table_name='annotations')
    op.drop_index('ix_annotations_video_id', table_name='annotations')
    op.drop_index('ix_annotations_task_id', table_name='annotations')
    op.drop_table('annotations')

    op.drop_index('ix_annotation_tasks_status', table_name='annotation_tasks')
    op.drop_index('ix_annotation_tasks_handler_id', table_name='annotation_tasks')
    op.drop_index('ix_annotation_tasks_video_id', table_name='annotation_tasks')
    op.drop_table('annotation_tasks')
