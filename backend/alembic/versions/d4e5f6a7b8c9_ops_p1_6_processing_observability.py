"""ops_p1_6_processing_observability

P1-6 处理过程可观测性（纯 additive）:
1. videos 新增 4 个 nullable 处理进度列（不回填、无默认值、无类型变更）
2. videos.processed_at 新增索引（统计窗口范围扫描）

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-10-19 10:00:00

注意:
    - 4 列全部 nullable，历史行保持 NULL，前端按“无数据”降级展示。
    - downgrade 逆序删除索引与 4 列，可无损回滚（新列无业务数据依赖）。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. 处理阶段进度列（全部 nullable，兼容历史行）
    op.add_column(
        'videos',
        sa.Column(
            'processing_stage',
            sa.String(length=32),
            nullable=True,
            comment='当前处理阶段: preparing/pose_inference/.../completed/retrying',
        ),
    )
    op.add_column(
        'videos',
        sa.Column(
            'processing_percent',
            sa.Float(),
            nullable=True,
            comment='阶段级进度百分比 0-100（null=未知/历史数据）',
        ),
    )
    op.add_column(
        'videos',
        sa.Column(
            'processing_started_at',
            sa.DateTime(timezone=True),
            nullable=True,
            comment='本次处理开始时间（用于已运行时长/stale 判定）',
        ),
    )
    op.add_column(
        'videos',
        sa.Column(
            'processing_duration_ms',
            sa.Integer(),
            nullable=True,
            comment='处理耗时（毫秒，started_at→processed_at）',
        ),
    )

    # 2. processed_at 索引（统计窗口 processed_at >= since 范围扫描）
    op.create_index('ix_videos_processed_at', 'videos', ['processed_at'], unique=False)


def downgrade() -> None:
    # 逆序回滚：先索引，后列（与 upgrade 相反）
    op.drop_index('ix_videos_processed_at', table_name='videos')
    op.drop_column('videos', 'processing_duration_ms')
    op.drop_column('videos', 'processing_started_at')
    op.drop_column('videos', 'processing_percent')
    op.drop_column('videos', 'processing_stage')
