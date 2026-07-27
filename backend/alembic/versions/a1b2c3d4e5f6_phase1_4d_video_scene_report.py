"""phase1_4d_video_scene_report

1. 添加 videos.scene + videos.report_path 列
2. behavior_class 枚举新增 SIT_UP / STAY 值（与 rule_engine P0 输出对齐）

Revision ID: a1b2c3d4e5f6
Revises: 5d7105cb28a9
Create Date: 2026-07-27 20:00:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '5d7105cb28a9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. scene: 测试场景（puppy_selection / obedience_trial）
    # 默认 obedience_trial（与已有视频兼容，Phase 1.4d 前的视频均视为科目测评）
    op.add_column(
        'videos',
        sa.Column(
            'scene',
            sa.String(length=32),
            nullable=False,
            server_default='obedience_trial',
            comment='测试场景: puppy_selection / obedience_trial',
        ),
    )
    # 2. report_path: PDF 报告相对路径
    op.add_column(
        'videos',
        sa.Column(
            'report_path',
            sa.String(length=256),
            nullable=True,
            comment='PDF 报告相对 reports_dir 的路径',
        ),
    )
    # 3. behavior_class 枚举新增 SIT_UP / STAY（大写，匹配 SQLAlchemy enum.name 惯例）
    # PostgreSQL ALTER TYPE ADD VALUE 不能在事务中执行，需要 op.execute 直接调用
    # IF NOT EXISTS 保证幂等（重复运行无副作用）
    op.execute("ALTER TYPE behavior_class ADD VALUE IF NOT EXISTS 'SIT_UP'")
    op.execute("ALTER TYPE behavior_class ADD VALUE IF NOT EXISTS 'STAY'")


def downgrade() -> None:
    # PostgreSQL 不支持从 enum 移除值，downgrade 不动 behavior_class
    op.drop_column('videos', 'report_path')
    op.drop_column('videos', 'scene')
