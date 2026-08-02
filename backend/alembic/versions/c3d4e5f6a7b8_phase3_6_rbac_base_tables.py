"""phase3_6_rbac_base_tables

1. 创建 bases 表（基地/多租户隔离单位）
2. 扩展 handlers 表：password_hash + base_id + is_superuser
3. user_role 枚举新增 VIEWER 值（查看者）
4. 创建 dog_handler_association 表（多对多：一只犬可分配多名训导员）
5. 创建 dog_base_association 表（多对多：一只犬可属于多个基地，跨基地协作场景）

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-08-02 10:00:00

RBAC 权限矩阵（Phase 3.6a）:
    ADMIN       : 全局超管，可管理所有基地/用户/模型
    MANAGER     : 基地管理员，可管理本基地用户 + 犬只 + 视频
    RESEARCHER  : 科研人员，本基地只读 + 标注权限
    HANDLER     : 训导员，仅自有犬只/视频读写
    VIEWER      : 查看者，全基地只读
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, Sequence[str], None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. bases 表（基地/多租户隔离）
    op.create_table(
        'bases',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False, comment='基地名称'),
        sa.Column('code', sa.String(length=32), nullable=False, comment='基地代号（唯一）'),
        sa.Column('description', sa.String(length=512), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('code', name='uq_bases_code'),
        sa.UniqueConstraint('name', name='uq_bases_name'),
    )
    op.create_index('ix_bases_code', 'bases', ['code'], unique=True)

    # 2. 扩展 handlers 表
    # 2a. password_hash: NULL = 未启用登录（兼容历史数据）
    op.add_column(
        'handlers',
        sa.Column(
            'password_hash',
            sa.String(length=255),
            nullable=True,
            comment='bcrypt 哈希（passlib CryptContext）',
        ),
    )
    # 2b. base_id: NULL = 全局账号（admin 跨基地）
    op.add_column(
        'handlers',
        sa.Column(
            'base_id',
            sa.Integer(),
            nullable=True,
            comment='所属基地 ID（NULL = 全局账号）',
        ),
    )
    op.create_foreign_key(
        'fk_handlers_base_id_bases',
        'handlers',
        'bases',
        ['base_id'],
        ['id'],
        ondelete='SET NULL',
    )
    op.create_index('ix_handlers_base_id', 'handlers', ['base_id'], unique=False)

    # 2c. is_superuser: 显式超管标志（绕过所有 RBAC 检查）
    op.add_column(
        'handlers',
        sa.Column(
            'is_superuser',
            sa.Boolean(),
            nullable=False,
            server_default=sa.text('false'),
            comment='超管标志（绕过 RBAC）',
        ),
    )

    # 3. user_role 枚举新增 VIEWER（查看者）
    # PostgreSQL ALTER TYPE ADD VALUE IF NOT EXISTS 保证幂等
    op.execute("ALTER TYPE user_role ADD VALUE IF NOT EXISTS 'VIEWER'")

    # 4. dog_handler_association（多对多）
    # 用例：警犬配备主训导员 + 副训导员；交接期一只犬同时关联两人
    op.create_table(
        'dog_handler_association',
        sa.Column('dog_id', sa.Integer(), nullable=False),
        sa.Column('handler_id', sa.Integer(), nullable=False),
        sa.Column(
            'assignment_role',
            sa.Enum('PRIMARY', 'SECONDARY', name='dog_handler_role'),
            nullable=False,
            server_default='PRIMARY',
            comment='主训导员/副训导员',
        ),
        sa.Column('assigned_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('notes', sa.String(length=256), nullable=True),
        sa.ForeignKeyConstraint(['dog_id'], ['dogs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['handler_id'], ['handlers.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('dog_id', 'handler_id', name='pk_dog_handler_association'),
    )
    op.create_index('ix_dog_handler_assoc_dog_id', 'dog_handler_association', ['dog_id'])
    op.create_index('ix_dog_handler_assoc_handler_id', 'dog_handler_association', ['handler_id'])

    # 5. dog_base_association（多对多：跨基地协作）
    # 用例：联合训练期间一只犬可临时进入其他基地
    op.create_table(
        'dog_base_association',
        sa.Column('dog_id', sa.Integer(), nullable=False),
        sa.Column('base_id', sa.Integer(), nullable=False),
        sa.Column(
            'access_type',
            sa.Enum('PERMANENT', 'TEMPORARY', name='dog_base_access'),
            nullable=False,
            server_default='PERMANENT',
        ),
        sa.Column('granted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True, comment='临时访问过期时间'),
        sa.ForeignKeyConstraint(['dog_id'], ['dogs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['base_id'], ['bases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('dog_id', 'base_id', name='pk_dog_base_association'),
    )
    op.create_index('ix_dog_base_assoc_dog_id', 'dog_base_association', ['dog_id'])
    op.create_index('ix_dog_base_assoc_base_id', 'dog_base_association', ['base_id'])

    # 6. dogs 表新增 home_base_id（主基地）
    op.add_column(
        'dogs',
        sa.Column(
            'home_base_id',
            sa.Integer(),
            nullable=True,
            comment='犬只主基地 ID',
        ),
    )
    op.create_foreign_key(
        'fk_dogs_home_base_id_bases',
        'dogs',
        'bases',
        ['home_base_id'],
        ['id'],
        ondelete='SET NULL',
    )
    op.create_index('ix_dogs_home_base_id', 'dogs', ['home_base_id'], unique=False)


def downgrade() -> None:
    # dogs.home_base_id
    op.drop_index('ix_dogs_home_base_id', table_name='dogs')
    op.drop_constraint('fk_dogs_home_base_id_bases', 'dogs', type_='foreignkey')
    op.drop_column('dogs', 'home_base_id')

    # dog_base_association
    op.drop_index('ix_dog_base_assoc_base_id', table_name='dog_base_association')
    op.drop_index('ix_dog_base_assoc_dog_id', table_name='dog_base_association')
    op.drop_table('dog_base_association')
    op.execute("DROP TYPE IF EXISTS dog_base_access")

    # dog_handler_association
    op.drop_index('ix_dog_handler_assoc_handler_id', table_name='dog_handler_association')
    op.drop_index('ix_dog_handler_assoc_dog_id', table_name='dog_handler_association')
    op.drop_table('dog_handler_association')
    op.execute("DROP TYPE IF EXISTS dog_handler_role")

    # handlers 扩展字段
    op.drop_index('ix_handlers_base_id', table_name='handlers')
    op.drop_constraint('fk_handlers_base_id_bases', 'handlers', type_='foreignkey')
    op.drop_column('handlers', 'is_superuser')
    op.drop_column('handlers', 'base_id')
    op.drop_column('handlers', 'password_hash')

    # bases 表
    op.drop_index('ix_bases_code', table_name='bases')
    op.drop_table('bases')

    # PostgreSQL 不支持从 enum 移除值，downgrade 不动 user_role（VIEWER 保留）
