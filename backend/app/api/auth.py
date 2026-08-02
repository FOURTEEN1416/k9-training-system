"""鉴权 API 路由（Phase 3.6b）。

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 3.6b

端点:
    POST /api/auth/login           登录获取 JWT
    GET  /api/auth/me              获取当前用户信息
    POST /api/auth/change-password 修改密码
    POST /api/auth/register        注册新用户（仅 ADMIN）
    GET  /api/auth/users           用户列表（仅 MANAGER+）
    GET  /api/auth/users/{id}      用户详情（仅 MANAGER+ 或本人）
    PUT  /api/auth/users/{id}      更新用户（仅 ADMIN 或本人改自己）
    DELETE /api/auth/users/{id}    禁用用户（仅 ADMIN）
"""
from __future__ import annotations

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.core.deps import (
    get_current_handler, get_optional_handler, require_role_hierarchy,
)
from backend.app.core.security import (
    AuthenticationFailedError, hash_password, verify_password,
)
from backend.app.models.handler import Handler, UserRole
from backend.app.schemas.common import (
    HandlerRead, HandlerRegisterRequest, LoginRequest,
    PasswordChangeRequest, TokenResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

DbSession = Annotated[AsyncSession, Depends(get_db)]
CurrentHandler = Annotated[Handler, Depends(get_current_handler)]


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: DbSession) -> TokenResponse:
    """邮箱 + 密码登录，返回 JWT access token。

    安全性:
        - 统一返回 401 "邮箱或密码错误"，不区分邮箱是否存在
        - 失败时记录 warning 日志（不记录明文密码）
        - 已禁用账号（is_active=False）禁止登录
    """
    result = await db.execute(
        select(Handler).where(Handler.email == payload.email)
    )
    handler = result.scalar_one_or_none()

    # 统一错误响应（防止邮箱枚举）
    if handler is None or not handler.is_active:
        logger.warning(f"登录失败: 邮箱 {payload.email} 不存在或账号已禁用")
        raise AuthenticationFailedError()

    if not handler.password_hash:
        logger.warning(f"登录失败: 用户 {handler.id} ({handler.email}) 未设置密码")
        raise AuthenticationFailedError("账号未启用密码登录，请联系管理员")

    if not verify_password(payload.password, handler.password_hash):
        logger.warning(f"登录失败: 用户 {handler.id} ({handler.email}) 密码错误")
        raise AuthenticationFailedError()

    # 签发 JWT
    from backend.app.core.security import create_access_token
    token = create_access_token(
        handler_id=handler.id,
        role=handler.role.value,
        base_id=handler.base_id,
        is_superuser=handler.is_superuser,
    )
    logger.info(f"登录成功: 用户 {handler.id} ({handler.email}) 角色={handler.role.value}")

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.auth_access_token_expire_minutes * 60,
        handler=HandlerRead.model_validate(handler, from_attributes=True),
    )


@router.get("/me", response_model=HandlerRead)
async def get_me(me: CurrentHandler) -> Handler:
    """获取当前登录用户信息。"""
    return me


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    payload: PasswordChangeRequest,
    me: CurrentHandler,
    db: DbSession,
) -> None:
    """修改当前用户密码。

    要求:
        - 旧密码验证通过
        - 新密码 ≥ 8 字符
        - 不允许未设置密码的账号通过此接口设置密码（用 /auth/reset-password）
    """
    if not me.password_hash:
        raise HTTPException(
            status_code=400,
            detail="账号未设置密码，请联系管理员重置",
        )
    if not verify_password(payload.old_password, me.password_hash):
        raise HTTPException(status_code=400, detail="旧密码错误")
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="新密码至少 8 个字符")

    me.password_hash = hash_password(payload.new_password)
    await db.flush()
    logger.info(f"用户 {me.id} ({me.email}) 修改密码")


@router.post(
    "/register",
    response_model=HandlerRead,
    status_code=status.HTTP_201_CREATED,
)
async def register_user(
    payload: HandlerRegisterRequest,
    db: DbSession,
    me: Annotated[Handler, Depends(require_role_hierarchy(UserRole.ADMIN))],
) -> Handler:
    """注册新用户（仅 ADMIN 或超管）。

    Args:
        payload: 注册信息（email/role/base_id 等）
        me: 当前登录用户（必须 ADMIN）

    Returns:
        新创建的 Handler 对象
    """
    # 校验角色值
    try:
        role_enum = UserRole(payload.role.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"非法角色: {payload.role}，允许: {[r.value for r in UserRole]}",
        )

    # 校验邮箱唯一
    existing = await db.execute(select(Handler).where(Handler.email == payload.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status_code=409, detail=f"邮箱 {payload.email} 已存在")

    # 校验密码长度
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="密码至少 8 个字符")

    # 校验 base_id 存在（若提供）
    if payload.base_id is not None:
        from backend.app.models.base_entity import Base_
        base = await db.get(Base_, payload.base_id)
        if base is None:
            raise HTTPException(status_code=400, detail=f"基地 {payload.base_id} 不存在")

    # 仅超管可创建超管
    is_superuser = payload.is_superuser and me.is_superuser

    handler = Handler(
        name=payload.name,
        email=payload.email,
        phone=payload.phone,
        role=role_enum,
        base_id=payload.base_id,
        is_superuser=is_superuser,
        is_active=True,
        password_hash=hash_password(payload.password),
    )
    db.add(handler)
    await db.flush()
    await db.refresh(handler)
    logger.info(
        f"用户 {me.id} ({me.email}) 创建新用户 {handler.id} ({handler.email}) 角色={handler.role.value}"
    )
    return handler


@router.get("/users", response_model=list[HandlerRead])
async def list_users(
    db: DbSession,
    me: Annotated[Handler, Depends(require_role_hierarchy(UserRole.MANAGER))],
    role: Optional[str] = None,
    base_id: Optional[int] = None,
    is_active: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Handler]:
    """列出用户（仅 MANAGER+）。

    MANAGER 只能查看本基地用户；ADMIN 可查看所有基地。
    """
    stmt = select(Handler).order_by(Handler.id.desc()).limit(limit).offset(offset)

    # MANAGER 仅本基地
    if me.role == UserRole.MANAGER and not me.is_superuser:
        if me.base_id is None:
            return []  # MANAGER 未分配基地，无权查看任何用户
        stmt = stmt.where(Handler.base_id == me.base_id)

    if role is not None:
        try:
            role_enum = UserRole(role.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"非法角色: {role}")
        stmt = stmt.where(Handler.role == role_enum)
    if base_id is not None:
        stmt = stmt.where(Handler.base_id == base_id)
    if is_active is not None:
        stmt = stmt.where(Handler.is_active.is_(is_active))

    result = await db.execute(stmt)
    return list(result.scalars().all())


@router.get("/users/{user_id}", response_model=HandlerRead)
async def get_user(
    user_id: int,
    db: DbSession,
    me: CurrentHandler,
) -> Handler:
    """获取用户详情（MANAGER+ 或本人）。"""
    handler = await db.get(Handler, user_id)
    if handler is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")

    # 本人可查
    if me.id == user_id:
        return handler

    # ADMIN 可查任何
    if me.is_superuser or me.role == UserRole.ADMIN:
        return handler

    # MANAGER 本基地可查
    if me.role == UserRole.MANAGER and me.base_id is not None:
        if handler.base_id == me.base_id:
            return handler
        raise HTTPException(status_code=403, detail="无权查看其他基地用户")

    raise HTTPException(status_code=403, detail="无权查看其他用户")


@router.put("/users/{user_id}", response_model=HandlerRead)
async def update_user(
    user_id: int,
    payload: HandlerRegisterRequest,
    db: DbSession,
    me: CurrentHandler,
) -> Handler:
    """更新用户信息（仅 ADMIN，或本人改自己非角色字段）。

    简化策略: 仅 ADMIN 可用此接口修改任何用户；非 ADMIN 用户请用 /auth/me + /auth/change-password。
    """
    if not (me.is_superuser or me.role == UserRole.ADMIN):
        raise HTTPException(status_code=403, detail="仅 ADMIN 可修改用户信息")

    handler = await db.get(Handler, user_id)
    if handler is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")

    try:
        role_enum = UserRole(payload.role.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"非法角色: {payload.role}")

    # 邮箱冲突检查
    if payload.email != handler.email:
        existing = await db.execute(select(Handler).where(Handler.email == payload.email))
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail=f"邮箱 {payload.email} 已存在")

    handler.name = payload.name
    handler.email = payload.email
    handler.phone = payload.phone
    handler.role = role_enum
    handler.base_id = payload.base_id
    # 仅超管可授予/撤销超管
    if me.is_superuser:
        handler.is_superuser = payload.is_superuser

    # 密码变更（非空时才更新）
    if payload.password:
        if len(payload.password) < 8:
            raise HTTPException(status_code=400, detail="密码至少 8 个字符")
        handler.password_hash = hash_password(payload.password)

    await db.flush()
    await db.refresh(handler)
    logger.info(f"用户 {me.id} 更新用户 {user_id}")
    return handler


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_user(
    user_id: int,
    db: DbSession,
    me: Annotated[Handler, Depends(require_role_hierarchy(UserRole.ADMIN))],
) -> None:
    """禁用用户（软删除，仅 ADMIN）。

    不可禁用自己；不可禁用其他超管（防误操作）。
    """
    if me.id == user_id:
        raise HTTPException(status_code=400, detail="不能禁用自己的账号")

    handler = await db.get(Handler, user_id)
    if handler is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")

    if handler.is_superuser and not me.is_superuser:
        raise HTTPException(status_code=403, detail="无权禁用超管账号")

    handler.is_active = False
    await db.flush()
    logger.info(f"用户 {me.id} 禁用用户 {user_id}")
