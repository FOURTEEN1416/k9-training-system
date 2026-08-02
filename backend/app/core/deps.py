"""FastAPI 依赖注入 + RBAC 权限检查。

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 3.6b

提供:
    - get_current_handler: 强制鉴权（必须提供有效 JWT）
    - get_optional_handler: 可选鉴权（无 JWT 返回 None，便于兼容现有路由）
    - require_roles(*roles): 角色级权限装饰器
    - require_dog_access: 资源级权限（犬只归属/基地归属）
    - require_base_access: 基地级权限（同上但用于 base 资源）

设计原则:
    - 依赖项不抛原生 HTTPException，而是抛 AuthError 子类
    - 上层通过 FastAPI exception_handler 统一转换（见 main.py）
    - 资源级权限查询 DB，避免在路由函数中重复查询
"""
from __future__ import annotations

import logging
from typing import Annotated, Callable, Optional

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.security import (
    AuthError, InsufficientPermissionError, InvalidTokenError,
    decode_access_token, verify_password,
)
from backend.app.models.dog import Dog
from backend.app.models.dog_associations import DogBaseAssociation, DogHandlerAssociation
from backend.app.models.handler import Handler, ROLE_HIERARCHY, UserRole

logger = logging.getLogger(__name__)


# FastAPI OpenAPI 安全方案（Bearer Token）
_bearer_scheme = HTTPBearer(
    bearerFormat="JWT",
    auto_error=False,  # 我们自己处理错误，便于区分 401 vs 403
)


async def _fetch_handler(db: AsyncSession, handler_id: int) -> Optional[Handler]:
    """按 ID 查询用户，仅返回 is_active=True 的账号。"""
    result = await db.execute(
        select(Handler).where(Handler.id == handler_id, Handler.is_active.is_(True))
    )
    return result.scalar_one_or_none()


async def get_current_handler(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer_scheme)],
) -> Handler:
    """强制鉴权：必须提供有效 JWT。

    用法:
        @router.get("/me")
        async def me(me: Handler = Depends(get_current_handler)):
            return me

    Raises:
        InvalidTokenError: token 缺失/无效/过期
        AuthError: 用户不存在/已禁用
    """
    if credentials is None or not credentials.credentials:
        raise InvalidTokenError("缺少认证令牌（Authorization: Bearer <token>）")

    payload = decode_access_token(credentials.credentials)
    handler = await _fetch_handler(db, payload.sub)
    if handler is None:
        raise AuthError("用户不存在或已被禁用", status_code=401)

    # 写入 request.state 便于下游日志/审计
    request.state.current_handler = handler
    return handler


async def get_optional_handler(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(_bearer_scheme)],
) -> Optional[Handler]:
    """可选鉴权：无 JWT 返回 None，有 JWT 验证并返回 Handler。

    用例: 现有路由（dogs/videos/scores）兼容旧客户端，无 token 时返回全部数据，
    有 token 时按 RBAC 过滤（路由函数内自行决定过滤逻辑）。
    """
    if credentials is None or not credentials.credentials:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
    except InvalidTokenError:
        return None
    handler = await _fetch_handler(db, payload.sub)
    if handler is not None:
        request.state.current_handler = handler
    return handler


# ============================================================
# 角色级权限
# ============================================================


def require_roles(*allowed_roles: UserRole) -> Callable:
    """角色级权限依赖工厂。

    用法:
        @router.post("/bases", dependencies=[Depends(require_roles(UserRole.ADMIN))])
        async def create_base(...):
            ...

    或:
        @router.get("/admin/users")
        async def list_users(
            me: Handler = Depends(require_roles(UserRole.ADMIN, UserRole.MANAGER)),
        ):
            ...

    逻辑:
        - is_superuser=True 直接通过
        - role 在 allowed_roles 中通过
        - 否则抛 InsufficientPermissionError
    """
    allowed_values = {r.value for r in allowed_roles}

    async def _check(me: Annotated[Handler, Depends(get_current_handler)]) -> Handler:
        if me.is_superuser:
            return me
        if me.role.value in allowed_values:
            return me
        raise InsufficientPermissionError(
            f"需要以下角色之一: {', '.join(r.value for r in allowed_roles)}（当前: {me.role.value}）"
        )

    return _check


def require_role_hierarchy(min_role: UserRole) -> Callable:
    """角色层级权限依赖工厂。

    用法:
        @router.get("/users", dependencies=[Depends(require_role_hierarchy(UserRole.MANAGER))])
        async def list_users(...):
            ...

    逻辑:
        - is_superuser=True 直接通过
        - 当前角色层级 ≥ min_role 通过
        - 否则抛 InsufficientPermissionError
    """
    min_level = ROLE_HIERARCHY[min_role]

    async def _check(me: Annotated[Handler, Depends(get_current_handler)]) -> Handler:
        if me.is_superuser:
            return me
        current_level = ROLE_HIERARCHY.get(me.role, 0)
        if current_level >= min_level:
            return me
        raise InsufficientPermissionError(
            f"角色权限不足（需要 ≥ {min_role.value}，当前 {me.role.value}）"
        )

    return _check


# ============================================================
# 资源级权限
# ============================================================


async def check_dog_access(
    db: AsyncSession,
    handler: Handler,
    dog_id: int,
    require_write: bool = False,
) -> Dog:
    """检查用户对犬只的访问权限。

    Args:
        db: 数据库会话
        handler: 当前用户
        dog_id: 犬只 ID
        require_write: True 需要写权限（仅主训导员/MANAGER+/ADMIN）

    Returns:
        Dog 对象（用于后续业务逻辑）

    Raises:
        InsufficientPermissionError: 无权限
        AuthError(404): 犬只不存在（统一 404 防止枚举攻击）
    """
    dog = await db.get(Dog, dog_id)
    if dog is None:
        raise AuthError("犬只不存在", status_code=404)

    # 超管 / ADMIN 全局可访问
    if handler.is_superuser or handler.role == UserRole.ADMIN:
        return dog

    # MANAGER 本基地可访问
    if handler.role == UserRole.MANAGER and handler.base_id is not None:
        if dog.home_base_id == handler.base_id:
            return dog
        # 检查临时基地访问
        if await _has_temp_base_access(db, dog_id, handler.base_id):
            return dog

    # RESEARCHER 本基地只读
    if handler.role == UserRole.RESEARCHER and not require_write:
        if handler.base_id is not None and dog.home_base_id == handler.base_id:
            return dog

    # VIEWER 本基地只读
    if handler.role == UserRole.VIEWER and not require_write:
        if handler.base_id is not None and dog.home_base_id == handler.base_id:
            return dog

    # HANDLER 自有犬只（dog.handler_id）+ 多对多关联表
    if handler.role == UserRole.HANDLER:
        is_owner = dog.handler_id == handler.id
        if not is_owner:
            is_associated = await _is_handler_associated_with_dog(db, handler.id, dog_id)
            if is_associated:
                # 副训导员写权限取决于 assignment_role（这里简化：PRIMARY 可写，SECONDARY 只读）
                if require_write:
                    is_primary = await _is_primary_handler(db, handler.id, dog_id)
                    if not is_primary:
                        raise InsufficientPermissionError(
                            "副训导员无写权限（仅主训导员可修改）"
                        )
                return dog
            raise InsufficientPermissionError("无权访问此犬只")
        return dog

    raise InsufficientPermissionError("无权访问此犬只")


def require_dog_access(
    dog_id_param: str = "dog_id",
    require_write: bool = False,
) -> Callable:
    """资源级权限依赖工厂（犬只）。

    用法:
        @router.get("/dogs/{dog_id}/scores")
        async def dog_scores(
            dog_id: int,
            me: Handler = Depends(get_current_handler),
            db: AsyncSession = Depends(get_db),
        ):
            dog = await check_dog_access(db, me, dog_id, require_write=False)
            ...

    或直接作为路径操作依赖（自动校验，但不返回 dog 对象）:
        @router.put("/dogs/{dog_id}", dependencies=[
            Depends(require_dog_access("dog_id", require_write=True))
        ])
        async def update_dog(dog_id: int, ...):
            ...
    """
    async def _check(
        request: Request,
        me: Annotated[Handler, Depends(get_current_handler)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> Dog:
        # 从路径参数提取 dog_id
        try:
            dog_id = int(request.path_params[dog_id_param])
        except (KeyError, ValueError):
            raise AuthError(f"路径参数 {dog_id_param} 不是有效的犬只 ID", status_code=400)
        dog = await check_dog_access(db, me, dog_id, require_write=require_write)
        request.state.accessed_dog = dog
        return dog

    return _check


async def check_base_access(
    handler: Handler,
    base_id: int,
    require_write: bool = False,
) -> None:
    """检查用户对基地的访问权限。

    Args:
        handler: 当前用户
        base_id: 基地 ID
        require_write: True 需要写权限（仅 ADMIN）

    Raises:
        InsufficientPermissionError: 无权限
    """
    if handler.is_superuser or handler.role == UserRole.ADMIN:
        return

    if require_write:
        raise InsufficientPermissionError("仅 ADMIN 可修改基地信息")

    # MANAGER/RESEARCHER/VIEWER/HANDLER 仅本基地只读
    if handler.base_id is None or handler.base_id != base_id:
        raise InsufficientPermissionError("无权访问此基地")


# ============================================================
# 内部辅助
# ============================================================


async def _has_temp_base_access(
    db: AsyncSession, dog_id: int, base_id: int
) -> bool:
    """检查犬只是否临时访问某基地（未过期）。"""
    from datetime import datetime, timezone
    result = await db.execute(
        select(DogBaseAssociation).where(
            DogBaseAssociation.dog_id == dog_id,
            DogBaseAssociation.base_id == base_id,
        )
    )
    assoc = result.scalar_one_or_none()
    if assoc is None:
        return False
    if assoc.expires_at is None:
        return True  # 永久访问
    return assoc.expires_at > datetime.now(tz=timezone.utc)


async def _is_handler_associated_with_dog(
    db: AsyncSession, handler_id: int, dog_id: int
) -> bool:
    """检查训导员是否通过关联表关联犬只。"""
    result = await db.execute(
        select(DogHandlerAssociation).where(
            DogHandlerAssociation.handler_id == handler_id,
            DogHandlerAssociation.dog_id == dog_id,
        )
    )
    return result.scalar_one_or_none() is not None


async def _is_primary_handler(
    db: AsyncSession, handler_id: int, dog_id: int
) -> bool:
    """检查训导员是否为犬只主训导员。"""
    from backend.app.models.dog_associations import DogHandlerRole
    result = await db.execute(
        select(DogHandlerAssociation).where(
            DogHandlerAssociation.handler_id == handler_id,
            DogHandlerAssociation.dog_id == dog_id,
            DogHandlerAssociation.assignment_role == DogHandlerRole.PRIMARY,
        )
    )
    return result.scalar_one_or_none() is not None
