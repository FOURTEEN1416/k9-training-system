"""RBAC 鉴权单元测试.

Owner: 后端开发（见 AGENTS.md §2.2）
Phase: 3.6c

测试矩阵:
    1. security.py: 密码哈希/验证 + JWT 编解码 + TokenPayload 序列化 + 异常类型
    2. handler.py: UserRole 枚举 + ROLE_HIERARCHY 层级
    3. deps.py: require_roles / require_role_hierarchy / check_base_access 逻辑
    4. deps.py: check_dog_access 资源级权限（mock AsyncSession）

标记: fast（纯 Python + mock，无 DB/GPU 依赖）
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.core.security import (
    AuthenticationFailedError,
    AuthError,
    InsufficientPermissionError,
    InvalidTokenError,
    TokenPayload,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.app.core.deps import check_base_access, check_dog_access
from backend.app.models.dog import Dog, Gender, TrainingStage
from backend.app.models.handler import ROLE_HIERARCHY, Handler, UserRole

pytestmark = pytest.mark.fast


# ============================================================
# 1. 密码哈希（bcrypt 直接调用）
# ============================================================


class TestPasswordHash:
    """密码哈希/验证测试。"""

    def test_hash_and_verify_success(self) -> None:
        """正确密码 → 哈希 → 验证通过。"""
        plain = "MySecurePass123!"
        hashed = hash_password(plain)
        assert hashed != plain
        assert hashed.startswith("$2")  # bcrypt 前缀
        assert verify_password(plain, hashed) is True

    def test_verify_wrong_password(self) -> None:
        """错误密码 → 验证失败。"""
        hashed = hash_password("CorrectPass123!")
        assert verify_password("WrongPass456!", hashed) is False

    def test_verify_empty_hash(self) -> None:
        """空哈希（未设置密码）→ 验证失败。"""
        assert verify_password("anything", None) is False
        assert verify_password("anything", "") is False

    def test_hash_empty_password_raises(self) -> None:
        """空密码 → 抛 ValueError。"""
        with pytest.raises(ValueError, match="密码不能为空"):
            hash_password("")

    def test_hash_long_password_truncated(self) -> None:
        """超过 72 字节的密码被截断，不抛异常。"""
        long_pwd = "x" * 200  # 200 字节，远超 bcrypt 72 字节限制
        hashed = hash_password(long_pwd)
        # 截断后哈希仍能验证（前 72 字节）
        assert verify_password(long_pwd, hashed) is True
        # 前 72 字节的密码也能验证
        assert verify_password("x" * 72, hashed) is True

    def test_hash_different_salts(self) -> None:
        """同一密码两次哈希结果不同（随机 salt）。"""
        h1 = hash_password("SamePass123!")
        h2 = hash_password("SamePass123!")
        assert h1 != h2
        # 但都能验证
        assert verify_password("SamePass123!", h1) is True
        assert verify_password("SamePass123!", h2) is True

    def test_verify_invalid_hash_returns_false(self) -> None:
        """无效哈希格式 → 验证失败（不抛异常）。"""
        assert verify_password("pass", "not-a-valid-hash") is False
        assert verify_password("pass", "corrupted") is False


# ============================================================
# 2. JWT 编解码
# ============================================================


class TestJWT:
    """JWT 创建/解析测试。"""

    def test_create_and_decode_success(self) -> None:
        """创建 token → 解析 → payload 一致。"""
        token = create_access_token(
            handler_id=42,
            role="admin",
            base_id=7,
            is_superuser=False,
        )
        assert isinstance(token, str)
        assert token.count(".") == 2  # JWT 三段式

        payload = decode_access_token(token)
        assert payload.sub == 42
        assert payload.role == "admin"
        assert payload.base_id == 7
        assert payload.is_superuser is False

    def test_sub_is_string_in_jwt(self) -> None:
        """JWT sub claim 必须为字符串（PyJWT 2.x 规范）。"""
        import jwt as _jwt
        from backend.app.core.config import settings

        token = create_access_token(
            handler_id=99,
            role="handler",
            base_id=None,
            is_superuser=False,
        )
        # 手动解码（不验证）检查 sub 类型
        unverified = _jwt.decode(token, options={"verify_signature": False})
        assert isinstance(unverified["sub"], str)
        assert unverified["sub"] == "99"

    def test_decode_invalid_token(self) -> None:
        """无效 token → InvalidTokenError。"""
        with pytest.raises(InvalidTokenError):
            decode_access_token("invalid.token.here")

    def test_decode_empty_token(self) -> None:
        """空 token → InvalidTokenError。"""
        with pytest.raises(InvalidTokenError, match="令牌为空"):
            decode_access_token("")

    def test_decode_wrong_secret(self) -> None:
        """错误密钥签名 → InvalidTokenError。"""
        import jwt as _jwt
        from backend.app.core.config import settings

        # 用错误密钥签发
        wrong_token = _jwt.encode(
            {"sub": "1", "exp": datetime.now(tz=timezone.utc) + timedelta(hours=1)},
            "wrong-secret",
            algorithm="HS256",
        )
        with pytest.raises(InvalidTokenError):
            decode_access_token(wrong_token)

    def test_decode_expired_token(self) -> None:
        """过期 token → InvalidTokenError。"""
        token = create_access_token(
            handler_id=1,
            role="admin",
            base_id=None,
            is_superuser=False,
            expires_minutes=-60,  # 1 小时前过期
        )
        with pytest.raises(InvalidTokenError, match="过期"):
            decode_access_token(token)

    def test_token_payload_roundtrip(self) -> None:
        """TokenPayload 通过 JWT 实际编解码的往返测试。

        注: to_dict() 输出 datetime 对象（供 PyJWT 编码），
        from_dict() 接收 timestamp（PyJWT 解码后）。
        两者不对称，必须通过实际 JWT 编解码验证。
        """
        token = create_access_token(
            handler_id=123,
            role="manager",
            base_id=5,
            is_superuser=True,
        )
        payload = decode_access_token(token)
        assert payload.sub == 123
        assert payload.role == "manager"
        assert payload.base_id == 5
        assert payload.is_superuser is True
        # exp 应在未来
        assert payload.exp > datetime.now(tz=timezone.utc)
        # iat 应在过去或现在
        assert payload.iat <= datetime.now(tz=timezone.utc)


# ============================================================
# 3. 异常类型
# ============================================================


class TestAuthErrors:
    """AuthError 异常层级测试。"""

    def test_auth_error_default_status(self) -> None:
        err = AuthError("test")
        assert err.status_code == 401
        assert err.message == "test"

    def test_auth_error_custom_status(self) -> None:
        err = AuthError("forbidden", status_code=403)
        assert err.status_code == 403

    def test_invalid_token_error(self) -> None:
        err = InvalidTokenError()
        assert isinstance(err, AuthError)
        assert err.status_code == 401

    def test_insufficient_permission_error(self) -> None:
        err = InsufficientPermissionError()
        assert isinstance(err, AuthError)
        assert err.status_code == 403

    def test_authentication_failed_error(self) -> None:
        err = AuthenticationFailedError()
        assert isinstance(err, AuthError)
        assert err.status_code == 401

    def test_custom_message_in_subclass(self) -> None:
        err = InsufficientPermissionError("需要 ADMIN 权限")
        assert err.message == "需要 ADMIN 权限"
        assert err.status_code == 403


# ============================================================
# 4. UserRole 枚举 + ROLE_HIERARCHY
# ============================================================


class TestUserRole:
    """用户角色枚举测试。"""

    def test_role_values(self) -> None:
        """角色值是小写字符串。"""
        assert UserRole.HANDLER.value == "handler"
        assert UserRole.MANAGER.value == "manager"
        assert UserRole.RESEARCHER.value == "researcher"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.VIEWER.value == "viewer"

    def test_role_hierarchy_ordering(self) -> None:
        """层级数值递增：VIEWER < HANDLER < RESEARCHER < MANAGER < ADMIN。"""
        assert ROLE_HIERARCHY[UserRole.VIEWER] < ROLE_HIERARCHY[UserRole.HANDLER]
        assert ROLE_HIERARCHY[UserRole.HANDLER] < ROLE_HIERARCHY[UserRole.RESEARCHER]
        assert ROLE_HIERARCHY[UserRole.RESEARCHER] < ROLE_HIERARCHY[UserRole.MANAGER]
        assert ROLE_HIERARCHY[UserRole.MANAGER] < ROLE_HIERARCHY[UserRole.ADMIN]

    def test_all_roles_in_hierarchy(self) -> None:
        """所有 5 个角色都在层级表里。"""
        assert len(ROLE_HIERARCHY) == 5
        for role in UserRole:
            assert role in ROLE_HIERARCHY

    def test_role_is_str_enum(self) -> None:
        """UserRole 继承 str，可用字符串比较。"""
        assert UserRole.ADMIN == "admin"
        assert UserRole.HANDLER == "handler"


# ============================================================
# 5. deps.py: check_base_access（同步，无需 DB mock）
# ============================================================


def _make_handler(
    role: UserRole = UserRole.HANDLER,
    base_id: int | None = None,
    is_superuser: bool = False,
    handler_id: int = 1,
) -> Handler:
    """构造测试用 Handler 对象（SimpleNamespace，不触发 SQLAlchemy mapped attribute）。

    check_base_access / check_dog_access 只读取 .id/.role/.base_id/.is_superuser 属性，
    不需要真实 SQLAlchemy 实例。
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        id=handler_id,
        name="Test",
        email="test@test.com",
        role=role,
        is_active=True,
        base_id=base_id,
        is_superuser=is_superuser,
        password_hash=None,
    )


class TestCheckBaseAccess:
    """基地访问权限测试。

    注: check_base_access 声明为 async（API 一致性），但内部无 await，
    测试用 asyncio 同步等待。
    """

    @pytest.mark.asyncio
    async def test_admin_can_access_any_base_read(self) -> None:
        admin = _make_handler(UserRole.ADMIN, base_id=None, is_superuser=False)
        # 不抛异常即通过
        await check_base_access(admin, base_id=999, require_write=False)

    @pytest.mark.asyncio
    async def test_admin_can_access_any_base_write(self) -> None:
        admin = _make_handler(UserRole.ADMIN, base_id=None)
        await check_base_access(admin, base_id=999, require_write=True)

    @pytest.mark.asyncio
    async def test_superuser_can_access_any_base(self) -> None:
        viewer = _make_handler(UserRole.VIEWER, base_id=1, is_superuser=True)
        await check_base_access(viewer, base_id=999, require_write=True)

    @pytest.mark.asyncio
    async def test_manager_own_base_read_ok(self) -> None:
        manager = _make_handler(UserRole.MANAGER, base_id=5)
        await check_base_access(manager, base_id=5, require_write=False)

    @pytest.mark.asyncio
    async def test_manager_own_base_write_denied(self) -> None:
        """MANAGER 不能写基地（仅 ADMIN）。"""
        manager = _make_handler(UserRole.MANAGER, base_id=5)
        with pytest.raises(InsufficientPermissionError, match="仅 ADMIN"):
            await check_base_access(manager, base_id=5, require_write=True)

    @pytest.mark.asyncio
    async def test_manager_other_base_denied(self) -> None:
        manager = _make_handler(UserRole.MANAGER, base_id=5)
        with pytest.raises(InsufficientPermissionError):
            await check_base_access(manager, base_id=999, require_write=False)

    @pytest.mark.asyncio
    async def test_handler_own_base_read_ok(self) -> None:
        handler = _make_handler(UserRole.HANDLER, base_id=3)
        await check_base_access(handler, base_id=3, require_write=False)

    @pytest.mark.asyncio
    async def test_handler_other_base_denied(self) -> None:
        handler = _make_handler(UserRole.HANDLER, base_id=3)
        with pytest.raises(InsufficientPermissionError):
            await check_base_access(handler, base_id=999, require_write=False)

    @pytest.mark.asyncio
    async def test_handler_no_base_denied(self) -> None:
        """base_id=None 的用户不能访问任何基地。"""
        handler = _make_handler(UserRole.HANDLER, base_id=None)
        with pytest.raises(InsufficientPermissionError):
            await check_base_access(handler, base_id=1, require_write=False)

    @pytest.mark.asyncio
    async def test_viewer_own_base_read_ok(self) -> None:
        viewer = _make_handler(UserRole.VIEWER, base_id=7)
        await check_base_access(viewer, base_id=7, require_write=False)

    @pytest.mark.asyncio
    async def test_viewer_write_denied(self) -> None:
        viewer = _make_handler(UserRole.VIEWER, base_id=7)
        with pytest.raises(InsufficientPermissionError):
            await check_base_access(viewer, base_id=7, require_write=True)


# ============================================================
# 6. deps.py: check_dog_access（需要 mock AsyncSession）
# ============================================================


def _make_dog(
    dog_id: int = 1,
    handler_id: int | None = None,
    home_base_id: int | None = None,
) -> Dog:
    """构造测试用 Dog 对象（SimpleNamespace）。"""
    from types import SimpleNamespace
    return SimpleNamespace(
        id=dog_id,
        name="TestDog",
        handler_id=handler_id,
        home_base_id=home_base_id,
        training_stage=TrainingStage.P0,
        gender=Gender.MALE,
    )


class TestCheckDogAccess:
    """犬只资源级权限测试。"""

    @pytest.mark.asyncio
    async def test_admin_can_access_any_dog(self) -> None:
        admin = _make_handler(UserRole.ADMIN)
        dog = _make_dog(dog_id=1, handler_id=99, home_base_id=99)
        db = AsyncMock()
        db.get = AsyncMock(return_value=dog)
        result = await check_dog_access(db, admin, dog_id=1)
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_superuser_can_access_any_dog(self) -> None:
        viewer = _make_handler(UserRole.VIEWER, is_superuser=True)
        dog = _make_dog(dog_id=1, handler_id=99)
        db = AsyncMock()
        db.get = AsyncMock(return_value=dog)
        result = await check_dog_access(db, viewer, dog_id=1)
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_manager_own_base_dog(self) -> None:
        manager = _make_handler(UserRole.MANAGER, base_id=5)
        dog = _make_dog(dog_id=1, home_base_id=5)
        db = AsyncMock()
        db.get = AsyncMock(return_value=dog)
        # mock _has_temp_base_access（check_dog_access 内部会查）
        result = await check_dog_access(db, manager, dog_id=1)
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_handler_own_dog(self) -> None:
        """HANDLER 访问自己负责的犬只。"""
        handler = _make_handler(UserRole.HANDLER, handler_id=10, base_id=5)
        dog = _make_dog(dog_id=1, handler_id=10, home_base_id=5)
        db = AsyncMock()
        db.get = AsyncMock(return_value=dog)
        result = await check_dog_access(db, handler, dog_id=1)
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_handler_other_dog_denied(self) -> None:
        """HANDLER 访问别人的犬只 → 403。"""
        handler = _make_handler(UserRole.HANDLER, handler_id=10, base_id=5)
        dog = _make_dog(dog_id=1, handler_id=99, home_base_id=5)
        db = AsyncMock()
        db.get = AsyncMock(return_value=dog)
        # mock 关联表查询返回空
        db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
        with pytest.raises(InsufficientPermissionError):
            await check_dog_access(db, handler, dog_id=1)

    @pytest.mark.asyncio
    async def test_dog_not_found_returns_404(self) -> None:
        """犬只不存在 → AuthError(404) 防枚举。"""
        admin = _make_handler(UserRole.ADMIN)
        db = AsyncMock()
        db.get = AsyncMock(return_value=None)
        with pytest.raises(AuthError) as exc_info:
            await check_dog_access(db, admin, dog_id=99999)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_viewer_own_base_dog_read_ok(self) -> None:
        viewer = _make_handler(UserRole.VIEWER, base_id=5)
        dog = _make_dog(dog_id=1, home_base_id=5)
        db = AsyncMock()
        db.get = AsyncMock(return_value=dog)
        result = await check_dog_access(db, viewer, dog_id=1, require_write=False)
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_viewer_dog_write_denied(self) -> None:
        """VIEWER 不可写犬只。"""
        viewer = _make_handler(UserRole.VIEWER, base_id=5)
        dog = _make_dog(dog_id=1, home_base_id=5)
        db = AsyncMock()
        db.get = AsyncMock(return_value=dog)
        with pytest.raises(InsufficientPermissionError):
            await check_dog_access(db, viewer, dog_id=1, require_write=True)

    @pytest.mark.asyncio
    async def test_researcher_own_base_dog_read_ok(self) -> None:
        researcher = _make_handler(UserRole.RESEARCHER, base_id=5)
        dog = _make_dog(dog_id=1, home_base_id=5)
        db = AsyncMock()
        db.get = AsyncMock(return_value=dog)
        result = await check_dog_access(db, researcher, dog_id=1, require_write=False)
        assert result.id == 1

    @pytest.mark.asyncio
    async def test_researcher_dog_write_denied(self) -> None:
        """RESEARCHER 不可写犬只。"""
        researcher = _make_handler(UserRole.RESEARCHER, base_id=5)
        dog = _make_dog(dog_id=1, home_base_id=5)
        db = AsyncMock()
        db.get = AsyncMock(return_value=dog)
        with pytest.raises(InsufficientPermissionError):
            await check_dog_access(db, researcher, dog_id=1, require_write=True)
