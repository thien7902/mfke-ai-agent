"""Custom decorators for handlers."""
from functools import wraps
from typing import Callable, Awaitable, Any

from telegram import Update
from telegram.ext import ContextTypes, CallbackContext

from src.bot.utils.config import config
from src.bot.services.permission_service import PermissionService
from src.bot.models.user import UserPermission, User


async def _is_admin(telegram_id: int) -> bool:
    """Check admin status: in config.admin_user_ids OR has ADMIN_ACCESS permission.

    Mirrors PermissionService.is_admin but works at import time without
    instantiating PermissionService (the decorator may run before the
    service is wired up). Looks up the user via PermissionService only
    when the id is not in the static admin list.
    """
    if telegram_id in config.admin_user_ids:
        return True
    try:
        svc = PermissionService()
        user = await svc.get_user(telegram_id)
        return user is not None and user.has_permission(UserPermission.ADMIN_ACCESS)
    except Exception:
        return False


def _get_update_from_args(args) -> Update:
    """Extract Update object from args, handling both bound methods and standalone functions."""
    # For bound methods: args[0] = self, args[1] = update
    # For standalone functions: args[0] = update
    if len(args) >= 2 and hasattr(args[1], 'effective_user'):
        return args[1]
    elif len(args) >= 1 and hasattr(args[0], 'effective_user'):
        return args[0]
    return None


def admin_only(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorator to restrict command to admin users only.

    Admin status is checked by _is_admin, which accepts users either in
    config.admin_user_ids OR with the UserPermission.ADMIN_ACCESS permission
    (typically granted via /grant). This mirrors PermissionService.is_admin
    so /shell and other @admin_only commands accept both kinds of admin.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Skip in test mode
        if getattr(config, 'test_mode', False):
            return await func(*args, **kwargs)

        update = _get_update_from_args(args)
        if update is None:
            return await func(*args, **kwargs)

        user_id = update.effective_user.id
        if not await _is_admin(user_id):
            await update.message.reply_text(
                "❌ This command is restricted to administrators only."
            )
            return
        return await func(*args, **kwargs)

    return wrapper


def rate_limit(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Decorator to rate limit user requests."""

    # Simple in-memory rate limiting (replace with Redis in production)
    user_requests: dict[int, list] = {}

    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Skip in test mode
        if getattr(config, 'test_mode', False):
            return await func(*args, **kwargs)

        update = _get_update_from_args(args)
        if update is None:
            return await func(*args, **kwargs)

        user_id = update.effective_user.id
        now = __import__("time").time()

        if user_id not in user_requests:
            user_requests[user_id] = []

        # Clean old requests (older than 1 minute)
        user_requests[user_id] = [
            req_time for req_time in user_requests[user_id] if now - req_time < 60
        ]

        if len(user_requests[user_id]) >= config.rate_limit_per_minute:
            await update.message.reply_text(
                f"⏱️ Rate limit exceeded. Please wait before sending more messages."
            )
            return

        user_requests[user_id].append(now)
        return await func(*args, **kwargs)

    return wrapper


def require_permission(permission: UserPermission):
    """Decorator to require a specific permission."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Skip in test mode
            if getattr(config, 'test_mode', False):
                return await func(*args, **kwargs)

            update = _get_update_from_args(args)
            if update is None:
                return await func(*args, **kwargs)

            user_id = update.effective_user.id
            permission_service = PermissionService()
            user = await permission_service.get_user(user_id)

            if not user or not user.has_permission(permission):
                await update.message.reply_text(
                    f"🔒 This action requires the '{permission.value}' permission."
                )
                return

            return await func(*args, **kwargs)

        return wrapper

    return decorator