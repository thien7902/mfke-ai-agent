"""Permission Service - MongoDB storage for user permissions and privilege requests."""
import logging
from typing import List, Optional

import structlog
from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection

from src.bot.utils.mongodb import MongoDB
from src.bot.models.user import User, UserPermission
from src.bot.models.permission_request import PermissionRequest, PermissionStatus

logger = structlog.get_logger(__name__)


class PermissionService:
    """Service for managing user permissions and privilege requests."""

    USERS_COLLECTION = "users"
    REQUESTS_COLLECTION = "permission_requests"

    def __init__(self):
        self._users_collection: Optional[Collection] = None
        self._requests_collection: Optional[Collection] = None

    @property
    def users_collection(self) -> Collection:
        """Get users collection, creating indexes if needed."""
        if self._users_collection is None:
            self._users_collection = MongoDB.get_collection(self.USERS_COLLECTION)
            self._create_user_indexes()
        return self._users_collection

    @property
    def requests_collection(self) -> Collection:
        """Get permission requests collection, creating indexes if needed."""
        if self._requests_collection is None:
            self._requests_collection = MongoDB.get_collection(self.REQUESTS_COLLECTION)
            self._create_request_indexes()
        return self._requests_collection

    def _create_user_indexes(self):
        """Create database indexes for users collection."""
        try:
            self._users_collection.create_index(
                [("telegram_id", ASCENDING)], unique=True
            )
            self._users_collection.create_index([("updated_at", DESCENDING)])
        except Exception as e:
            logger.warning("Failed to create user indexes", error=str(e))

    def _create_request_indexes(self):
        """Create database indexes for requests collection."""
        try:
            self._requests_collection.create_index(
                [("target_user_id", ASCENDING), ("status", ASCENDING)]
            )
            self._requests_collection.create_index(
                [("requester_id", ASCENDING), ("status", ASCENDING)]
            )
            self._requests_collection.create_index([("expires_at", ASCENDING)])
            self._requests_collection.create_index([("status", ASCENDING)])
        except Exception as e:
            logger.warning("Failed to create request indexes", error=str(e))

    # User management
    async def get_user(self, telegram_id: int) -> Optional[User]:
        """Get user by Telegram ID."""
        doc = self.users_collection.find_one({"telegram_id": telegram_id})
        if doc:
            return User.from_dict(doc)
        return None

    async def get_or_create_user(
        self,
        telegram_id: int,
        username: Optional[str] = None,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
    ) -> User:
        """Get existing user or create new one."""
        user = await self.get_user(telegram_id)
        if user:
            # Update info if provided
            if username:
                user.username = username
            if first_name:
                user.first_name = first_name
            if last_name:
                user.last_name = last_name
            await self.save_user(user)
            return user

        user = User(
            telegram_id=telegram_id,
            username=username,
            first_name=first_name,
            last_name=last_name,
        )
        return await self.save_user(user)

    async def save_user(self, user: User) -> User:
        """Save or update user in MongoDB."""
        doc = user.to_dict()
        if user._id:
            self.users_collection.replace_one({"_id": user._id}, doc)
        else:
            result = self.users_collection.insert_one(doc)
            user._id = result.inserted_id
        return user

    async def grant_permission(
        self, telegram_id: int, permission: UserPermission
    ) -> bool:
        """Grant a permission to a user."""
        user = await self.get_or_create_user(telegram_id)
        user.add_permission(permission)
        await self.save_user(user)
        return True

    async def revoke_permission(
        self, telegram_id: int, permission: UserPermission
    ) -> bool:
        """Revoke a permission from a user."""
        user = await self.get_user(telegram_id)
        if not user:
            return False
        user.remove_permission(permission)
        await self.save_user(user)
        return True

    async def list_user_permissions(self, telegram_id: int) -> List[UserPermission]:
        """List all permissions for a user."""
        user = await self.get_user(telegram_id)
        if not user:
            return []
        return user.permissions

    async def is_admin(self, telegram_id: int) -> bool:
        """Check if user is an admin (has admin_access permission or is in config)."""
        from src.bot.utils.config import config

        if telegram_id in config.admin_user_ids:
            return True

        user = await self.get_user(telegram_id)
        return user is not None and user.has_permission(UserPermission.ADMIN_ACCESS)

    # Permission request workflow
    async def create_permission_request(
        self,
        requester_id: int,
        target_user_id: int,
        permission: UserPermission,
        reason: Optional[str] = None,
    ) -> PermissionRequest:
        """Create a new permission request (admin requests permission for user)."""
        # Check if requester is admin
        if not await self.is_admin(requester_id):
            raise PermissionError("Only admins can request permissions for users")

        # Check if user already has permission
        target_user = await self.get_user(target_user_id)
        if target_user and target_user.has_permission(permission):
            raise ValueError(f"User already has {permission.value} permission")

        # Check for existing pending request
        existing = self.requests_collection.find_one({
            "target_user_id": target_user_id,
            "permission": permission.value,
            "status": PermissionStatus.PENDING.value,
        })
        if existing:
            raise ValueError("Pending request already exists for this permission")

        request = PermissionRequest(
            requester_id=requester_id,
            target_user_id=target_user_id,
            permission=permission,
            reason=reason,
        )

        doc = request.to_dict()
        result = self.requests_collection.insert_one(doc)
        request._id = result.inserted_id

        logger.info(
            "Permission request created",
            request_id=str(request._id),
            requester=requester_id,
            target=target_user_id,
            permission=permission.value,
        )

        return request

    async def get_pending_requests_for_user(self, user_id: int) -> List[PermissionRequest]:
        """Get all pending permission requests for a user."""
        docs = self.requests_collection.find({
            "target_user_id": user_id,
            "status": PermissionStatus.PENDING.value,
        }).sort("created_at", DESCENDING)
        return [PermissionRequest.from_dict(doc) for doc in docs]

    async def get_pending_requests_by_admin(self, admin_id: int) -> List[PermissionRequest]:
        """Get all pending permission requests created by an admin."""
        docs = self.requests_collection.find({
            "requester_id": admin_id,
            "status": PermissionStatus.PENDING.value,
        }).sort("created_at", DESCENDING)
        return [PermissionRequest.from_dict(doc) for doc in docs]

    async def get_request(self, request_id: str) -> Optional[PermissionRequest]:
        """Get a permission request by ID."""
        from bson import ObjectId
        doc = self.requests_collection.find_one({"_id": ObjectId(request_id)})
        if doc:
            return PermissionRequest.from_dict(doc)
        return None

    async def approve_request(self, request_id: str, user_id: int) -> bool:
        """Approve a permission request (user approves)."""
        request = await self.get_request(request_id)
        if not request:
            return False

        if request.target_user_id != user_id:
            raise PermissionError("You can only approve requests for yourself")

        if request.status != PermissionStatus.PENDING:
            raise ValueError("Request is not pending")

        if request.is_expired():
            request.status = PermissionStatus.EXPIRED
            await self._save_request(request)
            return False

        # Grant the permission
        request.approve()
        await self._save_request(request)
        await self.grant_permission(request.target_user_id, request.permission)

        logger.info(
            "Permission request approved",
            request_id=request_id,
            user_id=user_id,
            permission=request.permission.value,
        )

        return True

    async def deny_request(self, request_id: str, user_id: int) -> bool:
        """Deny a permission request (user denies)."""
        request = await self.get_request(request_id)
        if not request:
            return False

        if request.target_user_id != user_id:
            raise PermissionError("You can only deny requests for yourself")

        if request.status != PermissionStatus.PENDING:
            raise ValueError("Request is not pending")

        request.deny()
        await self._save_request(request)

        logger.info(
            "Permission request denied",
            request_id=request_id,
            user_id=user_id,
            permission=request.permission.value,
        )

        return True

    async def revoke_request(self, request_id: str, admin_id: int) -> bool:
        """Revoke a permission request (admin revokes)."""
        if not await self.is_admin(admin_id):
            raise PermissionError("Only admins can revoke requests")

        request = await self.get_request(request_id)
        if not request:
            return False

        if request.requester_id != admin_id:
            raise PermissionError("You can only revoke your own requests")

        request.revoke()
        await self._save_request(request)

        logger.info(
            "Permission request revoked by admin",
            request_id=request_id,
            admin_id=admin_id,
        )

        return True

    async def _save_request(self, request: PermissionRequest):
        """Save permission request to MongoDB."""
        doc = request.to_dict()
        if request._id:
            self.requests_collection.replace_one({"_id": request._id}, doc)
        else:
            result = self.requests_collection.insert_one(doc)
            request._id = result.inserted_id

    async def cleanup_expired_requests(self) -> int:
        """Clean up expired permission requests."""
        from datetime import datetime

        result = self.requests_collection.update_many(
            {
                "status": PermissionStatus.PENDING.value,
                "expires_at": {"$lt": datetime.utcnow()},
            },
            {"$set": {"status": PermissionStatus.EXPIRED.value}},
        )
        return result.modified_count