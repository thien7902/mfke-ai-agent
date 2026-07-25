"""User model with permissions."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
from bson import ObjectId


class UserPermission(str, Enum):
    """Available user permissions."""

    ADMIN_ACCESS = "admin_access"
    PREMIUM_FEATURES = "premium_features"
    AGENT_EXECUTION = "agent_execution"
    TOOL_USAGE = "tool_usage"
    CONVERSATION_MEMORY = "conversation_memory"
    STREAMING_RESPONSES = "streaming_responses"


@dataclass
class User:
    """User model with permissions."""

    telegram_id: int
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    permissions: List[UserPermission] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    _id: Optional[ObjectId] = None

    def has_permission(self, permission: UserPermission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions

    def add_permission(self, permission: UserPermission):
        """Add a permission to the user."""
        if permission not in self.permissions:
            self.permissions.append(permission)
            self.updated_at = datetime.utcnow()

    def remove_permission(self, permission: UserPermission):
        """Remove a permission from the user."""
        if permission in self.permissions:
            self.permissions.remove(permission)
            self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        data = {
            "telegram_id": self.telegram_id,
            "username": self.username,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "permissions": [p.value for p in self.permissions],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self._id:
            data["_id"] = self._id
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        """Create User from MongoDB document."""
        user = cls(
            telegram_id=data["telegram_id"],
            username=data.get("username"),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            permissions=[UserPermission(p) for p in data.get("permissions", [])],
            created_at=data.get("created_at", datetime.utcnow()),
            updated_at=data.get("updated_at", datetime.utcnow()),
        )
        user._id = data.get("_id")
        return user