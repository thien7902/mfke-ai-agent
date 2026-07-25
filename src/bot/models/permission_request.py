"""Permission request model."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from bson import ObjectId

from src.bot.models.user import UserPermission
from src.bot.utils.config import config


class PermissionStatus(str, Enum):
    """Status of a permission request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class PermissionRequest:
    """Permission request model for privilege escalation workflow."""

    requester_id: int  # Admin who requested the permission
    target_user_id: int  # User who needs to approve
    permission: UserPermission
    status: PermissionStatus = PermissionStatus.PENDING
    reason: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow()
        + timedelta(minutes=config.permission_request_expiry_minutes)
    )
    _id: Optional[ObjectId] = None

    def is_expired(self) -> bool:
        """Check if the request has expired."""
        return datetime.utcnow() > self.expires_at

    def approve(self):
        """Mark request as approved."""
        self.status = PermissionStatus.APPROVED
        self.updated_at = datetime.utcnow()

    def deny(self):
        """Mark request as denied."""
        self.status = PermissionStatus.DENIED
        self.updated_at = datetime.utcnow()

    def revoke(self):
        """Mark request as revoked."""
        self.status = PermissionStatus.REVOKED
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        data = {
            "requester_id": self.requester_id,
            "target_user_id": self.target_user_id,
            "permission": self.permission.value,
            "status": self.status.value,
            "reason": self.reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }
        if self._id:
            data["_id"] = self._id
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PermissionRequest":
        """Create PermissionRequest from MongoDB document."""
        req = cls(
            requester_id=data["requester_id"],
            target_user_id=data["target_user_id"],
            permission=UserPermission(data["permission"]),
            status=PermissionStatus(data["status"]),
            reason=data.get("reason"),
            created_at=data.get("created_at", datetime.utcnow()),
            updated_at=data.get("updated_at", datetime.utcnow()),
            expires_at=data.get("expires_at", datetime.utcnow()),
        )
        req._id = data.get("_id")
        return req