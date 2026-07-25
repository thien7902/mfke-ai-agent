"""Tests for Models."""
import pytest
from datetime import datetime, timedelta
from bson import ObjectId

from src.bot.models.user import User, UserPermission
from src.bot.models.conversation import Conversation, Message
from src.bot.models.permission_request import PermissionRequest, PermissionStatus


class TestUserModel:
    """Test User model."""

    def test_user_creation(self):
        """Test user creation."""
        user = User(
            telegram_id=123456789,
            username="testuser",
            first_name="Test",
            last_name="User",
        )

        assert user.telegram_id == 123456789
        assert user.username == "testuser"
        assert user.permissions == []

    def test_user_permissions(self):
        """Test permission management."""
        user = User(telegram_id=123456789)

        user.add_permission(UserPermission.ADMIN_ACCESS)
        assert user.has_permission(UserPermission.ADMIN_ACCESS)
        assert len(user.permissions) == 1

        user.add_permission(UserPermission.ADMIN_ACCESS)  # Duplicate
        assert len(user.permissions) == 1

        user.remove_permission(UserPermission.ADMIN_ACCESS)
        assert not user.has_permission(UserPermission.ADMIN_ACCESS)

    def test_user_to_dict(self):
        """Test user serialization."""
        user = User(
            telegram_id=123456789,
            username="testuser",
            permissions=[UserPermission.ADMIN_ACCESS, UserPermission.TOOL_USAGE],
        )
        user._id = ObjectId()

        data = user.to_dict()

        assert data["telegram_id"] == 123456789
        assert data["username"] == "testuser"
        assert "admin_access" in data["permissions"]
        assert "tool_usage" in data["permissions"]
        assert "_id" in data

    def test_user_from_dict(self):
        """Test user deserialization."""
        data = {
            "_id": ObjectId(),
            "telegram_id": 123456789,
            "username": "testuser",
            "permissions": ["admin_access", "tool_usage"],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        user = User.from_dict(data)

        assert user.telegram_id == 123456789
        assert user.username == "testuser"
        assert UserPermission.ADMIN_ACCESS in user.permissions
        assert UserPermission.TOOL_USAGE in user.permissions
        assert user._id == data["_id"]


class TestConversationModel:
    """Test Conversation model."""

    def test_conversation_creation(self):
        """Test conversation creation."""
        conv = Conversation(user_id=123456789)

        assert conv.user_id == 123456789
        assert conv.messages == []
        assert conv.holmes_session_id is None

    def test_add_message(self):
        """Test adding messages."""
        conv = Conversation(user_id=123456789)
        msg = Message(role="user", content="Hello")

        conv.add_message(msg)

        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Hello"

    def test_get_recent_messages(self):
        """Test getting recent messages."""
        conv = Conversation(user_id=123456789)
        for i in range(5):
            conv.add_message(Message(role="user", content=f"Message {i}"))

        recent = conv.get_recent_messages(limit=3)

        assert len(recent) == 3
        assert recent[0].content == "Message 2"
        assert recent[2].content == "Message 4"

    def test_to_holmes_format(self):
        """Test conversion to Holmes format."""
        conv = Conversation(user_id=123456789)
        conv.add_message(Message(role="user", content="Hello"))
        conv.add_message(Message(role="assistant", content="Hi!"))

        holmes_format = conv.to_holmes_format()

        assert len(holmes_format) == 2
        assert holmes_format[0]["role"] == "user"
        assert holmes_format[0]["content"] == "Hello"
        assert holmes_format[1]["role"] == "assistant"

    def test_conversation_serialization(self):
        """Test conversation to/from dict."""
        conv = Conversation(
            user_id=123456789,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ],
            holmes_session_id="session_123",
        )
        conv._id = ObjectId()

        data = conv.to_dict()
        restored = Conversation.from_dict(data)

        assert restored.user_id == conv.user_id
        assert len(restored.messages) == 2
        assert restored.holmes_session_id == "session_123"
        assert restored._id == conv._id


class TestMessageModel:
    """Test Message model."""

    def test_message_creation(self):
        """Test message creation."""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls == []
        assert msg.tool_call_id is None

    def test_message_with_tool_calls(self):
        """Test message with tool calls."""
        msg = Message(
            role="assistant",
            content="",
            tool_calls=[{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
            tool_call_id="call_1",
        )

        assert len(msg.tool_calls) == 1
        assert msg.tool_call_id == "call_1"

    def test_message_serialization(self):
        """Test message to/from dict."""
        msg = Message(
            role="user",
            content="Hello",
            metadata={"source": "telegram"},
            tool_calls=[{"id": "call_1"}],
            tool_call_id="call_1",
            name="test_function",
        )

        data = msg.to_dict()
        restored = Message.from_dict(data)

        assert restored.role == "user"
        assert restored.content == "Hello"
        assert restored.metadata == {"source": "telegram"}
        assert restored.tool_calls == [{"id": "call_1"}]
        assert restored.tool_call_id == "call_1"
        assert restored.name == "test_function"


class TestPermissionRequestModel:
    """Test PermissionRequest model."""

    def test_permission_request_creation(self):
        """Test permission request creation."""
        req = PermissionRequest(
            requester_id=987654321,
            target_user_id=123456789,
            permission=UserPermission.PREMIUM_FEATURES,
            reason="Test reason",
        )

        assert req.requester_id == 987654321
        assert req.target_user_id == 123456789
        assert req.permission == UserPermission.PREMIUM_FEATURES
        assert req.reason == "Test reason"
        assert req.status == PermissionStatus.PENDING
        assert not req.is_expired()

    def test_permission_request_expiry(self):
        """Test permission request expiry."""
        past = datetime.utcnow() - timedelta(minutes=5)
        req = PermissionRequest(
            requester_id=987654321,
            target_user_id=123456789,
            permission=UserPermission.PREMIUM_FEATURES,
            expires_at=past,
        )

        assert req.is_expired()

    def test_approve_deny_revoke(self):
        """Test request status changes."""
        req = PermissionRequest(
            requester_id=987654321,
            target_user_id=123456789,
            permission=UserPermission.PREMIUM_FEATURES,
        )

        req.approve()
        assert req.status == PermissionStatus.APPROVED

        req.deny()
        assert req.status == PermissionStatus.DENIED

        req.revoke()
        assert req.status == PermissionStatus.REVOKED

    def test_permission_request_serialization(self):
        """Test permission request to/from dict."""
        req = PermissionRequest(
            requester_id=987654321,
            target_user_id=123456789,
            permission=UserPermission.PREMIUM_FEATURES,
            reason="Test reason",
        )
        req._id = ObjectId()

        data = req.to_dict()
        restored = PermissionRequest.from_dict(data)

        assert restored.requester_id == req.requester_id
        assert restored.target_user_id == req.target_user_id
        assert restored.permission == req.permission
        assert restored.reason == req.reason
        assert restored.status == req.status
        assert restored._id == req._id