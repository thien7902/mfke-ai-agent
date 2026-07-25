"""Conversation and Message models."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from bson import ObjectId


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # For tool calls
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool/function messages

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create Message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.utcnow()),
            metadata=data.get("metadata", {}),
            tool_calls=data.get("tool_calls", []),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass
class Conversation:
    """Conversation model with Holmes-compatible memory."""

    user_id: int
    topic_id: int = 0  # Telegram topic/thread ID (0 for legacy/no topic)
    chat_id: int = 0   # Telegram chat ID (for group/forum chats)
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    holmes_session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    _id: Optional[ObjectId] = None

    def add_message(self, message: Message):
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def get_recent_messages(self, limit: int = 20) -> List[Message]:
        """Get recent messages for context."""
        return self.messages[-limit:]

    def to_holmes_format(self) -> List[dict]:
        """Convert messages to Holmes-compatible format."""
        return [msg.to_dict() for msg in self.messages]

    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB storage."""
        data = {
            "user_id": self.user_id,
            "topic_id": self.topic_id,
            "chat_id": self.chat_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata,
            "holmes_session_id": self.holmes_session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self._id:
            data["_id"] = self._id
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        """Create Conversation from MongoDB document."""
        conv = cls(
            user_id=data["user_id"],
            topic_id=data.get("topic_id", 0),
            chat_id=data.get("chat_id", 0),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            metadata=data.get("metadata", {}),
            holmes_session_id=data.get("holmes_session_id"),
            created_at=data.get("created_at", datetime.utcnow()),
            updated_at=data.get("updated_at", datetime.utcnow()),
        )
        conv._id = data.get("_id")
        return conv