"""Conversation Service - MongoDB storage for conversation history and Holmes memory."""
import logging
from typing import List, Optional

import structlog
from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection

from src.bot.utils.mongodb import MongoDB
from src.bot.models.conversation import Conversation, Message

logger = structlog.get_logger(__name__)


class ConversationService:
    """Service for managing conversations in MongoDB."""

    COLLECTION_NAME = "conversations"

    def __init__(self):
        self._collection: Optional[Collection] = None

    @property
    def collection(self) -> Collection:
        """Get conversations collection, creating indexes if needed."""
        if self._collection is None:
            self._collection = MongoDB.get_collection(self.COLLECTION_NAME)
            self._create_indexes()
        return self._collection

    def _create_indexes(self):
        """Create database indexes for efficient queries."""
        try:
            self._collection.create_index([("user_id", ASCENDING)])
            self._collection.create_index([("topic_id", ASCENDING), ("chat_id", ASCENDING)], unique=True, sparse=True)
            self._collection.create_index([("updated_at", DESCENDING)])
            self._collection.create_index([("holmes_session_id", ASCENDING)], sparse=True)
        except Exception as e:
            logger.warning("Failed to create indexes", error=str(e))

    async def get_conversation(self, user_id: int, topic_id: int = 0, chat_id: int = 0) -> Conversation:
        """Get or create conversation for a user/topic."""
        # Try to find by topic_id and chat_id first (for forum topics)
        if topic_id and chat_id:
            doc = self.collection.find_one({"topic_id": topic_id, "chat_id": chat_id})
            if doc:
                return Conversation.from_dict(doc)

        # Fallback to user_id (for private chats or legacy)
        doc = self.collection.find_one({"user_id": user_id})
        if doc:
            return Conversation.from_dict(doc)

        return Conversation(user_id=user_id, topic_id=topic_id, chat_id=chat_id)

    async def get_conversation_by_topic(self, topic_id: int, chat_id: int) -> Optional[Conversation]:
        """Get conversation by topic ID."""
        doc = self.collection.find_one({"topic_id": topic_id, "chat_id": chat_id})
        if doc:
            return Conversation.from_dict(doc)
        return None

    async def save_conversation(self, conversation: Conversation) -> Conversation:
        """Save or update conversation in MongoDB."""
        doc = conversation.to_dict()
        if conversation._id:
            self.collection.replace_one({"_id": conversation._id}, doc)
        else:
            # Use upsert with topic_id + chat_id as unique key
            if conversation.topic_id and conversation.chat_id:
                self.collection.update_one(
                    {"topic_id": conversation.topic_id, "chat_id": conversation.chat_id},
                    {"$set": doc},
                    upsert=True
                )
            else:
                result = self.collection.insert_one(doc)
                conversation._id = result.inserted_id
        return conversation

    async def add_message(self, user_id: int, message: Message, topic_id: int = 0, chat_id: int = 0) -> Conversation:
        """Add a message to user's conversation."""
        conversation = await self.get_conversation(user_id, topic_id, chat_id)
        conversation.add_message(message)
        return await self.save_conversation(conversation)

    async def get_recent_messages(
        self, user_id: int, limit: int = 20, topic_id: int = 0, chat_id: int = 0
    ) -> List[Message]:
        """Get recent messages for a user."""
        conversation = await self.get_conversation(user_id, topic_id, chat_id)
        return conversation.get_recent_messages(limit)

    async def clear_conversation(self, user_id: int, topic_id: int = 0, chat_id: int = 0) -> bool:
        """Clear conversation history for a user."""
        if topic_id and chat_id:
            result = self.collection.delete_one({"topic_id": topic_id, "chat_id": chat_id})
        else:
            result = self.collection.delete_one({"user_id": user_id})
        return result.deleted_count > 0

    async def get_conversation_count(self) -> int:
        """Get total number of conversations."""
        return self.collection.count_documents({})

    async def cleanup_old_conversations(self, days: int = 30) -> int:
        """Remove conversations older than specified days."""
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(days=days)
        result = self.collection.delete_many({"updated_at": {"$lt": cutoff}})
        return result.deleted_count

    async def get_all_user_ids(self) -> List[int]:
        """Get all user IDs that have conversations."""
        return self.collection.distinct("user_id")

    async def get_all_topics(self, chat_id: int) -> List[dict]:
        """Get all topics in a forum chat."""
        cursor = self.collection.find({"chat_id": chat_id}, {"topic_id": 1, "user_id": 1, "updated_at": 1})
        return list(cursor)