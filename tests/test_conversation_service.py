"""Tests for Conversation Service."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.bot.services.conversation_service import ConversationService
from src.bot.models.conversation import Conversation, Message


class TestConversationService:
    """Test ConversationService functionality."""

    @pytest.fixture
    def conversation_service(self):
        """Create ConversationService instance."""
        return ConversationService()

    @pytest.fixture
    def sample_conversation(self):
        """Create a sample conversation."""
        return Conversation(
            user_id=123456789,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
        )

    @pytest.mark.asyncio
    async def test_get_conversation_exists(self, conversation_service, sample_conversation):
        """Test getting existing conversation."""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = sample_conversation.to_dict()
        conversation_service._collection = mock_collection

        conv = await conversation_service.get_conversation(123456789)

        assert conv.user_id == 123456789
        assert len(conv.messages) == 2
        mock_collection.find_one.assert_called_once_with({"user_id": 123456789})

    @pytest.mark.asyncio
    async def test_get_conversation_new_topic_isolated(self, conversation_service):
        """Test that new forum topic gets isolated conversation (no fallback to user_id)."""
        mock_collection = MagicMock()
        # No document for the specific topic
        mock_collection.find_one.return_value = None
        conversation_service._collection = mock_collection

        conv = await conversation_service.get_conversation(123456789, topic_id=999, chat_id=888)

        assert conv.user_id == 123456789
        assert conv.topic_id == 999
        assert conv.chat_id == 888
        assert len(conv.messages) == 0
        # Should only query by topic_id+chat_id, not fall back to user_id
        mock_collection.find_one.assert_called_once_with({"topic_id": 999, "chat_id": 888})

    @pytest.mark.asyncio
    async def test_get_conversation_private_chat_fallback(self, conversation_service, sample_conversation):
        """Test that private chat (topic_id=0) falls back to user_id."""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = sample_conversation.to_dict()
        conversation_service._collection = mock_collection

        conv = await conversation_service.get_conversation(123456789, topic_id=0, chat_id=0)

        assert conv.user_id == 123456789
        assert conv.topic_id == 0
        assert conv.chat_id == 0
        assert len(conv.messages) == 2
        mock_collection.find_one.assert_called_once_with({"user_id": 123456789})

    @pytest.mark.asyncio
    async def test_save_conversation_new(self, conversation_service, sample_conversation):
        """Test saving new conversation."""
        mock_collection = MagicMock()
        mock_collection.insert_one.return_value = MagicMock(inserted_id="conv_id")
        conversation_service._collection = mock_collection

        conv = await conversation_service.save_conversation(sample_conversation)

        assert conv._id == "conv_id"
        mock_collection.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_conversation_update(self, conversation_service, sample_conversation):
        """Test updating existing conversation."""
        sample_conversation._id = "existing_id"
        mock_collection = MagicMock()
        mock_collection.replace_one.return_value = MagicMock()
        conversation_service._collection = mock_collection

        conv = await conversation_service.save_conversation(sample_conversation)

        assert conv._id == "existing_id"
        mock_collection.replace_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_message(self, conversation_service):
        """Test adding a message."""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        mock_collection.insert_one.return_value = MagicMock(inserted_id="conv_id")
        conversation_service._collection = mock_collection

        message = Message(role="user", content="New message")
        conv = await conversation_service.add_message(123456789, message)

        assert len(conv.messages) == 1
        assert conv.messages[0].content == "New message"

    @pytest.mark.asyncio
    async def test_get_recent_messages(self, conversation_service, sample_conversation):
        """Test getting recent messages."""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = sample_conversation.to_dict()
        conversation_service._collection = mock_collection

        messages = await conversation_service.get_recent_messages(123456789, limit=1)

        assert len(messages) == 1
        assert messages[0].content == "Hi there!"

    @pytest.mark.asyncio
    async def test_clear_conversation(self, conversation_service):
        """Test clearing conversation."""
        mock_collection = MagicMock()
        mock_collection.delete_one.return_value = MagicMock(deleted_count=1)
        conversation_service._collection = mock_collection

        result = await conversation_service.clear_conversation(123456789)

        assert result is True
        mock_collection.delete_one.assert_called_once_with({"user_id": 123456789})

    @pytest.mark.asyncio
    async def test_cleanup_old_conversations(self, conversation_service):
        """Test cleaning up old conversations."""
        mock_collection = MagicMock()
        mock_collection.delete_many.return_value = MagicMock(deleted_count=10)
        conversation_service._collection = mock_collection

        count = await conversation_service.cleanup_old_conversations(days=30)

        assert count == 10
        mock_collection.delete_many.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_all_user_ids(self, conversation_service):
        """Test getting all user IDs."""
        mock_collection = MagicMock()
        mock_collection.distinct.return_value = [123456789, 987654321]
        conversation_service._collection = mock_collection

        user_ids = await conversation_service.get_all_user_ids()

        assert user_ids == [123456789, 987654321]
        mock_collection.distinct.assert_called_once_with("user_id")