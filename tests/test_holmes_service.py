"""Tests for Holmes Service."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from src.bot.services.holmes_service import HolmesService
from src.bot.models.conversation import Conversation, Message
from src.bot.models.user import UserPermission
from holmes.core.tool_calling_llm import LLMResult


class TestHolmesService:
    """Test HolmesService functionality."""

    @pytest.fixture
    def holmes_service(self):
        """Create HolmesService instance."""
        return HolmesService()

    @pytest.fixture
    def sample_conversation(self):
        """Create a sample conversation."""
        return Conversation(
            user_id=123456789,
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi! How can I help?"),
            ],
        )

    @pytest.fixture
    def mock_llm_result(self):
        """Create a mock LLMResult."""
        return LLMResult(
            result="Test response",
            tool_calls=[],
            num_llm_calls=1,
            messages=[],
            metadata={},
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        )

    @pytest.mark.asyncio
    async def test_initialize(self, holmes_service):
        """Test Holmes initialization."""
        with patch("src.bot.services.holmes_service.DefaultLLM") as mock_llm_class, \
             patch("src.bot.services.holmes_service.ToolExecutor") as mock_executor_class, \
             patch("src.bot.services.holmes_service.ToolCallingLLM") as mock_tcllm_class, \
             patch("src.bot.services.holmes_service.TracingFactory") as mock_tracing_class, \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENAI_API_BASE": "https://api.openai.com/v1"}):

            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm

            mock_executor = MagicMock()
            mock_executor_class.return_value = mock_executor

            mock_tcllm = MagicMock()
            mock_tcllm_class.return_value = mock_tcllm

            mock_tracing = MagicMock()
            mock_tracing_class.return_value = mock_tracing

            await holmes_service.initialize()

            assert holmes_service._initialized is True
            mock_llm_class.assert_called_once()
            mock_executor_class.assert_called_once()
            mock_tcllm_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_non_streaming(self, holmes_service, sample_conversation, mock_llm_result):
        """Test non-streaming chat."""
        with patch("src.bot.services.holmes_service.DefaultLLM"), \
             patch("src.bot.services.holmes_service.ToolExecutor"), \
             patch("src.bot.services.holmes_service.ToolCallingLLM") as mock_tcllm_class, \
             patch("src.bot.services.holmes_service.TracingFactory"), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENAI_API_BASE": "https://api.openai.com/v1"}):

            mock_tcllm = MagicMock()
            mock_tcllm.call.return_value = mock_llm_result
            mock_tcllm_class.return_value = mock_tcllm

            await holmes_service.initialize()

            response = await holmes_service.chat(
                user_id=123456789,
                user_message="Test message",
                conversation=sample_conversation,
                user_permissions=[UserPermission.AGENT_EXECUTION],
                stream=False,
            )

            assert response == "Test response"
            assert len(sample_conversation.messages) == 3  # Original 2 + user message
            mock_tcllm.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_with_streaming(self, holmes_service, sample_conversation):
        """Test streaming chat."""
        with patch("src.bot.services.holmes_service.DefaultLLM"), \
             patch("src.bot.services.holmes_service.ToolExecutor"), \
             patch("src.bot.services.holmes_service.ToolCallingLLM") as mock_tcllm_class, \
             patch("src.bot.services.holmes_service.TracingFactory"), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENAI_API_BASE": "https://api.openai.com/v1"}):

            mock_tcllm = MagicMock()

            # Create mock stream events
            mock_event1 = MagicMock()
            mock_event1.event.name = "AI_MESSAGE"
            mock_event1.data = {"content": "Hello"}

            mock_event2 = MagicMock()
            mock_event2.event.name = "AI_MESSAGE"
            mock_event2.data = {"content": " world"}

            def mock_call_stream(*args, **kwargs):
                return [mock_event1, mock_event2]

            mock_tcllm.call_stream = mock_call_stream
            mock_tcllm_class.return_value = mock_tcllm

            await holmes_service.initialize()

            chunks = []
            # chat returns a coroutine that yields an async generator when stream=True
            async_gen = await holmes_service.chat(
                user_id=123456789,
                user_message="Test",
                conversation=sample_conversation,
                user_permissions=[UserPermission.STREAMING_RESPONSES],
                stream=True,
            )
            async for chunk in async_gen:
                chunks.append(chunk)

            assert chunks == ["Hello", " world"]

    def test_get_user_permissions(self, holmes_service):
        """Test permission conversion."""
        permissions = [
            UserPermission.AGENT_EXECUTION,
            UserPermission.TOOL_USAGE,
        ]
        features = holmes_service._get_user_permissions(permissions)

        assert features["agents"] is True
        assert features["tools"] is True
        assert features["memory"] is False
        assert features["streaming"] is False

    def test_convert_messages_to_llm_format(self, holmes_service, sample_conversation):
        """Test message conversion to LLM format."""
        llm_messages = holmes_service._convert_messages_to_llm_format(sample_conversation.messages)

        assert len(llm_messages) == 2
        assert llm_messages[0]["role"] == "user"
        assert llm_messages[0]["content"] == "Hello"
        assert llm_messages[1]["role"] == "assistant"
        assert llm_messages[1]["content"] == "Hi! How can I help?"

    @pytest.mark.asyncio
    async def test_execute_agent(self, holmes_service, sample_conversation, mock_llm_result):
        """Test agent execution."""
        with patch("src.bot.services.holmes_service.DefaultLLM"), \
             patch("src.bot.services.holmes_service.ToolExecutor"), \
             patch("src.bot.services.holmes_service.ToolCallingLLM") as mock_tcllm_class, \
             patch("src.bot.services.holmes_service.TracingFactory"), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENAI_API_BASE": "https://api.openai.com/v1"}):

            mock_tcllm = MagicMock()
            mock_tcllm.call.return_value = mock_llm_result
            mock_tcllm_class.return_value = mock_tcllm

            await holmes_service.initialize()

            result = await holmes_service.execute_agent(
                user_id=123456789,
                agent_name="test_agent",
                task="Do something",
                conversation=sample_conversation,
                user_permissions=[UserPermission.AGENT_EXECUTION],
            )

            assert result == "Test response"
            mock_tcllm.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_agent_no_permission(self, holmes_service, sample_conversation):
        """Test agent execution without permission."""
        with patch("src.bot.services.holmes_service.DefaultLLM"), \
             patch("src.bot.services.holmes_service.ToolExecutor"), \
             patch("src.bot.services.holmes_service.ToolCallingLLM"), \
             patch("src.bot.services.holmes_service.TracingFactory"), \
             patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "OPENAI_API_BASE": "https://api.openai.com/v1"}):

            await holmes_service.initialize()

            with pytest.raises(PermissionError):
                await holmes_service.execute_agent(
                    user_id=123456789,
                    agent_name="test_agent",
                    task="Do something",
                    conversation=sample_conversation,
                    user_permissions=[],  # No agent permission
                )