"""Tests for Handlers."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.bot.handlers.command_handler import CommandHandler
from src.bot.handlers.privilege_handler import PrivilegeHandler
from src.bot.handlers.message_handler import MessageHandler
from src.bot.models.user import User, UserPermission
from src.bot.models.permission_request import PermissionRequest, PermissionStatus


class TestCommandHandler:
    """Test CommandHandler functionality."""

    @pytest.fixture
    def command_handler(self, mock_permission_service):
        """Create CommandHandler instance."""
        return CommandHandler(mock_permission_service)

    @pytest.fixture
    def mock_permission_service(self):
        """Create mock permission service."""
        service = MagicMock()
        service.get_or_create_user = AsyncMock()
        service.is_admin = AsyncMock(return_value=False)
        service.get_user = AsyncMock(return_value=None)
        return service

    @pytest.mark.asyncio
    async def test_start_command(self, command_handler, mock_permission_service, mock_update, mock_context):
        """Test /start command."""
        mock_user = User(telegram_id=987654321, first_name="Admin")  # Match mock_update user ID
        mock_permission_service.get_or_create_user.return_value = mock_user

        await command_handler.start_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Hello Admin" in call_args
        assert "Available commands" in call_args

    @pytest.mark.asyncio
    async def test_start_command_admin(self, command_handler, mock_permission_service, mock_update, mock_context):
        """Test /start command for admin."""
        mock_user = User(telegram_id=123456789, first_name="Admin", permissions=[UserPermission.ADMIN_ACCESS])
        mock_permission_service.get_or_create_user.return_value = mock_user
        mock_permission_service.is_admin.return_value = True

        await command_handler.start_command(mock_update, mock_context)

        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Admin commands" in call_args

    @pytest.mark.asyncio
    async def test_help_command(self, command_handler, mock_permission_service, mock_update, mock_context):
        """Test /help command."""
        mock_permission_service.is_admin.return_value = False

        await command_handler.help_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Bot Help" in call_args

    @pytest.mark.asyncio
    async def test_permissions_command_no_perms(self, command_handler, mock_permission_service, mock_update, mock_context):
        """Test /permissions with no permissions."""
        mock_permission_service.get_user.return_value = None

        await command_handler.permissions_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "don't have any special permissions" in call_args


class TestPrivilegeHandler:
    """Test PrivilegeHandler functionality."""

    @pytest.fixture
    def privilege_handler(self, mock_permission_service):
        """Create PrivilegeHandler instance."""
        return PrivilegeHandler(mock_permission_service)

    @pytest.fixture
    def mock_permission_service(self):
        """Create mock permission service."""
        service = MagicMock()
        service.is_admin = AsyncMock(return_value=True)  # Admin by default for privilege tests
        service.create_permission_request = AsyncMock()
        service.revoke_permission = AsyncMock(return_value=True)
        service.list_user_permissions = AsyncMock(return_value=[])
        service.get_pending_requests_by_admin = AsyncMock(return_value=[])
        service.approve_request = AsyncMock(return_value=True)
        service.deny_request = AsyncMock(return_value=True)
        service.get_request = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_grant_command_missing_args(self, privilege_handler, mock_update, mock_context):
        """Test /grant with missing arguments."""
        mock_context.args = []

        await privilege_handler.grant_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Usage:" in call_args

    @pytest.mark.asyncio
    async def test_grant_command_success(self, privilege_handler, mock_permission_service, mock_update, mock_context):
        """Test successful /grant command."""
        mock_context.args = ["123456789", "premium_features"]

        mock_request = PermissionRequest(
            requester_id=987654321,
            target_user_id=123456789,
            permission=UserPermission.PREMIUM_FEATURES,
        )
        mock_request._id = "req_123"
        mock_permission_service.create_permission_request.return_value = mock_request

        await privilege_handler.grant_command(mock_update, mock_context)

        mock_permission_service.create_permission_request.assert_called_once()
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Permission request created" in call_args

    @pytest.mark.asyncio
    async def test_revoke_command_success(self, privilege_handler, mock_permission_service, mock_update, mock_context):
        """Test successful /revoke command."""
        mock_context.args = ["123456789", "premium_features"]

        await privilege_handler.revoke_command(mock_update, mock_context)

        mock_permission_service.revoke_permission.assert_called_once_with(
            123456789, UserPermission.PREMIUM_FEATURES
        )
        mock_update.message.reply_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_approve_command_success(self, privilege_handler, mock_permission_service, mock_update, mock_context):
        """Test successful /approve command."""
        mock_context.args = ["req_123"]

        await privilege_handler.approve_command(mock_update, mock_context)

        mock_permission_service.approve_request.assert_called_once_with("req_123", 987654321)
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "Permission granted" in call_args

    @pytest.mark.asyncio
    async def test_deny_command_success(self, privilege_handler, mock_permission_service, mock_update, mock_context):
        """Test successful /deny command."""
        mock_context.args = ["req_123"]

        await privilege_handler.deny_command(mock_update, mock_context)

        mock_permission_service.deny_request.assert_called_once_with("req_123", 987654321)
        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args[0][0]
        assert "denied" in call_args.lower()


class TestMessageHandler:
    """Test MessageHandler functionality."""

    @pytest.fixture
    def message_handler(self, mock_holmes_service, mock_conversation_service, mock_permission_service):
        """Create MessageHandler instance."""
        return MessageHandler(mock_holmes_service, mock_conversation_service, mock_permission_service)

    @pytest.fixture
    def mock_holmes_service(self):
        """Create mock Holmes service."""
        service = MagicMock()
        service.chat = AsyncMock(return_value="Test response")
        return service

    @pytest.fixture
    def mock_conversation_service(self):
        """Create mock conversation service."""
        service = MagicMock()
        service.get_conversation = AsyncMock()
        service.save_conversation = AsyncMock()
        return service

    @pytest.fixture
    def mock_permission_service(self):
        """Create mock permission service."""
        service = MagicMock()
        service.get_or_create_user = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_handle_message_regular(
        self, message_handler, mock_holmes_service, mock_conversation_service, mock_permission_service,
        mock_update, mock_context
    ):
        """Test handling regular message."""
        mock_user = User(telegram_id=123456789, permissions=[])
        mock_permission_service.get_or_create_user.return_value = mock_user

        mock_conversation = MagicMock()
        mock_conversation.messages = []
        mock_conversation_service.get_conversation.return_value = mock_conversation

        mock_update.message.text = "Hello bot"
        mock_context.bot.send_message = AsyncMock()

        await message_handler.handle_message(mock_update, mock_context)

        mock_holmes_service.chat.assert_called_once()
        mock_conversation_service.save_conversation.assert_called_once()
        mock_context.bot.send_message.assert_called_once()
        call_args = mock_context.bot.send_message.call_args[1]
        assert call_args["chat_id"] == mock_update.effective_chat.id
        assert call_args["text"] == "Test response"

    @pytest.mark.asyncio
    async def test_handle_message_skip_commands(
        self, message_handler, mock_permission_service, mock_update, mock_context
    ):
        """Test that commands are skipped."""
        mock_update.message.text = "/start"

        await message_handler.handle_message(mock_update, mock_context)

        # Should not call Holmes for commands
        mock_permission_service.get_or_create_user.assert_not_called()

    @pytest.mark.asyncio
    async def test_batch_combines_messages(
        self, message_handler, mock_holmes_service, mock_conversation_service,
        mock_permission_service, mock_update, mock_context, monkeypatch
    ):
        """Rapid consecutive messages in a topic flush as one combined Holmes call."""
        import src.bot.handlers.message_handler as mh

        # Real batching path (not test_mode immediate flush); tiny quiet period.
        monkeypatch.setattr(mh.config, "test_mode", False)
        monkeypatch.setattr(mh.config, "message_batch_quiet_period_seconds", 0.05)
        monkeypatch.setattr(mh.config, "message_batch_max_wait_seconds", 30)
        monkeypatch.setattr(mh.config, "message_batch_max_messages", 20)
        mh._topic_batches.clear()

        mock_user = User(telegram_id=123456789, permissions=[])
        mock_permission_service.get_or_create_user.return_value = mock_user
        mock_conversation = MagicMock()
        mock_conversation.messages = []
        mock_conversation_service.get_conversation.return_value = mock_conversation

        # Two quick messages from the same user in the same topic.
        mock_update.message.message_id = 101
        mock_update.message.text = "hello"
        await message_handler.handle_message(mock_update, mock_context)
        mock_update.message.message_id = 102
        mock_update.message.text = "world"
        await message_handler.handle_message(mock_update, mock_context)

        # Let the quiet-period timer fire.
        await asyncio.sleep(0.2)

        mock_holmes_service.chat.assert_called_once()
        kwargs = mock_holmes_service.chat.call_args.kwargs
        assert kwargs["user_message"] == "hello\n\nworld"
        mh._topic_batches.clear()

    @pytest.mark.asyncio
    async def test_stop_discards_pending_batch(
        self, message_handler, mock_holmes_service, mock_permission_service,
        mock_update, mock_context, monkeypatch
    ):
        """/stop while a batch is pending discards it and skips Holmes."""
        import src.bot.handlers.message_handler as mh

        monkeypatch.setattr(mh.config, "test_mode", False)
        monkeypatch.setattr(mh.config, "message_batch_quiet_period_seconds", 10)
        monkeypatch.setattr(mh.config, "message_batch_max_wait_seconds", 30)
        monkeypatch.setattr(mh.config, "message_batch_max_messages", 20)
        mh._topic_batches.clear()

        mock_update.message.text = "hello"
        await message_handler.handle_message(mock_update, mock_context)

        # /stop before the quiet period elapses -> discard.
        await message_handler.stop_command(mock_update, mock_context)

        mock_holmes_service.chat.assert_not_called()
        reply_args = mock_update.message.reply_text.call_args
        assert "Stopped" in reply_args[0][0]
        mh._topic_batches.clear()

    @pytest.mark.asyncio
    async def test_stop_in_flight_signals_cancel_event(
        self, message_handler, mock_holmes_service, mock_conversation_service,
        mock_permission_service, mock_update, mock_context, monkeypatch
    ):
        """/stop during dispatch sets the batch's cancel_event so Holmes aborts."""
        import src.bot.handlers.message_handler as mh

        # Immediate flush (test_mode) so a batch is in flight right away.
        monkeypatch.setattr(mh.config, "test_mode", True)
        mh._topic_batches.clear()

        captured = {}

        async def fake_chat(*args, **kwargs):
            ev = kwargs.get("cancel_event")
            captured["event"] = ev
            # Simulate /stop firing while Holmes is mid-call.
            if ev is not None:
                ev.set()
            return "partial"

        mock_holmes_service.chat = AsyncMock(side_effect=fake_chat)
        mock_user = User(telegram_id=123456789, permissions=[])
        mock_permission_service.get_or_create_user.return_value = mock_user
        mock_conversation = MagicMock()
        mock_conversation.messages = []
        mock_conversation_service.get_conversation.return_value = mock_conversation

        mock_update.message.text = "hello"
        await message_handler.handle_message(mock_update, mock_context)

        # cancel_event was threaded into holmes.chat and set by the stop path.
        assert captured.get("event") is not None
        assert captured["event"].is_set()
        mh._topic_batches.clear()

    @pytest.mark.asyncio
    async def test_stop_during_stream_shows_stopped_footer(
        self, message_handler, mock_holmes_service, mock_conversation_service,
        mock_permission_service, mock_update, mock_context, monkeypatch
    ):
        """When Holmes raises LLMInterruptedError mid-stream, the fix-up uses a 'Stopped' footer + status, not 'Stream interrupted'."""
        import src.bot.handlers.message_handler as mh
        from holmes.core.tool_calling_llm import LLMInterruptedError

        monkeypatch.setattr(mh.config, "test_mode", True)
        mh._topic_batches.clear()

        # Streaming: holmes.chat returns an async generator that raises LLMInterruptedError after one content chunk.
        async def fake_stream(*args, **kwargs):
            yield {"type": "content", "content": "partial answer"}
            raise LLMInterruptedError()

        mock_holmes_service.chat = AsyncMock(return_value=fake_stream())
        mock_user = User(telegram_id=123456789, permissions=[UserPermission.STREAMING_RESPONSES])
        mock_permission_service.get_or_create_user.return_value = mock_user
        mock_conversation = MagicMock()
        mock_conversation.messages = []
        mock_conversation_service.get_conversation.return_value = mock_conversation

        # Streaming needs real message objects whose edit_text we can inspect.
        main_msg = MagicMock(edit_text=AsyncMock())
        status_msg = MagicMock(edit_text=AsyncMock())
        mock_update.message.reply_text = AsyncMock(side_effect=[main_msg, status_msg])

        mock_update.message.text = "hello"
        await message_handler.handle_message(mock_update, mock_context)
        # Let the stream coroutine drain.
        await asyncio.sleep(0.05)

        # The main message's edit_text should contain the 'Stopped by user.' footer,
        # not the 'Stream interrupted' error text.
        edited_texts = [c.args[0] for c in main_msg.edit_text.call_args_list if c.args]
        assert any("🛑 Stopped by user." in t for t in edited_texts), edited_texts
        assert not any("Stream interrupted" in t for t in edited_texts), edited_texts
        mh._topic_batches.clear()