"""Tests for Handlers."""
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