"""Test configuration and fixtures."""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.bot.models.user import User, UserPermission
from src.bot.models.conversation import Conversation, Message
from src.bot.models.permission_request import PermissionRequest, PermissionStatus


@pytest.fixture
def mock_user():
    """Create a mock user."""
    return User(
        telegram_id=123456789,
        username="testuser",
        first_name="Test",
        last_name="User",
        permissions=[UserPermission.AGENT_EXECUTION, UserPermission.TOOL_USAGE],
    )


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user."""
    return User(
        telegram_id=987654321,
        username="admin",
        first_name="Admin",
        last_name="User",
        permissions=[UserPermission.ADMIN_ACCESS],
    )


@pytest.fixture
def mock_conversation():
    """Create a mock conversation."""
    return Conversation(
        user_id=123456789,
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ],
    )


@pytest.fixture
def mock_permission_request():
    """Create a mock permission request."""
    return PermissionRequest(
        requester_id=987654321,
        target_user_id=123456789,
        permission=UserPermission.PREMIUM_FEATURES,
        reason="User requested premium access",
    )


@pytest.fixture
def mock_update():
    """Create a mock Telegram Update."""
    update = MagicMock()
    update.effective_user = MagicMock()
    update.effective_user.id = 987654321  # Match admin ID in mock_config
    update.effective_user.username = "admin"
    update.effective_user.first_name = "Admin"
    update.effective_user.last_name = "User"
    update.effective_chat = MagicMock()
    update.effective_chat.id = 987654321
    update.effective_chat.type = "private"  # Default to private chat
    update.effective_chat.is_forum = False
    update.message = MagicMock()
    update.message.text = "Test message"
    update.message.reply_text = AsyncMock()
    update.message.edit_text = AsyncMock()
    update.message.message_thread_id = None  # No topic in private chat
    return update


@pytest.fixture
def mock_context():
    """Create a mock Telegram Context."""
    context = MagicMock()
    context.args = []
    context.bot = MagicMock()
    context.bot.send_message = AsyncMock()
    context.bot.send_chat_action = AsyncMock()
    return context


@pytest.fixture
def mock_callback_query():
    """Create a mock CallbackQuery."""
    query = MagicMock()
    query.from_user = MagicMock()
    query.from_user.id = 123456789
    query.data = "approve_test123"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


@pytest.fixture(autouse=True)
def mock_config():
    """Mock configuration for tests."""
    with patch("src.bot.utils.config.config") as mock:
        mock.telegram_bot_token = "test_token"
        mock.mongodb_uri = "mongodb://localhost:27017"
        mock.mongodb_database = "test_db"
        mock.admin_user_ids = [987654321]
        mock.holmes_config_path = "/tmp/holmes/config.yaml"
        mock.log_level = "DEBUG"
        mock.rate_limit_per_minute = 100
        mock.permission_request_expiry_minutes = 10
        mock.test_mode = True  # Enable test mode to skip decorators
        yield mock


@pytest.fixture(autouse=True)
def mock_mongodb():
    """Mock MongoDB for tests."""
    with patch("src.bot.utils.mongodb.MongoDB") as mock:
        mock_db = MagicMock()
        mock.get_database.return_value = mock_db
        mock.get_collection.return_value = MagicMock()
        yield mock