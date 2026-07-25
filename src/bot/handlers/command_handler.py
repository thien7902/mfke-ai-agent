"""Command Handler - Handle basic bot commands."""
import structlog
from telegram import Update
from telegram.ext import ContextTypes

from src.bot.services.permission_service import PermissionService
from src.bot.services.conversation_service import ConversationService
from src.bot.models.user import UserPermission
from src.bot.utils.decorators import rate_limit

logger = structlog.get_logger(__name__)


class CommandHandler:
    """Handle basic bot commands."""

    def __init__(self, permission_service: PermissionService, conversation_service: ConversationService = None):
        self.permissions = permission_service
        self.conversations = conversation_service

    @rate_limit
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user = update.effective_user
        user_model = await self.permissions.get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )

        # Check if we're in a forum topic
        chat = update.effective_chat
        topic_id = update.message.message_thread_id or 0

        welcome_text = (
            f"👋 Hello {user.first_name or 'there'}!\n\n"
            "I'm an AI assistant powered by HolmesGPT. "
            "I can help you with various tasks using AI agents and tools.\n\n"
            "📝 **Available commands:**\n"
            "/start - Show this welcome message\n"
            "/help - Show help information\n"
            "/new - Create a new conversation thread\n"
            "/permissions - View your permissions\n"
            "/clear - Clear conversation history\n\n"
            "Just send me a message to start chatting!"
        )

        if chat.type in ("group", "supergroup") and chat.is_forum:
            if topic_id == 0:
                welcome_text += "\n\n💡 **Note:** In this forum, please create a topic or use an existing one to chat with me."
            else:
                welcome_text += "\n\n💬 You're in a topic - your conversation is isolated here."

        # Add admin commands if user is admin
        if await self.permissions.is_admin(user.id):
            welcome_text += (
                "\n\n🔧 **Admin commands:**\n"
                "/grant <user_id> <permission> - Request permission for user\n"
                "/revoke <user_id> <permission> - Revoke user permission\n"
                "/user_permissions <user_id> - View user's permissions\n"
                "/pending - View pending permission requests\n"
            )

        await update.message.reply_text(welcome_text, message_thread_id=topic_id or None)

    @rate_limit
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        user = update.effective_user
        is_admin = await self.permissions.is_admin(user.id)
        topic_id = update.message.message_thread_id or 0

        help_text = (
            "🤖 **Bot Help**\n\n"
            "**How to use:**\n"
            "• Send any text message to chat with the AI\n"
            "• In forums: mention @botname in general chat to create a new thread\n"
            "• The AI has access to agents and tools based on your permissions\n\n"
            "**Commands:**\n"
            "/start - Welcome message\n"
            "/help - This help message\n"
            "/new - Create a new conversation thread (forum only)\n"
            "/permissions - View your current permissions\n"
            "/clear - Clear your conversation history\n"
        )

        if is_admin:
            help_text += (
                "\n**Admin Commands:**\n"
                "/grant <user_id> <permission> - Request a permission for a user\n"
                "/revoke <user_id> <permission> - Revoke a user's permission\n"
                "/user_permissions <user_id> - View another user's permissions\n"
                "/pending - View all pending permission requests\n\n"
                "**Available Permissions:**\n"
                "• admin_access - Full admin access\n"
                "• premium_features - Access to premium features\n"
                "• agent_execution - Execute AI agents\n"
                "• tool_usage - Use AI tools\n"
                "• conversation_memory - Persistent conversation memory\n"
                "• streaming_responses - Real-time streaming responses\n"
            )

        await update.message.reply_text(help_text, message_thread_id=topic_id or None)

    @rate_limit
    async def permissions_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /permissions command - show user's permissions."""
        user = update.effective_user
        user_model = await self.permissions.get_user(user.id)
        topic_id = update.message.message_thread_id or 0

        if not user_model or not user_model.permissions:
            await update.message.reply_text(
                "🔐 You don't have any special permissions yet.\n"
                "An admin can grant you permissions using /grant command.",
                message_thread_id=topic_id or None
            )
            return

        perm_text = "🔐 **Your Permissions:**\n\n"
        for perm in user_model.permissions:
            perm_text += f"• {perm.value}\n"

        await update.message.reply_text(perm_text, message_thread_id=topic_id or None)

    @rate_limit
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command - clear conversation history."""
        user = update.effective_user
        topic_id = update.message.message_thread_id or 0
        chat = update.effective_chat
        chat_id = chat.id if chat.type in ("group", "supergroup") and chat.is_forum else 0

        if self.conversations:
            await self.conversations.clear_conversation(user.id, topic_id=topic_id, chat_id=chat_id)
        else:
            # Fallback if conversation_service not injected
            conv_service = ConversationService()
            await conv_service.clear_conversation(user.id, topic_id=topic_id, chat_id=chat_id)

        await update.message.reply_text(
            "🗑️ Your conversation history has been cleared.",
            message_thread_id=topic_id or None
        )

    @rate_limit
    async def new_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /new command - create a new forum topic/thread."""
        user = update.effective_user
        chat = update.effective_chat
        topic_id = update.message.message_thread_id or 0

        if chat.type in ("group", "supergroup") and chat.is_forum:
            if topic_id != 0:
                # Already in a topic - create a new one
                await update.message.reply_text(
                    "🔄 You're already in a topic! Use /new in the general chat to create a fresh thread.",
                    message_thread_id=topic_id
                )
                return

            # In general chat - create new topic
            from src.bot.handlers.message_handler import _generate_topic_name

            topic_name = _generate_topic_name("New conversation", user.id)
            try:
                forum_topic = await context.bot.create_forum_topic(
                    chat_id=chat.id,
                    name=topic_name[:100],
                    icon_color=0x6FB9F0
                )
                new_topic_id = forum_topic.message_thread_id

                await context.bot.send_message(
                    chat_id=chat.id,
                    text=(
                        f"🆕 **New conversation started!**\n\n"
                        f"👋 Welcome {user.first_name or 'there'}! This is your fresh conversation topic.\n\n"
                        f"🤖 I'm ready to help. Just send me a message here!\n\n"
                        f"💡 Use /clear to reset conversation history."
                    ),
                    message_thread_id=new_topic_id
                )
            except Exception as e:
                logger.error("Failed to create new topic", user_id=user.id, error=str(e))
                await update.message.reply_text(
                    "❌ Failed to create new conversation topic. Please contact admin.",
                    message_thread_id=None
                )
        else:
            # Private chat - just clear conversation
            if self.conversations:
                await self.conversations.clear_conversation(user.id)
            else:
                conv_service = ConversationService()
                await conv_service.clear_conversation(user.id)

            await update.message.reply_text(
                "🆕 New conversation started! Your history has been cleared.",
                message_thread_id=None
            )