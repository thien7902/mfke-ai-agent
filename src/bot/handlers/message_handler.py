"""Message Handler - Process regular messages through Holmes."""
import logging
import random
import structlog
import time
from telegram import Update, ForumTopic
from telegram.ext import ContextTypes

from src.bot.services.holmes_service import HolmesService
from src.bot.services.conversation_service import ConversationService
from src.bot.services.permission_service import PermissionService
from src.bot.models.user import UserPermission
from src.bot.models.conversation import Conversation
from src.bot.utils.config import config

logger = structlog.get_logger(__name__)

# In-memory cache to prevent duplicate topic creation (user_id -> timestamp)
_recent_topic_creation: dict[int, float] = {}
_TOPIC_CREATION_COOLDOWN = 10  # seconds - increased from 5

# In-memory cache to prevent duplicate message processing (message_id -> timestamp)
_processed_messages: dict[int, float] = {}
_MESSAGE_PROCESSING_TTL = 30  # seconds


def _generate_topic_name(user_message: str, user_id: int) -> str:
    """Generate a topic name from user message: 2-3 word summary + random ID."""
    # Simple keyword extraction - take first few meaningful words
    words = user_message.strip().split()
    # Filter out common words, mentions, keep meaningful words
    meaningful_words = []
    for word in words:
        clean_word = word.strip('@#.,!?').lower()
        if len(clean_word) > 2 and clean_word not in {'the', 'and', 'for', 'you', 'are', 'who', 'what', 'how', 'why', 'when', 'where', 'can', 'please', 'help', 'bot', 'gpt', 'mfke'}:
            meaningful_words.append(clean_word)
            if len(meaningful_words) >= 3:
                break

    if meaningful_words:
        summary = ' '.join(meaningful_words[:3]).title()
    else:
        summary = "New Chat"

    # Add random 4-digit ID to prevent duplication
    random_id = random.randint(1000, 9999)
    return f"💬 {summary} #{random_id}"


class MessageHandler:
    """Handle regular text messages via Holmes."""

    def __init__(
        self,
        holmes_service: HolmesService,
        conversation_service: ConversationService,
        permission_service: PermissionService,
    ):
        self.holmes = holmes_service
        self.conversations = conversation_service
        self.permissions = permission_service

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming text message."""
        if not update.message or not update.message.text:
            return

        user = update.effective_user
        user_id = user.id
        message_text = update.message.text
        message_id = update.message.message_id

        # Skip commands (they start with /)
        if message_text.startswith("/"):
            return

        # Deduplication: skip if we've processed this message recently
        current_time = time.time()
        if message_id in _processed_messages:
            elapsed = current_time - _processed_messages[message_id]
            if elapsed < _MESSAGE_PROCESSING_TTL:
                logger.warning("Duplicate message ignored", message_id=message_id, user_id=user_id, elapsed=elapsed)
                return
        _processed_messages[message_id] = current_time

        # Clean up old entries
        _processed_messages.clear()  # Simple cleanup - in production use a proper TTL cache

        # Get chat and topic info
        chat = update.effective_chat
        chat_id = chat.id
        topic_id = update.message.message_thread_id or 0

        # Only process messages in configured chats (private or specific forum chat)
        if chat.type == "private":
            # Private chat - use user_id as key
            topic_id = 0
            chat_id = 0
        elif chat.type in ("group", "supergroup") and chat.is_forum:
            # Forum topic - use topic_id
            if not topic_id:
                # Message in general chat, not a topic - ignore or create topic
                await self._handle_general_chat_message(update, context, user, chat_id)
                return
        else:
            # Regular group - ignore
            return

        logger.info("Received message", user_id=user_id, chat_id=chat_id, topic_id=topic_id, text=message_text[:50])

        try:
            # Get or create user
            user_model = await self.permissions.get_or_create_user(
                telegram_id=user_id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
            )

            # Get conversation (by topic for forums, by user for private)
            conversation = await self.conversations.get_conversation(
                user_id, topic_id=topic_id, chat_id=chat_id
            )

            # Get user permissions
            user_permissions = user_model.permissions

            # Send typing indicator
            await context.bot.send_chat_action(
                chat_id=chat_id or update.effective_chat.id, action="typing", message_thread_id=topic_id or None
            )

            # Check if user has streaming permission
            use_streaming = UserPermission.STREAMING_RESPONSES in user_permissions

            if use_streaming:
                await self._handle_streaming_response(
                    update, conversation, user_permissions, message_text, chat_id, topic_id
                )
            else:
                await self._handle_regular_response(
                    update, conversation, user_permissions, message_text, chat_id, topic_id
                )

        except Exception as e:
            # Safely convert error to string (handles non-serializable objects like ToolCallResult)
            try:
                error_str = str(e)
            except Exception:
                error_str = f"<unserializable error: {type(e).__name__}>"
            logger.error("Error handling message", user_id=user_id, error=error_str)
            await update.message.reply_text(
                "❌ An error occurred while processing your message. Please try again.",
                message_thread_id=topic_id or None
            )

    async def handle_topic_created(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle when a new forum topic is created."""
        if not update.message or not update.message.forum_topic_created:
            return

        chat_id = update.effective_chat.id
        topic_id = update.message.message_thread_id
        topic_name = update.message.forum_topic_created.name

        logger.info("Forum topic created", chat_id=chat_id, topic_id=topic_id, name=topic_name)

        # Try to extract user_id from topic name (format: "💬 Username" or "User 123456")
        user_id = None
        if topic_name.startswith("💬 "):
            # Try to find user by name
            username = topic_name[3:].strip()
            # We can't easily get user_id from name, so we'll wait for first message
            pass

        # Just log it for now - the first message in the topic will create the conversation
        await update.message.reply_text(
            f"📝 New topic created: {topic_name}\n"
            f"Send a message here to start chatting with the AI!",
            message_thread_id=topic_id
        )

    async def _handle_general_chat_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, user, chat_id: int):
        """Handle messages in general chat (not in a topic) - create a new topic for each mention."""
        user_message = update.message.text
        bot_username = context.bot.username or "mfke_gpt_bot"

        # Only create topic if user mentions the bot (@mfke_gpt_bot)
        if f"@{bot_username}" not in user_message:
            # Not mentioning bot - ignore silently in general chat
            return

        # Check cooldown to prevent duplicate topic creation
        user_id = user.id
        current_time = time.time()
        if user_id in _recent_topic_creation:
            elapsed = current_time - _recent_topic_creation[user_id]
            if elapsed < _TOPIC_CREATION_COOLDOWN:
                logger.warning("Topic creation cooldown active", user_id=user_id, elapsed=elapsed)
                await update.message.reply_text(
                    f"⏳ Please wait a moment before creating another topic.",
                    message_thread_id=None
                )
                return

        # Remove the mention from the message for processing
        clean_message = user_message.replace(f"@{bot_username}", "").strip()

        # Mark topic creation time
        _recent_topic_creation[user_id] = current_time

        # Create a NEW forum topic for each mention (don't reuse existing topics)
        try:
            topic_name = _generate_topic_name(clean_message or "Hello", user.id)
            forum_topic = await context.bot.create_forum_topic(
                chat_id=chat_id,
                name=topic_name[:100],  # Max 100 chars
                icon_color=0x6FB9F0  # Blue color
            )

            topic_id = forum_topic.message_thread_id

            # Send welcome message in the new topic
            welcome_msg = await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    f"👋 Welcome {user.first_name or 'there'}! This is your new conversation topic.\n\n"
                    f"🤖 I'm ready to help. Just send me a message here!\n\n"
                    f"💡 Use /clear to reset conversation history."
                ),
                message_thread_id=topic_id
            )

            logger.info("Created new topic for user", user_id=user.id, topic_id=topic_id, chat_id=chat_id, topic_name=topic_name)

            # Process the user's message in the new topic by creating a synthetic update
            # We'll reply directly in the new topic instead of re-processing through handle_message
            # Get or create user
            user_model = await self.permissions.get_or_create_user(
                telegram_id=user.id,
                username=user.username,
                first_name=user.first_name,
                last_name=user.last_name,
            )

            # Get conversation for the new topic
            conversation = await self.conversations.get_conversation(
                user.id, topic_id=topic_id, chat_id=chat_id
            )

            # Get user permissions
            user_permissions = user_model.permissions

            # Send typing indicator in the new topic
            await context.bot.send_chat_action(
                chat_id=chat_id, action="typing", message_thread_id=topic_id
            )

            # Check if user has streaming permission
            use_streaming = UserPermission.STREAMING_RESPONSES in user_permissions

            # Use clean message (without @mention) for processing
            process_message = clean_message if clean_message else "Hello"

            if use_streaming:
                await self._handle_streaming_response_in_topic(
                    context, conversation, user_permissions, process_message, chat_id, topic_id, user.id
                )
            else:
                await self._handle_regular_response_in_topic(
                    context, conversation, user_permissions, process_message, chat_id, topic_id, user.id
                )

        except Exception as e:
            logger.error("Failed to create forum topic", user_id=user.id, error=str(e))
            await update.message.reply_text(
                "❌ Failed to create conversation topic. Please contact admin.",
                message_thread_id=None
            )

    async def _handle_regular_response(
        self,
        update: Update,
        conversation: Conversation,
        user_permissions: list,
        message_text: str,
        chat_id: int,
        topic_id: int,
    ):
        """Handle non-streaming response."""
        response = await self.holmes.chat(
            user_id=update.effective_user.id,
            user_message=message_text,
            conversation=conversation,
            user_permissions=user_permissions,
            stream=False,
        )

        # Save conversation
        await self.conversations.save_conversation(conversation)

        # Send response
        await update.message.reply_text(response, message_thread_id=topic_id or None)

    async def _handle_streaming_response(
        self,
        update: Update,
        conversation: Conversation,
        user_permissions: list,
        message_text: str,
        chat_id: int,
        topic_id: int,
    ):
        """Handle streaming response."""
        # Send initial empty message to edit
        message = await update.message.reply_text("🤔 Thinking...", message_thread_id=topic_id or None)

        full_response = ""
        try:
            async_gen = await self.holmes.chat(
                user_id=update.effective_user.id,
                user_message=message_text,
                conversation=conversation,
                user_permissions=user_permissions,
                stream=True,
            )
            async for chunk in async_gen:
                full_response += chunk
                # Edit message every few chunks to avoid rate limits
                if len(full_response) % 50 == 0:
                    try:
                        await message.edit_text(full_response + "▌", message_thread_id=topic_id or None)
                    except Exception:
                        pass  # Ignore edit conflicts

            # Final edit
            await message.edit_text(full_response, message_thread_id=topic_id or None)

        except Exception as e:
            logger.error("Streaming error", error=str(e))
            await message.edit_text(full_response + "\n\n⚠️ Stream interrupted", message_thread_id=topic_id or None)

        # Save conversation
        await self.conversations.save_conversation(conversation)

    async def _handle_regular_response_in_topic(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        conversation: Conversation,
        user_permissions: list,
        message_text: str,
        chat_id: int,
        topic_id: int,
        user_id: int,
    ):
        """Handle non-streaming response in a specific topic (without update object)."""
        response = await self.holmes.chat(
            user_id=user_id,
            user_message=message_text,
            conversation=conversation,
            user_permissions=user_permissions,
            stream=False,
        )

        # Save conversation
        await self.conversations.save_conversation(conversation)

        # Send response in the topic
        await context.bot.send_message(
            chat_id=chat_id,
            text=response,
            message_thread_id=topic_id
        )

    async def _handle_streaming_response_in_topic(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        conversation: Conversation,
        user_permissions: list,
        message_text: str,
        chat_id: int,
        topic_id: int,
        user_id: int,
    ):
        """Handle streaming response in a specific topic (without update object)."""
        # Send initial empty message to edit
        message = await context.bot.send_message(
            chat_id=chat_id,
            text="🤔 Thinking...",
            message_thread_id=topic_id
        )

        full_response = ""
        try:
            async_gen = await self.holmes.chat(
                user_id=user_id,
                user_message=message_text,
                conversation=conversation,
                user_permissions=user_permissions,
                stream=True,
            )
            async for chunk in async_gen:
                full_response += chunk
                # Edit message every few chunks to avoid rate limits
                if len(full_response) % 50 == 0:
                    try:
                        await message.edit_text(full_response + "▌", message_thread_id=topic_id)
                    except Exception:
                        pass  # Ignore edit conflicts

            # Final edit
            await message.edit_text(full_response, message_thread_id=topic_id)

        except Exception as e:
            logger.error("Streaming error", error=str(e))
            await message.edit_text(full_response + "\n\n⚠️ Stream interrupted", message_thread_id=topic_id)

        # Save conversation
        await self.conversations.save_conversation(conversation)