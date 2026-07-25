"""Main bot entry point."""
import asyncio
import logging
import signal
import sys

import structlog
from telegram.ext import (
    Application,
    CommandHandler as TelegramCommandHandler,
    MessageHandler as TelegramMessageHandler,
    CallbackQueryHandler,
    filters,
)

from src.bot.utils.config import config
from src.bot.utils.mongodb import MongoDB
from src.bot.services.holmes_service import HolmesService
from src.bot.services.conversation_service import ConversationService
from src.bot.services.permission_service import PermissionService
from src.bot.handlers.message_handler import MessageHandler
from src.bot.handlers.command_handler import CommandHandler as BotCommandHandler
from src.bot.handlers.privilege_handler import PrivilegeHandler

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Configure standard logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=getattr(logging, config.log_level.upper(), logging.INFO),
)

logger = structlog.get_logger(__name__)


class TelegramBot:
    """Main Telegram bot application."""

    def __init__(self):
        self.application: Application = None
        self.holmes_service: HolmesService = None
        self.conversation_service: ConversationService = None
        self.permission_service: PermissionService = None

    async def initialize(self):
        """Initialize all services and the Telegram application."""
        logger.info("Initializing bot...")

        # Initialize MongoDB connection
        MongoDB.get_database()
        logger.info("MongoDB connected")

        # Initialize services
        self.holmes_service = HolmesService()
        await self.holmes_service.initialize()

        self.conversation_service = ConversationService()
        self.permission_service = PermissionService()

        # Create Telegram application
        self.application = (
            Application.builder()
            .token(config.telegram_bot_token)
            .build()
        )

        # Initialize handlers
        msg_handler = MessageHandler(
            self.holmes_service,
            self.conversation_service,
            self.permission_service,
        )
        cmd_handler = BotCommandHandler(self.permission_service, self.conversation_service)
        priv_handler = PrivilegeHandler(self.permission_service)

        # Register command handlers
        self.application.add_handler(TelegramCommandHandler("start", cmd_handler.start_command))
        self.application.add_handler(TelegramCommandHandler("help", cmd_handler.help_command))
        self.application.add_handler(TelegramCommandHandler("permissions", cmd_handler.permissions_command))
        self.application.add_handler(TelegramCommandHandler("clear", cmd_handler.clear_command))
        self.application.add_handler(TelegramCommandHandler("new", cmd_handler.new_command))

        # Privilege commands
        self.application.add_handler(TelegramCommandHandler("grant", priv_handler.grant_command))
        self.application.add_handler(TelegramCommandHandler("revoke", priv_handler.revoke_command))
        self.application.add_handler(TelegramCommandHandler("user_permissions", priv_handler.user_permissions_command))
        self.application.add_handler(TelegramCommandHandler("pending", priv_handler.pending_command))
        self.application.add_handler(TelegramCommandHandler("approve", priv_handler.approve_command))
        self.application.add_handler(TelegramCommandHandler("deny", priv_handler.deny_command))

        # Callback query handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(priv_handler.handle_callback_query))

        # Message handler for regular text (non-commands)
        self.application.add_handler(
            TelegramMessageHandler(filters.TEXT & ~filters.COMMAND, msg_handler.handle_message)
        )

        # Forum topic created handler
        self.application.add_handler(
            TelegramMessageHandler(filters.StatusUpdate.FORUM_TOPIC_CREATED, msg_handler.handle_topic_created)
        )

        # Error handler
        self.application.add_error_handler(self._error_handler)

        logger.info("Bot initialized successfully")

    async def _error_handler(self, update: object, context: "ContextTypes.DEFAULT_TYPE"):
        """Global error handler."""
        logger.error("Unhandled error", error=str(context.error), update=str(update)[:200])

    async def start(self):
        """Start the bot."""
        if not self.application:
            await self.initialize()

        logger.info("Starting bot...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True  # Clear any pending updates from other instances
        )
        logger.info("Bot started successfully")

    async def stop(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")

        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()

        if self.holmes_service:
            await self.holmes_service.close()

        MongoDB.close()

        logger.info("Bot stopped")


async def main():
    """Main entry point."""
    bot = TelegramBot()

    # Setup signal handlers for graceful shutdown (Unix only)
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(bot.stop())

    # Signal handlers not available on Windows
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass

    try:
        await bot.initialize()
        await bot.start()

        # Keep running until stopped
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        sys.exit(1)
    finally:
        await bot.stop()


if __name__ == "__main__":
    # Import Update here to avoid circular import
    from telegram import Update
    from telegram.ext import ContextTypes

    asyncio.run(main())