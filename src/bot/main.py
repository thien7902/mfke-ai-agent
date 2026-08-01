"""Main bot entry point."""
import asyncio
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor

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
from src.bot.utils.health import health_checker
from src.bot.services.holmes_service import HolmesService, _holmes_executor
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
        self.last_update_time: float = 0  # Track last received update

    async def initialize(self):
        """Initialize all services and the Telegram application."""
        logger.info("Initializing bot...")

        # Initialize MongoDB connection
        MongoDB.get_database()
        logger.info("MongoDB connected")
        health_checker.set_mongodb_connected(True)

        # Initialize services
        self.holmes_service = HolmesService()
        await self.holmes_service.initialize()
        health_checker.set_holmes_ready(True)

        self.conversation_service = ConversationService()
        self.permission_service = PermissionService()

        # Create Telegram application
        self.application = (
            Application.builder()
            .token(config.telegram_bot_token)
            .concurrent_updates(True)  # Enable concurrent update processing
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

        # Add update tracker to monitor connection health (runs before all other handlers)
        self.application.add_handler(
            TelegramMessageHandler(filters.ALL, self._update_tracker), group=-1
        )

        # Forum topic created handler
        self.application.add_handler(
            TelegramMessageHandler(filters.StatusUpdate.FORUM_TOPIC_CREATED, msg_handler.handle_topic_created)
        )

        # Error handler
        self.application.add_error_handler(self._error_handler)

        logger.info("Bot initialized successfully")

    async def _update_tracker(self, update: object, context: "ContextTypes.DEFAULT_TYPE"):
        """Track when we receive updates to detect silent connection failures."""
        import time
        self.last_update_time = time.time()

    async def _error_handler(self, update: object, context: "ContextTypes.DEFAULT_TYPE"):
        """Global error handler with connection recovery."""
        error_str = str(context.error)
        logger.error("Unhandled error", error=error_str, update=str(update)[:200])

        # Check for network/connection errors and attempt recovery
        if any(keyword in error_str.lower() for keyword in ['connection', 'timeout', 'network', 'unreachable']):
            logger.warning("Detected connection error, checking connections...")
            try:
                # Test MongoDB connection
                MongoDB.get_database().command('ping')
            except Exception as e:
                logger.error("MongoDB connection failed, reconnecting", error=str(e))
                MongoDB.close()
                MongoDB.get_database()

    async def start(self):
        """Start the bot."""
        if not self.application:
            await self.initialize()

        logger.info("Starting bot...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,  # Clear any pending updates from other instances
            pool_timeout=30,  # Connection timeout for long polling
            connect_timeout=30,  # Initial connection timeout
            read_timeout=30,  # Read timeout between updates
            write_timeout=30,  # Write timeout for sending updates
        )
        health_checker.set_bot_alive(True)
        logger.info("Bot started successfully")

    async def stop(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        health_checker.set_bot_alive(False)

        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()

        if self.holmes_service:
            await self.holmes_service.close()
            health_checker.set_holmes_ready(False)

        # Shutdown the Holmes thread pool executor
        _holmes_executor.shutdown(wait=True)

        MongoDB.close()
        health_checker.set_mongodb_connected(False)

        logger.info("Bot stopped")


async def main():
    """Main entry point."""
    # Start health check server first
    await health_checker.start()

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

        # Keep running until stopped with periodic health checks
        last_health_check = asyncio.get_event_loop().time()
        import time
        bot.last_update_time = time.time()  # Initialize

        while True:
            await asyncio.sleep(60)  # Check every minute

            # Periodic connection health check
            current_time = asyncio.get_event_loop().time()
            if current_time - last_health_check >= 300:  # Every 5 minutes
                try:
                    # Ping MongoDB to ensure connection is alive
                    MongoDB.get_database().command('ping')
                    logger.debug("MongoDB connection healthy")
                except Exception as e:
                    logger.error("MongoDB connection lost, reconnecting", error=str(e))
                    MongoDB.close()
                    MongoDB.get_database()  # Reconnect

                # Check if Telegram polling is stuck (no updates for 2+ hours during active hours)
                # This detects silent connection failures
                time_since_update = time.time() - bot.last_update_time
                if time_since_update > 7200:  # 2 hours with no activity
                    logger.warning(
                        "No Telegram updates received for 2+ hours - possible silent connection failure",
                        hours_idle=time_since_update / 3600
                    )
                    # Send a test message to ourselves to verify connection
                    try:
                        me = await bot.application.bot.get_me()
                        logger.info("Telegram connection verified", bot_username=me.username)
                        bot.last_update_time = time.time()  # Reset timer after check
                    except Exception as e:
                        logger.error("Telegram connection test failed - restarting polling", error=str(e))
                        # Restart the polling connection
                        try:
                            await bot.application.updater.stop()
                            await asyncio.sleep(2)
                            await bot.application.updater.start_polling(
                                allowed_updates=Update.ALL_TYPES,
                                drop_pending_updates=True,
                                pool_timeout=30,
                                connect_timeout=30,
                                read_timeout=30,
                                write_timeout=30,
                            )
                            logger.info("Telegram polling restarted successfully")
                            bot.last_update_time = time.time()
                        except Exception as restart_error:
                            logger.error("Failed to restart polling", error=str(restart_error))

                last_health_check = current_time

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error("Fatal error", error=str(e))
        sys.exit(1)
    finally:
        await bot.stop()
        await health_checker.stop()


if __name__ == "__main__":
    # Import Update here to avoid circular import
    from telegram import Update
    from telegram.ext import ContextTypes

    asyncio.run(main())