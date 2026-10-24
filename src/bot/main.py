"""Main bot entry point."""
import asyncio
import logging
import os
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
from src.bot.services.memory_service import MemoryService
from src.bot.handlers.message_handler import MessageHandler
from src.bot.handlers.command_handler import CommandHandler as BotCommandHandler
from src.bot.commands.add_memory import AddMemoryCommand
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

# Fatal polling errors: PTB logs these via the `telegram.ext.Updater` logger
# and never routes them through application.add_error_handler. So we attach
# a logging handler on that logger to detect them and exit — Docker/K8s
# restarts the container with a fresh network/TLS environment. Same restart
# semantics as the 90-min idle watchdog at main.py:285.
FATAL_POLLING_ERRORS = (
    'certificate_verify_failed',
    'unable to get local issuer certificate',
    'ssl: handshake failed',
    'tls handshake',
)


class FatalPollingDetector(logging.Handler):
    """Exit the process when PTB logs a fatal TLS/certificate error.

    A missing CA cert or broken TLS handshake is not transient — a fresh
    container is the only useful recovery.
    """

    def emit(self, record):
        try:
            msg = record.getMessage().lower()
        except Exception:
            return
        if any(sig in msg for sig in FATAL_POLLING_ERRORS):
            logger.error(
                "Fatal TLS/certificate error in PTB Updater - exiting for container restart",
                error=record.getMessage(),
            )
            health_checker.set_bot_alive(False)
            os._exit(1)


class TelegramBot:
    """Main Telegram bot application."""

    def __init__(self):
        self.application: Application = None
        self.holmes_service: HolmesService = None
        self.conversation_service: ConversationService = None
        self.permission_service: PermissionService = None
        self.memory_service: MemoryService = None
        self.last_update_time: float = 0  # Track last received update
        self.polling_restarts: int = 0    # Consecutive in-place restarts with no updates since

    async def initialize(self):
        """Initialize all services and the Telegram application."""
        logger.info("Initializing bot...")

        # Initialize MongoDB connection
        MongoDB.get_database()
        logger.info("MongoDB connected")
        health_checker.set_mongodb_connected(True)

        # Initialize services
        self.memory_service = MemoryService()
        # Memory init is best-effort: if FalkorDB is unreachable or graphiti
        # fails to come up, the bot still runs — memory retrieval just returns
        # [] and /addmemory replies with an error. Memory start failures are
        # logged but never fatal.
        try:
            await self.memory_service.initialize()
        except Exception as e:
            logger.error(
                "memory_init_failed_bot_continues",
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            # Also print to stdout so it's impossible to miss in docker logs
            # even when structlog formatting is being filtered by log level.
            import traceback
            print(f"[MEMORY INIT FAILED] {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

        self.holmes_service = HolmesService(memory_service=self.memory_service)
        await self.holmes_service.initialize()
        health_checker.set_holmes_ready(True)

        self.conversation_service = ConversationService()
        self.permission_service = PermissionService()

        # Create Telegram application
        # Defaults (PTB HTTPXRequest) are read/write=5s, pool=1s, pool_size=1 — too tight
        # for concurrent send/edit_message_text under .concurrent_updates(True). The
        # start_polling timeouts below only cover getUpdates, not bot API sends, so raise
        # the general request's timeouts and pool size here to avoid telegram.error.TimedOut
        # ("Timed out") on long/concurrent responses.
        self.application = (
            Application.builder()
            .token(config.telegram_bot_token)
            .concurrent_updates(True)  # Enable concurrent update processing
            .connection_pool_size(8)  # Multiple concurrent bot API requests
            .read_timeout(60)  # Large send_message payloads can exceed the 5s default
            .write_timeout(60)
            .connect_timeout(15)
            .pool_timeout(15)  # Waiting for a free pooled connection shouldn't blow up in 1s
            .build()
        )

        # Initialize handlers
        msg_handler = MessageHandler(
            self.holmes_service,
            self.conversation_service,
            self.permission_service,
        )
        cmd_handler = BotCommandHandler(
            self.permission_service, self.conversation_service, holmes_service=self.holmes_service
        )
        priv_handler = PrivilegeHandler(self.permission_service)
        add_memory_handler = AddMemoryCommand(self.memory_service, self.permission_service)

        # Register command handlers
        self.application.add_handler(TelegramCommandHandler("start", cmd_handler.start_command))
        self.application.add_handler(TelegramCommandHandler("help", cmd_handler.help_command))
        self.application.add_handler(TelegramCommandHandler("permissions", cmd_handler.permissions_command))
        self.application.add_handler(TelegramCommandHandler("clear", cmd_handler.clear_command))
        self.application.add_handler(TelegramCommandHandler("new", cmd_handler.new_command))
        self.application.add_handler(TelegramCommandHandler("stop", msg_handler.stop_command))
        self.application.add_handler(TelegramCommandHandler("ping", msg_handler.ping_command))
        self.application.add_handler(TelegramCommandHandler("shell", cmd_handler.shell_command))
        self.application.add_handler(TelegramCommandHandler("reconcile", cmd_handler.reconcile_command))
        self.application.add_handler(TelegramCommandHandler("kshell", cmd_handler.kshell_command))
        # /claude — admin only; routes a one-shot turn through the enterprise
        # model configured via CLAUDE_ENTERPRISE_MODEL (key in
        # ~/.holmes/model_list.yaml). See CommandHandler.claude_command.
        self.application.add_handler(TelegramCommandHandler("claude", cmd_handler.claude_command))
        # /addmemory — admin only; ingests replied message text or attached file
        # content as a knowledge episode into Graphiti (see src/bot/commands/add_memory.py)
        self.application.add_handler(TelegramCommandHandler("addmemory", add_memory_handler.handle))

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

        # Document/photo uploads with caption starting with `/addmemory` —
        # PTB's CommandHandler only matches commands in `message.text`, not in
        # `message.caption`, so a file upload captioned "/addmemory" would
        # otherwise fall through to handle_file as a normal user message. This
        # handler catches that case and routes to AddMemoryCommand.handle_upload.
        # Registered before the generic handle_file handler so it wins the match.
        #
        # The CaptionRegex filter matches the command form, including an optional
        # "/addmemory@BotName" suffix and any text after it. The handler parses
        # trailing args as extra caption text to ingest alongside the file.
        self.application.add_handler(
            TelegramMessageHandler(
                (filters.Document.ALL | filters.PHOTO)
                & filters.CaptionRegex(r"^/addmemory(\s|$|@)"),
                add_memory_handler.handle_upload,
            ),
            group=1,  # Separate group so a non-/addmemory caption still falls through to handle_file
        )

        # Document/photo uploads: download, extract text, and run through the
        # same batch/streaming pipeline as text messages. See MessageHandler.handle_file.
        self.application.add_handler(
            TelegramMessageHandler(filters.Document.ALL | filters.PHOTO, msg_handler.handle_file)
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
        # A real update arrived — the polling connection (and any previous in-place
        # restart) worked. Clear the consecutive-restart counter so the watchdog
        # starts fresh if polling sticks again later.
        self.polling_restarts = 0

    async def _error_handler(self, update: object, context: "ContextTypes.DEFAULT_TYPE"):
        """Global error handler with connection recovery."""
        error_str = str(context.error)
        logger.error("Unhandled error", error=error_str, update=str(update)[:200])

        error_lower = error_str.lower()

        # Recoverable network errors — try to restore MongoDB and keep going.
        # Fatal TLS/cert errors during polling are caught by FatalPollingDetector
        # (PTB routes polling errors to its own logger, not this handler).
        if any(keyword in error_lower for keyword in [
            'connection', 'timeout', 'network', 'unreachable',
        ]):
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

        if self.memory_service:
            await self.memory_service.close()

        # Shutdown the Holmes thread pool executor
        _holmes_executor.shutdown(wait=True)

        MongoDB.close()
        health_checker.set_mongodb_connected(False)

        logger.info("Bot stopped")


async def main():
    """Main entry point."""
    # Attach the fatal-polling detector to PTB's Updater logger. Polling errors
    # (SSL/cert/TLS handshake failures during getUpdates) are logged here and
    # never reach application.add_error_handler, so we hook the logger stream.
    logging.getLogger("telegram.ext.Updater").addHandler(FatalPollingDetector())

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

                # Check if Telegram polling is stuck (no updates for N minutes).
                # This detects silent connection failures.
                #
                # Three-tier recovery (30 / 60 / 90 min):
                #   30 min → in-place restart (attempt #1)
                #   60 min → in-place restart (attempt #2; restart #1 didn't help)
                #   90 min → exit process so Docker's `restart: unless-stopped` starts a
                #            fresh container with fresh httpx/TCP connections
                #
                # We can't rely on exceptions from `stop()`/`start_polling()` to detect a
                # failed restart: PTB suppresses the `PoolTimeout` from its internal
                # `_get_updates_cleanup` during `stop()`, and `start_polling()` returns
                # normally even when the resulting polling task can't reach Telegram.
                # "No updates arrived since the last restart" is the only reliable signal.
                # `polling_restarts` is reset to 0 in `_update_tracker` when a real update
                # arrives, confirming the restart worked.
                time_since_update = time.time() - bot.last_update_time
                if time_since_update >= 5400:  # 90 minutes — two restarts already failed
                    logger.error(
                        "No Telegram updates for 90+ minutes - exiting for container restart",
                        minutes_idle=time_since_update / 60,
                        consecutive_restarts=bot.polling_restarts,
                    )
                    raise SystemExit(1)
                elif time_since_update >= 1800:  # 30+ minutes — try in-place restart
                    bot.polling_restarts += 1
                    logger.warning(
                        "No Telegram updates received for 30+ minutes - restarting polling",
                        minutes_idle=time_since_update / 60,
                        attempt=bot.polling_restarts,
                    )
                    try:
                        await bot.application.updater.stop()
                        await asyncio.sleep(2)
                        await bot.application.updater.start_polling(
                            allowed_updates=Update.ALL_TYPES,
                            drop_pending_updates=False,  # Keep backlog — don't lose user messages
                            pool_timeout=30,
                            connect_timeout=30,
                            read_timeout=30,
                            write_timeout=30,
                        )
                        logger.info("Telegram polling restarted successfully")
                        # NOTE: do NOT reset last_update_time here. If the restart worked,
                        # _update_tracker will update it when the next real update arrives,
                        # and _update_tracker resets polling_restarts to 0 at the same time.
                        # If the restart failed, last_update_time stays stale and the next
                        # watchdog trip will either restart again (at 60 min) or exit (at 90).
                    except Exception as restart_error:
                        logger.error(
                            "Failed to restart polling - exiting for container restart",
                            error=str(restart_error)
                        )
                        raise SystemExit(1)

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