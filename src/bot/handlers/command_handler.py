"""Command Handler - Handle basic bot commands."""
import asyncio
import shlex
import structlog
from telegram import Update
from telegram.ext import ContextTypes

from src.bot.services.permission_service import PermissionService
from src.bot.services.conversation_service import ConversationService
from src.bot.models.user import UserPermission
from src.bot.utils.decorators import admin_only, rate_limit

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

        # Disable private chats
        chat = update.effective_chat
        if chat.type == "private":
            await update.message.reply_text(
                "🚫 Private chats are disabled. Please use the bot in a group forum topic.",
                message_thread_id=None
            )
            return

        user_model = await self.permissions.get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )

        # Check if we're in a forum topic
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
            "/clear - Clear conversation history\n"
            "/stop - Stop a pending batch or in-flight request in this topic\n"
            "/ping - Show the current agent status in this topic\n\n"
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
                "/shell <command> - Execute a shell command in the bot container\n"
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

        # Disable private chats
        chat = update.effective_chat
        if chat.type == "private":
            await update.message.reply_text(
                "🚫 Private chats are disabled. Please use the bot in a group forum topic.",
                message_thread_id=None
            )
            return

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
            "/stop - Stop a pending batch or in-flight request in this topic\n"
            "/ping - Show the current agent status in this topic\n"
        )

        if is_admin:
            help_text += (
                "\n**Admin Commands:**\n"
                "/shell <command> - Execute a shell command in the bot container\n"
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
                "• tool_approval - Require approval before tool execution\n"
            )

        await update.message.reply_text(help_text, message_thread_id=topic_id or None)

    @rate_limit
    async def permissions_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /permissions command - show user's permissions."""
        user = update.effective_user

        # Disable private chats
        chat = update.effective_chat
        if chat.type == "private":
            await update.message.reply_text(
                "🚫 Private chats are disabled. Please use the bot in a group forum topic.",
                message_thread_id=None
            )
            return

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

        # Disable private chats
        chat = update.effective_chat
        if chat.type == "private":
            await update.message.reply_text(
                "🚫 Private chats are disabled. Please use the bot in a group forum topic.",
                message_thread_id=None
            )
            return

        topic_id = update.message.message_thread_id or 0
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

        # Disable private chats
        if chat.type == "private":
            await update.message.reply_text(
                "🚫 Private chats are disabled. Please use the bot in a group forum topic.",
                message_thread_id=None
            )
            return

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
            # Regular group (non-forum) - not supported
            await update.message.reply_text(
                "🚫 This command only works in forum-enabled groups.",
                message_thread_id=None
            )

    @admin_only
    async def shell_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /shell <command> — execute a shell command in the bot container.

        Admin-only. Replies immediately with an acknowledgement (like /ping),
        then runs the command in the background and edits the ack message with
        the captured output once it completes. This keeps the handler
        non-blocking — the user can /ping or send other messages while the
        shell command runs.
        """
        if not update.message:
            return

        topic_id = update.message.message_thread_id or 0

        # Reconstruct the command string from context.args. PTB strips the
        # /shell prefix and splits the rest on whitespace; join back so users
        # can use either /shell kubectl get pods or /shell "kubectl get pods".
        if not context.args:
            await update.message.reply_text(
                "Usage: /shell <command>\nExample: /shell kubectl get pods -A",
                message_thread_id=topic_id or None,
            )
            return

        if len(context.args) == 1:
            command = context.args[0]
        else:
            command = " ".join(context.args)

        if not command.strip():
            await update.message.reply_text(
                "❌ Empty command.",
                message_thread_id=topic_id or None,
            )
            return

        # Send immediate acknowledgement so the handler returns right away.
        COMMAND_PREVIEW = 200
        cmd_preview = command if len(command) <= COMMAND_PREVIEW else "…" + command[-(COMMAND_PREVIEW - 1):]
        ack = await update.message.reply_text(
            f"⏳ Running...\n\n🔧 Command:\n`{cmd_preview}`",
            message_thread_id=topic_id or None,
            parse_mode="Markdown",
        )

        # Run the command in the background and edit the ack with the result.
        # Don't await — fire-and-forget so the handler returns immediately.
        asyncio.create_task(
            self._run_shell_and_edit(ack, command, topic_id)
        )

    async def _run_shell_and_edit(self, ack_message, command: str, topic_id: int):
        """Run the shell command and edit the ack message with the result.

        Runs in a background task spawned by shell_command. Captures stdout+
        stderr via /bin/bash -c, enforces a 120s timeout (OpenStack / kubectl
        calls can be slow), and edits the ack message with the output (truncated
        to fit a single Telegram message).
        """
        import os
        import signal

        MAX_OUTPUT = 3800
        TIMEOUT_SECONDS = 120

        try:
            # start_new_session=True lets us kill the whole process group on
            # timeout — otherwise SIGKILL only hits bash, leaving openstack/
            # kubectl children alive and holding the stdout pipe open.
            proc = await asyncio.create_subprocess_shell(
                command,
                executable="/bin/bash",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                await self._edit_shell_reply(
                    ack_message, command,
                    f"⏱️ Command timed out after {TIMEOUT_SECONDS}s.",
                )
                return

            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            output = output.strip()

            # Format the header so output stands out:
            #   🔧 Command:
            #   <command — truncated if huge>
            #
            #   📤 Output:
            # When the command line is very long (openstack env exports are
            # 300+ chars), truncate it to keep the visible header readable;
            # we keep the tail since the meaningful part is usually at the end
            # (the actual subcommand like `openstack server show <id>`).
            COMMAND_PREVIEW = 200
            if len(command) > COMMAND_PREVIEW:
                cmd_preview = "…" + command[-(COMMAND_PREVIEW - 1):]
            else:
                cmd_preview = command

            header = f"🔧 Command:\n`{cmd_preview}`\n\n📤 Output:\n```\n"
            if proc.returncode != 0:
                header = f"🔧 Command:\n`{cmd_preview}`\n\n[exit code: {proc.returncode}]\n\n📤 Output:\n```\n"

            # Output is wrapped in a Markdown code block so Telegram renders it
            # monospace — important for ASCII tables (openstack server list,
            # kubectl get pods, etc.) which look garbled in proportional font.
            # The closing fence adds 4 chars (newline + ```).
            CODE_FENCE_CLOSE = "\n```"
            # Telegram message limit is 4096 chars. Compute the output budget
            # AFTER the header so the full reply always fits — otherwise
            # edit_text silently fails and the ack stays on "Running..." forever.
            TELEGRAM_LIMIT = 4096
            FOOTER_TEMPLATE = "\n\n… (output truncated, full length was {n} chars)"
            footer_reserve = len(FOOTER_TEMPLATE.format(n=10 ** 12))  # worst-case width
            output_budget = max(0, TELEGRAM_LIMIT - len(header) - len(CODE_FENCE_CLOSE) - footer_reserve - 1)

            # Escape any backticks in the output so they don't break the
            # Markdown code block when rendered.
            safe_output = output.replace("```", "\\`\\`\\`")

            if len(safe_output) > output_budget:
                truncated = safe_output[-output_budget:]
                footer = "\n\n… (output truncated, full length was {} chars)".format(len(output))
                reply = header + truncated + CODE_FENCE_CLOSE + footer
            elif safe_output:
                reply = header + safe_output + CODE_FENCE_CLOSE
            else:
                # Empty output: drop the opening fence so we don't leave
                # a stray "```" hanging.
                header_no_fence = header.rstrip("\n`")
                reply = header_no_fence + "(no output)"

            # Guard against Telegram limit one more time — if the header is so
            # long it eats the whole budget, just truncate header side too.
            if len(reply) > TELEGRAM_LIMIT:
                reply = reply[-(TELEGRAM_LIMIT - 1):]

            await self._edit_shell_reply(ack_message, command, reply, parse_mode="Markdown")
        except Exception as e:
            logger.error("/shell failed", command=command, error=str(e))
            await self._edit_shell_reply(
                ack_message, command, f"❌ Failed to execute: {e}"
            )

    async def _edit_shell_reply(self, ack_message, command: str, body: str, parse_mode: str = None):
        """Edit the ack message with the result; fall back to a new reply on failure.

        If parse_mode is set (e.g. "Markdown"), try with it first. On any
        failure, retry as plain text so the user always sees something — a
        garbled unformatted message is better than a stuck "Running...".
        """
        # First attempt: with requested parse_mode (e.g. Markdown for code
        # fences around command output).
        if parse_mode:
            try:
                await ack_message.edit_text(body, parse_mode=parse_mode)
                return
            except Exception:
                pass
        # Second attempt: plain text.
        try:
            await ack_message.edit_text(body)
            return
        except Exception:
            pass
        # Final fallback: reply as a new message (in case edit failed because
        # the ack was deleted or is unrecoverable).
        try:
            await ack_message.reply_text(
                body,
                message_thread_id=ack_message.message_thread_id or None,
            )
        except Exception:
            pass