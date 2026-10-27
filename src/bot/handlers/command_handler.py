"""Command Handler - Handle basic bot commands."""
import asyncio
import shlex
import structlog
import threading
from typing import Optional
from telegram import Update
from telegram.ext import ContextTypes

from src.bot.services.permission_service import PermissionService
from src.bot.services.conversation_service import ConversationService
from src.bot.services.holmes_service import HolmesService
from src.bot.models.user import UserPermission
from src.bot.utils.config import config
from src.bot.utils.decorators import admin_only, rate_limit
from src.bot.utils.kubeconfig_crypto import (
    migrate_kubeconfigs,
    decrypt_to_tempfile,
    cleanup_tempfile,
)
from src.bot.utils.telegram_helpers import send_long_message, collect_vision_images

logger = structlog.get_logger(__name__)


class CommandHandler:
    """Handle basic bot commands."""

    def __init__(
        self,
        permission_service: PermissionService,
        conversation_service: ConversationService = None,
        holmes_service: Optional[HolmesService] = None,
    ):
        self.permissions = permission_service
        self.conversations = conversation_service
        self.holmes = holmes_service

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
            "/ping - Show the current agent status in this topic\n"
            "/sendfile [name] - Send a generated file, or list sendable files\n"
            "/reconcile <shoot_name> - Trigger Gardener reconcile for a shoot\n\n"
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
                "/kshell [kc_name] <command> - Run command with a decrypted kubeconfig (/kshell alone to list)\n"
                "/claude <prompt> - Run a one-shot turn on the enterprise model (reads images)\n"
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
            "/sendfile [name] - Send a generated file, or list the sendable files\n"
            "/reconcile <shoot_name> - Trigger Gardener reconcile for a shoot\n"
        )

        if is_admin:
            help_text += (
                "\n**Admin Commands:**\n"
                "/shell <command> - Execute a shell command in the bot container\n"
                "/kshell [kc_name] <command> - Run command with a decrypted kubeconfig (/kshell alone to list)\n"
                "/claude <prompt> - Run a one-shot turn on the enterprise model (reads images)\n"
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

    @admin_only
    async def kshell_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /kshell — run a shell command with KUBECONFIG decrypted on demand.

        Two modes:
          /kshell                         — list available kubeconfigs (shows the
                                            .enc filenames without the extension).
          /kshell <kc_name> <command...>   — decrypt <kc_name>.enc to a 0600 tempfile,
                                            run `bash -c <command>` with KUBECONFIG
                                            set to the tempfile, then shred the
                                            tempfile. Returned like /shell: immediate
                                            ack, background task edits the ack with
                                            the captured output.

        Admin-only. The kubeconfigs live encrypted in
        /root/.holmes/kubeconfig-only-admin-can-run/ — this is the only command
        (besides /reconcile) that decrypts them, so /shell cannot casually exfil
        via `cat` or `kubectl --kubeconfig <file>.enc`.
        """
        if not update.message:
            return

        import glob
        import os
        topic_id = update.message.message_thread_id or 0
        KCONFIG_DIR = "/root/.holmes/kubeconfig-only-admin-can-run"

        # /kshell alone → list. Also ensure migration has happened, so the
        # listing reflects the encrypted state (not stale plaintext files).
        try:
            migrate_kubeconfigs(KCONFIG_DIR)
        except Exception:
            pass
        if os.path.isdir(KCONFIG_DIR):
            kconfigs = sorted(
                os.path.basename(p)[:-4]
                for p in glob.glob(os.path.join(KCONFIG_DIR, "*.enc"))
                if os.path.isfile(p)
            )
        else:
            kconfigs = []

        if not context.args:
            if not kconfigs:
                await update.message.reply_text(
                    "📂 No kubeconfigs found in "
                    f"`{KCONFIG_DIR}`.\n\n"
                    "Usage: /kshell <kc_name> <command>",
                    message_thread_id=topic_id or None,
                    parse_mode="Markdown",
                )
                return
            listing = "\n".join(f"• `{n}`" for n in kconfigs)
            await update.message.reply_text(
                "📂 **Available kubeconfigs** (pick one "
                "for `/kshell <name> <command>`):\n\n" + listing,
                message_thread_id=topic_id or None,
                parse_mode="Markdown",
            )
            return

        # Parse: first arg = kc_name, rest = command.
        kc_name = context.args[0].strip()
        if len(context.args) == 1:
            # Just the kc_name, no command — show usage.
            await update.message.reply_text(
                f"Usage: /kshell {kc_name} <command>\n"
                f"Example: /kshell {kc_name} kubectl get pods -A",
                message_thread_id=topic_id or None,
            )
            return
        command = " ".join(context.args[1:]).strip()
        if not command:
            await update.message.reply_text(
                "❌ Empty command.",
                message_thread_id=topic_id or None,
            )
            return

        # Resolve the encrypted kubeconfig path. Accept either bare
        # "han-gardener" or "han-gardener.enc".
        kc_stem = kc_name[:-4] if kc_name.endswith(".enc") else kc_name
        enc_path = os.path.join(KCONFIG_DIR, kc_stem + ".enc")
        if not os.path.exists(enc_path):
            await update.message.reply_text(
                f"❌ Kubeconfig `{kc_stem}` not found.\n\n"
                f"Use `/kshell` alone to list available names.",
                message_thread_id=topic_id or None,
                parse_mode="Markdown",
            )
            return

        COMMAND_PREVIEW = 200
        cmd_preview = command if len(command) <= COMMAND_PREVIEW else "…" + command[-(COMMAND_PREVIEW - 1):]
        ack = await update.message.reply_text(
            f"⏳ Running with kubeconfig `{kc_stem}`...\n\n"
            f"🔧 Command:\n`{cmd_preview}`",
            message_thread_id=topic_id or None,
            parse_mode="Markdown",
        )

        asyncio.create_task(
            self._run_kshell_and_edit(ack, enc_path, kc_stem, command, topic_id)
        )

    async def _run_kshell_and_edit(
        self, ack_message, enc_path: str, kc_stem: str, command: str, topic_id: int
    ):
        """Run the kshell command: decrypt kubeconfig, run, edit ack, shred tempfile.

        Inherits /shell's output formatting (Markdown code block, 120s timeout,
        4096-char budget, fallback chain) — but the kubectl command runs in
        a bash subshell with KUBECONFIG=<tempfile> exported.
        """
        import os
        import signal

        MAX_OUTPUT = 3800
        TIMEOUT_SECONDS = 120

        tmp_kc = decrypt_to_tempfile(enc_path)
        if tmp_kc is None:
            await self._edit_shell_reply(
                ack_message, command, f"❌ Failed to decrypt kubeconfig `{kc_stem}`.",
                parse_mode="Markdown",
            )
            return

        try:
            # Wrap the user's command in `export KUBECONFIG=...; <cmd>` so
            # kubectl (and any subshells/pipes) see the decrypted kubeconfig.
            full_cmd = (
                f"export KUBECONFIG={shlex.quote(tmp_kc)}; "
                f"{command}"
            )

            proc = await asyncio.create_subprocess_shell(
                full_cmd,
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

            COMMAND_PREVIEW = 200
            if len(command) > COMMAND_PREVIEW:
                cmd_preview = "…" + command[-(COMMAND_PREVIEW - 1):]
            else:
                cmd_preview = command

            header = (
                f"🔧 Kubeconfig: `{kc_stem}`\n"
                f"🔧 Command:\n`{cmd_preview}`\n\n"
                f"📤 Output:\n```\n"
            )
            if proc.returncode != 0:
                header = (
                    f"🔧 Kubeconfig: `{kc_stem}`\n"
                    f"🔧 Command:\n`{cmd_preview}`\n\n"
                    f"[exit code: {proc.returncode}]\n\n"
                    f"📤 Output:\n```\n"
                )

            CODE_FENCE_CLOSE = "\n```"
            TELEGRAM_LIMIT = 4096
            FOOTER_TEMPLATE = "\n\n… (output truncated, full length was {n} chars)"
            footer_reserve = len(FOOTER_TEMPLATE.format(n=10 ** 12))
            output_budget = max(0, TELEGRAM_LIMIT - len(header) - len(CODE_FENCE_CLOSE) - footer_reserve - 1)

            safe_output = output.replace("```", "\\`\\`\\`")

            if len(safe_output) > output_budget:
                truncated = safe_output[-output_budget:]
                footer = "\n\n… (output truncated, full length was {} chars)".format(len(output))
                reply = header + truncated + CODE_FENCE_CLOSE + footer
            elif safe_output:
                reply = header + safe_output + CODE_FENCE_CLOSE
            else:
                header_no_fence = header.rstrip("\n`")
                reply = header_no_fence + "(no output)"

            if len(reply) > TELEGRAM_LIMIT:
                reply = reply[-(TELEGRAM_LIMIT - 1):]

            await self._edit_shell_reply(ack_message, command, reply, parse_mode="Markdown")
        except Exception as e:
            logger.error("/kshell failed", command=command, error=str(e))
            await self._edit_shell_reply(
                ack_message, command, f"❌ Failed to execute: {e}"
            )
        finally:
            cleanup_tempfile(tmp_kc)

    @rate_limit
    async def reconcile_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reconcile <shoot_name> — annotate the Gardener shoot for reconcile.

        Works like /shell: replies immediately with an ack, then runs
        in the background and edits the ack with the result.

        Flow:
          1. chmod -R 777 /root/.holmes/kubeconfig-only-admin-can-run
          2. Traverse every kubeconfig in that dir; for each, run
             `kubectl --context <ctx> annotate shoot <name> -n <ns>
              gardener.cloud/operation=reconcile --overwrite`
          3. Show which shoots succeeded (with kubeconfig + namespace)
          4. If none matched, reply that no shoot matched <name>
          5. chmod -R 000 /root/.holmes/kubeconfig-only-admin-can-run
        """
        if not update.message:
            return

        topic_id = update.message.message_thread_id or 0

        if not context.args:
            await update.message.reply_text(
                "Usage: /reconcile <shoot_name>",
                message_thread_id=topic_id or None,
            )
            return

        shoot_name = context.args[0].strip()
        if not shoot_name:
            await update.message.reply_text(
                "❌ Empty shoot name.",
                message_thread_id=topic_id or None,
            )
            return

        ack = await update.message.reply_text(
            f"⏳ Reconciling shoot `{shoot_name}`...",
            message_thread_id=topic_id or None,
            parse_mode="Markdown",
        )

        asyncio.create_task(
            self._run_reconcile_and_edit(ack, shoot_name, topic_id)
        )

    async def _run_reconcile_and_edit(self, ack_message, shoot_name: str, topic_id: int):
        """Run the reconcile flow and edit the ack with the result.

        For each kubeconfig in /root/.holmes/kubeconfig-only-admin-can-run/:
          1. Use its context (typically `garden`) to look up the shoot's
             namespace via `kubectl get shoot <name> -A -o jsonpath=...`.
          2. If found, annotate the shoot in that namespace with
             `gardener.cloud/operation=reconcile --overwrite`.

        Kubeconfigs are stored as `.enc` files (encrypted at rest with the
        hardcoded passphrase in kubeconfig_crypto); each one is decrypted to
        a fresh /tmp tempfile for the duration of the kubectl calls and
        shredded in finally.

        The namespace isn't derivable from the context (Gardener seed
        kubeconfigs have a single `garden` context), so the lookup step is
        required. Returns a summary: ✅ list (kubeconfig + context + namespace
        + annotate output), or "no shoot matched <name>" if no kubeconfig
        found the shoot.
        """
        import glob
        import os
        import signal

        KCONFIG_DIR = "/root/.holmes/kubeconfig-only-admin-can-run"
        TIMEOUT_SECONDS = 120
        MAX_OUTPUT = 2000

        async def run_subprocess(cmd: str) -> tuple[int, str]:
            """Run cmd via bash, return (returncode, combined stdout+stderr)."""
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
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
                        pass
                    return -1, f"⏱️ Timed out after {TIMEOUT_SECONDS}s."
                out = stdout.decode("utf-8", errors="replace") if stdout else ""
                return proc.returncode, out
            except Exception as e:
                return -2, str(e)

        def truncate(text: str, limit: int) -> str:
            if len(text) <= limit:
                return text
            return text[-limit:] + f"\n\n… (output truncated, full length was {len(text)} chars)"

        results: list[str] = []   # collected ✅/❌ lines
        any_lookup_attempt: bool = False  # did we try at least one kubeconfig?
        lookup_errors: list[str] = []  # kubeconfig-level lookup errors, surfaced
                                        # when nothing matched so the user can diagnose
                                        # (e.g. chmod 777 didn't take, cert expired,
                                        # cluster unreachable).

        async def process_kubeconfig(kc: str) -> tuple[bool, list[str]]:
            """Lookup + annotate one kubeconfig. Returns (did_attempt, result_lines).

            Decrypts `kc` (a `.enc` file) to a 0600 tempfile, runs kubectl
            against the tempfile, then shreds the tempfile in finally. This
            is the on-demand decrypt path that keeps kubeconfigs encrypted
            at rest between invocations.

            Runs get-contexts → for each context, `kubectl get shoot -A` to find
            the namespace, then `kubectl annotate shoot ... operation=reconcile`.
            `did_attempt` is True if at least one lookup was issued (even if the
            shoot wasn't found here) — used to distinguish "no match anywhere"
            from "no kubeconfigs at all".

            If lookup fails (non-zero rc AND empty namespaces), records the error
            output in lookup_errors so we can surface it in the "no match" reply
            — otherwise the user gets a misleading "No shoot matched" when the
            real problem is chmod failure, permission denied, or unreachable API.
            """
            kc_name = os.path.basename(kc)
            if kc_name.endswith(".enc"):
                kc_name = kc_name[:-4]
            tmp_kc = decrypt_to_tempfile(kc)
            if tmp_kc is None:
                lookup_errors.append(f"{kc_name}: gpg decrypt failed")
                return False, []
            kc_q = shlex.quote(tmp_kc)
            lines: list[str] = []
            did_attempt = False
            try:
                # Get contexts — if empty, fall back to no --context so kubectl
                # uses the default context (typically `garden` for Gardener kubeconfigs).
                # get-contexts also implicitly reads the file, so a permission error
                # on the kubeconfig (mode 000 after chmod failed) surfaces here.
                rc, ctx_out = await run_subprocess(
                    f"KUBECONFIG={kc_q} kubectl config get-contexts -o name"
                )
                if rc != 0 and ctx_out.strip():
                    # get-contexts failed with an error message — record and skip
                    # this kubeconfig. Don't fall through to the empty-contexts
                    # branch below; that would try to reach an unreachable API
                    # with a kubeconfig we already know is broken.
                    lookup_errors.append(f"{kc_name}: {truncate(ctx_out.strip(), 400)}")
                    return did_attempt, lines
                contexts = [c.strip() for c in ctx_out.splitlines() if c.strip()] if rc == 0 else []
                if not contexts:
                    contexts = [None]

                for ctx in contexts:
                    ctx_arg = f"--context {shlex.quote(ctx)}" if ctx else ""

                    # Look up the shoot's namespace. `kubectl get shoot <name> -A`
                    # is invalid (a named resource can't be retrieved across all
                    # namespaces), so fetch all shoots with `-A` and filter client-
                    # side by name. jsonpath emits `namespace name` per line so we
                    # can both match by name and read the namespace.
                    did_attempt = True
                    lookup_cmd = (
                        f"KUBECONFIG={kc_q} kubectl {ctx_arg} "
                        f"get shoot -A "
                        f"-o jsonpath='{{range .items[*]}}{{.metadata.namespace}}{{\" \"}}{{.metadata.name}}{{\"\\n\"}}{{end}}'"
                    )
                    rc, ns_out = await run_subprocess(lookup_cmd)

                    namespaces: list[str] = []
                    if rc == 0:
                        for line in ns_out.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(" ", 1)
                            if len(parts) == 2 and parts[1] == shoot_name:
                                namespaces.append(parts[0])

                    if not namespaces:
                        # Shoot not found in this kubeconfig's garden. If rc != 0,
                        # the lookup itself failed (not just "no match") — record
                        # the error output so we can surface it later. If rc==0
                        # and no matches, the shoot genuinely doesn't exist here —
                        # skip silently.
                        if rc != 0 and ns_out.strip():
                            err_ctx = f" (context: `{ctx}`)" if ctx else ""
                            lookup_errors.append(
                                f"{kc_name}{err_ctx}: {truncate(ns_out.strip(), 400)}"
                            )
                        continue

                    for ns in namespaces:
                        annotate_cmd = (
                            f"KUBECONFIG={kc_q} kubectl {ctx_arg} -n {shlex.quote(ns)} "
                            f"annotate shoot {shlex.quote(shoot_name)} "
                            f"gardener.cloud/operation=reconcile --overwrite"
                        )
                        rc, out = await run_subprocess(annotate_cmd)
                        out = out.strip()

                        header = f"✅ {kc_name}"
                        if ctx:
                            header += f" (context: `{ctx}`)"
                        header += f" — namespace `{ns}`"

                        # Gardener annotate success prints
                        # "shoot.core.gardener.cloud/<name> annotated" or "patched".
                        # Treat rc==0 with a not-found message as a failure too.
                        if rc == 0 and ("annotated" in out or "patched" in out or not out):
                            line = header
                            if out:
                                line += f"\n```\n{truncate(out, MAX_OUTPUT)}\n```"
                            lines.append(line)
                        else:
                            err_line = header.replace("✅", "❌", 1)
                            if out:
                                err_line += f"\n```\n{truncate(out, MAX_OUTPUT)}\n```"
                            lines.append(err_line)

                return did_attempt, lines
            finally:
                cleanup_tempfile(tmp_kc)

        try:
            # Step 1: chmod 777 so we can read the kubeconfigs. The dir starts
            # at mode 000 by design (locked down between invocations) — if chmod
            # fails, every kubectl call below will fail with permission denied
            # and the user only sees "No shoot matched", which is misleading.
            # Surface the chmod failure explicitly instead.
            rc, chmod_out = await run_subprocess(f"chmod -R 777 {KCONFIG_DIR}")
            if rc != 0:
                await self._edit_shell_reply(
                    ack_message, f"reconcile {shoot_name}",
                    f"❌ `chmod -R 777 {KCONFIG_DIR}` failed (exit {rc}):\n"
                    f"```\n{truncate(chmod_out.strip(), 1000)}\n```",
                    parse_mode="Markdown",
                )
                return

            # Step 2: encrypt any plaintext kubeconfigs at rest. Idempotent —
            # a no-op on subsequent runs because the glob only picks up .enc
            # files. If this fails (crypto error, bad permissions), we proceed
            # anyway: existing .enc files are picked up below, plaintext ones
            # are silently dropped (operator notices via "No shoot matched").
            try:
                migrate_kubeconfigs(KCONFIG_DIR)
            except Exception as migrate_err:
                logger.warning("kubeconfig migration failed", error=str(migrate_err))

            kconfigs = sorted(
                p for p in glob.glob(os.path.join(KCONFIG_DIR, "*.enc"), recursive=False)
                if os.path.isfile(p)
            )

            # Run all kubeconfigs in parallel — each is independent (own KUBECONFIG,
            # own kubectl process). This turns an O(N×120s) sequential wait into
            # O(120s) when kubectl calls are the bottleneck.
            per_kc = await asyncio.gather(
                *(process_kubeconfig(kc) for kc in kconfigs),
                return_exceptions=True,
            )

            for outcome in per_kc:
                if isinstance(outcome, Exception):
                    # One kubeconfig's processing blew up — record as a failure
                    # line rather than killing the whole reconcile.
                    results.append(f"❌ (exception) `{outcome}`")
                    continue
                did_attempt, lines = outcome
                if did_attempt:
                    any_lookup_attempt = True
                results.extend(lines)

        finally:
            # Always lock down again
            await run_subprocess(f"chmod -R 000 {KCONFIG_DIR}")

        # Build the summary
        successes = [r for r in results if r.startswith("✅")]
        failures = [r for r in results if r.startswith("❌")]

        if successes:
            reply_parts = [f"🎯 Reconciled shoot `{shoot_name}`:\n\n"]
            reply_parts.extend(successes)
            if failures:
                reply_parts.append("\n\n⚠️ Failures:")
                reply_parts.extend(failures)
            reply = "\n".join(reply_parts)
        elif any_lookup_attempt:
            # No matches AND there were lookup errors — surface the errors so
            # the user can diagnose (chmod didn't take, certs expired, cluster
            # unreachable, etc.) instead of the opaque "No shoot matched" line.
            if lookup_errors:
                # Limit to first 5 kubeconfig errors to stay within Telegram's
                # 4096-char cap; each line is truncated to ~400 chars above anyway.
                sample = lookup_errors[:5]
                extra = f"\n\n…and {len(lookup_errors) - 5} more" if len(lookup_errors) > 5 else ""
                reply = (
                    f"❌ No shoot matched `{shoot_name}` in any kubeconfig.\n\n"
                    f"🔍 Lookup errors (showing {len(sample)} of {len(lookup_errors)}):"
                    f"{extra}\n\n"
                    + "\n\n".join(f"• {e}" for e in sample)
                )
            else:
                reply = f"❌ No shoot matched `{shoot_name}` in any kubeconfig."
        else:
            reply = f"❌ No kubeconfigs found under `{KCONFIG_DIR}`."

        # Keep within Telegram limit
        TELEGRAM_LIMIT = 4096
        if len(reply) > TELEGRAM_LIMIT:
            reply = reply[-(TELEGRAM_LIMIT - 1):]

        # Reuse the shell reply editor (Markdown so the code blocks render)
        await self._edit_shell_reply(ack_message, f"reconcile {shoot_name}", reply, parse_mode="Markdown")

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

    @staticmethod
    def _claude_prompt(message, context) -> str:
        """Prompt text for /claude, from message.text or the file caption.

        An image attached with `/claude` as its caption arrives as
        `message.caption` with `message.text=None` (see handle_claude_upload),
        so the command args are empty there and the caption is the only source.
        """
        args = " ".join(context.args).strip() if getattr(context, "args", None) else ""
        if args:
            return args
        caption = (getattr(message, "caption", None) or "").strip()
        if not caption:
            return ""
        # Drop the leading "/claude" or "/claude@BotName" token.
        parts = caption.split(maxsplit=1)
        if parts and parts[0].lower().startswith("/claude"):
            return parts[1].strip() if len(parts) > 1 else ""
        return caption

    @staticmethod
    def _claude_has_image(message) -> bool:
        """True when the message (or the one it replies to) carries an image."""
        from src.bot.utils.telegram_helpers import _image_target_from_message
        return (
            _image_target_from_message(message) is not None
            or _image_target_from_message(getattr(message, "reply_to_message", None)) is not None
        )

    @admin_only
    @rate_limit
    async def claude_upload_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Entry point for an image whose caption starts with `/claude`.

        PTB's CommandHandler only matches `message.text`, and Telegram sets
        `text=None` / `caption="/claude ..."` for a photo or image document.
        Without this handler such an upload would fall through to
        MessageHandler.handle_file and be answered by the default model, which
        is not what /claude asks for. Mirrors AddMemoryCommand.handle_upload.
        """
        await self.claude_command(update, context)

    @admin_only
    @rate_limit
    async def claude_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /claude <prompt> — route a one-shot turn through the enterprise model.

        Admin-only. The model key is configured via CLAUDE_ENTERPRISE_MODEL (a
        key in ~/.holmes/model_list.yaml); Holmes' LLMModelRegistry resolves
        it to api_base + api_key, so no credentials are duplicated here. The
        prompt and reply are appended to the same topic conversation as normal
        turns, so follow-up plain messages on the default model still see
        this turn in history.

        Images are read natively by the enterprise model: attach a screenshot
        with `/claude` as its caption, or reply to an image with `/claude ...`.
        The bytes go straight into the message as image_url blocks — no OCR.
        Set ENTERPRISE_MODEL_SUPPORTS_IMAGES=false for a text-only endpoint.
        """
        if not update.message:
            return

        chat = update.effective_chat
        topic_id = update.message.message_thread_id or 0

        if chat.type == "private":
            await update.message.reply_text(
                "🚫 Private chats are disabled. Please use the bot in a group forum topic.",
                message_thread_id=None,
            )
            return

        if not (chat.type in ("group", "supergroup") and chat.is_forum and topic_id):
            await update.message.reply_text(
                "ℹ️ /claude only works inside a forum topic.",
                message_thread_id=topic_id or None,
            )
            return

        if not context.args and not self._claude_has_image(update.message):
            await update.message.reply_text(
                "Usage: /claude <prompt>\n"
                "Example: /claude summarize the latest pod logs\n"
                "Images: attach a screenshot with the caption `/claude <prompt>`, "
                "or reply to an image with `/claude <prompt>`.",
                message_thread_id=topic_id or None,
            )
            return

        prompt = self._claude_prompt(update.message, context)
        if not prompt:
            if self._claude_has_image(update.message):
                # Image-only turn (`/claude` as the caption, no text): the model
                # sees the image itself, so a neutral instruction is enough.
                prompt = "(_image uploaded — please inspect it)"
            else:
                await update.message.reply_text(
                    "❌ Empty prompt.",
                    message_thread_id=topic_id or None,
                )
                return

        if self.holmes is None:
            await update.message.reply_text(
                "❌ Holmes service is not wired up. /claude is unavailable.",
                message_thread_id=topic_id or None,
            )
            return

        if not self.holmes.enterprise_model_configured():
            await update.message.reply_text(
                "❌ Enterprise model not configured. Set CLAUDE_ENTERPRISE_MODEL "
                "to a key from ~/.holmes/model_list.yaml.",
                message_thread_id=topic_id or None,
            )
            return

        user = update.effective_user
        chat_id = chat.id

        user_model = await self.permissions.get_or_create_user(
            telegram_id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
        )

        if self.conversations is not None:
            conv_service = self.conversations
        else:
            conv_service = ConversationService()
        conversation = await conv_service.get_conversation(
            user.id, topic_id=topic_id, chat_id=chat_id
        )

        # Lazy-import to avoid a circular import at module load.
        from src.bot.handlers.message_handler import _set_investigation_status, _clear_investigation

        try:
            from holmes.core.tool_calling_llm import LLMInterruptedError as _LLMInterrupted
        except Exception:
            _LLMInterrupted = ()

        ack = await update.message.reply_text(
            f"🧭 Routing to enterprise model...\n\n📝 Prompt:\n`{prompt[:200]}`",
            message_thread_id=topic_id or None,
            parse_mode="Markdown",
        )

        # Native image reading: attach the screenshot's bytes to the message so
        # the enterprise model sees it as an image_url block (no OCR). The
        # `images` kwarg on chat_enterprise persists them in message metadata,
        # so history replay rehydrates the same multimodal content array.
        images = None
        if getattr(config, "enterprise_model_supports_images", True):
            try:
                images = await collect_vision_images(context.bot, update.message)
            except Exception as e:
                logger.warning("/claude image attach failed", user_id=user.id, error=str(e))
                images = None
            if images:
                try:
                    await ack.edit_text(
                        f"🧭 Routing to enterprise model with {len(images)} image(s)...\n\n"
                        f"📝 Prompt:\n`{prompt[:200]}`",
                        parse_mode="Markdown",
                    )
                except Exception:
                    pass

        # Streaming via enterprise LLM, but deliver only final text (matches
        # /ping-capable regular-response flow).
        _set_investigation_status(chat_id, topic_id, "🧭 Enterprise model...")
        cancel_event = threading.Event()
        response = ""
        try:
            async_gen = await self.holmes.chat_enterprise(
                user_id=user.id,
                user_message=prompt,
                conversation=conversation,
                user_permissions=user_model.permissions,
                stream=True,
                telegram_context=context,
                chat_id=chat_id,
                topic_id=topic_id,
                cancel_event=cancel_event,
                images=images,
            )
            async for event in async_gen:
                event_type = event.get("type", "unknown")
                if event_type == "content":
                    response += event.get("content", "")
                    _set_investigation_status(
                        chat_id, topic_id, response_preview=response[-300:],
                    )
                elif event_type == "task_update":
                    _set_investigation_status(
                        chat_id, topic_id, None, tasks=event.get("tasks", []),
                    )
                elif event_type == "tool_call":
                    tool_name = event.get("tool_name", "unknown")
                    _set_investigation_status(
                        chat_id, topic_id,
                        status_text=f"🔧 Using {tool_name}...",
                        last_tool=tool_name,
                        last_activity=f"🔧 {tool_name}",
                    )
                elif event_type == "tool_result":
                    tool_name = event.get("tool_name", "unknown")
                    _set_investigation_status(
                        chat_id, topic_id,
                        status_text=f"✅ Completed {tool_name}",
                        last_activity=f"✅ {tool_name}",
                    )
        except _LLMInterrupted:
            if cancel_event is not None:
                cancel_event.set()
        except Exception as e:
            logger.error("/claude failed", user_id=user.id, error=str(e))
            _clear_investigation(chat_id, topic_id)
            await ack.edit_text(
                f"❌ Enterprise model call failed: {e}",
                parse_mode=None,
            )
            return

        if cancel_event.is_set() and "🛑 Stopped by user." not in response:
            response = (response or "") + "\n\n🛑 Stopped by user."

        _clear_investigation(chat_id, topic_id)

        await conv_service.save_conversation(conversation)

        if not response.strip():
            response = "(no output from enterprise model)"

        try:
            await ack.delete()
        except Exception:
            pass

        try:
            await send_long_message(
                context.bot,
                chat_id,
                response,
                message_thread_id=topic_id or None,
            )
        except Exception as e:
            logger.warning("Failed to deliver /claude response", user_id=user.id, error=str(e))
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="⚠️ The enterprise response was generated but couldn't be delivered. Please try again.",
                    message_thread_id=topic_id or None,
                )
            except Exception:
                pass