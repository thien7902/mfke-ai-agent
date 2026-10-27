"""`/sendfile` — send a file the bot generated back into the chat.

The assistant normally delivers files on its own via the `chat_send_file`
Holmes tool. This command is the manual path: it is what you reach for when
the model produced a file without sending it, or when you want to re-send a
previously generated one.

Usage:
- `/sendfile`                 → list what is currently in the outbox
- `/sendfile <path>`          → send that file (relative paths resolve against the outbox)
- reply to a message with `/sendfile <path>`

The same policy as the tool applies (see src/bot/utils/file_sender.py): only
files inside the outbox, never credential-shaped names, never over the size cap.
"""
import structlog
from telegram import Update
from telegram.ext import ContextTypes

from src.bot.utils import file_sender
from src.bot.utils.decorators import rate_limit

logger = structlog.get_logger(__name__)


class SendFileCommand:
    """Handler for `/sendfile`."""

    @rate_limit
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
                "ℹ️ /sendfile only works inside a forum topic.",
                message_thread_id=topic_id or None,
            )
            return

        raw_path = " ".join(context.args).strip() if context.args else ""
        if not raw_path:
            await self._list_outbox(update, topic_id)
            return

        # Prune before resolving: pruning after the check could delete the very
        # file that was just validated (it may be older than the TTL), turning a
        # clear "file not found" into a confusing Telegram upload error.
        file_sender.prune_outbox()
        try:
            resolved = file_sender.resolve_sendable_path(raw_path)
        except file_sender.SendPolicyError as e:
            await update.message.reply_text(
                f"❌ {e}",
                message_thread_id=topic_id or None,
            )
            return

        try:
            await file_sender.send_document_async(
                context.bot, chat.id, str(resolved), topic_id=topic_id,
            )
        except Exception as e:  # noqa: BLE001 - surface any Telegram error to the user
            logger.error(
                "sendfile_command_failed",
                path=str(resolved), chat_id=chat.id, topic_id=topic_id, error=str(e),
            )
            await update.message.reply_text(
                f"❌ Couldn't send {resolved.name}: {e}",
                message_thread_id=topic_id or None,
            )

    async def _list_outbox(self, update: Update, topic_id: int) -> None:
        # Resolve through file_sender so there is a single source of truth for
        # both the directory and the test-time patchability (see its docstring).
        outbox = file_sender.outbox_dir()
        entries = file_sender.list_outbox()[:50]
        # Plain text, no parse_mode: filenames routinely contain Markdown
        # metacharacters (report_[1].txt, a_b.csv) that would break the markup
        # and make Telegram reject the whole message.
        if not entries:
            await update.message.reply_text(
                f"📂 The sendable directory is empty: {outbox}\n\n"
                "Ask the assistant for a file (e.g. \"make me a csv of the pods\") "
                "and it will be created there, or write one yourself with /shell.",
                message_thread_id=topic_id or None,
            )
            return

        lines = [f"📂 Files in {outbox}:", ""]
        for path, size, _mtime in entries:
            lines.append(f"• {path.name} — {file_sender.format_size(size)}")
        lines.append("")
        lines.append("Send one with: /sendfile <name>")
        await update.message.reply_text(
            "\n".join(lines),
            message_thread_id=topic_id or None,
        )
