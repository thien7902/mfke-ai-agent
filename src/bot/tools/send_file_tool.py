"""Holmes tool: deliver a file the assistant produced to the current chat.

The bot could already read files the user uploads, but had no way to send one
back — so "generate me an xlsx of the shoots" ended with the content pasted
into the chat. This tool closes that loop: Holmes writes the deliverable into
the outbox directory (bash / python / write_file all work there) and then calls
`chat_send_file` with its path, and the bot uploads it as a Telegram document
in the topic the conversation is happening in.

Chat routing comes from `ToolInvokeContext.request_context`, which the bot
fills in per turn (`chat_id`, `topic_id`). A tool call with no such context
(the CLI path, or a unit test) replies with an error instead of sending to an
arbitrary chat.

Send policy (outbox containment, credential-name refusal, size cap) lives in
`src.bot.utils.file_sender` so `/sendfile` shares it.
"""
from typing import Dict, Optional

import structlog
from holmes.core.tools import (
    StructuredToolResult,
    StructuredToolResultStatus,
    Tool,
    ToolInvokeContext,
    ToolParameter,
)

from src.bot.utils import file_sender

logger = structlog.get_logger(__name__)


def _as_int(value) -> int:
    """Coerce a request_context value to int; 0 when absent/unparseable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class SendFileTool(Tool):
    """Upload a file from the bot's outbox to the Telegram conversation."""

    name: str = "chat_send_file"
    description: str = (
        "Send a file to the user as a Telegram document attachment. "
        "Use this whenever the user asks for a file (txt, csv, xlsx, json, yaml, "
        "pdf, zip, a script, a report ...) instead of pasting its content in chat. "
        "Write the file into the sendable directory first (its absolute path is "
        "shown by list_chat_files, or use the CHAT FILE OUTPUT rule in the system "
        "prompt), then call this tool with that path. A relative path is resolved "
        "against the sendable directory. Returns confirmation with the file size."
    )
    parameters: Dict[str, ToolParameter] = {
        "path": ToolParameter(
            description=(
                "Absolute path of the file to send, or a path relative to the "
                "sendable outbox directory. The file must already exist and be "
                "under the outbox."
            ),
            type="string",
            required=True,
        ),
        "caption": ToolParameter(
            description="Optional short caption shown under the file (max 1024 chars).",
            type="string",
            required=False,
        ),
    }
    user_description: Optional[str] = "Sent file: {{path}}"

    def get_parameterized_one_liner(self, params: Dict) -> str:
        return f"chat_send_file(path={(params or {}).get('path', '')!r})"

    def _invoke(self, params: dict, context: ToolInvokeContext) -> StructuredToolResult:
        raw_path = str((params or {}).get("path") or "").strip()
        caption = (params or {}).get("caption")
        if isinstance(caption, str):
            caption = caption.strip() or None
        else:
            caption = None

        request_context = getattr(context, "request_context", None) or {}
        chat_id = _as_int(request_context.get("chat_id"))
        topic_id = _as_int(request_context.get("topic_id"))

        if not chat_id:
            return self._error(
                params,
                "No chat is associated with this tool call, so there is nowhere to "
                "send the file. Files can only be sent during a Telegram conversation.",
            )

        if not raw_path:
            return self._error(params, "Missing 'path'. Provide the file to send.")

        # Prune before resolving, not after — see the same note in
        # src/bot/commands/send_file.py: pruning last could delete the file we
        # just validated and turn a clear message into a Telegram upload error.
        file_sender.prune_outbox()
        try:
            resolved = file_sender.resolve_sendable_path(raw_path)
        except file_sender.SendPolicyError as e:
            return self._error(params, str(e))
        except Exception as e:  # noqa: BLE001 - path handling must never crash a turn
            logger.warning("send_file_resolve_failed", error=str(e), path=raw_path)
            return self._error(params, f"Could not prepare '{raw_path}': {e}")

        try:
            file_sender.send_document_sync(
                chat_id, str(resolved),
                topic_id=topic_id,
                caption=caption,
            )
        except file_sender.SenderNotConfiguredError as e:
            logger.warning("send_file_not_configured", error=str(e))
            return self._error(params, str(e))
        except TimeoutError as e:
            logger.warning("send_file_timeout", path=str(resolved), error=str(e))
            return self._error(params, str(e))
        except Exception as e:  # noqa: BLE001 - Telegram errors are many and varied
            logger.error(
                "send_file_failed",
                path=str(resolved), chat_id=chat_id, topic_id=topic_id, error=str(e),
            )
            return self._error(params, f"Telegram rejected the upload: {e}")

        size = resolved.stat().st_size
        logger.info(
            "send_file_delivered",
            path=str(resolved), size=size, chat_id=chat_id, topic_id=topic_id,
        )
        return StructuredToolResult(
            status=StructuredToolResultStatus.SUCCESS,
            data=(
                f"Sent '{resolved.name}' ({file_sender.format_size(size)}) to the chat."
            ),
            params=params,
            invocation=self.get_parameterized_one_liner(params),
        )

    def _error(self, params: dict, message: str) -> StructuredToolResult:
        return StructuredToolResult(
            status=StructuredToolResultStatus.ERROR,
            error=message,
            params=params,
            invocation=self.get_parameterized_one_liner(params),
        )


class ListChatFilesTool(Tool):
    """List files in the outbox — gives the model the directory it may send from."""

    name: str = "list_chat_files"
    description: str = (
        "List files currently in the sendable outbox directory (newest first), "
        "with their sizes and full paths. Use it to discover the outbox path or to "
        "check what is available before calling chat_send_file. Max 50 entries."
    )
    parameters: Dict[str, ToolParameter] = {}
    user_description: Optional[str] = "Listed sendable files"

    def get_parameterized_one_liner(self, params: Dict) -> str:
        return "list_chat_files()"

    def _invoke(self, params: dict, context: ToolInvokeContext) -> StructuredToolResult:
        outbox = file_sender.outbox_dir()
        entries = file_sender.list_outbox()[:50]
        if not entries:
            data = (
                f"The sendable directory is empty: {outbox}\n"
                "Write your deliverable there, then call chat_send_file with its path."
            )
        else:
            lines = [f"Sendable directory: {outbox}"]
            for path, size, _mtime in entries:
                lines.append(f"- {path} ({file_sender.format_size(size)})")
            data = "\n".join(lines)
        return StructuredToolResult(
            status=StructuredToolResultStatus.SUCCESS,
            data=data,
            params=params or {},
            invocation="list_chat_files()",
        )
