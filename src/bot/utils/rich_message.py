"""Rich Messages & native table support for the Telegram bot.

Telegram Bot API 10.1 (June 2026) added `sendRichMessage` accepting GFM
markdown (with tables) and HTML directly. python-telegram-bot v21.4 does not
wrap this endpoint, so we use `Bot._post` — the same private hook PTB uses
internally for every Bot API method. When PTB releases a wrapper, only this
file needs to change.

Two delivery paths:

1. **Rich path** (only when `has_rich_table(text)` is True or `prefer_rich=True`):
   call `sendRichMessage` with `rich_message={"markdown": text}`. If the
   server rejects the endpoint (self-hosted Bot API lagging behind), fall
   through to the HTML path.

2. **HTML path** (default): convert markdown to Telegram's subset HTML and
   send via `sendMessage` with `parse_mode="HTML"`. Tables degrade to a
   `<pre>`-formatted monospace grid with column separators.

Module API:
- `send_rich_message(bot, chat_id, text, ...)` — raw `sendRichMessage` with
  fallback to `sendMessage` HTML.
- `send_rich_reply(bot, chat_id, text, ...)` — auto-route: rich if table
  present, else HTML. Pass `prefer_rich=True/False` to force.
- `has_rich_table(text)` — GFM table detector (header row + separator).
- `markdown_to_html(text)` — markdown → Telegram subset HTML converter.
"""
import logging
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


TELEGRAM_MESSAGE_LIMIT = 4096


# ---------------------------------------------------------------------------
# GFM table detection
# ---------------------------------------------------------------------------

# A GFM table requires:
# - A header row with at least one `|`
# - A separator row of cells where each cell is `---`, `:--`, `--:`, `:--:` etc.
_SEPARATOR_CELL = re.compile(r"^\s*:?-{2,}:?\s*$")


def has_rich_table(text: str) -> bool:
    """True if `text` contains a GFM table.

    A GFM table is: header row with `|`, separator row matching the column
    shape with `---` cells, then optionally body rows. We check the first two
    lines.
    """
    if not text:
        return False
    lines = text.splitlines()
    for i, line in enumerate(lines[:-1]):
        if "|" not in line:
            continue
        if _is_separator_row(lines[i + 1]):
            header_cells = [c.strip() for c in line.split("|")]
            header_cells = [c for c in header_cells if c]
            if header_cells:
                return True
    return False


# ---------------------------------------------------------------------------
# Markdown → Telegram subset HTML
# ---------------------------------------------------------------------------

def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_FENCED_CODE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_ITALIC = re.compile(r"(?<!\w)[\*_]([^*\n_]+)[\*_](?!\w)")
_STRIKE = re.compile(r"~~([^~\n]+)~~")
_SPOILER = re.compile(r"\|\|([^|\n]+)\|\|")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _convert_table_to_pre_html(table_text: str) -> str:
    """Render a GFM table block as a `<pre>` monospace aligned grid."""
    lines = table_text.strip().splitlines()
    rows: List[List[str]] = []
    for line in lines:
        if not line.strip():
            continue
        cells = [c.strip() for c in line.split("|")]
        # Drop the empty edges when the row uses leading/trailing pipes.
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]
        rows.append(cells)

    if not rows or len(rows) < 2:
        return _escape_html(table_text)

    # Drop separator row(s) — they're index 1 typically.
    body_rows = []
    for i, row in enumerate(rows):
        if i == 1 and all(_SEPARATOR_CELL.match(c) for c in row if c):
            continue
        body_rows.append(row)

    if not body_rows:
        return _escape_html(table_text)

    # Detect ragged rows (varying column counts) → bail to plain escaped <pre>.
    width = len(body_rows[0])
    if any(len(r) != width for r in body_rows):
        return f"<pre>{_escape_html(table_text.strip())}</pre>"

    col_w = [0] * width
    for r in body_rows:
        for i, c in enumerate(r):
            col_w[i] = max(col_w[i], len(c))

    out_lines: List[str] = []
    for i, r in enumerate(body_rows):
        padded = " | ".join(c.ljust(col_w[i]) for i, c in enumerate(r))
        out_lines.append(padded)
        if i == 0:
            # Dashed separator under header.
            sep = "-+-".join("-" * w for w in col_w)
            out_lines.append(sep)

    return f"<pre>{_escape_html(chr(10).join(out_lines))}</pre>"


def _is_separator_row(line: str) -> bool:
    """True if `line` is a GFM table separator row (`---`, `:--`, `--:`, `:--:`)."""
    if "|" not in line:
        return False
    cells = [c.strip() for c in line.split("|")]
    # Drop empty edges (table without leading/trailing pipe).
    cells = [c for c in cells if c]
    if not cells:
        return False
    return all(_SEPARATOR_CELL.match(c) for c in cells)


def _extract_table_blocks(text: str) -> List[tuple]:
    """Split `text` into (is_table, chunk) tuples at GFM table boundaries.

    A table block is the header row + separator + body rows. Body rows are
    consecutive lines containing `|`. The block ends at the first blank line
    or line without `|`.
    """
    chunks: List[tuple] = []
    lines = text.splitlines(keepends=True)
    i = 0
    buf: List[str] = []

    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines) and "|" in line and _is_separator_row(lines[i + 1]):
            if buf:
                chunks.append((False, "".join(buf)))
                buf = []
            block = [line, lines[i + 1]]
            j = i + 2
            while j < len(lines):
                nxt_line = lines[j]
                if nxt_line.strip() == "" or "|" not in nxt_line:
                    break
                block.append(nxt_line)
                j += 1
            chunks.append((True, "".join(block)))
            i = j
            continue
        buf.append(line)
        i += 1
    if buf:
        chunks.append((False, "".join(buf)))
    return chunks


def markdown_to_html(text: str) -> str:
    """Convert markdown to Telegram subset HTML.

    Telegram HTML supports: <b>, <i>, <u>, <s>, <code>, <pre>, <a>,
    <blockquote>, <tg-spoiler>. No <table>, <ul>, <h1>. Tables degrade to
    <pre> aligned grid.
    """
    if not text:
        return ""

    # Pull out fenced code blocks first so inline transforms don't mangle them.
    code_blocks: List[str] = []

    def _stash_fence(m):
        lang = m.group(1)
        body = m.group(2)
        cls = f' class="language-{lang}"' if lang else ""
        code_blocks.append(f"<pre><code{cls}>{_escape_html(body)}</code></pre>")
        return f"\x00CODE{len(code_blocks) - 1}\x00"

    text = _FENCED_CODE.sub(_stash_fence, text)

    # Extract tables and stash placeholders so inline transforms don't touch them.
    table_blocks: List[str] = []
    parts = _extract_table_blocks(text)
    rebuilt: List[str] = []
    for is_table, chunk in parts:
        if is_table:
            table_blocks.append(_convert_table_to_pre_html(chunk))
            rebuilt.append(f"\x00TABLE{len(table_blocks) - 1}\x00")
        else:
            rebuilt.append(chunk)
    text = "".join(rebuilt)

    # Inline code first (so its contents aren't bolded/etc).
    def _stash_inline(m):
        idx = len(code_blocks)
        code_blocks.append(f"<code>{_escape_html(m.group(1))}</code>")
        return f"\x00CODE{idx}\x00"

    text = _INLINE_CODE.sub(_stash_inline, text)

    # Escape the rest — anything not yet stashed is "user text".
    text = _escape_html(text)

    # Re-stash the placeholders (now escaped to \x00... \x00 — they survive
    # because we escaped & < > off-context; the placeholders have none of
    # those chars, but the fenced/inline code blocks we stashed earlier had
    # their content pre-escaped, so we can restore them directly).
    # Note: _escape_html doesn't touch \x00.

    # Heading → <b>Heading</b>
    def _heading_repl(m):
        return f"<b>{m.group(2).strip()}</b>"

    text = _HEADING.sub(_heading_repl, text)

    # Bold, italic, strike, spoiler, link
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    text = _STRIKE.sub(r"<s>\1</s>", text)
    text = _SPOILER.sub(r'<span class="tg-spoiler">\1</span>', text)
    text = _LINK.sub(r'<a href="\2">\1</a>', text)

    # Restore code blocks and tables.
    for i, html in enumerate(code_blocks):
        text = text.replace(f"\x00CODE{i}\x00", html)
    for i, html in enumerate(table_blocks):
        text = text.replace(f"\x00TABLE{i}\x00", html)

    return text.strip()


# ---------------------------------------------------------------------------
# Telegram delivery
# ---------------------------------------------------------------------------

def _split_html(text: str, max_length: int = TELEGRAM_MESSAGE_LIMIT) -> List[str]:
    """Split a long HTML string at paragraph/newline boundaries.

    Naive — we split on `\n\n`, then `\n`, then by hard length if needed.
    We don't try to balance HTML tags; if a chunk becomes invalid it just
    renders raw, which is fine for the fallback path.
    """
    if len(text) <= max_length:
        return [text] if text else []
    chunks: List[str] = []
    remaining = text
    while len(remaining) > max_length:
        idx = remaining.rfind("\n\n", 0, max_length)
        if idx == -1 or idx < max_length // 2:
            idx = remaining.rfind("\n", 0, max_length)
        if idx == -1 or idx < max_length // 2:
            idx = max_length
        chunks.append(remaining[:idx])
        remaining = remaining[idx:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


async def send_rich_message(
    bot,
    chat_id: int,
    text: str,
    message_thread_id: Optional[int] = None,
    reply_to_message_id: Optional[int] = None,
    prefer_rich: bool = True,
    disable_notification: bool = False,
) -> List:
    """Try `sendRichMessage` with `markdown` payload; fall back to HTML on failure.

    Returns the list of sent Message objects (one per split chunk).
    """
    sent: List = []
    if not text:
        return sent

    # Rich path — single chunk, sendRichMessage handles its own layout.
    if prefer_rich:
        try:
            data = {
                "chat_id": chat_id,
                "rich_message": {"markdown": text},
            }
            if message_thread_id:
                data["message_thread_id"] = message_thread_id
            if reply_to_message_id:
                data["reply_to_message_id"] = reply_to_message_id
            if disable_notification:
                data["disable_notification"] = True
            # PTB's private _post hook is the standard way to call any Bot
            # API method. Returns the raw API response dict.
            resp = await bot._post("sendRichMessage", data=data)
            # If we got here without raising, the endpoint accepted the
            # request — do NOT fall through to the HTML path. (Earlier
            # code checked `if msg is not None`, but the API can return
            # `result: null` on success and that sent a duplicate raw
            # message via the HTML fallback — bug: "bot trả về 2 tin
            # nhắn nội dung y hệt nhau, 1 đẹp 1 xấu".)
            sent.append(resp)
            return sent
        except Exception as e:
            logger.warning(
                "sendRichMessage failed, falling back to HTML: "
                "error=%s error_type=%s",
                str(e), type(e).__name__,
            )
            # fall through

    # HTML fallback path.
    html = markdown_to_html(text)
    for chunk in _split_html(html):
        kwargs = {
            "chat_id": chat_id,
            "text": chunk,
            "parse_mode": "HTML",
        }
        if message_thread_id:
            kwargs["message_thread_id"] = message_thread_id
        if reply_to_message_id:
            kwargs["reply_to_message_id"] = reply_to_message_id
        if disable_notification:
            kwargs["disable_notification"] = True
        try:
            msg = await bot.send_message(**kwargs)
            sent.append(msg)
        except Exception as e:
            logger.warning("HTML fallback send_message failed: %s", str(e))
            # Last resort — plain text.
            try:
                kwargs.pop("parse_mode", None)
                msg = await bot.send_message(**kwargs)
                sent.append(msg)
            except Exception as e2:
                logger.error("Plain text fallback failed too: %s", str(e2))
    return sent


async def send_rich_reply(
    bot,
    chat_id: int,
    text: str,
    message_thread_id: Optional[int] = None,
    reply_to_message_id: Optional[int] = None,
    prefer_rich: Optional[bool] = None,
    disable_notification: bool = False,
) -> List:
    """Auto-route: rich path if a table is present (or `prefer_rich=True`),
    else HTML path.

    `prefer_rich=None` → auto-detect via `has_rich_table`.
    `prefer_rich=True` → force rich endpoint.
    `prefer_rich=False` → force HTML endpoint.
    """
    if prefer_rich is None:
        prefer_rich = has_rich_table(text)
    return await send_rich_message(
        bot,
        chat_id=chat_id,
        text=text,
        message_thread_id=message_thread_id,
        reply_to_message_id=reply_to_message_id,
        prefer_rich=prefer_rich,
        disable_notification=disable_notification,
    )


async def edit_or_send_rich(
    bot,
    chat_id: int,
    message,
    text: str,
    message_thread_id: Optional[int] = None,
    prefer_rich: Optional[bool] = None,
) -> List:
    """Edit an existing streaming message in place with HTML formatting;
    if the edit fails (too long, bad chars, can't parse), delete the
    streaming message and send a fresh rich-reply instead.

    Avoids the duplicate-message bug where the streaming-edit shows raw
    markdown AND a fallback rich-reply shows the formatted version —
    we now either edit-in-place OR replace-via-delete, never both.

    `message` is the python-telegram-bot Message object to edit.
    """
    if not text:
        return []
    html = markdown_to_html(text)
    sent: List = []
    try:
        # Telegram's edit_message_text splits internally — if the body
        # is > 4096 chars, the edit throws. Use the first chunk for the
        # edit and send remaining chunks as new rich messages.
        chunks = _split_html(html) if len(html) > TELEGRAM_MESSAGE_LIMIT else [html]
        await message.edit_text(chunks[0], parse_mode="HTML")
        sent.append(message)
        for chunk in chunks[1:]:
            sent.extend(await send_rich_message(
                bot, chat_id=chat_id, text=chunk,
                message_thread_id=message_thread_id,
                prefer_rich=prefer_rich,
            ))
    except Exception as e:
        logger.warning(
            "edit_text failed, replacing streaming message with rich reply: "
            "error=%s error_type=%s",
            str(e), type(e).__name__,
        )
        try:
            await message.delete()
        except Exception as del_err:
            logger.warning("Failed to delete streaming message: %s", str(del_err))
        sent.extend(await send_rich_message(
            bot, chat_id=chat_id, text=text,
            message_thread_id=message_thread_id,
            prefer_rich=prefer_rich if prefer_rich is not None else has_rich_table(text),
        ))
    return sent
