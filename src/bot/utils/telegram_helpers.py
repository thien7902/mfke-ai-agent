"""Telegram Helper Utilities."""
from typing import List


TELEGRAM_MAX_MESSAGE_LENGTH = 4096


def split_message(text: str, max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> List[str]:
    """
    Split a long message into chunks that fit within Telegram's message length limit.

    Attempts to split at natural boundaries (newlines, spaces) to avoid breaking words.

    Args:
        text: The message text to split
        max_length: Maximum length per chunk (default: Telegram's 4096 limit)

    Returns:
        List of message chunks, each <= max_length characters
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_length:
        # Try to find a good split point within the limit
        split_idx = max_length

        # Prefer splitting at double newline (paragraph break)
        double_newline_idx = remaining.rfind('\n\n', 0, max_length)
        if double_newline_idx > max_length * 0.5:  # Only if it's not too early
            split_idx = double_newline_idx + 2  # Include the newlines
        else:
            # Try single newline
            newline_idx = remaining.rfind('\n', 0, max_length)
            if newline_idx > max_length * 0.5:
                split_idx = newline_idx + 1  # Include the newline
            else:
                # Try space
                space_idx = remaining.rfind(' ', 0, max_length)
                if space_idx > max_length * 0.5:
                    split_idx = space_idx + 1  # Include the space

        chunk = remaining[:split_idx]
        chunks.append(chunk)
        remaining = remaining[split_idx:]

    # Add the remaining part
    if remaining:
        chunks.append(remaining)

    return chunks


async def send_long_message(
    bot,
    chat_id: int,
    text: str,
    message_thread_id: int = None,
    **kwargs
) -> List:
    """
    Send a potentially long message by splitting it into multiple Telegram messages.

    Args:
        bot: Telegram bot instance
        chat_id: Chat ID to send to
        text: Message text (will be split if > 4096 chars)
        message_thread_id: Optional forum topic thread ID
        **kwargs: Additional arguments passed to send_message (parse_mode, etc.)

    Returns:
        List of sent Message objects
    """
    chunks = split_message(text)
    sent_messages = []

    for i, chunk in enumerate(chunks):
        # For the first chunk, we might want to reply to a specific message
        # For subsequent chunks, just send as new messages
        if i == 0 and 'reply_to_message_id' in kwargs:
            # Only reply to the original message for the first chunk
            message = await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                message_thread_id=message_thread_id,
                **kwargs
            )
        else:
            # Remove reply_to_message_id for subsequent chunks
            send_kwargs = {k: v for k, v in kwargs.items() if k != 'reply_to_message_id'}
            message = await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                message_thread_id=message_thread_id,
                **send_kwargs
            )
        sent_messages.append(message)

    return sent_messages


async def edit_or_send_long_message(
    bot,
    chat_id: int,
    text: str,
    message_id: int = None,
    message_thread_id: int = None,
    **kwargs
) -> List:
    """
    Edit an existing message or send a new one, handling long messages by splitting.

    If message_id is provided, attempts to edit that message with the first chunk,
    then sends remaining chunks as new messages.

    Args:
        bot: Telegram bot instance
        chat_id: Chat ID
        text: Message text (will be split if > 4096 chars)
        message_id: Optional message ID to edit (for streaming responses)
        message_thread_id: Optional forum topic thread ID
        **kwargs: Additional arguments

    Returns:
        List of Message objects (edited/sent)
    """
    chunks = split_message(text)
    sent_messages = []

    if not chunks:
        return sent_messages

    if message_id:
        # Edit the first chunk
        try:
            await bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=chunks[0],
                **kwargs
            )
            # Create a dummy message object for the edited message
            # We can't get the actual message object from edit_message_text easily
            sent_messages.append(None)  # Placeholder
        except Exception:
            # If edit fails, send as new message
            message = await bot.send_message(
                chat_id=chat_id,
                text=chunks[0],
                message_thread_id=message_thread_id,
                **kwargs
            )
            sent_messages.append(message)

        # Send remaining chunks as new messages
        for chunk in chunks[1:]:
            message = await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                message_thread_id=message_thread_id,
                **kwargs
            )
            sent_messages.append(message)
    else:
        # No message_id, send all as new messages
        sent_messages = await send_long_message(
            bot, chat_id, text, message_thread_id, **kwargs
        )

    return sent_messages