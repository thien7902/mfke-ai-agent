"""Tests for Telegram helper utilities."""
import pytest
from src.bot.utils.telegram_helpers import split_message, TELEGRAM_MAX_MESSAGE_LENGTH


class TestSplitMessage:
    """Test the split_message function."""

    def test_short_message(self):
        """Test that short messages are returned as-is."""
        text = "Hello, world!"
        result = split_message(text)
        assert result == ["Hello, world!"]

    def test_exact_limit(self):
        """Test message exactly at the limit."""
        text = "a" * TELEGRAM_MAX_MESSAGE_LENGTH
        result = split_message(text)
        assert result == [text]
        assert len(result[0]) == TELEGRAM_MAX_MESSAGE_LENGTH

    def test_over_limit_by_one(self):
        """Test message just over the limit splits into two."""
        text = "a" * (TELEGRAM_MAX_MESSAGE_LENGTH + 1)
        result = split_message(text)
        assert len(result) == 2
        assert len(result[0]) <= TELEGRAM_MAX_MESSAGE_LENGTH
        assert len(result[1]) <= TELEGRAM_MAX_MESSAGE_LENGTH
        assert result[0] + result[1] == text

    def test_splits_at_newline(self):
        """Test that splitting prefers newlines when they're in a reasonable position."""
        # Create a message with a newline at around 60% of the limit, total > limit
        prefix = "x" * int(TELEGRAM_MAX_MESSAGE_LENGTH * 0.6)
        text = prefix + "\nLine 2\n" + "y" * 2000
        result = split_message(text)
        assert len(result) == 2
        # First chunk should end with the newline
        assert result[0].endswith("\n")

    def test_splits_at_space(self):
        """Test that splitting prefers spaces when no newlines."""
        # Create a long message with spaces
        text = "word " * (TELEGRAM_MAX_MESSAGE_LENGTH // 5 + 10)
        result = split_message(text)
        assert len(result) >= 2
        # Chunks should not break in the middle of "word "
        for chunk in result[:-1]:  # All but last
            assert chunk.endswith(" ") or chunk.endswith("word ")

    def test_very_long_message(self):
        """Test splitting a very long message into many chunks."""
        text = "This is a test sentence. " * 500  # ~12000 chars
        result = split_message(text)
        assert len(result) >= 3
        for chunk in result:
            assert len(chunk) <= TELEGRAM_MAX_MESSAGE_LENGTH
        # Reconstructed should match original
        assert "".join(result) == text

    def test_empty_string(self):
        """Test empty string returns empty list."""
        result = split_message("")
        assert result == [""]

    def test_custom_max_length(self):
        """Test with custom max length."""
        text = "Hello world, this is a test message."
        result = split_message(text, max_length=20)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 20