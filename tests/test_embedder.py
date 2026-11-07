"""Tests for src.bot.services.embedder.

Pattern: patch `openai.OpenAI` so no real API call. Assert the embeddings
client receives the right model + input and our wrappers translate
`resp.data[i].embedding` back correctly.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest


def _mock_embedding_response(*vecs):
    resp = MagicMock()
    resp.data = [MagicMock(embedding=v) for v in vecs]
    return resp


def test_embed_text_empty_returns_empty_list():
    from src.bot.services.embedder import embed_text
    with patch("src.bot.services.embedder.openai.OpenAI") as mock_openai:
        # Should NOT call the client at all.
        assert embed_text("") == []
        assert embed_text("   \n  ") == []
        mock_openai.assert_not_called()


def test_embed_text_missing_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from src.bot.services.embedder import embed_text
    with patch("src.bot.services.embedder.openai.OpenAI") as mock_openai:
        assert embed_text("hello") == []
        mock_openai.assert_not_called()


def test_embed_text_happy_path():
    from src.bot.services.embedder import embed_text
    with patch("src.bot.services.embedder.openai.OpenAI") as mock_openai, \
         patch.dict("os.environ", {"OPENAI_API_KEY": "k", "OPENAI_API_BASE": "http://x"}):
        client = MagicMock()
        client.embeddings.create.return_value = _mock_embedding_response([0.1, 0.2, 0.3])
        mock_openai.return_value = client
        out = embed_text("hello world")
        assert out == [0.1, 0.2, 0.3]
        # Assert model + input shape
        call = client.embeddings.create.call_args
        assert call.kwargs["input"] == "hello world"
        assert call.kwargs["model"] is not None  # graphiti_embedding_model default


def test_embed_text_truncates_long_input():
    from src.bot.services.embedder import embed_text
    with patch("src.bot.services.embedder.openai.OpenAI") as mock_openai, \
         patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
        client = MagicMock()
        client.embeddings.create.return_value = _mock_embedding_response([0.5])
        mock_openai.return_value = client
        long_text = "x" * 10000
        embed_text(long_text)
        sent = client.embeddings.create.call_args.kwargs["input"]
        assert len(sent) == 6000  # _MAX_CHARS_PER_TEXT


def test_embed_text_api_failure_returns_empty():
    from src.bot.services.embedder import embed_text
    with patch("src.bot.services.embedder.openai.OpenAI") as mock_openai, \
         patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
        client = MagicMock()
        client.embeddings.create.side_effect = RuntimeError("api down")
        mock_openai.return_value = client
        assert embed_text("hello") == []


def test_embed_chunks_empty_returns_empty_list():
    from src.bot.services.embedder import embed_chunks
    with patch("src.bot.services.embedder.openai.OpenAI") as mock_openai:
        assert embed_chunks([]) == []
        mock_openai.assert_not_called()


def test_embed_chunks_happy_path():
    from src.bot.services.embedder import embed_chunks
    with patch("src.bot.services.embedder.openai.OpenAI") as mock_openai, \
         patch.dict("os.environ", {"OPENAI_API_KEY": "k", "OPENAI_API_BASE": "http://x"}):
        client = MagicMock()
        # Return three vectors for three inputs.
        client.embeddings.create.return_value = _mock_embedding_response(
            [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]
        )
        mock_openai.return_value = client
        out = embed_chunks(["a", "b", "c"])
        assert out == [[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]]
        # API input should be the list of strings, in order.
        assert client.embeddings.create.call_args.kwargs["input"] == ["a", "b", "c"]


def test_embed_chunks_drops_empty_strings_from_input():
    from src.bot.services.embedder import embed_chunks
    with patch("src.bot.services.embedder.openai.OpenAI") as mock_openai, \
         patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
        client = MagicMock()
        # Only two non-empty inputs reach the API.
        client.embeddings.create.return_value = _mock_embedding_response([0.1], [0.2])
        mock_openai.return_value = client
        out = embed_chunks(["a", "", "  ", "b"])
        # Indices 0 and 3 should have vectors, 1 and 2 stay empty.
        assert out == [[0.1], [], [], [0.2]]


def test_embed_chunks_api_failure_returns_empty():
    from src.bot.services.embedder import embed_chunks
    with patch("src.bot.services.embedder.openai.OpenAI") as mock_openai, \
         patch.dict("os.environ", {"OPENAI_API_KEY": "k"}):
        client = MagicMock()
        client.embeddings.create.side_effect = RuntimeError("api down")
        mock_openai.return_value = client
        # Batched path swallows per-batch errors → vectors stay empty.
        out = embed_chunks(["a", "b"])
        assert out == [[], []]
