"""Tests for src/bot/services/vision_service.py.

Mocks the anthropic / openai SDK clients so tests don't need real API access.
"""
import base64
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from src.bot.services import vision_service


@pytest.fixture
def png_path(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


def test_returns_empty_when_vision_model_unset(png_path, monkeypatch):
    monkeypatch.setattr(vision_service.config, "vision_model", None)
    assert vision_service.describe_image(png_path) == ""


def test_returns_empty_when_no_api_key(png_path, monkeypatch):
    monkeypatch.setattr(vision_service.config, "vision_model", "gpt-4o")
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        assert vision_service.describe_image(png_path) == ""


def test_anthropic_path(png_path, monkeypatch):
    monkeypatch.setattr(vision_service.config, "vision_model", "claude-3-5-sonnet-20241022")
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}, clear=False), \
         patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
        os.environ.pop("OPENAI_API_KEY", None)

        fake_block = MagicMock()
        fake_block.text = "extracted text from image"
        fake_resp = MagicMock()
        fake_resp.content = [fake_block]
        fake_client = MagicMock()
        fake_client.messages.create.return_value = fake_resp

        with patch("anthropic.Anthropic", return_value=fake_client) as anthropic_ctor:
            result = vision_service.describe_image(png_path)
            assert result == "extracted text from image"
            # Verify call
            assert anthropic_ctor.called
            create_kwargs = fake_client.messages.create.call_args.kwargs
            assert create_kwargs["model"] == "claude-3-5-sonnet-20241022"
            content = create_kwargs["messages"][0]["content"]
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image"
            assert content[1]["source"]["media_type"] == "image/png"
            # data must be valid base64 of our bytes
            assert base64.b64decode(content[1]["source"]["data"]) == b"\x89PNG\r\n\x1a\n"


def test_openai_path(png_path, monkeypatch):
    monkeypatch.setattr(vision_service.config, "vision_model", "gpt-4o")
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        os.environ.pop("ANTHROPIC_API_KEY", None)

        fake_msg = MagicMock()
        fake_msg.content = "openai extracted text"
        fake_choice = MagicMock()
        fake_choice.message = fake_msg
        fake_resp = MagicMock()
        fake_resp.choices = [fake_choice]
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_resp

        with patch("openai.OpenAI", return_value=fake_client) as openai_ctor:
            result = vision_service.describe_image(png_path)
            assert result == "openai extracted text"
            assert openai_ctor.called
            create_kwargs = fake_client.chat.completions.create.call_args.kwargs
            assert create_kwargs["model"] == "gpt-4o"
            content = create_kwargs["messages"][0]["content"]
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"
            assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_falls_back_on_exception(png_path, monkeypatch):
    monkeypatch.setattr(vision_service.config, "vision_model", "gpt-4o")
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):
        os.environ.pop("ANTHROPIC_API_KEY", None)

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("api down")

        with patch("openai.OpenAI", return_value=fake_client):
            result = vision_service.describe_image(png_path)
            assert result == ""
