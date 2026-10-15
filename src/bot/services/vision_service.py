"""Vision pre-step for image uploads.

Uses the same provider as the main Holmes LLM (Anthropic if ANTHROPIC_API_KEY
is set, OpenAI-compatible if OPENAI_API_KEY is set). Sends the image with a
"extract everything" prompt and returns the model's text output, which the
message handler then feeds into the normal Holmes text-only flow.

Falls back to OCR (tesseract) on any error or when VISION_MODEL is unset.

Provider selection mirrors `HolmesService._get_llm_config` in
`src/bot/services/holmes_service.py` — keep them in sync if env-var
precedence ever changes.
"""
import base64
import os
from pathlib import Path
from typing import Optional

import structlog
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_exponential

from src.bot.utils.config import config

logger = structlog.get_logger(__name__)

# Prompt sent alongside every image. Asks the model to be a "text-only LLM's
# eyes": extract readable text and describe diagrams/layouts so the downstream
# Holmes agent can reason over the image without vision capability.
_VISION_PROMPT = (
    "You are assisting a text-only AI assistant. Extract all readable text "
    "from this image verbatim. Also briefly describe any charts, diagrams, "
    "layouts, and visible objects that carry information. Be concise but "
    "complete — output only the extraction, no preamble."
)


def _provider() -> str:
    """Return 'anthropic', 'openai', or 'none' based on env vars."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


def _read_b64(path: Path) -> tuple[str, str]:
    """Read file and return (base64_data, data_mime)."""
    mime = _guess_mime(path)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return data, mime


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }.get(ext, "image/jpeg")


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def _call_anthropic(model: str, b64: str, mime: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _VISION_PROMPT},
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
            ],
        }],
    )
    parts = [block.text for block in resp.content if getattr(block, "text", None)]
    return "".join(parts).strip()


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def _call_openai(model: str, b64: str, mime: str) -> str:
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("OPENAI_API_BASE")
    client = openai.OpenAI(api_key=api_key, base_url=api_base) if api_base else openai.OpenAI(api_key=api_key)
    data_url = f"data:{mime};base64,{b64}"
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    )
    return (resp.choices[0].message.content or "").strip()


def describe_image(path: Path) -> str:
    """Call the configured vision model on the image. Returns "" on any failure.

    Caller should fall back to OCR when this returns "".
    """
    model = config.vision_model
    if not model:
        return ""

    provider = _provider()
    if provider == "none":
        logger.warning("VISION_MODEL set but no API key found; falling back to OCR")
        return ""

    try:
        b64, mime = _read_b64(path)
    except Exception as e:
        logger.warning("Failed to read image for vision", path=str(path), error=str(e))
        return ""

    try:
        if provider == "anthropic":
            return _call_anthropic(model, b64, mime)
        else:
            return _call_openai(model, b64, mime)
    except Exception as e:
        logger.warning("Vision model call failed; falling back to OCR",
                       provider=provider, model=model, error=str(e))
        return ""
