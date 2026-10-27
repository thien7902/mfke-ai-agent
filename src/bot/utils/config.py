"""Configuration management for the Telegram bot."""
import os
from dataclasses import dataclass
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Application configuration."""

    # Telegram
    telegram_bot_token: str

    # MongoDB
    mongodb_uri: str
    mongodb_database: str

    # Admin users
    admin_user_ids: List[int]

    # Holmes
    holmes_config_path: str
    holmes_system_prompt_additions: Optional[str] = None
    # Key into ~/.holmes/model_list.yaml for the /claude one-shot enterprise model.
    # When unset, /claude replies with a "not configured" error.
    claude_enterprise_model: Optional[str] = None

    # Logging
    log_level: str = "INFO"

    # Rate limiting
    rate_limit_per_minute: int = 30

    # Permission request expiry
    permission_request_expiry_minutes: int = 10

    # Message batching (per forum topic)
    message_batch_quiet_period_seconds: int = 10
    message_batch_max_wait_seconds: int = 30
    message_batch_max_messages: int = 20

    # File uploads: max extracted text fed to Holmes, tmp dir for downloads,
    # and an optional vision model name (same provider as main LLM) used as
    # a pre-step for image uploads. When unset, images fall back to OCR only.
    # Default is Qwen3.8-27B; override via VISION_MODEL env var if your
    # OpenAI-compatible endpoint expects a different model ID.
    file_upload_max_text_bytes: int = 10 * 1024 * 1024
    file_upload_tmp_dir: str = "/tmp/bot_uploads"
    vision_model: Optional[str] = "Qwen3.8-27B"

    # Set True when the main Holmes LLM accepts OpenAI-style image_url content
    # blocks (e.g. gpt-4o, claude-3-5-sonnet, qwen-vl-max). When True, image
    # uploads skip the vision pre-step / OCR and are attached directly to the
    # user message so the main model can read them. When False (default, safe),
    # images go through vision_service → OCR fallback into plain text.
    main_model_supports_images: bool = False

    # Outbound files (bot → user). The `chat_send_file` Holmes tool and the
    # /sendfile command may only send files from this directory — it is the
    # model's workspace, kept separate so a generated file cannot be aimed at
    # /root/.holmes/kubeconfig-only-admin-can-run or any other secret on disk.
    # Default lives under /root/.holmes, which docker-compose already mounts as
    # a persistent volume, so generated files survive container restarts.
    file_send_outbox_dir: str = "/root/.holmes/outbox"
    # Upper bound on a single upload; clamped to Telegram's 50 MB bot limit.
    file_send_max_bytes: int = 50 * 1024 * 1024
    # Deliverables older than this are pruned before each send. 0 disables.
    file_send_outbox_ttl_hours: int = 24
    # Keep a copy of each user upload under <outbox>/received/ so the
    # conversation can send the same file back ("send that xlsx to me again").
    # Set False to stop persisting uploads — they are then only readable
    # during the turn that received them.
    file_send_keep_uploads: bool = True

    # Graph memory (Graphiti). Enabled by default — MemoryService.init is
    # wrapped in try/except in main.py, so a FalkorDB/embedder failure only
    # disables memory, not the whole bot.
    graphiti_memory_enabled: bool = True
    # Embedded FalkorDB on-disk path. Used when graphiti_falkor_mode="embedded".
    # For the non-embedded Docker mode, set FALKOR_HOST=FALKOR_PORT instead.
    # Default under /root/.holmes so it survives container restarts when
    # the holmes_config volume is mounted (see docker-compose.yml).
    graphiti_db_path: str = "/root/.holmes/graphiti.db"
    # "embedded" (falkordblite, in-process, no Docker) or "server" (separate FalkorDB container).
    graphiti_falkor_mode: str = "embedded"
    graphiti_falkor_host: str = "localhost"
    graphiti_falkor_port: int = 6379
    graphiti_falkor_database: str = "default_db"
    # Group id used when ingesting /addmemory episodes. All global knowledge
    # shares one group so search is cross-conversation.
    graphiti_group_id: str = "global"
    # Embedder: must match the model served at the OpenAI-compatible endpoint
    # configured via OPENAI_API_BASE / OPENAI_API_KEY. Vietnamese_Embedding
    # (aiteamvn/vietnamese-embedding) — produces 1024-dim vectors.
    graphiti_embedding_model: str = "Vietnamese_Embedding"
    graphiti_embedding_dim: int = 1024


def load_config() -> Config:
    """Load configuration from environment variables."""
    admin_ids_str = os.getenv("ADMIN_USER_IDS", "")
    admin_user_ids = [
        int(uid.strip()) for uid in admin_ids_str.split(",") if uid.strip()
    ]

    return Config(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        mongodb_database=os.getenv("MONGODB_DATABASE", "telegram_bot"),
        admin_user_ids=admin_user_ids,
        holmes_config_path=os.getenv("HOLMES_CONFIG_PATH", "/root/.holmes/config.yaml"),
        holmes_system_prompt_additions=os.getenv("HOLMES_SYSTEM_PROMPT_ADDITIONS"),
        claude_enterprise_model=os.getenv("CLAUDE_ENTERPRISE_MODEL") or None,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")),
        permission_request_expiry_minutes=int(os.getenv("PERMISSION_REQUEST_EXPIRY_MINUTES", "10")),
        message_batch_quiet_period_seconds=int(os.getenv("MESSAGE_BATCH_QUIET_PERIOD_SECONDS", "10")),
        message_batch_max_wait_seconds=int(os.getenv("MESSAGE_BATCH_MAX_WAIT_SECONDS", "30")),
        message_batch_max_messages=int(os.getenv("MESSAGE_BATCH_MAX_MESSAGES", "20")),
        file_upload_max_text_bytes=int(os.getenv("FILE_UPLOAD_MAX_TEXT_BYTES", str(10 * 1024 * 1024))),
        file_upload_tmp_dir=os.getenv("FILE_UPLOAD_TMP_DIR", "/tmp/bot_uploads"),
        vision_model=os.getenv("VISION_MODEL") or "Qwen3.8-27B",
        main_model_supports_images=os.getenv("MAIN_MODEL_SUPPORTS_IMAGES", "").lower() in ("1", "true", "yes", "on"),
        file_send_outbox_dir=os.getenv("FILE_SEND_OUTBOX_DIR", "/root/.holmes/outbox"),
        file_send_max_bytes=int(os.getenv("FILE_SEND_MAX_BYTES", str(50 * 1024 * 1024))),
        file_send_outbox_ttl_hours=int(os.getenv("FILE_SEND_OUTBOX_TTL_HOURS", "24")),
        file_send_keep_uploads=os.getenv("FILE_SEND_KEEP_UPLOADS", "true").lower() in ("1", "true", "yes", "on"),
        graphiti_memory_enabled=os.getenv("GRAPHITI_MEMORY_ENABLED", "true").lower() in ("1", "true", "yes", "on"),
        graphiti_db_path=os.getenv("GRAPHITI_DB_PATH", "/root/.holmes/graphiti.db"),
        graphiti_falkor_mode=os.getenv("GRAPHITI_FALKOR_MODE", "embedded"),
        graphiti_falkor_host=os.getenv("GRAPHITI_FALKOR_HOST", "localhost"),
        graphiti_falkor_port=int(os.getenv("GRAPHITI_FALKOR_PORT", "6379")),
        graphiti_falkor_database=os.getenv("GRAPHITI_FALKOR_DATABASE", "default_db"),
        graphiti_group_id=os.getenv("GRAPHITI_GROUP_ID", "global"),
        graphiti_embedding_model=os.getenv("GRAPHITI_EMBEDDING_MODEL", "Vietnamese_Embedding"),
        graphiti_embedding_dim=int(os.getenv("GRAPHITI_EMBEDDING_DIM", "1024")),
    )


# Global config instance
config = load_config()