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

    # Logging
    log_level: str = "INFO"

    # Rate limiting
    rate_limit_per_minute: int = 30

    # Permission request expiry
    permission_request_expiry_minutes: int = 10


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
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")),
        permission_request_expiry_minutes=int(os.getenv("PERMISSION_REQUEST_EXPIRY_MINUTES", "10")),
    )


# Global config instance
config = load_config()