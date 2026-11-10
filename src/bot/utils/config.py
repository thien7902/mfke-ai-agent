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

    # Enterprise model used by /claude. It reads uploaded images natively
    # (attached as image_url blocks) by default — that is how /claude is meant
    # to be used with screenshots. Set ENTERPRISE_MODEL_SUPPORTS_IMAGES=false
    # if the configured endpoint is text-only, in which case /claude falls back
    # to the vision pre-step / OCR into plain text.
    enterprise_model_supports_images: bool = True

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

    # Cluster health probe. When enabled, the bot spawns a background
    # asyncio task that probes every kubeconfig in `cluster_probe_kubeconfig_dir`
    # once per minute via `kubectl get --raw=/readyz` (5s timeout) and posts a
    # status table to the forum chat's "general" topic. Each tick sends a fresh
    # message at the bottom of the topic and deletes the previous probe, so the
    # latest status is always visible without flooding the topic.
    cluster_probe_enabled: bool = False
    cluster_probe_kubeconfig_dir: str = "/root/kubeconfig"
    cluster_probe_chat_id: Optional[int] = None
    # Telegram forum "general" topic is id 1 by convention.
    cluster_probe_topic_id: int = 1

    # Background consolidator. After a forum topic is idle for
    # `consolidator_idle_seconds` after the last assistant response, the bot
    # runs a separate LLM call (cheap model, `consolidator_model`) over the
    # recent transcript and distills reusable facts. Facts are stored in a
    # dedicated MongoDB collection `distilled_facts` (NOT Graphiti) and
    # injected whole into Holmes' system prompt at every turn so the LLM
    # sees them with high priority. Graphiti / `/addmemory` /
    # `search_memory` tool remain unchanged for admin-curated knowledge.
    consolidator_enabled: bool = False
    consolidator_idle_seconds: int = 300
    consolidator_model: Optional[str] = None
    # Skip consolidation if the reloaded conversation has fewer than this
    # many messages (chats too short to extract anything useful).
    consolidator_min_turns: int = 4
    # Cap the number of recent messages passed to the consolidator LLM.
    consolidator_max_turns: int = 20
    # Max topics that may run the consolidator LLM concurrently. Per-topic
    # locks still prevent two fires on the SAME topic from overlapping;
    # this semaphore bounds cross-topic LLM call fan-out so a burst of idle
    # topics doesn't slam the model endpoint all at once.
    consolidator_max_concurrent: int = 5
    # Wall-clock cap on a single consolidator LLM call. A hung model call
    # would otherwise hold the cross-topic semaphore indefinitely.
    consolidator_llm_timeout_seconds: int = 60
    # Cosine threshold for deduplicating distilled facts at inject time.
    # Two facts with cosine ≥ this are treated as duplicates; the newest
    # one (per `created_at`) is kept, the older one dropped from the
    # system-prompt bullets. 0 disables dedup (~1.0 = only drop exact twins).
    # 0.92 is a conservative floor for BGE-m3 on Vietnamese paragraph facts.
    distilled_dedup_cosine: float = 0.92
    # How often (seconds) the background thread rebuilds the deduplicated
    # injectable-facts snapshot. With this on, each Holmes turn reads O(1)
    # from an in-memory cache instead of paying O(N²×1024) cosine cost.
    # 0 disables the background loop and falls back to "recompute per call"
    # (legacy behavior, multi-second latency for ~1k facts).
    distilled_dedup_cache_seconds: int = 60

    # Known-issues KB (FPT Container). Disabled by default. Enable after
    # running `python -m scripts.ingest_kb_docx <docx path>` once to seed
    # the `kb_rules` MongoDB collection with LLM-distilled if-then rules.
    # Injection happens at the snapshot tool output (NOT system prompt) —
    # see src/bot/services/kb_store.py + the /kb_match endpoint in
    # src/bot/utils/health.py. Reuses GRAPHITI_EMBEDDING_MODEL /
    # GRAPHITI_EMBEDDING_DIM for the embedder.
    kb_enabled: bool = False
    # Top-K rules injected per /kb_match call. Rules are one-line if-then; a
    # larger K surfaces more candidate rules for the LLM to reason about.
    kb_top_k: int = 50
    # Model used ONE-TIME at ingest to distill KB articles into if-then
    # rules. Falls back to CONSOLIDATOR_MODEL if unset. Recommended: a
    # cheap/gp-fast model — the distill runs strictly offline at ingest
    # time, ~329 calls total.
    kb_distill_model: Optional[str] = None

    # On-call precheck (admin-configured). Bot is an MCP HTTP/SSE client to
    # mcp-atlassian running separately. At shift start (08:00 morning /
    # 19:00 night) the bot lists all k8s Jira tickets in the lookback window
    # (no assignee filter), creates one Telegram forum topic per NEW ticket,
    # and auto-investigates using the main Holmes model under a read-only
    # system-prompt rule + non-admin user_id so admin_gate hard-blocks any
    # mutation. See src/bot/services/jira_oncall_service.py.
    oncall_enabled: bool = False
    oncall_chat_id: Optional[int] = None
    oncall_mcp_url: Optional[str] = None
    # JQL template; {since} is substituted with the shift-start timestamp.
    # Uses `created >=` (not `updated >=`) so only tickets created within
    # the shift window are fetched — comments/status changes on old
    # tickets do not trigger a precheck.
    oncall_jira_jql_template: str = (
        'category = "Cloud FKE (K8S)" AND statusCategory != Done '
        'AND created >= "{since}"'
    )
    # Worker poll interval (seconds). The oncall loop ticks every
    # `oncall_poll_interval_seconds` and fires while a shift is active
    # (08:00-19:00 morning, 19:00-08:00 night). No fire outside a shift
    # or when no user is registered for the current shift.
    oncall_poll_interval_seconds: int = 120
    oncall_shift_morning_hour: int = 8
    oncall_shift_night_hour: int = 19
    # Lookback = length of the LONGEST shift (night = 13h) + small buffer so
    # the morning-after-night handover does not miss a ticket created at
    # 18:59 the previous evening.
    oncall_lookback_minutes: int = 780
    oncall_topic_icon_color: int = 0x6FB9F0
    # Virtual Telegram user id for auto-investigations. MUST NOT be in
    # ADMIN_USER_IDS — keeps admin_gate's hard-block on mutations active.
    oncall_bot_user_id: int = 0



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
        enterprise_model_supports_images=os.getenv("ENTERPRISE_MODEL_SUPPORTS_IMAGES", "true").lower() in ("1", "true", "yes", "on"),
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
        cluster_probe_enabled=os.getenv("CLUSTER_PROBE_ENABLED", "").lower() in ("1", "true", "yes", "on"),
        cluster_probe_kubeconfig_dir=os.getenv("CLUSTER_PROBE_KUBECONFIG_DIR", "/root/kubeconfig"),
        cluster_probe_chat_id=int(os.getenv("CLUSTER_PROBE_CHAT_ID")) if os.getenv("CLUSTER_PROBE_CHAT_ID") else None,
        cluster_probe_topic_id=int(os.getenv("CLUSTER_PROBE_TOPIC_ID", "1")),
        consolidator_enabled=os.getenv("CONSOLIDATOR_ENABLED", "").lower() in ("1", "true", "yes", "on"),
        consolidator_idle_seconds=int(os.getenv("CONSOLIDATOR_IDLE_SECONDS", "300")),
        consolidator_model=os.getenv("CONSOLIDATOR_MODEL") or None,
        consolidator_min_turns=int(os.getenv("CONSOLIDATOR_MIN_TURNS", "4")),
        consolidator_max_turns=int(os.getenv("CONSOLIDATOR_MAX_TURNS", "20")),
        consolidator_max_concurrent=int(os.getenv("CONSOLIDATOR_MAX_CONCURRENT", "5")),
        consolidator_llm_timeout_seconds=int(os.getenv("CONSOLIDATOR_LLM_TIMEOUT_SECONDS", "60")),
        distilled_dedup_cosine=float(os.getenv("DISTILLED_DEDUP_COSINE", "0.92")),
        distilled_dedup_cache_seconds=int(os.getenv("DISTILLED_DEDUP_CACHE_SECONDS", "60")),
        kb_enabled=os.getenv("KB_ENABLED", "").lower() in ("1", "true", "yes", "on"),
        kb_top_k=int(os.getenv("KB_TOP_K", "50")),
        kb_distill_model=os.getenv("KB_DISTILL_MODEL") or os.getenv("CONSOLIDATOR_MODEL") or None,
        oncall_enabled=os.getenv("ONCALL_ENABLED", "").lower() in ("1", "true", "yes", "on"),
        oncall_chat_id=int(os.getenv("ONCALL_CHAT_ID")) if os.getenv("ONCALL_CHAT_ID") else None,
        oncall_mcp_url=os.getenv("ONCALL_MCP_URL") or None,
        oncall_jira_jql_template=os.getenv("ONCALL_JIRA_JQL_TEMPLATE") or (
            'category = "Cloud FKE (K8S)" AND statusCategory != Done '
            'AND created >= "{since}"'
        ),
        oncall_poll_interval_seconds=int(os.getenv("ONCALL_POLL_INTERVAL_SECONDS", "120")),
        oncall_shift_morning_hour=int(os.getenv("ONCALL_SHIFT_MORNING_HOUR", "8")),
        oncall_shift_night_hour=int(os.getenv("ONCALL_SHIFT_NIGHT_HOUR", "19")),
        oncall_lookback_minutes=int(os.getenv("ONCALL_LOOKBACK_MINUTES", "780")),
        oncall_topic_icon_color=int(os.getenv("ONCALL_TOPIC_ICON_COLOR", str(0x6FB9F0)), base=0),
        oncall_bot_user_id=int(os.getenv("ONCALL_BOT_USER_ID", "0")),
    )


# Global config instance
config = load_config()