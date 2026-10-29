# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered Telegram bot that fronts HolmesGPT (`holmesgpt` pip package) — a tool-calling LLM agent framework. The bot proxies Telegram chat messages to Holmes, exposes Holmes' tools/agents to users, persists conversation history in MongoDB, and (optionally) ingests/retrieves long-term knowledge via Graphiti + FalkorDB.

The bot is forum-aware (per-topic conversation state, message batching, progress topics), admin-gated (permission request/approve workflow via inline buttons), and runs in Docker with an embedded FalkorDB for graph memory.

## Common Commands

### Run

```bash
# Locally (requires TELEGRAM_BOT_TOKEN, ADMIN_USER_IDS, MONGODB_URI in env)
python -m src.bot.main

# Docker (production-shape; builds ubuntu:24.04, installs holmes via pip,
# embeds falkordblite, starts health server on :8080)
docker-compose up -d --build
docker-compose logs -f bot
docker-compose down
```

### Tests

```bash
# Full suite — pytest.ini sets asyncio_mode=auto, testpaths=tests
pytest tests/ -v

# Single file / single test
pytest tests/test_holmes_service.py -v
pytest tests/test_holmes_service.py::TestHolmesService::test_x -v

# Inside the Docker image (entrypoint dispatches `test` mode)
docker-compose run --rm bot test
```

Note on testing Holmes-coupled code: `import src.bot.main as x` binds `x` to the `main()` function defined in that module, not to the module itself. When you need to `patch.object` something on the module, use `importlib.import_module("src.bot.main")` instead of `import src.bot.main as main` (see `tests/conftest.py` for the pattern). This is recorded as a project memory.

### Holmes CLI (debugging from inside the container)

```bash
docker-compose exec bot holmes chat
docker-compose exec bot bash    # entrypoint `shell` mode
```

### Version bumps

`VERSION` is a single line like `v2.2.29`. Commits in this repo follow `vX.Y.Z: <summary>` — bump the file and use that prefix when committing.

## Architecture

### Request flow

```
Telegram update
  → main._update_tracker (resets polling-stick watchdog; group=-1, runs first)
  → CommandHandler / MessageHandler / PrivilegeHandler
  → MessageHandler.handle_message batches per-topic (debounce) → dispatch_batch
  → HolmesService.ask / ask_stream
  → ToolCallingLLM (Holmes) — may call tools, including search_memory
  → stream tokens back to chat (edit_message_text loop)
  → persist assistant + tool-result messages to ConversationService
```

### Module map (only the non-obvious bits)

- `src/bot/main.py` — entry point. Wires all services into a `python-telegram-bot` `Application`. Three things here are load-bearing and not obvious:
  - **`FatalPollingDetector`** (logging handler on `telegram.ext.Updater`): PTB routes polling SSL/TLS errors to its own logger, NOT to `application.add_error_handler`. On `certificate_verify_failed` / TLS handshake failures the handler calls `os._exit(1)` so Docker restarts a fresh container.
  - **30/60/90 min idle watchdog**: `last_update_time` is touched by `_update_tracker`. 30 min → in-place `start_polling` restart (#1), 60 min → restart #2, 90 min → `raise SystemExit(1)` for a full container restart. `polling_restarts` is reset to 0 by `_update_tracker` when a real update arrives — that's the only signal that a restart worked.
  - **`_error_handler`** only catches recoverable network errors (mongo reconnect). Fatal polling errors never reach it.
- `src/bot/services/holmes_service.py` — wraps Holmes. Two non-obvious things:
  - **`_patch_bash_kill_process_group`** monkey-patches Holmes' `execute_bash_command` in BOTH `holmes.plugins.toolsets.bash.common.bash` and `holmes.plugins.toolsets.bash.bash_toolset` (the second imports the function by reference, so patching only the source module leaves a stale bound reference). Without this, timeout-killed bash leaves `kubectl`/`curl` children holding the stdout pipe open and blocks the worker thread for minutes.
  - **`BASH_TIMEOUT_MAX_SECONDS = 30`** / **`NOHUP_TIMEOUT_MAX_SECONDS = 240`** clamp LLM-requested timeouts regardless of what the model asks for.
  - Holmes runs in a dedicated `ThreadPoolExecutor` (`_holmes_executor`) because Holmes' `ToolCallingLLM` is sync; the bot wraps it in `asyncio.run_in_executor`. Streaming tokens cross back over a queue.
  - LLM tool-result persistence is non-trivial — see `FIXES_SUMMARY.md` "Issue 1" before touching `_convert_llm_result_to_messages` / `_stream_chat`. The assistant `tool_calls` block MUST be persisted alongside matching `role="tool"` messages, otherwise Holmes' `_resolve_orphaned_tool_calls` injects fake "Tool execution was cancelled" messages on the next turn.
- `src/bot/services/memory_service.py` — Graphiti + FalkorDB. Owns its own event loop (Holmes tools are sync and call back via `MemoryService.run_sync(coro)`). Init is best-effort: exceptions in `initialize()` are caught in `main.py` so a FalkorDB/embedder failure only disables memory, never the bot.
- `src/bot/tools/search_memory_tool.py` — registers `search_memory` as a Holmes `Tool`. Tool-use policy is governed by a rule appended to `HOLMES_SYSTEM_PROMPT_ADDITIONS` (call ONLY after at least one investigative tool has produced concrete signals; query with distilled terms, not the raw user message). This description lives in the tool itself and in the system-prompt additions — keep them aligned.
- `src/bot/commands/add_memory.py` — `/addmemory` ingests replied-message text or attached files as Graphiti episodes. Chunks large inputs (>~4 KB) along markdown headers because Graphiti's per-episode entity extraction scales poorly with size. Admin-only via `@admin_only`.
- `src/bot/utils/kubeconfig_crypto.py` — encrypts plaintext kubeconfigs at rest on container start (entrypoint calls `migrate_kubeconfigs`). `/kshell` decrypts to a 0600 tempfile, runs `bash -c` with `KUBECONFIG=`, then shreds.
- `src/bot/utils/health.py` — `health_checker` runs an aiohttp server on :8080 with `/live` and `/ready`. Liveness is gated on `bot_alive`, MongoDB, Holmes readiness.
- `src/bot/handlers/message_handler.py` — per-topic message batching (`_PendingBatch` keyed by `(chat_id, topic_id)`), per-topic asyncio locks (`_topic_locks`), dedup via `_processed_messages`. `handle_file` extracts text via `file_extractor` and routes images through `vision_service` (or directly attaches image_url blocks when `MAIN_MODEL_SUPPORTS_IMAGES=true`).
- `src/bot/handlers/command_handler.py` — `/shell`, `/kshell`, `/reconcile`, `/claude`. `/claude` uses `CLAUDE_ENTERPRISE_MODEL` (a key into `~/.holmes/model_list.yaml`) — Holmes resolves it to api_base + key.

### Permission system

`PermissionService` + `PrivilegeHandler` + inline-button callbacks. Admin requests via `/grant <user_id> <perm>` → user gets inline approve/deny buttons → decision persisted + admin notified. `@admin_only` decorator (in `utils/decorators.py`) gates admin commands. Permission requests expire after `PERMISSION_REQUEST_EXPIRY_MINUTES`.

### Configuration

All config flows through `src/bot/utils/config.py` (single `Config` dataclass, loaded from env at import time as the module-level `config`). See `.env.example` for the authoritative list and comments. Key non-obvious env vars:

- `GRAPHITI_MEMORY_ENABLED` (default `true`) — toggles graph memory. Disable to skip FalkorDB entirely.
- `GRAPHITI_FALKOR_MODE` — `embedded` (default, falkordblite in-process, requires Python 3.12+) or `server` (separate FalkorDB container).
- `MAIN_MODEL_SUPPORTS_IMAGES` — when true, image uploads bypass the vision pre-step / OCR and are attached as `image_url` content blocks directly to the main LLM.
- `VISION_MODEL` — used by `vision_service` for image pre-step (default `Qwen3.8-27B`). Shares `OPENAI_API_BASE` / `OPENAI_API_KEY` with Holmes and the embedder.
- `CLAUDE_ENTERPRISE_MODEL` — key into `~/.holmes/model_list.yaml` for the `/claude` one-shot admin command.
- `HOLMES_SYSTEM_PROMPT_ADDITIONS` — appended to Holmes' system prompt. Includes a tool-use rule for `search_memory` (auto-added when memory is enabled) — modifying this rule affects when Holmes calls memory.

### Dependency pin to be aware of

`anthropic` is pinned `>=0.49.0,<1.0.0` because graphiti-core 0.31.2's `AnthropicClient` passes `temperature=` to `messages.create`, which was removed in anthropic SDK v1.0+. Do NOT upgrade `anthropic` past 1.0 until graphiti-core is upgraded (there is a vendored `graphiti_core-0.31.2-py3-none-any.whl` in the repo root used for the embedded install path).

## Conventions

- Structured logging via `structlog` with JSON renderer (configured in `main.py`). Use `logger = structlog.get_logger(__name__)`. Logs include structured kwargs — pass them as keyword args, not f-string interpolation where reasonable.
- The Docker image is `ubuntu:24.04` + Python 3.12 (falkordblite requires it). The image relies on `--break-system-packages` for system-wide pip installs.
- Tests live in `tests/` mirroring source layout; `conftest.py` provides fixtures (`mock_user`, `mock_admin_user`, `mock_update`, etc.). Prefer extending existing fixtures over building new ones.
- Commits are prefixed with the version: `v2.2.29: <summary>`. Bump `VERSION` in the same commit.
