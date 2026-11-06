#!/usr/bin/env bash
# Run the bot as a plain `docker run` container (no compose labels, no
# compose-project prefix). Reads .env for credentials and uses a named
# volume `telegram-bot-holmes` for /root/.holmes persistence across
# restarts (kubeconfigs, graphiti.db, etc.).
#
# Usage:
#   ./run.sh                # latest tag
#   ./run.sh v2.3.30        # specific version
#   ./run.sh --build        # build image first (uses .env's VERSION or 'latest')

set -euo pipefail

cd "$(dirname "$0")"

# --- load .env (only KEY=VALUE lines, skips comments and blanks) ---
if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Copy .env.example to .env and fill in." >&2
  exit 1
fi
declare -A ENV_VARS
while IFS='=' read -r key val; do
  [[ -z "${key:-}" || "${key:0:1}" == "#" ]] && continue
  # strip surrounding quotes
  val="${val#\"}"; val="${val%\"}"
  val="${val#\'}"; val="${val%\'}"
  ENV_VARS["$key"]="$val"
done < <(grep -v '^[[:space:]]*$' .env)

# --- figure out tag ---
TAG="${1:-latest}"
if [[ "$TAG" == "--build" ]]; then
  TAG="${ENV_VARS[VERSION]:-latest}"
  echo "Building image thienlm204608/mfke-ai-telegram-bot:$TAG..."
  docker build -t "thienlm204608/mfke-ai-telegram-bot:$TAG" .
fi

IMAGE="thienlm204608/mfke-ai-telegram-bot:$TAG"
CONTAINER_NAME="telegram-bot"
VOLUME_NAME="telegram-bot-holmes"

echo "Running $CONTAINER_NAME from $IMAGE..."

# --- remove any stale container with the same name (don't fail if absent) ---
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# --- ensure volume exists (no-op if already present) ---
docker volume create "$VOLUME_NAME" >/dev/null 2>&1 || true

# --- run via `--env-file` so all settings (Telegram token, MongoDB Atlas
# URI, OpenAI creds, ADMIN_USER_IDS, consolidator config, ...) flow straight
# from .env without us re-listing them here. MONGODB_URI in .env is the
# Atlas URI — we do NOT override it with a local-mongo URI like compose did.
# Network mode host so the container can reach Atlas and external APIs
# without a Docker bridge network (Windows Docker Desktop host networking
# has caveats — using bridge default + DNS works too, but --env-file alone
# is enough since the bot only needs outbound HTTPS).
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --env-file .env \
  -v "${VOLUME_NAME}:/root/.holmes" \
  -p 8080:8080 \
  "$IMAGE"

echo
echo "Container '$CONTAINER_NAME' started."
echo "  Image:  $IMAGE"
echo "  Volume: $VOLUME_NAME -> /root/.holmes"
echo "  Logs:   docker logs -f $CONTAINER_NAME"
