#!/bin/bash

# Entrypoint script for Telegram Bot with HolmesGPT
# Supports multiple modes: default (bot), holmes (Holmes CLI), shell, test

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check required environment variables
check_env() {
    local missing=()

    if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
        missing+=("TELEGRAM_BOT_TOKEN")
    fi

    if [ -z "$ADMIN_USER_IDS" ]; then
        log_warn "ADMIN_USER_IDS not set - admin commands will not work"
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required environment variables: ${missing[*]}"
        log_error "Please set them in .env file or docker-compose.yml"
        return 1
    fi

    return 0
}

# Function to initialize Holmes config
init_holmes() {
    log_info "Initializing HolmesGPT configuration..."

    if [ ! -f "/home/botuser/.holmes/config.yaml" ]; then
        log_info "Creating default Holmes config..."
        mkdir -p /home/botuser/.holmes

        cat > /home/botuser/.holmes/config.yaml << 'EOF'
# HolmesGPT Configuration
# See https://github.com/holmesgpt/holmes for full documentation

# Model configuration
model:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  temperature: 0.7
  max_tokens: 4096

# Agent configuration
agents:
  enabled: true
  default_agent: "general"

# Tool configuration
tools:
  enabled: true
  builtin:
    - web_search
    - code_execution
    - file_operations

# Memory configuration
memory:
  enabled: true
  type: "conversation"
  max_messages: 50

# Logging
logging:
  level: "INFO"
  format: "json"
EOF
        log_info "Default Holmes config created"
    else
        log_info "Holmes config already exists"
    fi
}

# Function to run the Telegram bot
run_bot() {
    log_info "Starting Telegram Bot..."

    # Check environment
    if ! check_env; then
        exit 1
    fi

    # Initialize Holmes
    init_holmes

    # Run the bot
    cd /app
    exec python3 -m src.bot.main
}

# Function to run Holmes CLI
run_holmes() {
    log_info "Running HolmesGPT CLI..."
    exec holmes "$@"
}

# Function to run shell
run_shell() {
    log_info "Starting interactive shell..."
    exec /bin/bash
}

# Function to run tests
run_tests() {
    log_info "Running tests..."
    cd /app
    exec python3 -m pytest tests/ -v
}

# Function to show help
show_help() {
    echo "Telegram Bot with HolmesGPT - Entrypoint"
    echo ""
    echo "Usage: $0 <command> [args...]"
    echo ""
    echo "Commands:"
    echo "  default, bot     Start the Telegram bot (default)"
    echo "  holmes           Run HolmesGPT CLI"
    echo "  shell            Start interactive shell"
    echo "  test             Run tests"
    echo "  help             Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  TELEGRAM_BOT_TOKEN    Required - Bot token from @BotFather"
    echo "  ADMIN_USER_IDS        Comma-separated list of admin user IDs"
    echo "  MONGODB_URI           MongoDB connection string (default: mongodb://mongodb:27017)"
    echo "  MONGODB_DATABASE      Database name (default: telegram_bot)"
    echo "  HOLMES_CONFIG_PATH    Path to Holmes config (default: /home/botuser/.holmes/config.yaml)"
    echo "  LOG_LEVEL             Log level (default: INFO)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start bot (default)"
    echo "  $0 bot                # Start bot"
    echo "  $0 holmes chat        # Run Holmes chat"
    echo "  $0 shell              # Interactive shell"
    echo "  $0 test               # Run tests"
}

# Main command dispatcher
case "${1:-default}" in
    default|bot)
        run_bot
        ;;
    holmes)
        shift
        run_holmes "$@"
        ;;
    shell)
        run_shell
        ;;
    test)
        run_tests
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac