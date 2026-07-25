# Telegram Bot with HolmesGPT

An AI-powered Telegram bot that uses HolmesGPT for intelligent conversations with full agent, tool, and memory capabilities. Features a privilege command system for permission management.

## Features

- 🤖 **HolmesGPT Integration**: Full access to agents, tools, conversation memory, and streaming responses
- 🔐 **Privilege Commands**: Admin-controlled permission system with user approval workflow
- 💾 **MongoDB Storage**: Persistent conversation history and permission management
- 🐳 **Docker Ready**: Complete containerized deployment with docker-compose
- ⚡ **Rate Limiting**: Configurable per-user rate limiting
- 📝 **Structured Logging**: JSON-formatted logs with structlog

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Telegram   │────▶│  Bot Core   │────▶│  HolmesGPT  │
│   Updates   │     │  (Handlers) │     │   Service   │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Conversa- │ │ Permis-  │ │  User    │
        │  tions   │ │  sions   │ │  Model   │
        └──────────┘ └──────────┘ └──────────┘
              │            │
              └────────────┼────────────┘
                           ▼
                    ┌─────────────┐
                    │  MongoDB    │
                    └─────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- MongoDB (included in docker-compose)

### Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your values:
   ```env
   TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
   MONGODB_URI=mongodb://mongodb:27017
   MONGODB_DATABASE=telegram_bot
   ADMIN_USER_IDS=123456789,987654321
   HOLMES_CONFIG_PATH=/home/botuser/.holmes/config.yaml
   LOG_LEVEL=INFO
   RATE_LIMIT_PER_MINUTE=30
   PERMISSION_REQUEST_EXPIRY_MINUTES=10
   ```

### Running with Docker Compose

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f bot

# Stop
docker-compose down
```

### Running Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export TELEGRAM_BOT_TOKEN=your_token
export ADMIN_USER_IDS=123456789
export MONGODB_URI=mongodb://localhost:27017

# Run
python -m src.bot.main
```

## Commands

### User Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and bot introduction |
| `/help` | Show help information |
| `/permissions` | View your current permissions |
| `/clear` | Clear conversation history |
| `/approve <request_id>` | Approve a permission request |
| `/deny <request_id>` | Deny a permission request |

### Admin Commands

| Command | Description |
|---------|-------------|
| `/grant <user_id> <permission>` | Request permission for a user |
| `/revoke <user_id> <permission>` | Revoke a user's permission |
| `/user_permissions <user_id>` | View another user's permissions |
| `/pending` | View pending permission requests |

### Available Permissions

- `admin_access` - Full admin access to bot commands
- `premium_features` - Access to premium features
- `agent_execution` - Execute AI agents
- `tool_usage` - Use AI tools
- `conversation_memory` - Persistent conversation memory
- `streaming_responses` - Real-time streaming responses

## Permission Workflow

1. **Admin requests permission** for a user using `/grant`
2. **Bot notifies user** with inline approve/deny buttons
3. **User approves/denies** via buttons or `/approve` `/deny` commands
4. **Permission granted/revoked** automatically
5. **Admin notified** of user's decision

## HolmesGPT Features

The bot leverages all HolmesGPT capabilities:

- **Agents**: Specialized AI agents for different tasks
- **Tools**: Web search, code execution, file operations
- **Memory**: Persistent conversation context
- **Streaming**: Real-time response streaming

Configure Holmes in `/home/botuser/.holmes/config.yaml` (auto-generated on first run).

## Development

### Project Structure

```
src/bot/
├── main.py                 # Entry point
├── handlers/
│   ├── message_handler.py  # Regular messages
│   ├── command_handler.py  # Basic commands
│   └── privilege_handler.py # Permission commands
├── services/
│   ├── holmes_service.py   # HolmesGPT integration
│   ├── conversation_service.py # MongoDB conversations
│   └── permission_service.py   # Permissions & requests
├── utils/
│   ├── config.py           # Configuration
│   ├── mongodb.py          # MongoDB connection
│   └── decorators.py       # Handler decorators
└── models/
    ├── user.py             # User & permissions
    ├── conversation.py     # Conversation & messages
    └── permission_request.py # Permission requests
```

### Running Tests

```bash
# In Docker
docker-compose run --rm bot test

# Locally
pytest tests/ -v
```

## Deployment

### Production Considerations

1. **Use a proper MongoDB instance** (not the docker-compose one)
2. **Set up proper logging aggregation** (ELK, Datadog, etc.)
3. **Configure rate limiting** with Redis for distributed deployments
4. **Use secrets management** for tokens (Docker secrets, Vault, etc.)
5. **Set up monitoring** (Prometheus, Grafana)
6. **Configure HolmesGPT** with your preferred model provider

### Environment Variables for Production

```env
TELEGRAM_BOT_TOKEN=your_production_token
MONGODB_URI=mongodb://your-mongodb-cluster:27017
MONGODB_DATABASE=telegram_bot_prod
ADMIN_USER_IDS=123456789,987654321
HOLMES_CONFIG_PATH=/home/botuser/.holmes/config.yaml
LOG_LEVEL=WARNING
RATE_LIMIT_PER_MINUTE=60
PERMISSION_REQUEST_EXPIRY_MINUTES=15
```

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request