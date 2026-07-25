# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered Telegram bot project (mfke-ai-telegram-bot). The bot likely integrates with AI services to provide intelligent responses to Telegram users.

## Common Development Commands

### Setup & Installation
```bash
# Install dependencies (when package.json exists)
npm install

# Or if using Python
pip install -r requirements.txt
```

### Development
```bash
# Run in development mode (when implemented)
npm run dev
# or
python bot.py

# Run tests (when implemented)
npm test
# or
pytest
```

### Building & Deployment
```bash
# Build for production (when applicable)
npm run build

# Lint code (when configured)
npm run lint
# or
flake8 .
```

## Expected Project Structure

```
mfke-ai-telegram-bot/
├── src/                    # Source code
│   ├── bot/               # Bot core logic
│   ├── handlers/          # Message/command handlers
│   ├── services/          # External service integrations (AI APIs, databases)
│   ├── utils/             # Utility functions
│   └── config/            # Configuration management
├── tests/                 # Test files
├── .env.example           # Environment variables template
├── package.json / requirements.txt  # Dependencies
└── README.md              # Project documentation
```

## Key Configuration

- **Environment Variables**: Create `.env` file from `.env.example` with:
  - `TELEGRAM_BOT_TOKEN` - Bot token from @BotFather
  - `AI_API_KEY` - API key for AI service (OpenAI, Anthropic, etc.)
  - `DATABASE_URL` - Database connection string (if used)

## Development Notes

- Use webhook or long-polling for receiving Telegram updates
- Implement proper error handling and logging
- Consider rate limiting for AI API calls
- Store user sessions/conversation history appropriately
- Follow Telegram Bot API best practices

## Testing Strategy

- Unit tests for handlers and utilities
- Integration tests for AI service integration
- Mock Telegram API for isolated testing

---

*This CLAUDE.md will be updated as the project develops with actual commands, structure, and conventions.*