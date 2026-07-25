"""Init for handlers package."""
from src.bot.handlers.message_handler import MessageHandler
from src.bot.handlers.command_handler import CommandHandler
from src.bot.handlers.privilege_handler import PrivilegeHandler

__all__ = [
    "MessageHandler",
    "CommandHandler",
    "PrivilegeHandler",
]