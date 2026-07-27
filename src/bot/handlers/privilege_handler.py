"""Privilege Handler - Handle privilege commands for permission management."""
import structlog
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.bot.services.permission_service import PermissionService
from src.bot.models.user import UserPermission
from src.bot.models.permission_request import PermissionStatus
from src.bot.utils.decorators import admin_only, rate_limit

logger = structlog.get_logger(__name__)


class PrivilegeHandler:
    """Handle privilege commands for permission management."""

    def __init__(self, permission_service: PermissionService):
        self.permissions = permission_service

    @admin_only
    @rate_limit
    async def grant_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /grant command - admin requests permission for user."""
        args = context.args
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: /grant <user_id> <permission>\n\n"
                "Available permissions:\n"
                "• admin_access\n"
                "• premium_features\n"
                "• agent_execution\n"
                "• tool_usage\n"
                "• conversation_memory\n"
                "• streaming_responses"
            )
            return

        try:
            target_user_id = int(args[0])
            permission_str = args[1]
            permission = UserPermission(permission_str)
        except ValueError:
            await update.message.reply_text("❌ Invalid user ID or permission.")
            return
        except Exception:
            await update.message.reply_text("❌ Invalid permission. Use /help to see available permissions.")
            return

        requester_id = update.effective_user.id

        # Get current chat info for forum topic notification
        chat = update.effective_chat
        chat_id = chat.id
        topic_id = update.message.message_thread_id or 0

        try:
            request = await self.permissions.create_permission_request(
                requester_id=requester_id,
                target_user_id=target_user_id,
                permission=permission,
            )

            # Notify the target user with approve/deny buttons in the same forum topic
            await self._notify_user_for_approval(request, context, chat_id=chat_id, topic_id=topic_id)

            await update.message.reply_text(
                f"✅ Permission request created!\n\n"
                f"📋 **Request ID:** {request._id}\n"
                f"👤 **Target User:** {target_user_id}\n"
                f"🔐 **Permission:** {permission.value}\n\n"
                f"The user has been notified in this topic to approve or deny this request."
            )

        except ValueError as e:
            await update.message.reply_text(f"❌ {str(e)}")
        except PermissionError as e:
            await update.message.reply_text(f"❌ {str(e)}")
        except Exception as e:
            logger.error("Error creating permission request", error=str(e))
            await update.message.reply_text("❌ An error occurred. Please try again.")

    @admin_only
    @rate_limit
    async def revoke_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /revoke command - admin revokes user permission."""
        args = context.args
        if len(args) < 2:
            await update.message.reply_text(
                "Usage: /revoke <user_id> <permission>"
            )
            return

        try:
            target_user_id = int(args[0])
            permission_str = args[1]
            permission = UserPermission(permission_str)
        except ValueError:
            await update.message.reply_text("❌ Invalid user ID or permission.")
            return

        success = await self.permissions.revoke_permission(target_user_id, permission)

        if success:
            await update.message.reply_text(
                f"✅ Revoked {permission.value} from user {target_user_id}"
            )
        else:
            await update.message.reply_text(
                f"❌ User {target_user_id} not found or didn't have that permission"
            )

    @admin_only
    @rate_limit
    async def user_permissions_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /user_permissions command - admin views user's permissions."""
        args = context.args
        if len(args) < 1:
            await update.message.reply_text("Usage: /user_permissions <user_id>")
            return

        try:
            target_user_id = int(args[0])
        except ValueError:
            await update.message.reply_text("❌ Invalid user ID.")
            return

        user_perms = await self.permissions.list_user_permissions(target_user_id)

        if not user_perms:
            await update.message.reply_text(
                f"🔐 User {target_user_id} has no permissions."
            )
            return

        perm_text = f"🔐 **Permissions for user {target_user_id}:**\n\n"
        for perm in user_perms:
            perm_text += f"• {perm.value}\n"

        await update.message.reply_text(perm_text)

    @admin_only
    @rate_limit
    async def pending_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /pending command - admin views pending requests."""
        admin_id = update.effective_user.id
        requests = await self.permissions.get_pending_requests_by_admin(admin_id)

        if not requests:
            await update.message.reply_text("📭 No pending permission requests.")
            return

        text = "📋 **Pending Permission Requests:**\n\n"
        for req in requests:
            text += (
                f"🆔 **ID:** {req._id}\n"
                f"👤 **User:** {req.target_user_id}\n"
                f"🔐 **Permission:** {req.permission.value}\n"
                f"📝 **Reason:** {req.reason or 'None'}\n"
                f"⏰ **Expires:** {req.expires_at.strftime('%Y-%m-%d %H:%M')}\n\n"
            )

        await update.message.reply_text(text)

    @rate_limit
    async def approve_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle /approve command - user approves permission request."""
        # Disable private chats
        chat = update.effective_chat
        if chat.type == "private":
            await update.message.reply_text(
                "🚫 Private chats are disabled. Please use the bot in a group forum topic.",
                message_thread_id=None
            )
            return

        args = context.args
        if len(args) < 1:
            await update.message.reply_text("Usage: /approve <request_id>")
            return

        request_id = args[0]
        user_id = update.effective_user.id

        try:
            success = await self.permissions.approve_request(request_id, user_id)
            if success:
                await update.message.reply_text(
                    "✅ Permission granted! You now have the new permission."
                )
                # Notify admin
                await self._notify_admin_of_decision(request_id, True, context)
            else:
                await update.message.reply_text(
                    "❌ Request not found, already processed, or expired."
                )
        except PermissionError as e:
            await update.message.reply_text(f"❌ {str(e)}")
        except Exception as e:
            logger.error("Error approving request", error=str(e))
            await update.message.reply_text("❌ An error occurred.")

    @rate_limit
    async def deny_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /deny command - user denies permission request."""
        # Disable private chats
        chat = update.effective_chat
        if chat.type == "private":
            await update.message.reply_text(
                "🚫 Private chats are disabled. Please use the bot in a group forum topic.",
                message_thread_id=None
            )
            return

        args = context.args
        if len(args) < 1:
            await update.message.reply_text("Usage: /deny <request_id>")
            return

        request_id = args[0]
        user_id = update.effective_user.id

        try:
            success = await self.permissions.deny_request(request_id, user_id)
            if success:
                await update.message.reply_text("✅ Permission request denied.")
                # Notify admin
                await self._notify_admin_of_decision(request_id, False, context)
            else:
                await update.message.reply_text(
                    "❌ Request not found, already processed, or expired."
                )
        except PermissionError as e:
            await update.message.reply_text(f"❌ {str(e)}")
        except Exception as e:
            logger.error("Error denying request", error=str(e))
            await update.message.reply_text("❌ An error occurred.")

    async def _notify_user_for_approval(
        self, request, context: ContextTypes.DEFAULT_TYPE, chat_id: int = None, topic_id: int = None
    ):
        """Send notification to target user with approve/deny buttons in a forum topic."""
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Approve", callback_data=f"approve_{request._id}"),
                InlineKeyboardButton("❌ Deny", callback_data=f"deny_{request._id}"),
            ]
        ])

        text = (
            f"🔔 **Permission Request**\n\n"
            f"An admin has requested the **{request.permission.value}** permission for you.\n\n"
            f"📋 **Request ID:** {request._id}\n"
            f"📝 **Reason:** {request.reason or 'None provided'}\n\n"
            f"Please choose:"
        )

        try:
            # Send to the forum topic where the grant was initiated
            await context.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=keyboard,
                message_thread_id=topic_id
            )
        except Exception as e:
            logger.warning("Failed to notify user in topic", user_id=request.target_user_id, error=str(e))

    async def _notify_admin_of_decision(
        self, request_id: str, approved: bool, context: ContextTypes.DEFAULT_TYPE
    ):
        """Notify the requesting admin of user's decision."""
        request = await self.permissions.get_request(request_id)
        if not request:
            return

        decision = "approved ✅" if approved else "denied ❌"
        text = (
            f"🔔 **Permission Request Update**\n\n"
            f"User {request.target_user_id} has **{decision}** your request "
            f"for **{request.permission.value}** permission.\n\n"
            f"📋 **Request ID:** {request_id}"
        )

        try:
            await context.bot.send_message(
                chat_id=request.requester_id,
                text=text,
            )
        except Exception as e:
            logger.warning("Failed to notify admin", admin_id=request.requester_id, error=str(e))

    async def handle_callback_query(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Handle inline keyboard callbacks for approve/deny."""
        query = update.callback_query
        await query.answer()

        # Disable private chats for callbacks too
        chat = update.effective_chat
        if chat.type == "private":
            await query.edit_message_text(
                "🚫 Private chats are disabled. Please use the bot in a group forum topic."
            )
            return

        user_id = query.from_user.id
        data = query.data

        if data.startswith("approve_"):
            request_id = data[8:]  # Remove "approve_"
            try:
                success = await self.permissions.approve_request(request_id, user_id)
                if success:
                    await query.edit_message_text(
                        "✅ You have approved the permission request!\n"
                        "The permission has been granted."
                    )
                    await self._notify_admin_of_decision(request_id, True, context)
                else:
                    await query.edit_message_text(
                        "❌ This request is no longer valid or has expired."
                    )
            except Exception as e:
                logger.error("Callback approve error", error=str(e))
                await query.edit_message_text("❌ An error occurred.")

        elif data.startswith("deny_"):
            request_id = data[5:]  # Remove "deny_"
            try:
                success = await self.permissions.deny_request(request_id, user_id)
                if success:
                    await query.edit_message_text(
                        "❌ You have denied the permission request."
                    )
                    await self._notify_admin_of_decision(request_id, False, context)
                else:
                    await query.edit_message_text(
                        "❌ This request is no longer valid or has expired."
                    )
            except Exception as e:
                logger.error("Callback deny error", error=str(e))
                await query.edit_message_text("❌ An error occurred.")