"""Tests for Permission Service."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from src.bot.services.permission_service import PermissionService
from src.bot.models.user import User, UserPermission
from src.bot.models.permission_request import PermissionRequest, PermissionStatus


class TestPermissionService:
    """Test PermissionService functionality."""

    @pytest.fixture
    def permission_service(self):
        """Create PermissionService instance."""
        return PermissionService()

    @pytest.fixture
    def mock_user(self):
        """Create a mock user."""
        return User(
            telegram_id=123456789,
            username="testuser",
            permissions=[UserPermission.AGENT_EXECUTION],
        )

    @pytest.fixture
    def mock_admin_user(self):
        """Create a mock admin user."""
        return User(
            telegram_id=987654321,
            username="admin",
            permissions=[UserPermission.ADMIN_ACCESS],
        )

    @pytest.mark.asyncio
    async def test_get_or_create_user(self, permission_service, mock_user):
        """Test getting or creating a user."""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        mock_collection.insert_one.return_value = MagicMock(inserted_id="test_id")

        permission_service._users_collection = mock_collection

        user = await permission_service.get_or_create_user(
            telegram_id=123456789,
            username="testuser",
            first_name="Test",
        )

        assert user.telegram_id == 123456789
        assert user.username == "testuser"
        mock_collection.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_grant_permission(self, permission_service, mock_user):
        """Test granting a permission."""
        mock_collection = MagicMock()
        # First call returns user without permission, second call returns user with permission
        user_with_perm = User(
            telegram_id=123456789,
            username="testuser",
            permissions=[UserPermission.AGENT_EXECUTION, UserPermission.PREMIUM_FEATURES],
        )
        mock_collection.find_one.side_effect = [mock_user.to_dict(), user_with_perm.to_dict()]
        mock_collection.replace_one.return_value = MagicMock()

        permission_service._users_collection = mock_collection

        result = await permission_service.grant_permission(
            123456789, UserPermission.PREMIUM_FEATURES
        )

        assert result is True
        # Verify replace_one was called with updated user
        mock_collection.replace_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_permission(self, permission_service, mock_user):
        """Test revoking a permission."""
        mock_user.add_permission(UserPermission.PREMIUM_FEATURES)
        from bson import ObjectId
        mock_user._id = ObjectId()

        mock_collection = MagicMock()
        # Return user with permission (with _id)
        user_dict = mock_user.to_dict()
        mock_collection.find_one.return_value = user_dict
        mock_collection.replace_one.return_value = MagicMock()

        permission_service._users_collection = mock_collection

        result = await permission_service.revoke_permission(
            123456789, UserPermission.PREMIUM_FEATURES
        )

        assert result is True
        mock_collection.replace_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_admin_from_config(self, permission_service):
        """Test admin check from config list."""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = None
        permission_service._users_collection = mock_collection

        # User ID in config.admin_user_ids
        result = await permission_service.is_admin(987654321)
        assert result is True

        # User ID not in config
        result = await permission_service.is_admin(111111111)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_admin_from_permission(self, permission_service, mock_admin_user):
        """Test admin check from user permission."""
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = mock_admin_user.to_dict()
        permission_service._users_collection = mock_collection

        # User has ADMIN_ACCESS permission
        result = await permission_service.is_admin(987654321)
        assert result is True

    @pytest.mark.asyncio
    async def test_create_permission_request(self, permission_service, mock_admin_user):
        """Test creating a permission request."""
        # Mock admin check
        with patch.object(permission_service, "is_admin", return_value=True):
            # Mock target user without permission
            mock_target_user = User(telegram_id=123456789)
            mock_users_collection = MagicMock()
            mock_users_collection.find_one.return_value = mock_target_user.to_dict()

            mock_requests_collection = MagicMock()
            mock_requests_collection.find_one.return_value = None  # No existing request
            mock_requests_collection.insert_one.return_value = MagicMock(inserted_id="req_id")

            permission_service._users_collection = mock_users_collection
            permission_service._requests_collection = mock_requests_collection

            request = await permission_service.create_permission_request(
                requester_id=987654321,
                target_user_id=123456789,
                permission=UserPermission.PREMIUM_FEATURES,
                reason="Test reason",
            )

            assert request.requester_id == 987654321
            assert request.target_user_id == 123456789
            assert request.permission == UserPermission.PREMIUM_FEATURES
            assert request.reason == "Test reason"
            assert request.status == PermissionStatus.PENDING
            mock_requests_collection.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_permission_request_not_admin(self, permission_service):
        """Test creating request without admin permission."""
        with patch.object(permission_service, "is_admin", return_value=False):
            with pytest.raises(PermissionError):
                await permission_service.create_permission_request(
                    requester_id=111111111,
                    target_user_id=123456789,
                    permission=UserPermission.PREMIUM_FEATURES,
                )

    @pytest.mark.asyncio
    async def test_approve_request(self, permission_service):
        """Test approving a permission request."""
        from bson import ObjectId
        request_id = str(ObjectId())
        request = PermissionRequest(
            requester_id=987654321,
            target_user_id=123456789,
            permission=UserPermission.PREMIUM_FEATURES,
        )
        request._id = ObjectId(request_id)

        mock_requests_collection = MagicMock()
        mock_requests_collection.find_one.return_value = request.to_dict()
        mock_requests_collection.replace_one.return_value = MagicMock()

        permission_service._requests_collection = mock_requests_collection

        # Mock grant_permission
        with patch.object(permission_service, "grant_permission", return_value=True) as mock_grant:
            result = await permission_service.approve_request(request_id, 123456789)

            assert result is True
            # Check the request passed to replace_one has APPROVED status
            call_args = mock_requests_collection.replace_one.call_args
            saved_request = call_args[0][1]  # Second arg is the document
            assert saved_request["status"] == PermissionStatus.APPROVED.value
            mock_grant.assert_called_once_with(123456789, UserPermission.PREMIUM_FEATURES)

    @pytest.mark.asyncio
    async def test_deny_request(self, permission_service):
        """Test denying a permission request."""
        from bson import ObjectId
        request_id = str(ObjectId())
        request = PermissionRequest(
            requester_id=987654321,
            target_user_id=123456789,
            permission=UserPermission.PREMIUM_FEATURES,
        )
        request._id = ObjectId(request_id)

        mock_requests_collection = MagicMock()
        mock_requests_collection.find_one.return_value = request.to_dict()
        mock_requests_collection.replace_one.return_value = MagicMock()

        permission_service._requests_collection = mock_requests_collection

        result = await permission_service.deny_request(request_id, 123456789)

        assert result is True
        # Check the request passed to replace_one has DENIED status
        call_args = mock_requests_collection.replace_one.call_args
        saved_request = call_args[0][1]  # Second arg is the document
        assert saved_request["status"] == PermissionStatus.DENIED.value

    @pytest.mark.asyncio
    async def test_approve_wrong_user(self, permission_service):
        """Test approving request for wrong user."""
        from bson import ObjectId
        request_id = str(ObjectId())
        request = PermissionRequest(
            requester_id=987654321,
            target_user_id=123456789,
            permission=UserPermission.PREMIUM_FEATURES,
        )
        request._id = ObjectId(request_id)

        mock_requests_collection = MagicMock()
        mock_requests_collection.find_one.return_value = request.to_dict()
        permission_service._requests_collection = mock_requests_collection

        with pytest.raises(PermissionError):
            await permission_service.approve_request(request_id, 999999999)  # Wrong user

    @pytest.mark.asyncio
    async def test_cleanup_expired_requests(self, permission_service):
        """Test cleaning up expired requests."""
        mock_requests_collection = MagicMock()
        mock_requests_collection.update_many.return_value = MagicMock(modified_count=5)
        permission_service._requests_collection = mock_requests_collection

        count = await permission_service.cleanup_expired_requests()

        assert count == 5
        mock_requests_collection.update_many.assert_called_once()