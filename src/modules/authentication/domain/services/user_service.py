"""
User management service for CRUD operations.
Handles creating, reading, updating, and deleting users.
"""

from typing import List, Tuple

from src.modules.authentication.domain.models.user import User
from src.modules.authentication.domain.models.enums import Role
from src.modules.authentication.presentation.dtos.user_dtos import CreateUserRequest, UserResponse, \
    GetUserRequest, UpdateUserRequest, DeleteUserRequest, ListUsersRequest
from src.modules.authentication.domain.repositories.user_repository import UserRepository
from src.shared.exceptions.exceptions import ConflictException, NotFoundException, ValidationException

from src.shared.response.error_detail import ErrorDetail


class UserService:
    """
    Service for user CRUD operations.
    """

    def __init__(self):
        self.user_repository = UserRepository()

    def create_user(self, request: CreateUserRequest) -> UserResponse:
        """
        Create a new user.

        Args:
            request: CreateUserRequest DTO

        Returns:
            UserResponse DTO

        Raises:
            ConflictException: If email already exists
        """
        # Check if email already exists
        existing_user = self.user_repository.find_by_email(request.email)
        if existing_user:
            error = ErrorDetail(
                title="User Creation Failed",
                code="EMAIL_EXISTS",
                status=409
            )
            error.add_field_error("email", "Email already registered")
            raise ConflictException(
                message="This email is already registered",
                error_detail=error
            )

        # Create user
        user = User(
            email=request.email,
            first_name=request.first_name,
            last_name=request.last_name,
            role=Role[request.role]
        )
        user.set_password(request.password)

        # Save to database
        saved_user = self.user_repository.create(user)

        # Convert to response DTO
        return UserResponse.model_validate(saved_user)

    def get_user(self, request: GetUserRequest) -> UserResponse:
        """
        Get a single user by ID.

        Args:
            request: GetUserRequest DTO

        Returns:
            UserResponse DTO

        Raises:
            NotFoundException: If user not found
        """
        user = self.user_repository.find_by_id(request.user_id)

        if not user:
            error = ErrorDetail(
                title="User Not Found",
                code="USER_NOT_FOUND",
                status=404,
                details=[f"User with ID {request.user_id} does not exist"]
            )
            raise NotFoundException(
                message="The user you're looking for doesn't exist",
                error_detail=error
            )

        return UserResponse.model_validate(user)

    def update_user(self, request: UpdateUserRequest) -> UserResponse:
        """
        Update user details.

        Args:
            request: UpdateUserRequest DTO

        Returns:
            UserResponse DTO

        Raises:
            NotFoundException: If user not found
            ConflictException: If email already exists
        """
        # Find user
        user = self.user_repository.find_by_id(request.user_id)
        if not user:
            error = ErrorDetail(
                title="User Not Found",
                code="USER_NOT_FOUND",
                status=404,
                details=[f"User with ID {request.user_id} does not exist"]
            )
            raise NotFoundException(
                message="The user you're trying to update doesn't exist",
                error_detail=error
            )

        # Check email uniqueness if email is being updated
        if request.email and request.email != user.email:
            existing = self.user_repository.find_by_email(request.email)
            if existing:
                error = ErrorDetail(
                    title="Update Failed",
                    code="EMAIL_EXISTS",
                    status=409
                )
                error.add_field_error("email", "Email already in use")
                raise ConflictException(
                    message="This email is already in use by another user",
                    error_detail=error
                )
            user.email = request.email

        # Update fields
        if request.first_name:
            user.first_name = request.first_name

        if request.last_name:
            user.last_name = request.last_name

        if request.role:
            user.role = Role[request.role]

        if request.password:
            user.set_password(request.password)

        # Save changes
        updated_user = self.user_repository.update(user)

        return UserResponse.model_validate(updated_user)

    def delete_user(self, request: DeleteUserRequest) -> None:
        """
        Soft delete a user.

        Args:
            request: DeleteUserRequest DTO

        Raises:
            NotFoundException: If user not found
        """
        user = self.user_repository.find_by_id(request.user_id)

        if not user:
            error = ErrorDetail(
                title="User Not Found",
                code="USER_NOT_FOUND",
                status=404,
                details=[f"User with ID {request.user_id} does not exist"]
            )
            raise NotFoundException(
                message="The user you're trying to delete doesn't exist",
                error_detail=error
            )

        # Soft delete
        self.user_repository.delete(user)

    def list_users(self, request: ListUsersRequest) -> Tuple[List[UserResponse], int]:
        """
        List users with pagination and optional filters.

        Args:
            request: ListUsersRequest DTO

        Returns:
            Tuple of (list of UserResponse DTOs, total count)
        """
        # Build filter dictionary
        filters = {}
        if request.role:
            filters['role'] = Role[request.role]

        # Get total count
        total = self.user_repository.count(filters)

        # Get paginated users
        pagination = self.user_repository.paginate(
            page=request.page,
            per_page=request.per_page,
            include_deleted=False
        )

        # Apply role filter if specified
        if request.role:
            users = [u for u in pagination.items if u.role == Role[request.role]]
        else:
            users = pagination.items

        # Apply search filter if specified
        if request.search:
            search_lower = request.search.lower()
            users = [
                u for u in users
                if search_lower in u.email.lower() or
                   search_lower in u.first_name.lower() or
                   search_lower in u.last_name.lower()
            ]

        # Convert to response DTOs
        user_responses = [UserResponse.model_validate(user) for user in users]

        return user_responses, total

    def get_user_by_email(self, email: str) -> UserResponse:
        """
        Get user by email address.

        Args:
            email: User email address

        Returns:
            UserResponse DTO

        Raises:
            NotFoundException: If user not found
        """
        user = self.user_repository.find_by_email(email)

        if not user:
            error = ErrorDetail(
                title="User Not Found",
                code="USER_NOT_FOUND",
                status=404,
                details=[f"User with email {email} does not exist"]
            )
            raise NotFoundException(
                message="No user found with this email address",
                error_detail=error
            )

        return UserResponse.model_validate(user)

    def restore_user(self, user_id: str) -> UserResponse:
        """
        Restore a soft-deleted user.

        Args:
            user_id: User ID to restore

        Returns:
            UserResponse DTO

        Raises:
            NotFoundException: If user not found
        """
        # Find user including deleted ones
        user = self.user_repository.find_by_id(user_id, include_deleted=True)

        if not user:
            error = ErrorDetail(
                title="User Not Found",
                code="USER_NOT_FOUND",
                status=404,
                details=[f"User with ID {user_id} does not exist"]
            )
            raise NotFoundException(
                message="The user you're trying to restore doesn't exist",
                error_detail=error
            )

        # Check if already active
        if not user.is_deleted:
            error = ErrorDetail(
                title="User Already Active",
                code="USER_ACTIVE",
                status=400,
                details=["User is not deleted"]
            )
            raise ValidationException(
                message="This user is already active",
                error_detail=error
            )

        # Restore user
        restored_user = self.user_repository.restore(user)

        return UserResponse.model_validate(restored_user)