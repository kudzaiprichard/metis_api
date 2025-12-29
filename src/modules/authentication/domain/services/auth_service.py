"""
Simplified authentication service for JWT-based auth.
Handles user registration, login, logout, and token refresh.
"""

from src.modules.authentication.domain.models.user import User
from src.modules.authentication.domain.models.enums import Role, TokenType
from src.modules.authentication.presentation.dtos.auth_dtos import RegisterRequest, UserResponse, LoginRequest, \
    LogoutRequest
from src.modules.authentication.domain.repositories.user_repository import UserRepository
from src.modules.authentication.domain.services.jwt_service import JwtService
from src.shared.exceptions.exceptions import ConflictException, AuthenticationException, NotFoundException
from src.shared.response.error_detail import ErrorDetail


class AuthenticationService:
    """
    Service for handling authentication operations.
    """

    def __init__(self):
        self.user_repository = UserRepository()
        self.jwt_service = JwtService()

    def register(self, request: RegisterRequest) -> tuple[UserResponse, dict]:
        """
        Register a new user and return user data with tokens.

        Args:
            request: RegisterRequest DTO with user data

        Returns:
            Tuple of (UserResponse, tokens_dict)

        Raises:
            ConflictException: If email already exists
        """
        # Check if email already exists
        existing_user = self.user_repository.find_by_email(request.email)
        if existing_user:
            error = ErrorDetail(
                title="Registration Failed",
                code="EMAIL_EXISTS",
                status=409
            )
            error.add_field_error("email", "Email already registered")
            raise ConflictException(error_detail=error)

        # Create user
        user = User(
            email=request.email,
            first_name=request.first_name,
            last_name=request.last_name,
            role=Role[request.role] if request.role else Role.DOCTOR
        )
        user.set_password(request.password)

        # Save user to database
        saved_user = self.user_repository.create(user)

        # Generate JWT tokens
        tokens = self.jwt_service.create_token_pair(saved_user)

        # Convert to response DTO
        user_response = UserResponse.model_validate(saved_user)

        return user_response, tokens

    def login(self, request: LoginRequest) -> tuple[UserResponse, dict]:
        """
        Authenticate user and generate tokens.

        Args:
            request: LoginRequest DTO with credentials

        Returns:
            Tuple of (UserResponse, tokens_dict)

        Raises:
            AuthenticationException: If credentials invalid
        """
        # Find user by email
        user = self.user_repository.find_by_email(request.email)

        # Check credentials
        if not user or not user.check_password(request.password):
            error = ErrorDetail(
                title="Login Failed",
                code="INVALID_CREDENTIALS",
                status=401,
                details=["Invalid email or password"]
            )
            raise AuthenticationException(error_detail=error)

        # Revoke old tokens and generate new ones
        self.jwt_service.revoke_all_user_tokens(user.id)
        tokens = self.jwt_service.create_token_pair(user)

        # Convert to response DTO
        user_response = UserResponse.model_validate(user)

        return user_response, tokens

    def logout(self, request: LogoutRequest) -> None:
        """
        Logout user by revoking all their tokens.

        Args:
            request: LogoutRequest DTO with user_id
        """
        # Revoke all user tokens
        self.jwt_service.revoke_all_user_tokens(request.user_id)

    def logout_with_token(self, token: str) -> None:
        """
        Logout user by revoking a specific token.

        Args:
            token: JWT token string to revoke
        """
        self.jwt_service.revoke_token(token)

    def refresh_token(self, refresh_token: str) -> dict:
        """
        Generate new access token using refresh token.

        Args:
            refresh_token: Valid refresh token string

        Returns:
            New tokens dictionary

        Raises:
            AuthenticationException: If refresh token invalid
        """
        # Verify refresh token
        payload = self.jwt_service.verify_token(refresh_token, TokenType.REFRESH)

        # Get user from token payload
        user = self.user_repository.find_by_id(payload['sub'])
        if not user:
            error = ErrorDetail(
                title="User Not Found",
                code="USER_NOT_FOUND",
                status=404,
                details=["User associated with token not found"]
            )
            raise NotFoundException(error_detail=error)

        # Revoke old refresh token and generate new token pair
        self.jwt_service.revoke_token(refresh_token)
        tokens = self.jwt_service.create_token_pair(user)

        return tokens

    def get_current_user(self, token: str) -> UserResponse:
        """
        Get current user from access token.

        Args:
            token: Valid access token string

        Returns:
            UserResponse DTO

        Raises:
            AuthenticationException: If token invalid
        """
        # Verify access token
        payload = self.jwt_service.verify_token(token, TokenType.ACCESS)

        # Get user from token payload
        user = self.user_repository.find_by_id(payload['sub'])
        if not user:
            error = ErrorDetail(
                title="User Not Found",
                code="USER_NOT_FOUND",
                status=404,
                details=["User associated with token not found"]
            )
            raise NotFoundException(error_detail=error)

        # Convert to response DTO
        return UserResponse.model_validate(user)