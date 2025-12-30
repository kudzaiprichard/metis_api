"""
Project-specific implementations of auth interfaces.
Adapts your JwtService and UserRepository to work with generic JwtAuthDecorators.
"""

from typing import Dict, Any, Optional

from src.modules.authentication.domain.models.user import User
from src.modules.authentication.domain.models.enums import TokenType
from src.modules.authentication.domain.services.jwt_service import JwtService
from src.modules.authentication.domain.repositories.user_repository import UserRepository
from src.shared.security.interfaces import ITokenVerifier, IUserProvider


class JwtTokenVerifier(ITokenVerifier):
    """
    Adapts JwtService to ITokenVerifier interface.

    This adapter translates between the generic interface
    and your specific JWT service implementation.
    """

    def __init__(self, jwt_service: JwtService):
        """
        Initialize token verifier with JWT service.

        Args:
            jwt_service: Your project's JWT service instance
        """
        self.jwt_service = jwt_service

    def verify_access_token(self, token: str) -> Dict[str, Any]:
        """
        Verify an access token using your JWT service.

        Args:
            token: JWT access token string

        Returns:
            Dictionary containing token payload with user info

        Raises:
            AuthenticationException: If token is invalid, expired, or revoked
        """
        # Delegate to your JWT service
        payload = self.jwt_service.verify_token(token, TokenType.ACCESS)
        return payload

    def verify_refresh_token(self, token: str) -> Dict[str, Any]:
        """
        Verify a refresh token using your JWT service.

        Args:
            token: JWT refresh token string

        Returns:
            Dictionary containing token payload with user info

        Raises:
            AuthenticationException: If token is invalid, expired, or revoked
        """
        # Delegate to your JWT service
        payload = self.jwt_service.verify_token(token, TokenType.REFRESH)
        return payload


class DatabaseUserProvider(IUserProvider):
    """
    Adapts UserRepository to IUserProvider interface.

    This adapter translates between the generic interface
    and your specific user repository implementation.
    """

    def __init__(self, user_repository: UserRepository):
        """
        Initialize user provider with user repository.

        Args:
            user_repository: Your project's user repository instance
        """
        self.user_repository = user_repository

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID from database.

        Args:
            user_id: Unique user identifier

        Returns:
            User object if found, None otherwise
        """
        # Delegate to your user repository
        return self.user_repository.find_by_id(user_id)