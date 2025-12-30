"""
Generic authentication interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ITokenVerifier(ABC):
    """
    Interface for token verification.
    Implement this to adapt your token verification logic.
    """

    @abstractmethod
    def verify_access_token(self, token: str) -> Dict[str, Any]:
        """
        Verify an access token and return its payload.

        Args:
            token: JWT token string

        Returns:
            Dictionary containing token payload with user info

        Raises:
            AuthenticationException: If token is invalid, expired, or revoked

        Expected payload structure:
            {
                'sub': 'user_id',
                'email': 'user@example.com',
                'first_name': 'John',
                'last_name': 'Doe',
                'role': 'ADMIN',
                'jti': 'token_id',
                'iat': 1234567890,
                'exp': 1234567890
            }
        """
        pass

    @abstractmethod
    def verify_refresh_token(self, token: str) -> Dict[str, Any]:
        """
        Verify a refresh token and return its payload.

        Args:
            token: JWT refresh token string

        Returns:
            Dictionary containing token payload with user info

        Raises:
            AuthenticationException: If token is invalid, expired, or revoked

        Expected payload structure: Same as verify_access_token
        """
        pass


class IUserProvider(ABC):
    """
    Interface for user data provider.
    Implement this to adapt your user repository/database.
    """

    @abstractmethod
    def get_user_by_id(self, user_id: str) -> Optional[Any]:
        """
        Get user by ID from your data source.

        Args:
            user_id: Unique user identifier

        Returns:
            User object if found, None otherwise

        Note:
            The returned user object should have basic attributes like:
            - id
            - email
            - first_name, last_name
            - role
            - Any other attributes you need
        """
        pass