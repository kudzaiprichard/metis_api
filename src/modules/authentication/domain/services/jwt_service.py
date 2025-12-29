from datetime import datetime
from typing import Dict, Any
from flask import current_app
import jwt

from src.modules.authentication.domain.models.enums import TokenType
from src.modules.authentication.domain.models.token import Token
from src.modules.authentication.domain.models.user import User
from src.modules.authentication.domain.repositories.token_repository import TokenRepository
from src.shared.exceptions.exceptions import AuthenticationException, ValidationException
from src.shared.response.error_detail import ErrorDetail


class JwtService:
    """
    Service for handling JWT token operations.
    """

    def __init__(self):
        self.token_repository = TokenRepository()

    def create_token_pair(self, user: User) -> Dict[str, Dict[str, Any]]:
        """
        Create both access and refresh tokens for a user.

        Args:
            user: User object

        Returns:
            Dictionary containing both access_token and refresh_token data
        """
        access_token = self._create_token(user, TokenType.ACCESS)
        refresh_token = self._create_token(user, TokenType.REFRESH)

        return {
            'access_token': access_token,
            'refresh_token': refresh_token
        }

    def verify_token(self, token: str, token_type: TokenType) -> Dict[str, Any]:
        """
        Verify and decode a JWT token.

        Args:
            token: JWT token string
            token_type: Type of token (ACCESS or REFRESH)

        Returns:
            Decoded token payload

        Raises:
            AuthenticationException: If token is invalid, expired, or revoked
        """
        try:
            secret_key = self._get_secret_key(token_type)
            algorithm = current_app.config['JWT_ALGORITHM']

            # Decode JWT token
            payload = jwt.decode(token, secret_key, algorithms=[algorithm])

            # Verify token type matches
            if payload.get('type') != token_type.value:
                error = ErrorDetail(
                    title="Invalid Token Type",
                    code="INVALID_TOKEN_TYPE",
                    status=401,
                    details=[f"Expected {token_type.value} token"]
                )
                raise AuthenticationException(error_detail=error)

            # Check token exists in database
            token_record = self.token_repository.find_by_token(token)
            if not token_record:
                error = ErrorDetail(
                    title="Token Not Found",
                    code="TOKEN_NOT_FOUND",
                    status=401,
                    details=["Token does not exist in system"]
                )
                raise AuthenticationException(error_detail=error)

            # Verify token is still valid (not revoked/expired)
            if not token_record.is_valid():
                error = ErrorDetail(
                    title="Token Invalid",
                    code="TOKEN_REVOKED",
                    status=401,
                    details=["Token has been revoked or expired"]
                )
                raise AuthenticationException(error_detail=error)

            return payload

        except jwt.ExpiredSignatureError:
            # Mark token as expired in database
            token_record = self.token_repository.find_by_token(token)
            if token_record:
                token_record.is_expired = True
                self.token_repository.update(token_record)

            error = ErrorDetail(
                title="Token Expired",
                code="TOKEN_EXPIRED",
                status=401,
                details=["Token has expired, please login again"]
            )
            raise AuthenticationException(error_detail=error)

        except jwt.InvalidSignatureError:
            error = ErrorDetail(
                title="Invalid Token Signature",
                code="INVALID_SIGNATURE",
                status=401,
                details=["Token signature is invalid"]
            )
            raise AuthenticationException(error_detail=error)

        except jwt.DecodeError:
            error = ErrorDetail(
                title="Token Decode Error",
                code="TOKEN_DECODE_ERROR",
                status=401,
                details=["Token could not be decoded"]
            )
            raise AuthenticationException(error_detail=error)

        except jwt.InvalidTokenError as e:
            error = ErrorDetail(
                title="Invalid Token",
                code="INVALID_TOKEN",
                status=401,
                details=[f"Token validation failed: {str(e)}"]
            )
            raise AuthenticationException(error_detail=error)

        except AuthenticationException:
            # Re-raise our custom exceptions
            raise

        except Exception as e:
            # Catch any unexpected errors
            error = ErrorDetail(
                title="Token Validation Failed",
                code="TOKEN_VALIDATION_ERROR",
                status=401,
                details=[f"Unexpected error during token validation: {str(e)}"]
            )
            raise AuthenticationException(error_detail=error)

    def revoke_token(self, token: str) -> None:
        """
        Revoke a token.

        Args:
            token: Token string to revoke

        Raises:
            ValidationException: If token not found
        """
        token_record = self.token_repository.find_by_token(token)

        if not token_record:
            error = ErrorDetail(
                title="Token Not Found",
                code="TOKEN_NOT_FOUND",
                status=400,
                details=["Token does not exist"]
            )
            raise ValidationException(error_detail=error)

        token_record.is_revoked = True
        self.token_repository.update(token_record)

    def revoke_all_user_tokens(self, user_id: str) -> None:
        """
        Revoke all tokens for a user.

        Args:
            user_id: User ID
        """
        # Find all tokens for the user
        tokens = self.token_repository.find_many_by({'user_id': user_id})

        # Filter and mark tokens that need to be revoked
        tokens_to_revoke = []
        for token in tokens:
            if not token.is_revoked:
                token.is_revoked = True
                tokens_to_revoke.append(token)

        # Bulk update all tokens in a single transaction
        if tokens_to_revoke:
            self.token_repository.update_many(tokens_to_revoke)

    def refresh_access_token(self, refresh_token: str, user: User) -> Dict[str, Any]:
        """
        Generate a new access token using a valid refresh token.

        Args:
            refresh_token: Valid refresh token string
            user: User object

        Returns:
            New access token data

        Raises:
            AuthenticationException: If refresh token is invalid
        """
        # Verify the refresh token
        payload = self.verify_token(refresh_token, TokenType.REFRESH)

        # Ensure token belongs to the user
        if payload.get('sub') != user.id:
            error = ErrorDetail(
                title="Token Mismatch",
                code="TOKEN_USER_MISMATCH",
                status=401,
                details=["Token does not belong to this user"]
            )
            raise AuthenticationException(error_detail=error)

        # Create new access token
        return self._create_token(user, TokenType.ACCESS)

    def _create_token(self, user: User, token_type: TokenType) -> Dict[str, Any]:
        """
        Create a token for a user.

        Args:
            user: User object
            token_type: Type of token to create

        Returns:
            Dictionary containing token data
        """
        import uuid

        secret_key = self._get_secret_key(token_type)
        algorithm = current_app.config['JWT_ALGORITHM']

        # Determine expiration based on token type
        if token_type == TokenType.ACCESS:
            expires_delta = current_app.config['JWT_ACCESS_TOKEN_EXPIRES']
        else:
            expires_delta = current_app.config['JWT_REFRESH_TOKEN_EXPIRES']

        issued_at = datetime.now()
        expires_at = issued_at + expires_delta

        # Generate unique JWT ID
        jti = str(uuid.uuid4())

        # Build JWT payload
        payload = {
            'sub': user.id,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'role': user.role.value,
            'type': token_type.value,
            'iat': int(issued_at.timestamp()),
            'exp': int(expires_at.timestamp()),
            'jti': jti
        }

        # Encode JWT token
        token_string = jwt.encode(payload, secret_key, algorithm=algorithm)

        # Create token record in database
        token_record = Token(
            user_id=user.id,
            token=token_string,
            token_type=token_type,
            expires_at=expires_at,
            is_expired=False,
            is_revoked=False
        )
        self.token_repository.create(token_record)

        return {
            'token': token_string,
            'token_type': 'Bearer',
            'expires_at': expires_at.isoformat(),
            'created_at': issued_at.isoformat()
        }

    def _get_secret_key(self, token_type: TokenType) -> str:
        """Get the appropriate secret key based on token type."""
        if token_type == TokenType.ACCESS:
            return current_app.config['JWT_SECRET_KEY']
        return current_app.config['JWT_REFRESH_SECRET_KEY']