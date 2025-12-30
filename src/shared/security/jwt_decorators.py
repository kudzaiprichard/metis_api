"""
Generic JWT authentication decorators.
"""

from functools import wraps
from flask import request, g
from typing import Optional, Callable

from src.shared.exceptions.exceptions import AuthenticationException, AuthorizationException
from src.shared.response.error_detail import ErrorDetail
from src.shared.security.interfaces import ITokenVerifier, IUserProvider


class JwtAuthDecorators:
    """
    JWT authentication and authorization decorators.

    Works with any token verification and user provider implementation.

    Usage:
        # In your app setup
        token_verifier = YourTokenVerifier()
        user_provider = YourUserProvider()

        jwt_auth = JwtAuthDecorators(
            token_verifier=token_verifier,
            user_provider=user_provider
        )

        # In your routes
        @jwt_auth.jwt_access_required
        def protected_route():
            user_id = jwt_auth.get_current_user_id()
            return {'user_id': user_id}
    """

    def __init__(
            self,
            token_verifier: ITokenVerifier,
            user_provider: IUserProvider
    ):
        """
        Initialize JWT auth decorators with dependency injection.

        Args:
            token_verifier: Implementation of ITokenVerifier interface
            user_provider: Implementation of IUserProvider interface
        """
        self.token_verifier = token_verifier
        self.user_provider = user_provider

    def jwt_access_required(self, f: Callable) -> Callable:
        """
        Decorator to require valid JWT access token.

        Extracts token from Authorization header, verifies it,
        and stores user info in Flask's g object for request context.

        Usage:
            @jwt_auth.jwt_access_required
            def protected_route():
                user_id = jwt_auth.get_current_user_id()
                return {'user_id': user_id}

        Raises:
            AuthenticationException: If token is missing, invalid, or expired
        """

        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract token from Authorization header
            token = self._extract_token_from_header()

            # Verify token using injected verifier
            payload = self.token_verifier.verify_access_token(token)

            # Store user info in Flask's g object for request context
            self._store_user_in_context(payload, token)

            return f(*args, **kwargs)

        return decorated_function

    def jwt_refresh_required(self, f: Callable) -> Callable:
        """
        Decorator to require valid JWT refresh token.

        Extracts token from Authorization header, verifies it as refresh token,
        and stores user info in Flask's g object.

        Usage:
            @jwt_auth.jwt_refresh_required
            def refresh_route():
                user_id = jwt_auth.get_current_user_id()
                return {'user_id': user_id}

        Raises:
            AuthenticationException: If token is missing, invalid, or expired
        """

        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract token from Authorization header
            token = self._extract_token_from_header()

            # Verify token using injected verifier
            payload = self.token_verifier.verify_refresh_token(token)

            # Store user info in Flask's g object for request context
            self._store_user_in_context(payload, token)

            return f(*args, **kwargs)

        return decorated_function

    def jwt_required(self, f: Callable) -> Callable:
        """
        Alias for jwt_access_required (backward compatibility).

        Deprecated: Use jwt_access_required instead for clarity.
        """
        return self.jwt_access_required(f)

    def role_required(self, *allowed_roles: str) -> Callable:
        """
        Decorator to require specific role(s).
        Must be used together with @jwt_access_required or @jwt_refresh_required.

        Args:
            *allowed_roles: One or more role values that are allowed
                           (e.g., 'ADMIN', 'USER', 'DOCTOR', 'ML_ENGINEER')

        Usage:
            @jwt_auth.jwt_access_required
            @jwt_auth.role_required('ADMIN', 'MODERATOR')
            def admin_route():
                return {'message': 'Admin access granted'}

        Raises:
            AuthenticationException: If user is not authenticated
            AuthorizationException: If user doesn't have required role
        """

        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Check if user is authenticated
                if not hasattr(g, 'current_user_id') or g.current_user_id is None:
                    self._raise_authentication_required()

                # Get current user role
                current_role = getattr(g, 'current_user_role', None)

                if not current_role:
                    self._raise_role_not_found()

                # Check if user has required role (case-insensitive)
                normalized_allowed = [role.upper() for role in allowed_roles]

                if current_role.upper() not in normalized_allowed:
                    self._raise_insufficient_permissions(allowed_roles, current_role)

                return f(*args, **kwargs)

            return decorated_function

        return decorator

    # ============ Helper Methods ============

    def get_current_token(self) -> Optional[str]:
        """
        Get the current JWT token string from request context.

        Returns:
            Token string if authenticated, None otherwise
        """
        return getattr(g, 'current_token', None)

    def get_current_user(self):
        """
        Get current authenticated user object from database.
        Fetches fresh data from the data source via user provider.

        Returns:
            User object if authenticated, None otherwise
        """
        user_id = self.get_current_user_id()
        if not user_id:
            return None

        return self.user_provider.get_user_by_id(user_id)

    def get_current_user_id(self) -> Optional[str]:
        """
        Get current user ID from request context (lightweight, no DB query).

        Returns:
            User ID string if authenticated, None otherwise
        """
        return getattr(g, 'current_user_id', None)

    def get_current_user_email(self) -> Optional[str]:
        """
        Get current user email from request context (lightweight).

        Returns:
            User email string if authenticated, None otherwise
        """
        return getattr(g, 'current_user_email', None)

    def get_current_user_role(self) -> Optional[str]:
        """
        Get current user role from request context (lightweight).

        Returns:
            User role string if authenticated, None otherwise
        """
        return getattr(g, 'current_user_role', None)

    def get_current_user_first_name(self) -> Optional[str]:
        """
        Get current user first name from request context (lightweight).

        Returns:
            User first name string if authenticated, None otherwise
        """
        return getattr(g, 'current_user_first_name', None)

    def get_current_user_last_name(self) -> Optional[str]:
        """
        Get current user last name from request context (lightweight).

        Returns:
            User last name string if authenticated, None otherwise
        """
        return getattr(g, 'current_user_last_name', None)

    def get_current_user_name(self) -> Optional[str]:
        """
        Get current user full name from request context (lightweight).

        Returns:
            User full name string if authenticated, None otherwise
        """
        first = getattr(g, 'current_user_first_name', '')
        last = getattr(g, 'current_user_last_name', '')
        return f"{first} {last}".strip() if first or last else None

    def is_authenticated(self) -> bool:
        """
        Check if current request is authenticated.

        Returns:
            True if authenticated, False otherwise
        """
        return self.get_current_user_id() is not None

    def has_role(self, role_value: str) -> bool:
        """
        Check if current user has a specific role.

        Args:
            role_value: The role value to check against

        Returns:
            True if current user has the specified role, False otherwise
        """
        current_role = self.get_current_user_role()
        if not current_role:
            return False
        return current_role.upper() == role_value.upper()

    # ============ Private Helper Methods ============

    def _extract_token_from_header(self) -> str:
        """
        Extract JWT token from Authorization header.

        Returns:
            Token string

        Raises:
            AuthenticationException: If header is missing or malformed
        """
        # Import here to avoid circular dependency

        auth_header = request.headers.get('Authorization', '')

        if not auth_header:
            error = ErrorDetail(
                title="Authorization Required",
                code="MISSING_AUTH_HEADER",
                status=401,
                details=["Authorization header is missing"]
            )
            raise AuthenticationException(error_detail=error)

        if not auth_header.startswith('Bearer '):
            error = ErrorDetail(
                title="Invalid Authorization Header",
                code="INVALID_AUTH_FORMAT",
                status=401,
                details=["Authorization header must start with 'Bearer '"]
            )
            raise AuthenticationException(error_detail=error)

        token = auth_header.replace('Bearer ', '', 1).strip()

        if not token:
            error = ErrorDetail(
                title="Token Missing",
                code="MISSING_TOKEN",
                status=401,
                details=["Token is missing from Authorization header"]
            )
            raise AuthenticationException(error_detail=error)

        return token

    def _store_user_in_context(self, payload: dict, token: str) -> None:
        """
        Store user information from token payload in Flask's g object.

        Args:
            payload: Token payload dictionary
            token: Original token string
        """
        g.current_user_id = payload.get('sub')
        g.current_user_email = payload.get('email')
        g.current_user_first_name = payload.get('first_name')
        g.current_user_last_name = payload.get('last_name')
        g.current_user_role = payload.get('role')
        g.token_jti = payload.get('jti')
        g.token_iat = payload.get('iat')
        g.token_exp = payload.get('exp')
        g.current_token = token

    def _raise_authentication_required(self) -> None:
        """Raise exception for missing authentication."""

        error = ErrorDetail(
            title="Authentication Required",
            code="AUTH_REQUIRED",
            status=401,
            details=["You must be authenticated. Use @jwt_access_required decorator first."]
        )
        raise AuthenticationException(error_detail=error)

    def _raise_role_not_found(self) -> None:
        """Raise exception for missing role information."""

        error = ErrorDetail(
            title="Role Not Found",
            code="ROLE_NOT_FOUND",
            status=403,
            details=["User role information is missing"]
        )
        raise AuthorizationException(error_detail=error)

    def _raise_insufficient_permissions(self, allowed_roles: tuple, current_role: str) -> None:
        """Raise exception for insufficient permissions."""

        error = ErrorDetail(
            title="Access Denied",
            code="INSUFFICIENT_PERMISSIONS",
            status=403,
            details=[f"Required role(s): {', '.join(allowed_roles)}. Your role: {current_role}"]
        )
        raise AuthorizationException(error_detail=error)