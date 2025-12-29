"""
Simplified exception system for API responses.
Each exception wraps an ErrorDetail object for consistent error handling.
"""

from typing import Optional, List, Dict
from src.shared.response.error_detail import ErrorDetail


class AppException(Exception):
    """
    Base exception for all application errors.
    Wraps ErrorDetail for consistent API error responses.

    Usage:
        error = ErrorDetail(
            title="User Not Found",
            code="USER_NOT_FOUND",
            status=404,
            details=["User with ID 123 does not exist"]
        )
        raise AppException(error_detail=error)
    """

    def __init__(self, message: str = "", error_detail: Optional[ErrorDetail] = None):
        """
        Initialize exception with ErrorDetail.

        Args:
            message: Short error message (optional, for logging)
            error_detail: ErrorDetail object with full error information
        """
        self.message = message
        self.error_detail = error_detail or ErrorDetail(
            title="Application Error",
            code="APP_ERROR",
            status=500,
            details=[message] if message else []
        )
        super().__init__(self.message)


class NotFoundException(AppException):
    """
    Raised when a requested resource is not found (404).

    Usage:
        error = ErrorDetail(
            title="User Not Found",
            code="USER_NOT_FOUND",
            status=404,
            details=[f"User with ID {user_id} not found"]
        )
        raise NotFoundException(error_detail=error)

        # Or shorthand
        raise NotFoundException(f"User {user_id} not found")
    """

    def __init__(self, message: str = "Resource not found", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(
                title="Not Found",
                code="NOT_FOUND",
                status=404,
                details=[message]
            )
        super().__init__(message, error_detail)


class ValidationException(AppException):
    """
    Raised when input validation fails (400).

    Usage:
        # With field errors (from Pydantic)
        error = ErrorDetail(
            title="Validation Failed",
            code="VALIDATION_ERROR",
            status=400
        )
        error.add_field_error("email", "Invalid format")
        error.add_field_error("password", "Too short")
        raise ValidationException(error_detail=error)

        # Or shorthand
        raise ValidationException("Invalid input data")
    """

    def __init__(self, message: str = "Validation failed", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(
                title="Validation Failed",
                code="VALIDATION_ERROR",
                status=400,
                details=[message]
            )
        super().__init__(message, error_detail)


class AuthenticationException(AppException):
    """
    Raised when authentication fails (401).

    Usage:
        error = ErrorDetail(
            title="Authentication Failed",
            code="INVALID_CREDENTIALS",
            status=401,
            details=["Invalid email or password"]
        )
        raise AuthenticationException(error_detail=error)
    """

    def __init__(self, message: str = "Authentication required", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(
                title="Authentication Failed",
                code="AUTH_FAILED",
                status=401,
                details=[message]
            )
        super().__init__(message, error_detail)


class AuthorizationException(AppException):
    """
    Raised when user doesn't have permission (403).

    Usage:
        error = ErrorDetail(
            title="Access Denied",
            code="INSUFFICIENT_PERMISSIONS",
            status=403,
            details=["Admin access required"]
        )
        raise AuthorizationException(error_detail=error)
    """

    def __init__(self, message: str = "Access forbidden", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(
                title="Access Denied",
                code="FORBIDDEN",
                status=403,
                details=[message]
            )
        super().__init__(message, error_detail)


class ConflictException(AppException):
    """
    Raised when operation conflicts with current state (409).

    Usage:
        error = ErrorDetail(
            title="Email Already Exists",
            code="EMAIL_CONFLICT",
            status=409
        )
        error.add_field_error("email", "Email already registered")
        raise ConflictException(error_detail=error)
    """

    def __init__(self, message: str = "Resource conflict", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(
                title="Conflict",
                code="CONFLICT",
                status=409,
                details=[message]
            )
        super().__init__(message, error_detail)


class BadRequestException(AppException):
    """
    Raised when request is malformed (400).

    Usage:
        error = ErrorDetail(
            title="Bad Request",
            code="INVALID_REQUEST",
            status=400,
            details=["Missing required field: user_id"]
        )
        raise BadRequestException(error_detail=error)
    """

    def __init__(self, message: str = "Bad request", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(
                title="Bad Request",
                code="BAD_REQUEST",
                status=400,
                details=[message]
            )
        super().__init__(message, error_detail)


class InternalServerException(AppException):
    """
    Raised when internal server error occurs (500).

    Usage:
        error = ErrorDetail(
            title="Database Error",
            code="DB_CONNECTION_FAILED",
            status=500,
            details=["Could not connect to database"]
        )
        raise InternalServerException(error_detail=error)
    """

    def __init__(self, message: str = "Internal server error", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(
                title="Internal Server Error",
                code="INTERNAL_ERROR",
                status=500,
                details=[message]
            )
        super().__init__(message, error_detail)


class ServiceUnavailableException(AppException):
    """
    Raised when service is temporarily unavailable (503).

    Usage:
        error = ErrorDetail(
            title="Service Unavailable",
            code="SERVICE_DOWN",
            status=503,
            details=["Payment service is temporarily unavailable"]
        )
        raise ServiceUnavailableException(error_detail=error)
    """

    def __init__(self, message: str = "Service unavailable", error_detail: Optional[ErrorDetail] = None):
        if error_detail is None:
            error_detail = ErrorDetail(
                title="Service Unavailable",
                code="SERVICE_UNAVAILABLE",
                status=503,
                details=[message]
            )
        super().__init__(message, error_detail)


# Utility function for quick exception creation
def create_exception(
    exception_type: str,
    message: str,
    code: Optional[str] = None,
    details: Optional[List[str]] = None,
    field_errors: Optional[Dict[str, List[str]]] = None
) -> AppException:
    """
    Factory function to create exceptions quickly.

    Args:
        exception_type: Type of exception ('not_found', 'validation', 'auth', etc.)
        message: Error message
        code: Error code (optional)
        details: List of detail messages (optional)
        field_errors: Field-specific errors (optional)

    Usage:
        raise create_exception(
            'not_found',
            'User not found',
            code='USER_NOT_FOUND',
            details=['User with ID 123 does not exist']
        )

        raise create_exception(
            'validation',
            'Validation failed',
            code='VALIDATION_ERROR',
            field_errors={'email': ['Invalid format']}
        )
    """
    exception_map = {
        'not_found': (NotFoundException, 404, "NOT_FOUND"),
        'validation': (ValidationException, 400, "VALIDATION_ERROR"),
        'auth': (AuthenticationException, 401, "AUTH_FAILED"),
        'authorization': (AuthorizationException, 403, "FORBIDDEN"),
        'conflict': (ConflictException, 409, "CONFLICT"),
        'bad_request': (BadRequestException, 400, "BAD_REQUEST"),
        'internal': (InternalServerException, 500, "INTERNAL_ERROR"),
        'unavailable': (ServiceUnavailableException, 503, "SERVICE_UNAVAILABLE"),
    }

    exception_class, status, default_code = exception_map.get(
        exception_type,
        (AppException, 500, "APP_ERROR")
    )

    error_detail = ErrorDetail(
        title=message,
        code=code or default_code,
        status=status,
        details=details,
        field_errors=field_errors
    )

    return exception_class(message=message, error_detail=error_detail)