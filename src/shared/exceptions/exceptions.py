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
        raise AppException(error_detail=error, message="The user could not be found")
    """

    def __init__(self, message: str = "An error occurred", error_detail: Optional[ErrorDetail] = None):
        """
        Initialize exception with ErrorDetail.

        Args:
            message: User-friendly error message
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
        raise NotFoundException(
            message="The user you're looking for doesn't exist",
            error_detail=error
        )

        # Or shorthand with default message
        raise NotFoundException()
    """

    def __init__(self, message: str = "The requested resource was not found", error_detail: Optional[ErrorDetail] = None):
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
        raise ValidationException(
            message="Please check your input and try again",
            error_detail=error
        )

        # Or shorthand
        raise ValidationException()
    """

    def __init__(self, message: str = "Please check your input and try again", error_detail: Optional[ErrorDetail] = None):
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
        raise AuthenticationException(
            message="Please log in to continue",
            error_detail=error
        )
    """

    def __init__(self, message: str = "Please log in to continue", error_detail: Optional[ErrorDetail] = None):
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
        raise AuthorizationException(
            message="You don't have permission to perform this action",
            error_detail=error
        )
    """

    def __init__(self, message: str = "You don't have permission to perform this action", error_detail: Optional[ErrorDetail] = None):
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
        raise ConflictException(
            message="This resource already exists",
            error_detail=error
        )
    """

    def __init__(self, message: str = "This resource already exists", error_detail: Optional[ErrorDetail] = None):
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
        raise BadRequestException(
            message="Your request could not be processed",
            error_detail=error
        )
    """

    def __init__(self, message: str = "Your request could not be processed", error_detail: Optional[ErrorDetail] = None):
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
        raise InternalServerException(
            message="Something went wrong. Please try again later",
            error_detail=error
        )
    """

    def __init__(self, message: str = "Something went wrong. Please try again later", error_detail: Optional[ErrorDetail] = None):
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
        raise ServiceUnavailableException(
            message="The service is temporarily unavailable. Please try again later",
            error_detail=error
        )
    """

    def __init__(self, message: str = "The service is temporarily unavailable. Please try again later", error_detail: Optional[ErrorDetail] = None):
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
        message: User-friendly error message
        code: Error code (optional)
        details: List of detail messages (optional)
        field_errors: Field-specific errors (optional)

    Usage:
        raise create_exception(
            'not_found',
            'The user you are looking for does not exist',
            code='USER_NOT_FOUND',
            details=['User with ID 123 does not exist']
        )

        raise create_exception(
            'validation',
            'Please check your input and try again',
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
        title=message if not details else details[0],
        code=code or default_code,
        status=status,
        details=details,
        field_errors=field_errors
    )

    return exception_class(message=message, error_detail=error_detail)