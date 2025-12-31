from typing import TypeVar, Generic, Optional, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel

from src.shared.response.error_detail import ErrorDetail

T = TypeVar('T')


@dataclass(frozen=True)
class ApiResponse(Generic[T]):
    """
    Generic API response wrapper for consistent service responses.
    Contains either success data or error information, never both.

    Works seamlessly with both Pydantic models and regular dataclasses.

    Usage:
        # Success response with Pydantic model
        user_dto = UserResponse.from_orm(user)
        response = ApiResponse.success(user_dto, "User created successfully")

        # Failure response
        error = ErrorDetail(
            title="Validation Failed",
            code="VALIDATION_ERROR",
            status=400
        )
        response = ApiResponse.failure(error, "Failed to create user")

        # Check response
        if response.is_success():
            process_data(response.value)
        else:
            handle_error(response.error)

        # Return to client
        return jsonify(response.to_dict()), response.get_status_code()
    """
    error: Optional[ErrorDetail] = None
    message: Optional[str] = None
    value: Optional[T] = None

    def __post_init__(self):
        """Validate that response doesn't have both error and value."""
        if self.error is not None and self.value is not None:
            raise ValueError("ApiResponse cannot have both error and value")

    def is_success(self) -> bool:
        """Check if response represents a successful operation."""
        return self.error is None and self.value is not None

    def is_failure(self) -> bool:
        """Check if response represents a failed operation."""
        return self.error is not None

    def has_message(self) -> bool:
        """Check if response has a message."""
        return self.message is not None and self.message.strip() != ""

    def get_status_code(self) -> int:
        """Get HTTP status code from error or default to 200 for success."""
        if self.error:
            return self.error.status
        return 200 if self.is_success() else 400

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary format.
        Handles both Pydantic models (with .dict()) and regular objects (with .to_dict()).
        """
        result: Dict[str, Any] = {'success': self.is_success()}

        if self.error:
            result['error'] = self.error.to_dict()

        if self.message:
            result['message'] = self.message

        if self.value is not None:
            # Handle Pydantic models
            if isinstance(self.value, BaseModel):
                result['value'] = self.value.dict()
            # Handle objects with to_dict method
            elif hasattr(self.value, 'to_dict'):
                result['value'] = self.value.to_dict()
            # Handle primitive types and lists
            else:
                result['value'] = self.value

        return result

    def __bool__(self) -> bool:
        """Allow if response: checks to verify success."""
        return self.is_success()

    @classmethod
    def success(cls, value: T, message: Optional[str] = None) -> 'ApiResponse[T]':
        """
        Create a successful response.

        Args:
            value: Response data (can be Pydantic model, dict, list, etc.)
            message: Optional success message

        Returns:
            ApiResponse with success data

        Usage:
            # With Pydantic model
            user_dto = UserResponse.from_orm(user)
            response = ApiResponse.success(user_dto, "User created successfully")

            # With dict
            response = ApiResponse.success({"status": "ok"}, "Operation completed")

            # With list
            users = [UserResponse.from_orm(u) for u in user_list]
            response = ApiResponse.success(users, "Users retrieved")
        """
        return cls(value=value, message=message)

    @classmethod
    def failure(cls, error: ErrorDetail, message: str) -> 'ApiResponse[T]':
        """
        Create a failure response.

        Args:
            error: Error detail object
            message: User-friendly error message

        Returns:
            ApiResponse with error information

        Usage:
            # Validation error
            error = ErrorDetail(
                title="Validation Failed",
                code="VALIDATION_ERROR",
                status=400
            )
            error.add_field_error("email", "Invalid format")
            response = ApiResponse.failure(error, "Validation failed")

            # Business logic error
            error = ErrorDetail(
                title="Not Found",
                code="USER_NOT_FOUND",
                status=404,
                details=["User with ID 123 not found"]
            )
            response = ApiResponse.failure(error)
        """
        return cls(error=error, message=message)