from typing import TypeVar, Generic, Optional, List, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel

from src.shared.response.error_detail import ErrorDetail
from src.shared.response.api_response import ApiResponse

T = TypeVar('T')


@dataclass(frozen=True)
class PaginatedResponse(ApiResponse[List[T]]):
    """
    Specialized response for paginated data.
    Extends ApiResponse with pagination metadata.
    Works seamlessly with Pydantic models.

    Usage:
        # Success with pagination (Pydantic models)
        users = [UserResponse.from_orm(u) for u in user_list]
        response = PaginatedResponse.success(
            value=users,
            page=1,
            total=100,
            page_size=10,
            message="Users retrieved successfully"
        )

        # Check pagination info
        if response.has_more_pages():
            next_page = response.page + 1

        # Return to client
        return jsonify(response.to_dict()), response.get_status_code()
    """
    page: Optional[int] = None
    total: Optional[int] = None
    page_size: Optional[int] = None

    def total_pages(self) -> Optional[int]:
        """Calculate total number of pages."""
        if self.total is not None and self.page_size is not None and self.page_size > 0:
            return (self.total + self.page_size - 1) // self.page_size
        return None

    def has_more_pages(self) -> bool:
        """Check if there are more pages available."""
        total_pages = self.total_pages()
        if total_pages is not None and self.page is not None:
            return self.page < total_pages
        return False

    def has_previous_page(self) -> bool:
        """Check if there is a previous page."""
        return self.page is not None and self.page > 1

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary format including pagination metadata.
        Handles Pydantic models in the list automatically.
        """
        result = super().to_dict()

        # Convert list of Pydantic models if needed
        if self.value is not None and isinstance(self.value, list):
            if self.value and isinstance(self.value[0], BaseModel):
                result['value'] = [item.dict() for item in self.value]

        pagination_info: Dict[str, Any] = {}

        if self.page is not None:
            pagination_info['page'] = self.page

        if self.total is not None:
            pagination_info['total'] = self.total

        if self.page_size is not None:
            pagination_info['page_size'] = self.page_size

        total_pages = self.total_pages()
        if total_pages is not None:
            pagination_info['total_pages'] = total_pages

        if pagination_info:
            result['pagination'] = pagination_info

        return result

    @classmethod
    def success(
            cls,
            value: List[T],
            page: Optional[int] = None,
            total: Optional[int] = None,
            page_size: Optional[int] = None,
            message: Optional[str] = None
    ) -> 'PaginatedResponse[T]':
        """
        Create a successful paginated response.

        Args:
            value: List of response data (can be Pydantic models)
            page: Current page number
            total: Total number of items across all pages
            page_size: Number of items per page
            message: Optional success message

        Returns:
            PaginatedResponse with success data and pagination info

        Usage:
            # With Pydantic models
            users = [UserResponse.from_orm(u) for u in user_list]
            response = PaginatedResponse.success(
                value=users,
                page=1,
                total=100,
                page_size=10,
                message="Users retrieved successfully"
            )

            # API response JSON will look like:
            # {
            #   "message": "Users retrieved successfully",
            #   "value": [...list of users...],
            #   "pagination": {
            #     "page": 1,
            #     "total": 100,
            #     "page_size": 10,
            #     "total_pages": 10
            #   }
            # }
        """
        return cls(
            value=value,
            message=message,
            page=page,
            total=total,
            page_size=page_size
        )

    @classmethod
    def failure(
            cls,
            error: ErrorDetail,
            message: Optional[str] = None
    ) -> 'PaginatedResponse[T]':
        """
        Create a failure paginated response.

        Args:
            error: Error detail object
            message: Optional error message

        Returns:
            PaginatedResponse with error information

        Usage:
            error = ErrorDetail(
                title="Database Error",
                code="DB_CONNECTION_FAILED",
                status=500
            )
            response = PaginatedResponse.failure(error)
        """
        return cls(error=error, message=message)