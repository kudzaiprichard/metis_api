"""
Global error handlers for Flask application.
Catches all exceptions and returns consistent API responses.
"""

from flask import Flask, jsonify, request
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from src.shared.exceptions.exceptions import AppException
from src.shared.response.api_response import ApiResponse
from src.shared.response.error_detail import ErrorDetail


def register_error_handlers(app: Flask) -> None:
    """
    Register all error handlers with the Flask app.

    Usage:
        app = Flask(__name__)
        register_error_handlers(app)
    """

    @app.errorhandler(AppException)
    def handle_app_exception(e: AppException):
        """
        Handle all custom application exceptions.

        This catches:
        - NotFoundException
        - ValidationException
        - AuthenticationException
        - AuthorizationException
        - ConflictException
        - BadRequestException
        - InternalServerException
        - ServiceUnavailableException
        """
        response = ApiResponse.failure(
            error=e.error_detail,
            message=e.message
        )
        return jsonify(response.to_dict()), e.error_detail.status

    @app.errorhandler(ValidationError)
    def handle_pydantic_validation_error(e: ValidationError):
        """
        Handle Pydantic validation errors.
        Converts Pydantic errors to our ErrorDetail format.
        """
        error_detail = ErrorDetail(
            title="Validation Failed",
            code="VALIDATION_ERROR",
            status=400
        )

        # Convert Pydantic errors to field_errors
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            error_detail.add_field_error(field, error['msg'])

        response = ApiResponse.failure(
            error_detail,
            message="Please check your input and try again"
        )
        return jsonify(response.to_dict()), 400

    @app.errorhandler(HTTPException)
    def handle_http_exception(e: HTTPException):
        """
        Handle Werkzeug HTTP exceptions (404, 405, etc.).
        These are raised by Flask itself.
        """
        error_detail = ErrorDetail(
            title=e.name,
            code=e.name.upper().replace(' ', '_'),
            status=e.code,
            details=[e.description]
        )

        # Map status codes to user-friendly messages
        user_messages = {
            400: "Please check your request and try again",
            401: "Please log in to continue",
            403: "You don't have permission to perform this action",
            404: "The page you're looking for doesn't exist",
            405: "This action is not allowed",
            500: "Something went wrong. Please try again later",
            503: "The service is temporarily unavailable"
        }

        message = user_messages.get(e.code, "An error occurred. Please try again")

        response = ApiResponse.failure(error_detail, message=message)
        return jsonify(response.to_dict()), e.code

    @app.errorhandler(404)
    def handle_not_found(e):
        """Handle 404 Not Found (route doesn't exist)."""
        error_detail = ErrorDetail(
            title="Route Not Found",
            code="ROUTE_NOT_FOUND",
            status=404,
            details=[f"The requested URL {request.path} was not found"]
        )

        response = ApiResponse.failure(
            error_detail,
            message="The page you're looking for doesn't exist"
        )
        return jsonify(response.to_dict()), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        """Handle 405 Method Not Allowed."""
        error_detail = ErrorDetail(
            title="Method Not Allowed",
            code="METHOD_NOT_ALLOWED",
            status=405,
            details=[f"Method {request.method} not allowed for {request.path}"]
        )

        response = ApiResponse.failure(
            error_detail,
            message="This action is not allowed"
        )
        return jsonify(response.to_dict()), 405

    @app.errorhandler(Exception)
    def handle_unexpected_error(e: Exception):
        """
        Catch-all handler for unexpected errors.
        This should be last resort - log these for debugging.
        """
        # Log the error for debugging
        app.logger.error(f"Unexpected error: {str(e)}", exc_info=True)

        error_detail = ErrorDetail(
            title="Internal Server Error",
            code="INTERNAL_ERROR",
            status=500,
            details=["An unexpected error occurred. Please try again later."]
        )

        response = ApiResponse.failure(
            error_detail,
            message="Something went wrong. Please try again later"
        )
        return jsonify(response.to_dict()), 500