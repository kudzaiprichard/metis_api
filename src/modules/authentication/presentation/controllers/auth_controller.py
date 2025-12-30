"""
Authentication API controller.
Uses centralized auth setup for clean imports.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth, auth_service
from src.modules.authentication.presentation.dtos.auth_dtos import RegisterRequest, LoginRequest, LogoutRequest
from src.shared.response.api_response import ApiResponse

# Create blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/api/v1/auth')


# =============================================================================
# REGISTRATION ENDPOINT
# =============================================================================

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user."""
    request_dto = RegisterRequest(**request.json)

    user_dto, tokens = auth_service.register(request_dto)

    response_data = {
        'user': user_dto.model_dump(),
        'tokens': tokens
    }

    response = ApiResponse.success(
        value=response_data,
        message="User registered successfully"
    )

    return jsonify(response.to_dict()), 201


# =============================================================================
# LOGIN ENDPOINT
# =============================================================================

@auth_bp.route('/login', methods=['POST'])
def login():
    """Authenticate user and generate tokens."""
    request_dto = LoginRequest(**request.json)

    user_dto, tokens = auth_service.login(request_dto)

    response_data = {
        'user': user_dto.model_dump(),
        'tokens': tokens
    }

    response = ApiResponse.success(
        value=response_data,
        message="Login successful"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# LOGOUT ENDPOINT
# =============================================================================

@auth_bp.route('/logout', methods=['POST'])
@jwt_auth.jwt_access_required
def logout():
    """Logout user by revoking all their tokens."""
    user_id = jwt_auth.get_current_user_id()

    request_dto = LogoutRequest(user_id=user_id)
    auth_service.logout(request_dto)

    response = ApiResponse.success(
        value={"logged_out": True},
        message="Logout successful"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# REFRESH TOKEN ENDPOINT
# =============================================================================

@auth_bp.route('/refresh', methods=['POST'])
@jwt_auth.jwt_refresh_required
def refresh_token():
    """Generate new tokens using refresh token."""
    refresh_token = jwt_auth.get_current_token()

    tokens = auth_service.refresh_token(refresh_token)

    response = ApiResponse.success(
        value={'tokens': tokens},
        message="Tokens refreshed successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET CURRENT USER ENDPOINT
# =============================================================================

@auth_bp.route('/me', methods=['GET'])
@jwt_auth.jwt_access_required
def get_current_user():
    """Get current authenticated user profile."""
    token = jwt_auth.get_current_token()

    user_dto = auth_service.get_current_user(token)

    response = ApiResponse.success(
        value=user_dto.model_dump(),
        message="User retrieved successfully"
    )

    return jsonify(response.to_dict()), 200