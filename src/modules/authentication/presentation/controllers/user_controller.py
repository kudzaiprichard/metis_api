"""
User Management API controller.
Uses centralized auth setup for clean imports.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth, user_service
from src.modules.authentication.presentation.dtos.user_dtos import CreateUserRequest, ListUsersRequest, GetUserRequest, \
    UpdateUserRequest, DeleteUserRequest
from src.shared.exceptions.exceptions import ValidationException
from src.shared.response.api_response import ApiResponse
from src.shared.response.paginated_response import PaginatedResponse
from src.shared.response.error_detail import ErrorDetail

# Create blueprint
user_bp = Blueprint('users', __name__, url_prefix='/api/v1/users')


# =============================================================================
# CREATE USER ENDPOINT
# =============================================================================

@user_bp.route('', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR', 'ML_ENGINEER')
def create_user():
    """Create a new user (admin functionality)."""
    request_dto = CreateUserRequest(**request.json)

    user_dto = user_service.create_user(request_dto)

    response = ApiResponse.success(
        value=user_dto.model_dump(),
        message="User created successfully"
    )

    return jsonify(response.to_dict()), 201


# =============================================================================
# LIST USERS ENDPOINT
# =============================================================================

@user_bp.route('', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR', 'ML_ENGINEER')
def list_users():
    """List all users with pagination and filters."""
    request_dto = ListUsersRequest(
        page=request.args.get('page', 1, type=int),
        per_page=request.args.get('per_page', 20, type=int),
        role=request.args.get('role', None, type=str),
        search=request.args.get('search', None, type=str)
    )

    users, total = user_service.list_users(request_dto)

    users_data = [user.model_dump() for user in users]

    response = PaginatedResponse.success(
        value=users_data,
        page=request_dto.page,
        total=total,
        page_size=request_dto.per_page,
        message="Users retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET SINGLE USER ENDPOINT
# =============================================================================

@user_bp.route('/<string:user_id>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR', 'ML_ENGINEER')
def get_user(user_id):
    """Get a single user by ID."""
    request_dto = GetUserRequest(user_id=user_id)

    user_dto = user_service.get_user(request_dto)

    response = ApiResponse.success(
        value=user_dto.model_dump(),
        message="User retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# UPDATE USER ENDPOINT
# =============================================================================

@user_bp.route('/<string:user_id>', methods=['PUT'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR', 'ML_ENGINEER')
def update_user(user_id):
    """Update user details."""
    request_data = request.json or {}
    request_data['user_id'] = user_id

    request_dto = UpdateUserRequest(**request_data)

    user_dto = user_service.update_user(request_dto)

    response = ApiResponse.success(
        value=user_dto.model_dump(),
        message="User updated successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# DELETE USER ENDPOINT
# =============================================================================

@user_bp.route('/<string:user_id>', methods=['DELETE'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR', 'ML_ENGINEER')
def delete_user(user_id):
    """Soft delete a user."""
    # Prevent self-deletion
    current_user_id = jwt_auth.get_current_user_id()
    if user_id == current_user_id:
        error = ErrorDetail(
            title="Invalid Operation",
            code="CANNOT_DELETE_SELF",
            status=400,
            details=["You cannot delete your own account"]
        )
        raise ValidationException(error_detail=error)

    request_dto = DeleteUserRequest(user_id=user_id)

    user_service.delete_user(request_dto)

    response = ApiResponse.success(
        value={"deleted": True, "user_id": user_id},
        message="User deleted successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# RESTORE USER ENDPOINT
# =============================================================================

@user_bp.route('/<string:user_id>/restore', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR', 'ML_ENGINEER')
def restore_user(user_id):
    """Restore a soft-deleted user."""
    user_dto = user_service.restore_user(user_id)

    response = ApiResponse.success(
        value=user_dto.model_dump(),
        message="User restored successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET USER BY EMAIL ENDPOINT
# =============================================================================

@user_bp.route('/email/<string:email>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR', 'ML_ENGINEER')
def get_user_by_email(email):
    """Get user by email address."""
    user_dto = user_service.get_user_by_email(email)

    response = ApiResponse.success(
        value=user_dto.model_dump(),
        message="User retrieved successfully"
    )

    return jsonify(response.to_dict()), 200