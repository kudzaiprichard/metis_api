"""
Follow-up API controller.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth
from src.modules.treatment_decisions.domain.services.follow_up_service import FollowUpService
from src.modules.treatment_decisions.presentation.dtos.follow_up_dtos import ScheduleFollowUpRequest, \
    CompleteFollowUpRequest, UpdateFollowUpRequest, CancelFollowUpRequest, GetFollowUpRequest, \
    GetPatientFollowUpsRequest, GetUpcomingFollowUpsRequest
from src.shared.response.api_response import ApiResponse

# Create blueprint
follow_up_bp = Blueprint('follow_ups', __name__, url_prefix='/api/v1/follow-ups')

# Initialize service
follow_up_service = FollowUpService()


# =============================================================================
# SCHEDULE FOLLOW-UP ENDPOINT
# =============================================================================

@follow_up_bp.route('/schedule', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def schedule_follow_up():
    """Schedule a follow-up appointment."""
    request_dto = ScheduleFollowUpRequest(**request.json)

    follow_up_dto = follow_up_service.schedule_follow_up(request_dto)

    response = ApiResponse.success(
        value=follow_up_dto.model_dump(),
        message="Follow-up scheduled successfully"
    )

    return jsonify(response.to_dict()), 201


# =============================================================================
# COMPLETE FOLLOW-UP ENDPOINT
# =============================================================================

@follow_up_bp.route('/<string:follow_up_id>/complete', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def complete_follow_up(follow_up_id):
    """Record a completed follow-up visit."""
    request_dto = CompleteFollowUpRequest(**request.json)

    # Get current user ID from JWT
    current_user_id = jwt_auth.get_current_user_id()

    follow_up_dto = follow_up_service.complete_follow_up(follow_up_id, request_dto, current_user_id)

    response = ApiResponse.success(
        value=follow_up_dto.model_dump(),
        message="Follow-up completed successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# UPDATE FOLLOW-UP ENDPOINT
# =============================================================================

@follow_up_bp.route('/<string:follow_up_id>', methods=['PUT'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def update_follow_up(follow_up_id):
    """Update a scheduled follow-up."""
    request_dto = UpdateFollowUpRequest(**request.json)

    follow_up_dto = follow_up_service.update_follow_up(follow_up_id, request_dto)

    response = ApiResponse.success(
        value=follow_up_dto.model_dump(),
        message="Follow-up updated successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# CANCEL FOLLOW-UP ENDPOINT
# =============================================================================

@follow_up_bp.route('/<string:follow_up_id>/cancel', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def cancel_follow_up(follow_up_id):
    """Cancel a follow-up appointment."""
    request_dto = CancelFollowUpRequest(follow_up_id=follow_up_id)

    follow_up_service.cancel_follow_up(request_dto)

    response = ApiResponse.success(
        value={"cancelled": True, "follow_up_id": follow_up_id},
        message="Follow-up cancelled successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET SINGLE FOLLOW-UP ENDPOINT
# =============================================================================

@follow_up_bp.route('/<string:follow_up_id>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_follow_up(follow_up_id):
    """Get follow-up by ID."""
    request_dto = GetFollowUpRequest(follow_up_id=follow_up_id)

    follow_up_dto = follow_up_service.get_follow_up(request_dto)

    response = ApiResponse.success(
        value=follow_up_dto.model_dump(),
        message="Follow-up retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET PATIENT FOLLOW-UPS ENDPOINT
# =============================================================================

@follow_up_bp.route('/patient/<string:patient_id>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_patient_follow_ups(patient_id):
    """Get all follow-ups for a patient."""
    status = request.args.get('status', None, type=str)

    request_dto = GetPatientFollowUpsRequest(
        patient_id=patient_id,
        status=status
    )

    follow_ups = follow_up_service.get_patient_follow_ups(request_dto)

    follow_ups_data = [follow_up.model_dump() for follow_up in follow_ups]

    response = ApiResponse.success(
        value=follow_ups_data,
        message="Patient follow-ups retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET UPCOMING FOLLOW-UPS ENDPOINT
# =============================================================================

@follow_up_bp.route('/upcoming', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_upcoming_follow_ups():
    """Get upcoming scheduled follow-ups."""
    request_dto = GetUpcomingFollowUpsRequest(
        page=request.args.get('page', 1, type=int),
        per_page=request.args.get('per_page', 20, type=int)
    )

    follow_ups = follow_up_service.get_upcoming_follow_ups(request_dto)

    follow_ups_data = [follow_up.model_dump() for follow_up in follow_ups]

    response = ApiResponse.success(
        value=follow_ups_data,
        message="Upcoming follow-ups retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# DELETE FOLLOW-UP ENDPOINT
# =============================================================================

@follow_up_bp.route('/<string:follow_up_id>', methods=['DELETE'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def delete_follow_up(follow_up_id):
    """Soft delete a follow-up."""
    follow_up_service.delete_follow_up(follow_up_id)

    response = ApiResponse.success(
        value={"deleted": True, "follow_up_id": follow_up_id},
        message="Follow-up deleted successfully"
    )

    return jsonify(response.to_dict()), 200