"""
Patient Timeline API controller.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth
from src.modules.treatment_decisions.domain.services.patient_timeline_service import PatientTimelineService
from src.modules.treatment_decisions.presentation.dtos.timeline_dtos import GetPatientTimelineRequest
from src.shared.response.api_response import ApiResponse

# Create blueprint
timeline_bp = Blueprint('timeline', __name__, url_prefix='/api/v1/patients')

# Initialize service
timeline_service = PatientTimelineService()


# =============================================================================
# GET PATIENT TIMELINE ENDPOINT
# =============================================================================

@timeline_bp.route('/<string:patient_id>/timeline', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_patient_timeline(patient_id):
    """Get complete patient timeline."""
    limit = request.args.get('limit', None, type=int)

    request_dto = GetPatientTimelineRequest(
        patient_id=patient_id,
        limit=limit
    )

    timeline_dto = timeline_service.get_patient_timeline(request_dto)

    response = ApiResponse.success(
        value=timeline_dto.model_dump(),
        message="Patient timeline retrieved successfully"
    )

    return jsonify(response.to_dict()), 200