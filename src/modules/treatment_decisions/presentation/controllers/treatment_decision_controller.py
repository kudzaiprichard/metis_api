"""
Treatment Decision API controller.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth
from src.modules.treatment_decisions.domain.services.treatment_decision_service import TreatmentDecisionService, \
    RecordTreatmentDecisionRequest, UpdateTreatmentOutcomeRequest, GetTreatmentDecisionRequest, \
    GetPatientDecisionsRequest, ListTreatmentDecisionsRequest
from src.shared.response.api_response import ApiResponse
from src.shared.response.paginated_response import PaginatedResponse

# Create blueprint
treatment_decision_bp = Blueprint('treatment_decisions', __name__, url_prefix='/api/v1/treatment-decisions')

# Initialize service
treatment_decision_service = TreatmentDecisionService()


# =============================================================================
# RECORD TREATMENT DECISION ENDPOINT
# =============================================================================

@treatment_decision_bp.route('', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def record_decision():
    """Record doctor's treatment decision."""
    request_dto = RecordTreatmentDecisionRequest(**request.json)

    # Get current user ID from JWT
    current_user_id = jwt_auth.get_current_user_id()

    decision_dto = treatment_decision_service.record_decision(request_dto, current_user_id)

    response = ApiResponse.success(
        value=decision_dto.model_dump(),
        message="Treatment decision recorded successfully"
    )

    return jsonify(response.to_dict()), 201


# =============================================================================
# UPDATE TREATMENT OUTCOME ENDPOINT
# =============================================================================

@treatment_decision_bp.route('/<string:decision_id>/outcome', methods=['PUT'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def update_outcome(decision_id):
    """Update treatment outcome after follow-up."""
    request_dto = UpdateTreatmentOutcomeRequest(**request.json)

    decision_dto = treatment_decision_service.update_outcome(decision_id, request_dto)

    response = ApiResponse.success(
        value=decision_dto.model_dump(),
        message="Treatment outcome updated successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET SINGLE DECISION ENDPOINT
# =============================================================================

@treatment_decision_bp.route('/<string:decision_id>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_decision(decision_id):
    """Get treatment decision by ID."""
    request_dto = GetTreatmentDecisionRequest(decision_id=decision_id)

    decision_dto = treatment_decision_service.get_decision(request_dto)

    response = ApiResponse.success(
        value=decision_dto.model_dump(),
        message="Treatment decision retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET PATIENT DECISIONS ENDPOINT
# =============================================================================

@treatment_decision_bp.route('/patient/<string:patient_id>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_patient_decisions(patient_id):
    """Get all treatment decisions for a patient."""
    limit = request.args.get('limit', None, type=int)

    request_dto = GetPatientDecisionsRequest(
        patient_id=patient_id,
        limit=limit
    )

    decisions = treatment_decision_service.get_patient_decisions(request_dto)

    decisions_data = [decision.model_dump() for decision in decisions]

    response = ApiResponse.success(
        value=decisions_data,
        message="Patient treatment decisions retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# LIST DECISIONS ENDPOINT
# =============================================================================

@treatment_decision_bp.route('', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def list_decisions():
    """List all treatment decisions with pagination and filters."""
    request_dto = ListTreatmentDecisionsRequest(
        page=request.args.get('page', 1, type=int),
        per_page=request.args.get('per_page', 20, type=int),
        patient_id=request.args.get('patient_id', None, type=str),
        decision_type=request.args.get('decision_type', None, type=str)
    )

    decisions, total = treatment_decision_service.list_decisions(request_dto)

    decisions_data = [decision.model_dump() for decision in decisions]

    response = PaginatedResponse.success(
        value=decisions_data,
        page=request_dto.page,
        total=total,
        page_size=request_dto.per_page,
        message="Treatment decisions retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# DELETE DECISION ENDPOINT
# =============================================================================

@treatment_decision_bp.route('/<string:decision_id>', methods=['DELETE'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def delete_decision(decision_id):
    """Soft delete a treatment decision."""
    treatment_decision_service.delete_decision(decision_id)

    response = ApiResponse.success(
        value={"deleted": True, "decision_id": decision_id},
        message="Treatment decision deleted successfully"
    )

    return jsonify(response.to_dict()), 200