"""
Similar Patient API controller.
Handles endpoints for finding and viewing similar patient cases from Neo4j.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth
from src.modules.patients.domain.services.similar_patient_service import SimilarPatientService
from src.modules.patients.presentation.dtos.similar_patient_dtos import (
    FindSimilarPatientsRequest,
    FindSimilarPatientsGraphRequest,
    GetSimilarPatientDetailRequest
)
from src.shared.response.api_response import ApiResponse

# Create blueprint
similar_patients_bp = Blueprint('similar_patients', __name__, url_prefix='/api/v1/similar-patients')

# Initialize service
similar_patient_service = SimilarPatientService()


# =============================================================================
# FIND SIMILAR PATIENTS (TABULAR FORMAT)
# =============================================================================

@similar_patients_bp.route('/search', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def find_similar_patients():
    """
    Find similar patient cases in tabular format.

    Request body:
    {
        "patient_id": "uuid",
        "limit": 5,
        "treatment_filter": "Metformin",  # optional
        "min_similarity": 0.5  # optional
    }
    """
    request_dto = FindSimilarPatientsRequest(**request.json)

    similar_patients_dto = similar_patient_service.find_similar_patients(request_dto)

    response = ApiResponse.success(
        value=similar_patients_dto.model_dump(),
        message="Similar patients retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# FIND SIMILAR PATIENTS (GRAPH FORMAT)
# =============================================================================

@similar_patients_bp.route('/search/graph', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def find_similar_patients_graph():
    """
    Find similar patient cases in graph format for visualization.

    Request body:
    {
        "patient_id": "uuid",
        "limit": 5,
        "treatment_filter": "Metformin",  # optional
        "min_similarity": 0.5  # optional
    }
    """
    request_dto = FindSimilarPatientsGraphRequest(**request.json)

    graph_dto = similar_patient_service.find_similar_patients_graph(request_dto)

    response = ApiResponse.success(
        value=graph_dto.model_dump(),
        message="Similar patients graph retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET SIMILAR PATIENT DETAIL BY CASE ID
# =============================================================================

@similar_patients_bp.route('/<string:case_id>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_similar_patient_detail(case_id):
    """
    Get complete details of a similar patient case from Neo4j historical dataset.

    Path parameter:
        case_id: Patient ID from Neo4j (e.g., "P000123")
    """
    request_dto = GetSimilarPatientDetailRequest(case_id=case_id)

    patient_detail_dto = similar_patient_service.get_similar_patient_detail(request_dto)

    response = ApiResponse.success(
        value=patient_detail_dto.model_dump(),
        message="Patient case details retrieved successfully"
    )

    return jsonify(response.to_dict()), 200