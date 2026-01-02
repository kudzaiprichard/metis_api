"""
Patient Medical Data API controller.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth
from src.modules.patients.domain.services.patient_medical_data_service import PatientMedicalDataService
from src.modules.patients.presentation.dtos.patient_dtos import (
    CreatePatientMedicalDataRequest,
    UpdatePatientMedicalDataRequest
)
from src.shared.response.api_response import ApiResponse

# Create blueprint
patients_bp = Blueprint('patients', __name__, url_prefix='/api/v1/patients')

# Initialize service
medical_data_service = PatientMedicalDataService()


# =============================================================================
# CREATE PATIENT MEDICAL DATA ENDPOINT
# =============================================================================

@patients_bp.route('/<string:patient_id>/medical-data', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def create_medical_data(patient_id):
    """Create medical data for a patient."""
    request_data = request.json or {}
    request_data['patient_id'] = patient_id

    request_dto = CreatePatientMedicalDataRequest(**request_data)

    medical_data_dto = medical_data_service.create_medical_data(request_dto)

    response = ApiResponse.success(
        value=medical_data_dto.model_dump(),
        message="Medical data created successfully"
    )

    return jsonify(response.to_dict()), 201


# =============================================================================
# GET PATIENT MEDICAL DATA ENDPOINT
# =============================================================================

@patients_bp.route('/<string:patient_id>/medical-data', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_medical_data(patient_id):
    """Get medical data for a patient."""
    medical_data_dto = medical_data_service.get_medical_data(patient_id)

    response = ApiResponse.success(
        value=medical_data_dto.model_dump(),
        message="Medical data retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# UPDATE PATIENT MEDICAL DATA ENDPOINT
# =============================================================================

@patients_bp.route('/<string:patient_id>/medical-data', methods=['PUT'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def update_medical_data(patient_id):
    """Update medical data for a patient."""
    request_dto = UpdatePatientMedicalDataRequest(**request.json)

    medical_data_dto = medical_data_service.update_medical_data(patient_id, request_dto)

    response = ApiResponse.success(
        value=medical_data_dto.model_dump(),
        message="Medical data updated successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# DELETE PATIENT MEDICAL DATA ENDPOINT
# =============================================================================

@patients_bp.route('/<string:patient_id>/medical-data', methods=['DELETE'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def delete_medical_data(patient_id):
    """Soft delete patient medical data."""
    medical_data_service.delete_medical_data(patient_id)

    response = ApiResponse.success(
        value={"deleted": True, "patient_id": patient_id},
        message="Medical data deleted successfully"
    )

    return jsonify(response.to_dict()), 200