"""
Patient Medical Data API controller.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth
from src.modules.patients.domain.services.patient_medical_data_service import PatientMedicalDataService
from src.modules.patients.domain.services.patient_service import PatientService
from src.modules.patients.presentation.dtos.patient_dtos import (
    CreatePatientMedicalDataRequest,
    UpdatePatientMedicalDataRequest, ListPatientsRequest, CreatePatientRequest, GetPatientRequest,
    UpdatePatientContactRequest, DeletePatientRequest
)
from src.shared.response.api_response import ApiResponse
from src.shared.response.paginated_response import PaginatedResponse

# Create blueprint
patients_bp = Blueprint('patients', __name__, url_prefix='/api/v1/patients')

# Initialize service
medical_data_service = PatientMedicalDataService()
patient_service = PatientService()



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

# =============================================================================
# PATIENT CRUD ENDPOINTS
# =============================================================================

@patients_bp.route('', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def list_patients():
    """List patients with pagination and optional search."""
    request_dto = ListPatientsRequest(
        page=request.args.get('page', 1, type=int),
        per_page=request.args.get('per_page', 20, type=int),
        search=request.args.get('search', None)
    )

    patients, total = patient_service.list_patients(request_dto)

    patients_data = [patient.model_dump() for patient in patients]

    response = PaginatedResponse.success(
        value=patients_data,
        page=request_dto.page,
        total=total,
        page_size=request_dto.per_page,
        message="Patients retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


@patients_bp.route('', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def create_patient():
    """Create a new patient."""
    request_dto = CreatePatientRequest(**request.json)

    patient_dto = patient_service.create_patient(request_dto)

    response = ApiResponse.success(
        value=patient_dto.model_dump(),
        message="Patient created successfully"
    )

    return jsonify(response.to_dict()), 201


@patients_bp.route('/<string:patient_id>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_patient(patient_id):
    """Get a single patient by ID."""
    request_dto = GetPatientRequest(patient_id=patient_id)

    patient_dto = patient_service.get_patient(request_dto)

    response = ApiResponse.success(
        value=patient_dto.model_dump(),
        message="Patient retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


@patients_bp.route('/<string:patient_id>/detail', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_patient_detail(patient_id):
    """Get patient with medical data."""
    request_dto = GetPatientRequest(patient_id=patient_id)

    patient_dto = patient_service.get_patient_detail(request_dto)

    response = ApiResponse.success(
        value=patient_dto.model_dump(),
        message="Patient details retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


@patients_bp.route('/<string:patient_id>', methods=['PUT'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def update_patient(patient_id):
    """Update patient contact information."""
    request_dto = UpdatePatientContactRequest(**request.json)

    patient_dto = patient_service.update_patient_contact(patient_id, request_dto)

    response = ApiResponse.success(
        value=patient_dto.model_dump(),
        message="Patient updated successfully"
    )

    return jsonify(response.to_dict()), 200


@patients_bp.route('/<string:patient_id>', methods=['DELETE'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def delete_patient(patient_id):
    """Soft delete a patient."""
    request_dto = DeletePatientRequest(patient_id=patient_id)

    patient_service.delete_patient(request_dto)

    response = ApiResponse.success(
        value={"deleted": True, "patient_id": patient_id},
        message="Patient deleted successfully"
    )

    return jsonify(response.to_dict()), 200


@patients_bp.route('/<string:patient_id>/restore', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def restore_patient(patient_id):
    """Restore a soft-deleted patient."""
    patient_dto = patient_service.restore_patient(patient_id)

    response = ApiResponse.success(
        value=patient_dto.model_dump(),
        message="Patient restored successfully"
    )

    return jsonify(response.to_dict()), 200
