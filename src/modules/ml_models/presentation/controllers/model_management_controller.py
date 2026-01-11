"""
Model Management API controller.
Handles ML model CRUD operations.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth
from src.modules.ml_models.domain.services.model_manager_service import ModelManagerService
from src.modules.ml_models.presentation.dtos.model_dtos import (
    ListModelsRequest,
    ActivateModelRequest,
    DeleteModelRequest,
    CompareModelsRequest
)
from src.shared.response.api_response import ApiResponse

# Create blueprint
model_management_bp = Blueprint('model_management', __name__, url_prefix='/api/v1/ml/models')


# =============================================================================
# LAZY INITIALIZATION HELPER
# =============================================================================

def get_model_service():
    """
    Lazy initialization of ModelManagerService.
    Creates service instance on-demand after app initialization is complete.

    Returns:
        ModelManagerService: Initialized service instance
    """
    return ModelManagerService()


# =============================================================================
# LIST ALL MODELS ENDPOINT
# =============================================================================

@model_management_bp.route('', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def list_models():
    """List all available model versions with sorting."""
    model_service = get_model_service()

    request_dto = ListModelsRequest(
        sort_by=request.args.get('sort_by', 'version'),
        reverse=request.args.get('reverse', 'false').lower() == 'true'
    )

    models_dto = model_service.list_models(request_dto)

    response = ApiResponse.success(
        value=models_dto.model_dump(),
        message="Models retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET SINGLE MODEL ENDPOINT
# =============================================================================

@model_management_bp.route('/<string:version>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def get_model(version):
    """Get detailed information for a specific model version."""
    model_service = get_model_service()

    model_dto = model_service.get_model_info(version)

    response = ApiResponse.success(
        value=model_dto.model_dump(),
        message="Model information retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET ACTIVE MODEL ENDPOINT
# =============================================================================

@model_management_bp.route('/active', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def get_active_model():
    """Get currently active model version information."""
    model_service = get_model_service()

    active_model_dto = model_service.get_active_model()

    response = ApiResponse.success(
        value=active_model_dto.model_dump(),
        message="Active model retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# ACTIVATE MODEL ENDPOINT
# =============================================================================

@model_management_bp.route('/<string:version>/activate', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def activate_model(version):
    """Activate a specific model version."""
    model_service = get_model_service()

    request_dto = ActivateModelRequest(version=version)

    model_dto = model_service.activate_model(request_dto)

    response = ApiResponse.success(
        value=model_dto.model_dump(),
        message=f"Model version '{version}' activated successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# DELETE MODEL ENDPOINT
# =============================================================================

@model_management_bp.route('/<string:version>', methods=['DELETE'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def delete_model(version):
    """Delete a model version."""
    model_service = get_model_service()

    request_dto = DeleteModelRequest(
        version=version,
        delete_files=request.args.get('delete_files', 'true').lower() == 'true'
    )

    result = model_service.delete_model(request_dto)

    response = ApiResponse.success(
        value=result,
        message=f"Model version '{version}' deleted successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET MODEL STATUS ENDPOINT
# =============================================================================

@model_management_bp.route('/status', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def get_status():
    """Get model manager status."""
    model_service = get_model_service()

    status_dto = model_service.get_status()

    response = ApiResponse.success(
        value=status_dto.model_dump(),
        message="Model manager status retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# COMPARE MODELS ENDPOINT
# =============================================================================

@model_management_bp.route('/compare', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def compare_models():
    """Compare performance between two model versions."""
    model_service = get_model_service()

    request_dto = CompareModelsRequest(**request.json)

    comparison_dto = model_service.compare_models(request_dto)

    response = ApiResponse.success(
        value=comparison_dto.model_dump(),
        message="Model comparison completed successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET MODEL LINEAGE ENDPOINT
# =============================================================================

@model_management_bp.route('/<string:version>/lineage', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def get_lineage(version):
    """Get version lineage for a model."""
    model_service = get_model_service()

    lineage_dto = model_service.get_lineage(version)

    response = ApiResponse.success(
        value=lineage_dto.model_dump(),
        message="Model lineage retrieved successfully"
    )

    return jsonify(response.to_dict()), 200