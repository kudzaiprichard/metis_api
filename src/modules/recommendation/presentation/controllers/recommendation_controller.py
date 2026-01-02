"""
Prediction API controller.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth
from src.modules.recommendation.domain.services.prediction_service import PredictionService
from src.modules.recommendation.domain.services.prediction_management_service import PredictionManagementService
from src.modules.recommendation.presentation.dtos.prediction_dtos import (
    GeneratePredictionRequest,
    GetPredictionRequest,
    GetPatientPredictionsRequest,
    ListPredictionsRequest
)
from src.shared.response.api_response import ApiResponse
from src.shared.response.paginated_response import PaginatedResponse

# Create blueprint
recommendation_bp = Blueprint('recommendation', __name__, url_prefix='/api/v1/recommendation')

# Initialize services
prediction_service = PredictionService()
prediction_management_service = PredictionManagementService()


# =============================================================================
# GENERATE PREDICTION ENDPOINT
# =============================================================================

@recommendation_bp.route('/generate', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def generate_prediction():
    """Generate AI prediction for a patient."""
    request_dto = GeneratePredictionRequest(**request.json)

    # Get current user ID from JWT
    current_user_id = jwt_auth.get_current_user_id()

    prediction_dto = prediction_service.generate_prediction(request_dto, current_user_id)

    response = ApiResponse.success(
        value=prediction_dto.model_dump(),
        message="Prediction generated successfully"
    )

    return jsonify(response.to_dict()), 201


# =============================================================================
# GET SINGLE PREDICTION ENDPOINT
# =============================================================================

@recommendation_bp.route('/<string:prediction_id>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_prediction(prediction_id):
    """Get prediction by ID with full details."""
    request_dto = GetPredictionRequest(prediction_id=prediction_id)

    prediction_dto = prediction_management_service.get_prediction(request_dto)

    response = ApiResponse.success(
        value=prediction_dto.model_dump(),
        message="Prediction retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# GET PATIENT PREDICTIONS ENDPOINT
# =============================================================================

@recommendation_bp.route('/patient/<string:patient_id>', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def get_patient_predictions(patient_id):
    """Get all recommendation for a patient."""
    limit = request.args.get('limit', None, type=int)

    request_dto = GetPatientPredictionsRequest(
        patient_id=patient_id,
        limit=limit
    )

    recommendation = prediction_management_service.get_patient_predictions(request_dto)

    predictions_data = [pred.model_dump() for pred in recommendation]

    response = ApiResponse.success(
        value=predictions_data,
        message="Patient recommendation retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# LIST PREDICTIONS ENDPOINT
# =============================================================================

@recommendation_bp.route('', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def list_predictions():
    """List all recommendation with pagination and filters."""
    request_dto = ListPredictionsRequest(
        page=request.args.get('page', 1, type=int),
        per_page=request.args.get('per_page', 20, type=int),
        patient_id=request.args.get('patient_id', None, type=str)
    )

    recommendation, total = prediction_management_service.list_predictions(request_dto)

    predictions_data = [pred.model_dump() for pred in recommendation]

    response = PaginatedResponse.success(
        value=predictions_data,
        page=request_dto.page,
        total=total,
        page_size=request_dto.per_page,
        message="Predictions retrieved successfully"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# DELETE PREDICTION ENDPOINT
# =============================================================================

@recommendation_bp.route('/<string:prediction_id>', methods=['DELETE'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('DOCTOR')
def delete_prediction(prediction_id):
    """Soft delete a prediction."""
    prediction_management_service.delete_prediction(prediction_id)

    response = ApiResponse.success(
        value={"deleted": True, "prediction_id": prediction_id},
        message="Prediction deleted successfully"
    )

    return jsonify(response.to_dict()), 200