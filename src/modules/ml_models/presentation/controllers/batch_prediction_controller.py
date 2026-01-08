"""
Batch Prediction API controller.
Handles batch predictions for model validation.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth
from src.modules.ml_models.application.services.batch_prediction_service import BatchPredictionService
from src.modules.ml_models.presentation.dtos.batch_prediction_dtos import (
    BatchPredictionRequest
)
from src.shared.response.api_response import ApiResponse

# Create blueprint
batch_prediction_bp = Blueprint('batch_prediction', __name__, url_prefix='/api/v1/ml/batch-predictions')

# Initialize service
batch_service = BatchPredictionService()


# =============================================================================
# BATCH PREDICTION ENDPOINT
# =============================================================================

@batch_prediction_bp.route('', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def process_batch():
    """Process batch predictions for model validation."""
    request_dto = BatchPredictionRequest(**request.json)

    result_dto = batch_service.process_batch(request_dto)

    response = ApiResponse.success(
        value=result_dto.model_dump(),
        message=f"Batch prediction completed: {result_dto.accuracy}% accuracy"
    )

    return jsonify(response.to_dict()), 200


# =============================================================================
# BATCH PREDICTION SUMMARY ENDPOINT
# =============================================================================

@batch_prediction_bp.route('/summary', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def get_summary():
    """Get summary statistics for batch predictions."""
    request_dto = BatchPredictionRequest(**request.json)

    summary_dto = batch_service.get_summary(request_dto)

    response = ApiResponse.success(
        value=summary_dto.model_dump(),
        message="Batch prediction summary generated successfully"
    )

    return jsonify(response.to_dict()), 200