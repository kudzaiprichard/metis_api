"""
Online Learning API controller.
Handles ML model training operations.
"""

from flask import Blueprint, request, jsonify

from src.config.auth_setup import jwt_auth
from src.modules.ml_models.application.services.online_learning_service import OnlineLearningService
from src.modules.ml_models.presentation.dtos.online_learning_dtos import (
    OnlineLearningRequest
)
from src.shared.response.api_response import ApiResponse

# Create blueprint
online_learning_bp = Blueprint('online_learning', __name__, url_prefix='/api/v1/ml/training')

# Initialize service
training_service = OnlineLearningService()


# =============================================================================
# TRAIN MODEL ENDPOINT
# =============================================================================

@online_learning_bp.route('/online-learning', methods=['POST'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def train_model():
    """Train a new model version using patient outcomes."""
    request_dto = OnlineLearningRequest(**request.json)

    result_dto = training_service.train_model(request_dto)

    if result_dto.success:
        response = ApiResponse.success(
            value=result_dto.model_dump(),
            message=f"Training completed successfully: Version {result_dto.version_number} created"
        )
        return jsonify(response.to_dict()), 201
    else:
        response = ApiResponse.success(
            value=result_dto.model_dump(),
            message="Training failed"
        )
        return jsonify(response.to_dict()), 200


# =============================================================================
# GET TRAINING STATUS ENDPOINT
# =============================================================================

@online_learning_bp.route('/status', methods=['GET'])
@jwt_auth.jwt_access_required
@jwt_auth.role_required('ML_ENGINEER')
def get_training_status():
    """Get current training status."""
    status_dto = training_service.get_training_status()

    response = ApiResponse.success(
        value=status_dto.model_dump(),
        message="Training status retrieved successfully"
    )

    return jsonify(response.to_dict()), 200