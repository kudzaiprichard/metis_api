"""
Online Learning service for ML model training.
Handles incremental model updates with patient outcomes.
"""

from typing import List, Dict
from src.modules.ml_models.presentation.dtos.online_learning_dtos import (
    OnlineLearningRequest,
    PatientOutcomeInput,
    OnlineLearningResponse,
    PerformanceMetrics,
    TrainingStatusResponse
)
from src.shared.exceptions.exceptions import (
    NotFoundException,
    ValidationException,
    InternalServerException,
    BadRequestException
)
from src.shared.response.error_detail import ErrorDetail

# Import ML Service Manager
from src.shared.ml.service_initializer import get_ml_service


class OnlineLearningService:
    """
    Service for online learning operations.
    Handles training new model versions from patient outcomes.
    """

    # Valid treatment names
    VALID_TREATMENTS = ['Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', 'Insulin']

    def __init__(self):
        """Initialize OnlineLearningService."""
        try:
            # Get shared ML service
            self.ml_service = get_ml_service()

        except RuntimeError as e:
            error = ErrorDetail(
                title="ML Service Not Available",
                code="ML_SERVICE_NOT_INITIALIZED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="ML service is not available",
                error_detail=error
            )

    def train_model(self, request: OnlineLearningRequest) -> OnlineLearningResponse:
        """
        Train a new model version using patient outcomes.

        Args:
            request: OnlineLearningRequest with outcomes and training config

        Returns:
            OnlineLearningResponse with training results

        Raises:
            ValidationException: If validation fails
            NotFoundException: If base version not found
            InternalServerException: If training fails
        """
        try:
            # Validate all inputs first
            self._validate_outcomes(request.outcomes)

            # Validate base version exists
            self._validate_base_version(request.base_version)

            # Convert outcomes to format expected by ML service
            formatted_outcomes = self._format_outcomes(request.outcomes)

            # Perform online learning
            result = self.ml_service.perform_online_learning(
                outcomes=formatted_outcomes,
                base_version=request.base_version,
                validate=request.validate,
                disable_ewc=request.disable_ewc,
                epochs=request.epochs
            )

            # Convert TrainingResult to response DTO
            if not result.success:
                return OnlineLearningResponse(
                    success=False,
                    version_number=None,
                    base_version=request.base_version,
                    outcomes_processed=0,
                    timestamp=result.timestamp,
                    error=result.error
                )

            # Extract performance metrics if available
            performance_before = None
            if result.performance_before:
                performance_before = PerformanceMetrics(
                    avg_reward=result.performance_before.get('avg_reward', 0.0),
                    accuracy=result.performance_before.get('accuracy', 0.0),
                    diversity=result.performance_before.get('diversity', 0),
                    success_rate=result.performance_before.get('success_rate', 0.0)
                )

            performance_after = None
            if result.performance_after:
                performance_after = PerformanceMetrics(
                    avg_reward=result.performance_after.get('avg_reward', 0.0),
                    accuracy=result.performance_after.get('accuracy', 0.0),
                    diversity=result.performance_after.get('diversity', 0),
                    success_rate=result.performance_after.get('success_rate', 0.0)
                )

            # Build training info
            training_info = {
                'epochs': request.epochs,
                'ewc_enabled': not request.disable_ewc,
                'validation_enabled': request.validate,
                'outcomes_count': len(request.outcomes)
            }

            return OnlineLearningResponse(
                success=True,
                version_number=result.version_number,
                base_version=request.base_version,
                outcomes_processed=result.outcomes_processed,
                performance_before=performance_before,
                performance_after=performance_after,
                timestamp=result.timestamp,
                error=None,
                training_info=training_info
            )

        except (ValidationException, NotFoundException):
            raise
        except ValueError as e:
            # Handle base version not found
            error = ErrorDetail(
                title="Invalid Base Version",
                code="INVALID_BASE_VERSION",
                status=400,
                details=[str(e)]
            )
            raise BadRequestException(
                message="Invalid base version specified",
                error_detail=error
            )
        except Exception as e:
            error = ErrorDetail(
                title="Online Learning Failed",
                code="TRAINING_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to train new model version",
                error_detail=error
            )

    def get_training_status(self) -> TrainingStatusResponse:
        """
        Get current training status.

        Returns:
            TrainingStatusResponse with current progress

        Raises:
            InternalServerException: If status retrieval fails
        """
        try:
            # This would require extending MLServiceManager to track training status
            # For now, return a simple response
            # TODO: Implement proper training status tracking if needed

            return TrainingStatusResponse(
                is_training=False,
                current_step=None,
                progress_percent=0,
                started_at=None,
                estimated_completion=None,
                version_number=None,
                outcomes_count=0,
                error=None
            )

        except Exception as e:
            error = ErrorDetail(
                title="Failed to Get Training Status",
                code="STATUS_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to retrieve training status",
                error_detail=error
            )

    def _format_outcomes(self, outcomes: List[PatientOutcomeInput]) -> List[Dict]:
        """
        Format outcomes for ML service.

        Args:
            outcomes: List of PatientOutcomeInput DTOs

        Returns:
            List of dicts in format expected by ML service
        """
        formatted = []

        for outcome in outcomes:
            patient_dict = {
                'age': float(outcome.age),
                'gender': outcome.gender,
                'ethnicity': outcome.ethnicity,
                'hba1c_baseline': float(outcome.hba1c_baseline),
                'diabetes_duration': float(outcome.diabetes_duration),
                'fasting_glucose': float(outcome.fasting_glucose),
                'c_peptide': float(outcome.c_peptide),
                'egfr': float(outcome.egfr),
                'bmi': float(outcome.bmi),
                'bp_systolic': int(outcome.bp_systolic),
                'bp_diastolic': int(outcome.bp_diastolic),
                'alt': float(outcome.alt),
                'ldl': float(outcome.ldl),
                'hdl': float(outcome.hdl),
                'triglycerides': float(outcome.triglycerides),
                'previous_prediabetes': outcome.previous_prediabetes,
                'hypertension': outcome.hypertension,
                'ckd': outcome.ckd,
                'cvd': outcome.cvd,
                'nafld': outcome.nafld,
                'retinopathy': outcome.retinopathy
            }

            formatted.append({
                'patient': patient_dict,
                'treatment_given': outcome.treatment_given,
                'reward': float(outcome.reward)
            })

        return formatted

    def _validate_base_version(self, base_version: str):
        """
        Validate that base version exists.

        Args:
            base_version: Base version to validate

        Raises:
            NotFoundException: If base version not found
        """
        available = [v['version_number'] for v in self.ml_service.list_all_versions()]

        if base_version not in available:
            error = ErrorDetail(
                title="Base Version Not Found",
                code="BASE_VERSION_NOT_FOUND",
                status=404,
                details=[
                    f"Base version '{base_version}' does not exist",
                    f"Available versions: {', '.join(available)}"
                ]
            )
            raise NotFoundException(
                message=f"Base version '{base_version}' not found",
                error_detail=error
            )

    def _validate_outcomes(self, outcomes: List[PatientOutcomeInput]):
        """
        Validate patient outcomes.

        Args:
            outcomes: List of outcome inputs

        Raises:
            ValidationException: If validation fails
        """
        errors = []

        # Validate treatments
        for i, outcome in enumerate(outcomes):
            if outcome.treatment_given not in self.VALID_TREATMENTS:
                errors.append(
                    f"Invalid treatment at index {i}: "
                    f"'{outcome.treatment_given}'. Must be one of: {', '.join(self.VALID_TREATMENTS)}"
                )

            # Validate gender
            if outcome.gender not in ['Male', 'Female']:
                errors.append(
                    f"Invalid gender at index {i}: "
                    f"'{outcome.gender}'. Must be 'Male' or 'Female'"
                )

            # Validate ethnicity
            valid_ethnicities = ['Caucasian', 'African', 'Asian', 'Hispanic', 'Other']
            if outcome.ethnicity not in valid_ethnicities:
                errors.append(
                    f"Invalid ethnicity at index {i}: "
                    f"'{outcome.ethnicity}'. Must be one of: {', '.join(valid_ethnicities)}"
                )

            # Validate reward is reasonable
            if outcome.reward < 0 or outcome.reward > 10:
                errors.append(
                    f"Invalid reward at index {i}: "
                    f"{outcome.reward}. Must be between 0.0 and 10.0"
                )

        if errors:
            error = ErrorDetail(
                title="Outcome Validation Failed",
                code="OUTCOME_VALIDATION_ERROR",
                status=400,
                details=errors
            )
            raise ValidationException(
                message="Outcome validation failed",
                error_detail=error
            )