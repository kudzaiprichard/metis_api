"""
Batch Prediction service for ML model testing and validation.
Handles batch predictions without explainability for speed.
Always uses specific model version (no active model cache).
"""

from typing import List
from src.modules.ml_models.presentation.dtos.batch_prediction_dtos import (
    BatchPredictionRequest,
    BatchPredictionInput,
    BatchPredictionResult,
    BatchPredictionResponse,
    BatchPredictionSummary
)
from src.shared.exceptions.exceptions import (
    NotFoundException,
    ValidationException,
    InternalServerException
)
from src.shared.response.error_detail import ErrorDetail

# Import ML Service Manager
from src.shared.ml.service_initializer import get_ml_service


class BatchPredictionService:
    """
    Service for batch predictions without explainability.
    Optimized for speed and validation testing.
    Always requires specific model version.
    """

    # Valid treatment names
    VALID_TREATMENTS = ['Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', 'Insulin']

    def __init__(self):
        """Initialize BatchPredictionService."""
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

    def process_batch(self, request: BatchPredictionRequest) -> BatchPredictionResponse:
        """
        Process batch predictions with specific model version.

        Args:
            request: BatchPredictionRequest with patient data and model version

        Returns:
            BatchPredictionResponse with results

        Raises:
            ValidationException: If validation fails
            NotFoundException: If model version not found
            InternalServerException: If prediction fails
        """
        try:
            # Validate all inputs first
            self._validate_batch_inputs(request.predictions)

            # Validate model version exists (REQUIRED)
            self._validate_model_version(request.model_version)

            # Process each prediction
            results = []
            correct_count = 0
            incorrect_count = 0

            for input_data in request.predictions:
                try:
                    result = self._process_single_prediction(
                        input_data=input_data,
                        model_version=request.model_version
                    )
                    results.append(result)

                    if result.is_correct:
                        correct_count += 1
                    else:
                        incorrect_count += 1

                except Exception as e:
                    # Log error but continue with other predictions
                    error_result = BatchPredictionResult(
                        id=input_data.id,
                        predicted_treatment="ERROR",
                        actual_treatment=input_data.actual_treatment,
                        is_correct=False,
                        confidence_score=0.0,
                        predicted_reduction=0.0,
                        all_q_values=None
                    )
                    results.append(error_result)
                    incorrect_count += 1

            # Calculate accuracy
            total = len(request.predictions)
            accuracy = (correct_count / total * 100) if total > 0 else 0.0

            return BatchPredictionResponse(
                total_predictions=total,
                correct_predictions=correct_count,
                incorrect_predictions=incorrect_count,
                accuracy=round(accuracy, 2),
                model_version_used=request.model_version,
                results=results
            )

        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            error = ErrorDetail(
                title="Batch Prediction Failed",
                code="BATCH_PREDICTION_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to process batch predictions",
                error_detail=error
            )

    def get_summary(self, request: BatchPredictionRequest) -> BatchPredictionSummary:
        """
        Get summary statistics for batch predictions.

        Args:
            request: BatchPredictionRequest

        Returns:
            BatchPredictionSummary with statistics

        Raises:
            ValidationException: If validation fails
            InternalServerException: If processing fails
        """
        try:
            # Process batch first
            response = self.process_batch(request)

            # Calculate per-treatment breakdown
            treatment_stats = {}
            for treatment in self.VALID_TREATMENTS:
                treatment_results = [
                    r for r in response.results
                    if self._normalize_treatment(r.actual_treatment) == self._normalize_treatment(treatment)
                ]

                if treatment_results:
                    correct = sum(1 for r in treatment_results if r.is_correct)
                    total = len(treatment_results)
                    accuracy = (correct / total * 100) if total > 0 else 0.0

                    treatment_stats[treatment] = {
                        "total": total,
                        "correct": correct,
                        "incorrect": total - correct,
                        "accuracy": round(accuracy, 2)
                    }
                else:
                    treatment_stats[treatment] = {
                        "total": 0,
                        "correct": 0,
                        "incorrect": 0,
                        "accuracy": 0.0
                    }

            return BatchPredictionSummary(
                total_predictions=response.total_predictions,
                correct_predictions=response.correct_predictions,
                incorrect_predictions=response.incorrect_predictions,
                accuracy=response.accuracy,
                model_version_used=response.model_version_used,
                treatment_breakdown=treatment_stats
            )

        except (ValidationException, NotFoundException):
            raise
        except Exception as e:
            error = ErrorDetail(
                title="Failed to Generate Summary",
                code="SUMMARY_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to generate batch prediction summary",
                error_detail=error
            )

    def _process_single_prediction(
        self,
        input_data: BatchPredictionInput,
        model_version: str
    ) -> BatchPredictionResult:
        """
        Process a single prediction with specific model version.

        Args:
            input_data: Single patient input data
            model_version: Model version to use (REQUIRED)

        Returns:
            BatchPredictionResult
        """
        # Convert input to patient features dict
        patient_features = {
            'age': float(input_data.age),
            'gender': input_data.gender,
            'ethnicity': input_data.ethnicity,
            'hba1c_baseline': float(input_data.hba1c_baseline),
            'diabetes_duration': float(input_data.diabetes_duration),
            'fasting_glucose': float(input_data.fasting_glucose),
            'c_peptide': float(input_data.c_peptide),
            'egfr': float(input_data.egfr),
            'bmi': float(input_data.bmi),
            'bp_systolic': int(input_data.bp_systolic),
            'bp_diastolic': int(input_data.bp_diastolic),
            'alt': float(input_data.alt),
            'ldl': float(input_data.ldl),
            'hdl': float(input_data.hdl),
            'triglycerides': float(input_data.triglycerides),
            'previous_prediabetes': input_data.previous_prediabetes,
            'hypertension': input_data.hypertension,
            'ckd': input_data.ckd,
            'cvd': input_data.cvd,
            'nafld': input_data.nafld,
            'retinopathy': input_data.retinopathy
        }

        # ALWAYS use predict_with_specific_version (no cache, guaranteed version accuracy)
        prediction_result = self.ml_service.predict_with_specific_version(
            patient_features=patient_features,
            version=model_version,
            include_explanation=False
        )

        # Extract prediction data
        prediction = prediction_result['prediction']
        predicted_treatment = prediction.recommended_treatment
        confidence_score = prediction.confidence_score
        predicted_reduction = prediction.predicted_reduction

        # Get all Q-values
        all_q_values = {}
        for q_val in prediction.all_q_values:
            all_q_values[q_val['treatment']] = float(q_val['q_value'])

        # Compare with actual treatment (normalized for safety)
        is_correct = (
            self._normalize_treatment(predicted_treatment) ==
            self._normalize_treatment(input_data.actual_treatment)
        )

        return BatchPredictionResult(
            id=input_data.id,
            predicted_treatment=predicted_treatment,
            actual_treatment=input_data.actual_treatment,
            is_correct=is_correct,
            confidence_score=float(confidence_score),
            predicted_reduction=float(predicted_reduction),
            all_q_values=all_q_values
        )

    def _normalize_treatment(self, treatment: str) -> str:
        """
        Normalize treatment name for safe comparison.

        Handles:
        - Case differences (Metformin vs metformin)
        - Leading/trailing whitespace
        - Extra spaces

        Args:
            treatment: Treatment name to normalize

        Returns:
            Normalized treatment name (lowercase, trimmed)

        Example:
            "Metformin " -> "metformin"
            "GLP-1" -> "glp-1"
            " SGLT-2  " -> "sglt-2"
        """
        return treatment.strip().lower()

    def _validate_model_version(self, model_version: str):
        """
        Validate that model version exists.

        Args:
            model_version: Model version to validate

        Raises:
            NotFoundException: If model version not found
        """
        available = [v['version_number'] for v in self.ml_service.list_all_versions()]

        if model_version not in available:
            error = ErrorDetail(
                title="Model Version Not Found",
                code="MODEL_NOT_FOUND",
                status=404,
                details=[
                    f"Model version '{model_version}' does not exist",
                    f"Available versions: {', '.join(available)}"
                ]
            )
            raise NotFoundException(
                message=f"Model version '{model_version}' not found",
                error_detail=error
            )

    def _validate_batch_inputs(self, predictions: List[BatchPredictionInput]):
        """
        Validate batch prediction inputs.

        Args:
            predictions: List of prediction inputs

        Raises:
            ValidationException: If validation fails
        """
        errors = []

        # Check for duplicate IDs
        ids = [p.id for p in predictions]
        duplicate_ids = [id for id in ids if ids.count(id) > 1]
        if duplicate_ids:
            errors.append(f"Duplicate IDs found: {', '.join(set(duplicate_ids))}")

        # Validate treatments (with normalization for better error messages)
        valid_treatments_normalized = [self._normalize_treatment(t) for t in self.VALID_TREATMENTS]

        for i, prediction in enumerate(predictions):
            if self._normalize_treatment(prediction.actual_treatment) not in valid_treatments_normalized:
                errors.append(
                    f"Invalid treatment at index {i} (ID: {prediction.id}): "
                    f"'{prediction.actual_treatment}'. Must be one of: {', '.join(self.VALID_TREATMENTS)}"
                )

            # Validate gender
            if prediction.gender not in ['Male', 'Female']:
                errors.append(
                    f"Invalid gender at index {i} (ID: {prediction.id}): "
                    f"'{prediction.gender}'. Must be 'Male' or 'Female'"
                )

            # Validate ethnicity
            valid_ethnicities = ['Caucasian', 'African', 'Asian', 'Hispanic', 'Other']
            if prediction.ethnicity not in valid_ethnicities:
                errors.append(
                    f"Invalid ethnicity at index {i} (ID: {prediction.id}): "
                    f"'{prediction.ethnicity}'. Must be one of: {', '.join(valid_ethnicities)}"
                )

        if errors:
            error = ErrorDetail(
                title="Batch Validation Failed",
                code="BATCH_VALIDATION_ERROR",
                status=400,
                details=errors
            )
            raise ValidationException(
                message="Batch prediction validation failed",
                error_detail=error
            )