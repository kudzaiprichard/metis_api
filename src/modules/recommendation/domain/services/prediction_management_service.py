"""
Prediction management service for retrieving and managing recommendations.
"""

from typing import List, Tuple

from src.modules.recommendation.presentation.dtos.prediction_dtos import (
    GetPredictionRequest,
    ListPredictionsRequest,
    PredictionResponse,
    PredictionDetailResponse,
    PatientSummaryResponse
)
from src.modules.recommendation.domain.repositories.prediction_repository import PredictionRepository
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.patient_medical_data_repository import PatientMedicalDataRepository

from src.shared.exceptions.exceptions import NotFoundException
from src.shared.response.error_detail import ErrorDetail


class PredictionManagementService:
    """
    Service for managing and retrieving recommendations.
    """

    def __init__(self):
        self.prediction_repository = PredictionRepository()
        self.patient_repository = PatientRepository()
        self.medical_data_repository = PatientMedicalDataRepository()

    def _build_patient_summary(self, medical_data_id: str) -> PatientSummaryResponse:
        """
        Build patient summary from medical data ID.
        Resolves patient through the medical data record.

        Args:
            medical_data_id: Medical data ID

        Returns:
            PatientSummaryResponse DTO
        """
        medical_data = self.medical_data_repository.find_by_id(medical_data_id)
        patient = self.patient_repository.find_by_id(medical_data.patient_id)

        return PatientSummaryResponse(
            id=patient.id,
            first_name=patient.first_name,
            last_name=patient.last_name,
            age=medical_data.age,
            gender=medical_data.gender.value
        )

    def get_prediction(self, request: GetPredictionRequest) -> PredictionDetailResponse:
        """
        Get prediction by ID with full details.

        Args:
            request: GetPredictionRequest DTO

        Returns:
            PredictionDetailResponse DTO

        Raises:
            NotFoundException: If prediction not found
        """
        prediction = self.prediction_repository.find_by_id(request.prediction_id)

        if not prediction:
            error = ErrorDetail(
                title="Prediction Not Found",
                code="PREDICTION_NOT_FOUND",
                status=404,
                details=[f"Prediction with ID {request.prediction_id} does not exist"]
            )
            raise NotFoundException(
                message="The prediction you're looking for doesn't exist",
                error_detail=error
            )

        # Build patient summary via medical_data_id
        patient_summary = self._build_patient_summary(prediction.medical_data_id)

        # Build response with patient info
        response_dict = prediction.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return PredictionDetailResponse(**response_dict)

    def list_predictions(self, request: ListPredictionsRequest) -> Tuple[List[PredictionResponse], int]:
        """
        List recommendations with pagination and optional patient filter.

        Args:
            request: ListPredictionsRequest DTO

        Returns:
            Tuple of (list of PredictionResponse DTOs, total count)
        """
        if request.patient_id:
            # Get predictions via join through medical data
            predictions = self.prediction_repository.find_by_patient_id(request.patient_id)
            total = len(predictions)

            # Manual pagination
            start = request.get_offset()
            end = start + request.per_page
            predictions = predictions[start:end]
        else:
            # Get all predictions with standard pagination
            total = self.prediction_repository.count()
            pagination = self.prediction_repository.paginate(
                page=request.page,
                per_page=request.per_page,
                include_deleted=False
            )
            predictions = pagination.items

        # Convert to response DTOs with patient info
        prediction_responses = []
        for pred in predictions:
            patient_summary = self._build_patient_summary(pred.medical_data_id)
            response_dict = pred.to_dict()
            response_dict['patient'] = patient_summary.model_dump()
            prediction_responses.append(PredictionResponse(**response_dict))

        return prediction_responses, total

    def delete_prediction(self, prediction_id: str) -> None:
        """
        Soft delete a prediction.

        Args:
            prediction_id: Prediction ID to delete

        Raises:
            NotFoundException: If prediction not found
        """
        prediction = self.prediction_repository.find_by_id(prediction_id)

        if not prediction:
            error = ErrorDetail(
                title="Prediction Not Found",
                code="PREDICTION_NOT_FOUND",
                status=404,
                details=[f"Prediction with ID {prediction_id} does not exist"]
            )
            raise NotFoundException(
                message="The prediction you're trying to delete doesn't exist",
                error_detail=error
            )

        self.prediction_repository.delete(prediction)