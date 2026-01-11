"""
Prediction management service for retrieving and managing recommendation.
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
    Service for managing and retrieving recommendation.
    """

    def __init__(self):
        self.prediction_repository = PredictionRepository()
        self.patient_repository = PatientRepository()
        self.medical_data_repository = PatientMedicalDataRepository()

    def _build_patient_summary(self, patient_id: str) -> PatientSummaryResponse:
        """
        Build patient summary for prediction responses.

        Args:
            patient_id: Patient ID

        Returns:
            PatientSummaryResponse DTO
        """
        patient = self.patient_repository.find_by_id(patient_id)
        medical_data = self.medical_data_repository.find_by_patient_id(patient_id)

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

        # Build patient summary
        patient_summary = self._build_patient_summary(prediction.patient_id)

        # Build response with patient info
        response_dict = prediction.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return PredictionDetailResponse(**response_dict)

    def list_predictions(self, request: ListPredictionsRequest) -> Tuple[List[PredictionResponse], int]:
        """
        List recommendation with pagination and optional filters.

        Args:
            request: ListPredictionsRequest DTO

        Returns:
            Tuple of (list of PredictionResponse DTOs, total count)
        """
        # Build filter dictionary
        filters = {}
        if request.patient_id:
            filters['patient_id'] = request.patient_id

        # Get total count
        total = self.prediction_repository.count(filters)

        # Get paginated recommendation
        pagination = self.prediction_repository.paginate(
            page=request.page,
            per_page=request.per_page,
            include_deleted=False
        )

        recommendation = pagination.items

        # Apply patient_id filter if specified
        if request.patient_id:
            recommendation = [p for p in recommendation if p.patient_id == request.patient_id]

        # Convert to response DTOs with patient info
        prediction_responses = []
        for pred in recommendation:
            patient_summary = self._build_patient_summary(pred.patient_id)
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

        # Soft delete (cascades to related records via relationships)
        self.prediction_repository.delete(prediction)