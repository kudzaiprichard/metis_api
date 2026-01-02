"""
Treatment decision service for recording doctor's treatment choices.
"""

from typing import List, Tuple

from src.modules.patients.domain.repositories.patient_medical_data_repository import PatientMedicalDataRepository
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.recommendation.domain.repositories.prediction_repository import PredictionRepository
from src.modules.treatment_decisions.domain.models.enums import DecisionType
from src.modules.treatment_decisions.domain.models.treatment_decision import TreatmentDecision
from src.modules.treatment_decisions.domain.repositories.treatment_decision_repository import \
    TreatmentDecisionRepository
from src.modules.treatment_decisions.presentation.dtos.follow_up_dtos import PatientSummaryResponse
from src.modules.treatment_decisions.presentation.dtos.treatment_decision_dtos import RecordTreatmentDecisionRequest, \
    TreatmentDecisionResponse, UpdateTreatmentOutcomeRequest, GetTreatmentDecisionRequest, GetPatientDecisionsRequest, \
    ListTreatmentDecisionsRequest
from src.shared.exceptions.exceptions import NotFoundException, ValidationException, ConflictException
from src.shared.response.error_detail import ErrorDetail




class TreatmentDecisionService:
    """
    Service for treatment decision operations.
    """

    def __init__(self):
        self.decision_repository = TreatmentDecisionRepository()
        self.prediction_repository = PredictionRepository()
        self.patient_repository = PatientRepository()
        self.medical_data_repository = PatientMedicalDataRepository()

    def _build_patient_summary(self, patient_id: str) -> PatientSummaryResponse:
        """
        Build patient summary for decision responses.

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

    def record_decision(self, request: RecordTreatmentDecisionRequest, decided_by_user_id: str) -> TreatmentDecisionResponse:
        """
        Record doctor's treatment decision.

        Args:
            request: RecordTreatmentDecisionRequest DTO
            decided_by_user_id: ID of the doctor making the decision

        Returns:
            TreatmentDecisionResponse DTO

        Raises:
            NotFoundException: If patient or prediction not found
            ConflictException: If decision already exists for this prediction
            ValidationException: If custom decision missing reasoning notes
        """
        # 1. Validate patient exists
        patient = self.patient_repository.find_by_id(request.patient_id)
        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {request.patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're recording decision for doesn't exist",
                error_detail=error
            )

        # 2. Validate prediction exists
        prediction = self.prediction_repository.find_by_id(request.prediction_id)
        if not prediction:
            error = ErrorDetail(
                title="Prediction Not Found",
                code="PREDICTION_NOT_FOUND",
                status=404,
                details=[f"Prediction with ID {request.prediction_id} does not exist"]
            )
            raise NotFoundException(
                message="The prediction you're recording decision for doesn't exist",
                error_detail=error
            )

        # 3. Check if decision already exists for this prediction
        existing_decision = self.decision_repository.find_by_prediction_id(request.prediction_id)
        if existing_decision:
            error = ErrorDetail(
                title="Decision Already Exists",
                code="DECISION_EXISTS",
                status=409,
                details=["A treatment decision has already been recorded for this prediction"]
            )
            raise ConflictException(
                message="Treatment decision already exists for this prediction",
                error_detail=error
            )

        # 4. Validate custom decision has reasoning notes
        if request.decision_type == 'custom' and not request.reasoning_notes:
            error = ErrorDetail(
                title="Validation Failed",
                code="REASONING_REQUIRED",
                status=400
            )
            error.add_field_error("reasoning_notes", "Reasoning notes are required for custom treatment decisions")
            raise ValidationException(
                message="Custom treatment decisions require reasoning notes",
                error_detail=error
            )

        # 5. Create treatment decision
        decision = TreatmentDecision(
            prediction_id=request.prediction_id,
            patient_id=request.patient_id,
            decided_by=decided_by_user_id,
            decision_type=DecisionType[request.decision_type.upper()],
            treatment_given=request.treatment_given,
            reasoning_notes=request.reasoning_notes,
            dosage=request.dosage
        )

        saved_decision = self.decision_repository.create(decision)

        # Build patient summary
        patient_summary = self._build_patient_summary(request.patient_id)

        # Build response with patient info
        response_dict = saved_decision.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return TreatmentDecisionResponse(**response_dict)

    def update_outcome(self, decision_id: str, request: UpdateTreatmentOutcomeRequest) -> TreatmentDecisionResponse:
        """
        Update treatment outcome after follow-up.

        Args:
            decision_id: Treatment decision ID
            request: UpdateTreatmentOutcomeRequest DTO

        Returns:
            TreatmentDecisionResponse DTO

        Raises:
            NotFoundException: If decision not found
        """
        decision = self.decision_repository.find_by_id(decision_id)

        if not decision:
            error = ErrorDetail(
                title="Decision Not Found",
                code="DECISION_NOT_FOUND",
                status=404,
                details=[f"Treatment decision with ID {decision_id} does not exist"]
            )
            raise NotFoundException(
                message="The treatment decision you're trying to update doesn't exist",
                error_detail=error
            )

        # Update outcome
        decision.observed_reduction = request.observed_reduction
        decision.outcome_recorded_at = decision.updated_at

        updated_decision = self.decision_repository.update(decision)

        # Build patient summary
        patient_summary = self._build_patient_summary(decision.patient_id)

        # Build response with patient info
        response_dict = updated_decision.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return TreatmentDecisionResponse(**response_dict)

    def get_decision(self, request: GetTreatmentDecisionRequest) -> TreatmentDecisionResponse:
        """
        Get treatment decision by ID.

        Args:
            request: GetTreatmentDecisionRequest DTO

        Returns:
            TreatmentDecisionResponse DTO

        Raises:
            NotFoundException: If decision not found
        """
        decision = self.decision_repository.find_by_id(request.decision_id)

        if not decision:
            error = ErrorDetail(
                title="Decision Not Found",
                code="DECISION_NOT_FOUND",
                status=404,
                details=[f"Treatment decision with ID {request.decision_id} does not exist"]
            )
            raise NotFoundException(
                message="The treatment decision you're looking for doesn't exist",
                error_detail=error
            )

        # Build patient summary
        patient_summary = self._build_patient_summary(decision.patient_id)

        # Build response with patient info
        response_dict = decision.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return TreatmentDecisionResponse(**response_dict)

    def get_patient_decisions(self, request: GetPatientDecisionsRequest) -> List[TreatmentDecisionResponse]:
        """
        Get all treatment decisions for a patient.

        Args:
            request: GetPatientDecisionsRequest DTO

        Returns:
            List of TreatmentDecisionResponse DTOs

        Raises:
            NotFoundException: If patient not found
        """
        # Validate patient exists
        patient = self.patient_repository.find_by_id(request.patient_id)
        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {request.patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're looking for doesn't exist",
                error_detail=error
            )

        # Get decisions
        decisions = self.decision_repository.find_by_patient_id(request.patient_id)

        # Apply limit if specified
        if request.limit:
            decisions = decisions[:request.limit]

        # Build patient summary once
        patient_summary = self._build_patient_summary(request.patient_id)

        # Convert to response DTOs with patient info
        decision_responses = []
        for decision in decisions:
            response_dict = decision.to_dict()
            response_dict['patient'] = patient_summary.model_dump()
            decision_responses.append(TreatmentDecisionResponse(**response_dict))

        return decision_responses

    def list_decisions(self, request: ListTreatmentDecisionsRequest) -> Tuple[List[TreatmentDecisionResponse], int]:
        """
        List treatment decisions with pagination and optional filters.

        Args:
            request: ListTreatmentDecisionsRequest DTO

        Returns:
            Tuple of (list of TreatmentDecisionResponse DTOs, total count)
        """
        # Build filter dictionary
        filters = {}
        if request.patient_id:
            filters['patient_id'] = request.patient_id
        if request.decision_type:
            filters['decision_type'] = DecisionType[request.decision_type.upper()]

        # Get total count
        total = self.decision_repository.count(filters)

        # Get paginated decisions
        pagination = self.decision_repository.paginate(
            page=request.page,
            per_page=request.per_page,
            include_deleted=False
        )

        decisions = pagination.items

        # Apply filters if specified
        if request.patient_id:
            decisions = [d for d in decisions if d.patient_id == request.patient_id]
        if request.decision_type:
            decision_type_enum = DecisionType[request.decision_type.upper()]
            decisions = [d for d in decisions if d.decision_type == decision_type_enum]

        # Convert to response DTOs with patient info
        decision_responses = []
        for decision in decisions:
            patient_summary = self._build_patient_summary(decision.patient_id)
            response_dict = decision.to_dict()
            response_dict['patient'] = patient_summary.model_dump()
            decision_responses.append(TreatmentDecisionResponse(**response_dict))

        return decision_responses, total

    def delete_decision(self, decision_id: str) -> None:
        """
        Soft delete a treatment decision.

        Args:
            decision_id: Treatment decision ID to delete

        Raises:
            NotFoundException: If decision not found
        """
        decision = self.decision_repository.find_by_id(decision_id)

        if not decision:
            error = ErrorDetail(
                title="Decision Not Found",
                code="DECISION_NOT_FOUND",
                status=404,
                details=[f"Treatment decision with ID {decision_id} does not exist"]
            )
            raise NotFoundException(
                message="The treatment decision you're trying to delete doesn't exist",
                error_detail=error
            )

        self.decision_repository.delete(decision)