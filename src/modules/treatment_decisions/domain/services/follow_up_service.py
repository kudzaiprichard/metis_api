# src/modules/monitoring/application/services/follow_up_service.py
"""
Follow-up service for scheduling and recording patient visits.
"""

from typing import List
from datetime import datetime

from src.modules.patients.domain.repositories.patient_medical_data_repository import PatientMedicalDataRepository
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.recommendation.presentation.dtos.prediction_dtos import PatientSummaryResponse
from src.modules.treatment_decisions.domain.models.enums import FollowUpStatus, PatientStatus, Adherence, \
    TreatmentAction
from src.modules.treatment_decisions.domain.models.follow_up import FollowUp
from src.modules.treatment_decisions.domain.repositories.follow_up_repository import FollowUpRepository
from src.modules.treatment_decisions.domain.repositories.treatment_decision_repository import \
    TreatmentDecisionRepository
from src.modules.treatment_decisions.presentation.dtos.follow_up_dtos import ScheduleFollowUpRequest, \
    FollowUpResponse, CompleteFollowUpRequest, UpdateFollowUpRequest, CancelFollowUpRequest, GetFollowUpRequest, \
    GetPatientFollowUpsRequest, GetUpcomingFollowUpsRequest
from src.shared.exceptions.exceptions import NotFoundException, ValidationException
from src.shared.response.error_detail import ErrorDetail


class FollowUpService:
    """
    Service for follow-up operations.
    """

    def __init__(self):
        self.follow_up_repository = FollowUpRepository()
        self.decision_repository = TreatmentDecisionRepository()
        self.patient_repository = PatientRepository()
        self.medical_data_repository = PatientMedicalDataRepository()

    def _build_patient_summary(self, patient_id: str) -> PatientSummaryResponse:
        """
        Build patient summary for follow-up responses.

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

    def schedule_follow_up(self, request: ScheduleFollowUpRequest) -> FollowUpResponse:
        """
        Schedule a follow-up appointment.

        Args:
            request: ScheduleFollowUpRequest DTO

        Returns:
            FollowUpResponse DTO

        Raises:
            NotFoundException: If patient or decision not found
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
                message="The patient you're scheduling follow-up for doesn't exist",
                error_detail=error
            )

        # Validate treatment decision exists
        decision = self.decision_repository.find_by_id(request.decision_id)
        if not decision:
            error = ErrorDetail(
                title="Decision Not Found",
                code="DECISION_NOT_FOUND",
                status=404,
                details=[f"Treatment decision with ID {request.decision_id} does not exist"]
            )
            raise NotFoundException(
                message="The treatment decision doesn't exist",
                error_detail=error
            )

        # Create follow-up
        follow_up = FollowUp(
            patient_id=request.patient_id,
            decision_id=request.decision_id,
            scheduled_date=request.scheduled_date,
            status=FollowUpStatus.SCHEDULED
        )

        saved_follow_up = self.follow_up_repository.create(follow_up)

        # Build patient summary
        patient_summary = self._build_patient_summary(request.patient_id)

        # Build response with patient info
        response_dict = saved_follow_up.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return FollowUpResponse(**response_dict)

    def complete_follow_up(self, follow_up_id: str, request: CompleteFollowUpRequest, recorded_by_user_id: str) -> FollowUpResponse:
        """
        Record a completed follow-up visit.

        Args:
            follow_up_id: Follow-up ID
            request: CompleteFollowUpRequest DTO
            recorded_by_user_id: ID of the doctor recording the visit

        Returns:
            FollowUpResponse DTO

        Raises:
            NotFoundException: If follow-up not found
            ValidationException: If action notes missing for adjust/change
        """
        follow_up = self.follow_up_repository.find_by_id(follow_up_id)

        if not follow_up:
            error = ErrorDetail(
                title="Follow-up Not Found",
                code="FOLLOWUP_NOT_FOUND",
                status=404,
                details=[f"Follow-up with ID {follow_up_id} does not exist"]
            )
            raise NotFoundException(
                message="The follow-up you're trying to complete doesn't exist",
                error_detail=error
            )

        # Validate action notes for adjust/change
        if request.treatment_action in ['adjust', 'change'] and not request.action_notes:
            error = ErrorDetail(
                title="Validation Failed",
                code="ACTION_NOTES_REQUIRED",
                status=400
            )
            error.add_field_error("action_notes", "Action notes are required when adjusting or changing treatment")
            raise ValidationException(
                message="Action notes required for treatment adjustments",
                error_detail=error
            )

        # Update follow-up
        follow_up.status = FollowUpStatus.COMPLETED
        follow_up.recorded_by = recorded_by_user_id
        follow_up.visit_date = request.visit_date
        follow_up.hba1c_new = request.hba1c_new
        follow_up.weight_new = request.weight_new
        follow_up.egfr_new = request.egfr_new
        follow_up.bp_systolic_new = request.bp_systolic_new
        follow_up.bp_diastolic_new = request.bp_diastolic_new
        follow_up.patient_status = PatientStatus[request.patient_status.upper()] if request.patient_status else None
        follow_up.adherence = Adherence[request.adherence.upper()] if request.adherence else None
        follow_up.adverse_events = request.adverse_events
        follow_up.patient_feedback = request.patient_feedback
        follow_up.treatment_action = TreatmentAction[request.treatment_action.upper()] if request.treatment_action else None
        follow_up.action_notes = request.action_notes

        updated_follow_up = self.follow_up_repository.update(follow_up)

        # Update treatment decision outcome if HbA1c provided
        if request.hba1c_new:
            decision = self.decision_repository.find_by_id(follow_up.decision_id)
            if decision:
                # Calculate observed reduction (baseline - new)
                # Note: This needs patient's baseline HbA1c
                # For now, just update the timestamp
                decision.outcome_recorded_at = datetime.now()
                self.decision_repository.update(decision)

        # Build patient summary
        patient_summary = self._build_patient_summary(follow_up.patient_id)

        # Build response with patient info
        response_dict = updated_follow_up.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return FollowUpResponse(**response_dict)

    def update_follow_up(self, follow_up_id: str, request: UpdateFollowUpRequest) -> FollowUpResponse:
        """
        Update a scheduled follow-up.

        Args:
            follow_up_id: Follow-up ID
            request: UpdateFollowUpRequest DTO

        Returns:
            FollowUpResponse DTO

        Raises:
            NotFoundException: If follow-up not found
        """
        follow_up = self.follow_up_repository.find_by_id(follow_up_id)

        if not follow_up:
            error = ErrorDetail(
                title="Follow-up Not Found",
                code="FOLLOWUP_NOT_FOUND",
                status=404,
                details=[f"Follow-up with ID {follow_up_id} does not exist"]
            )
            raise NotFoundException(
                message="The follow-up you're trying to update doesn't exist",
                error_detail=error
            )

        # Update scheduled date if provided
        if request.scheduled_date:
            follow_up.scheduled_date = request.scheduled_date

        updated_follow_up = self.follow_up_repository.update(follow_up)

        # Build patient summary
        patient_summary = self._build_patient_summary(follow_up.patient_id)

        # Build response with patient info
        response_dict = updated_follow_up.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return FollowUpResponse(**response_dict)

    def cancel_follow_up(self, request: CancelFollowUpRequest) -> None:
        """
        Cancel a follow-up appointment.

        Args:
            request: CancelFollowUpRequest DTO

        Raises:
            NotFoundException: If follow-up not found
        """
        follow_up = self.follow_up_repository.find_by_id(request.follow_up_id)

        if not follow_up:
            error = ErrorDetail(
                title="Follow-up Not Found",
                code="FOLLOWUP_NOT_FOUND",
                status=404,
                details=[f"Follow-up with ID {request.follow_up_id} does not exist"]
            )
            raise NotFoundException(
                message="The follow-up you're trying to cancel doesn't exist",
                error_detail=error
            )

        follow_up.status = FollowUpStatus.CANCELLED
        self.follow_up_repository.update(follow_up)

    def get_follow_up(self, request: GetFollowUpRequest) -> FollowUpResponse:
        """
        Get follow-up by ID.

        Args:
            request: GetFollowUpRequest DTO

        Returns:
            FollowUpResponse DTO

        Raises:
            NotFoundException: If follow-up not found
        """
        follow_up = self.follow_up_repository.find_by_id(request.follow_up_id)

        if not follow_up:
            error = ErrorDetail(
                title="Follow-up Not Found",
                code="FOLLOWUP_NOT_FOUND",
                status=404,
                details=[f"Follow-up with ID {request.follow_up_id} does not exist"]
            )
            raise NotFoundException(
                message="The follow-up you're looking for doesn't exist",
                error_detail=error
            )

        # Build patient summary
        patient_summary = self._build_patient_summary(follow_up.patient_id)

        # Build response with patient info
        response_dict = follow_up.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return FollowUpResponse(**response_dict)

    def get_patient_follow_ups(self, request: GetPatientFollowUpsRequest) -> List[FollowUpResponse]:
        """
        Get all follow-ups for a patient.

        Args:
            request: GetPatientFollowUpsRequest DTO

        Returns:
            List of FollowUpResponse DTOs

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

        # Get follow-ups
        if request.status:
            status_enum = FollowUpStatus[request.status.upper()]
            follow_ups = self.follow_up_repository.find_by_status(status_enum)
            follow_ups = [f for f in follow_ups if f.patient_id == request.patient_id]
        else:
            follow_ups = self.follow_up_repository.find_by_patient_id(request.patient_id)

        # Build patient summary once
        patient_summary = self._build_patient_summary(request.patient_id)

        # Convert to response DTOs with patient info
        follow_up_responses = []
        for follow_up in follow_ups:
            response_dict = follow_up.to_dict()
            response_dict['patient'] = patient_summary.model_dump()
            follow_up_responses.append(FollowUpResponse(**response_dict))

        return follow_up_responses

    def get_upcoming_follow_ups(self, request: GetUpcomingFollowUpsRequest) -> List[FollowUpResponse]:
        """
        Get upcoming scheduled follow-ups.

        Args:
            request: GetUpcomingFollowUpsRequest DTO

        Returns:
            List of FollowUpResponse DTOs
        """
        follow_ups = self.follow_up_repository.find_upcoming()

        # Apply pagination
        start = request.get_offset()
        end = start + request.per_page
        paginated_follow_ups = follow_ups[start:end]

        # Convert to response DTOs with patient info
        follow_up_responses = []
        for follow_up in paginated_follow_ups:
            patient_summary = self._build_patient_summary(follow_up.patient_id)
            response_dict = follow_up.to_dict()
            response_dict['patient'] = patient_summary.model_dump()
            follow_up_responses.append(FollowUpResponse(**response_dict))

        return follow_up_responses

    def delete_follow_up(self, follow_up_id: str) -> None:
        """
        Soft delete a follow-up.

        Args:
            follow_up_id: Follow-up ID to delete

        Raises:
            NotFoundException: If follow-up not found
        """
        follow_up = self.follow_up_repository.find_by_id(follow_up_id)

        if not follow_up:
            error = ErrorDetail(
                title="Follow-up Not Found",
                code="FOLLOWUP_NOT_FOUND",
                status=404,
                details=[f"Follow-up with ID {follow_up_id} does not exist"]
            )
            raise NotFoundException(
                message="The follow-up you're trying to delete doesn't exist",
                error_detail=error
            )

        self.follow_up_repository.delete(follow_up)