"""
Patient timeline service for aggregating complete patient journey.
"""
from src.modules.patients.domain.repositories.patient_medical_data_repository import PatientMedicalDataRepository
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.recommendation.domain.repositories.prediction_repository import PredictionRepository
from src.modules.treatment_decisions.domain.repositories.follow_up_repository import FollowUpRepository
from src.modules.treatment_decisions.domain.repositories.treatment_decision_repository import \
    TreatmentDecisionRepository
from src.modules.treatment_decisions.presentation.dtos.timeline_dtos import TimelineEvent, GetPatientTimelineRequest, \
    PatientTimelineResponse, TimelinePatientSummary
from src.shared.exceptions.exceptions import NotFoundException
from src.shared.response.error_detail import ErrorDetail


class PatientTimelineService:
    """
    Service for patient timeline aggregation.
    """

    def __init__(self):
        self.patient_repository = PatientRepository()
        self.medical_data_repository = PatientMedicalDataRepository()
        self.prediction_repository = PredictionRepository()
        self.decision_repository = TreatmentDecisionRepository()
        self.follow_up_repository = FollowUpRepository()

    def get_patient_timeline(self, request: GetPatientTimelineRequest) -> PatientTimelineResponse:
        """
        Get complete patient timeline.

        Args:
            request: GetPatientTimelineRequest DTO

        Returns:
            PatientTimelineResponse DTO

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

        medical_data = self.medical_data_repository.find_by_patient_id(request.patient_id)

        # Build patient summary
        patient_summary = TimelinePatientSummary(
            id=patient.id,
            first_name=patient.first_name,
            last_name=patient.last_name,
            age=medical_data.age if medical_data else 0,
            gender=medical_data.gender.value if medical_data else "Unknown"
        )

        # Collect timeline events
        timeline_events = []

        # 1. Patient created event
        timeline_events.append(TimelineEvent(
            type="patient_created",
            timestamp=patient.created_at,
            data={
                "initial_hba1c": float(medical_data.hba1c_baseline) if medical_data else None,
                "created_by": "Doctor"  # TODO: Get actual doctor name
            }
        ))

        # 2. Predictions generated events
        predictions = self.prediction_repository.find_by_patient_id(request.patient_id)
        for pred in predictions:
            timeline_events.append(TimelineEvent(
                type="prediction_generated",
                timestamp=pred.created_at,
                data={
                    "prediction_id": pred.id,
                    "model_version": pred.model_version,
                    "recommended_treatment": pred.recommended_treatment.value,
                    "predicted_reduction": float(pred.predicted_reduction),
                    "confidence_score": float(pred.confidence_score)
                }
            ))

        # 3. Treatment decision events
        decisions = self.decision_repository.find_by_patient_id(request.patient_id)
        for decision in decisions:
            timeline_events.append(TimelineEvent(
                type="treatment_decision",
                timestamp=decision.decided_at,
                data={
                    "decision_id": decision.id,
                    "decision_type": decision.decision_type.value,
                    "treatment_given": decision.treatment_given,
                    "dosage": decision.dosage,
                    "decided_by": "Doctor"  # TODO: Get actual doctor name
                }
            ))

        # 4. Follow-up events
        follow_ups = self.follow_up_repository.find_by_patient_id(request.patient_id)
        for follow_up in follow_ups:
            if follow_up.status.value == 'scheduled':
                timeline_events.append(TimelineEvent(
                    type="followup_scheduled",
                    timestamp=follow_up.created_at,
                    data={
                        "followup_id": follow_up.id,
                        "scheduled_date": follow_up.scheduled_date.isoformat(),
                        "scheduled_by": "Doctor"  # TODO: Get actual doctor name
                    }
                ))
            elif follow_up.status.value == 'completed':
                timeline_events.append(TimelineEvent(
                    type="followup_completed",
                    timestamp=follow_up.created_at,
                    data={
                        "followup_id": follow_up.id,
                        "hba1c_new": float(follow_up.hba1c_new) if follow_up.hba1c_new else None,
                        "patient_status": follow_up.patient_status.value if follow_up.patient_status else None,
                        "adherence": follow_up.adherence.value if follow_up.adherence else None,
                        "treatment_action": follow_up.treatment_action.value if follow_up.treatment_action else None,
                        "recorded_by": "Doctor"  # TODO: Get actual doctor name
                    }
                ))

        # Sort timeline by timestamp (most recent first)
        timeline_events.sort(key=lambda x: x.timestamp, reverse=True)

        # Apply limit if specified
        if request.limit:
            timeline_events = timeline_events[:request.limit]

        return PatientTimelineResponse(
            patient=patient_summary,
            timeline=timeline_events,
            total_events=len(timeline_events)
        )