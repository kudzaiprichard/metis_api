# src/modules/treatment_decisions/application/dtos/treatment_decision_dtos.py
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from decimal import Decimal
from src.modules.treatment_decisions.presentation.dtos.patient_summary_dto import PatientSummaryResponse


# ============ Shared Treatment Decision Responses ============

class TreatmentDecisionResponse(BaseModel):
    """Standard treatment decision response DTO."""
    id: str
    prediction_id: str
    patient_id: str
    patient: PatientSummaryResponse
    decided_by: str
    decision_type: str
    treatment_given: str
    reasoning_notes: Optional[str]
    dosage: Optional[str]
    observed_reduction: Optional[Decimal]
    outcome_recorded_at: Optional[datetime]
    used_for_training: bool
    decided_at: datetime
    created_at: datetime

    model_config = {
        'from_attributes': True
    }


# ============ Record Treatment Decision ============

class RecordTreatmentDecisionRequest(BaseModel):
    """DTO for recording a treatment decision."""
    prediction_id: str = Field(..., min_length=1)
    patient_id: str = Field(..., min_length=1)
    decision_type: str = Field(..., description="Must be 'accepted' or 'custom'")
    treatment_given: str = Field(..., min_length=1)
    reasoning_notes: Optional[str] = Field(None, description="Required if decision_type='custom'")
    dosage: Optional[str] = Field(None, max_length=100)

    @classmethod
    def model_validate(cls, value):
        """Validate that reasoning_notes is provided for custom decisions."""
        instance = super().model_validate(value)
        if instance.decision_type == 'custom' and not instance.reasoning_notes:
            raise ValueError("reasoning_notes is required when decision_type is 'custom'")
        return instance

    def validate_decision_type(self, v):
        """Validate decision type is valid."""
        if v not in ['accepted', 'custom']:
            raise ValueError("decision_type must be 'accepted' or 'custom'")
        return v


# ============ Update Treatment Outcome ============

class UpdateTreatmentOutcomeRequest(BaseModel):
    """DTO for updating treatment outcome after follow-up."""
    observed_reduction: Decimal = Field(..., ge=-10.0, le=10.0, description="Actual HbA1c reduction (%)")


# ============ Get Single Decision ============

class GetTreatmentDecisionRequest(BaseModel):
    """DTO for getting a single treatment decision by ID."""
    decision_id: str = Field(..., min_length=1)


# ============ Get Patient Decisions ============

class GetPatientDecisionsRequest(BaseModel):
    """DTO for getting all decisions for a patient."""
    patient_id: str = Field(..., min_length=1)
    limit: Optional[int] = Field(None, ge=1, le=100, description="Max number of decisions to return")


# ============ List Treatment Decisions (Pagination) ============

class ListTreatmentDecisionsRequest(BaseModel):
    """DTO for listing treatment decisions with pagination."""
    page: int = Field(default=1, ge=1, description="Page number (starts at 1)")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page (max 100)")
    patient_id: Optional[str] = None
    decision_type: Optional[str] = None

    def get_offset(self) -> int:
        """Calculate database offset for pagination."""
        return (self.page - 1) * self.per_page

    def get_limit(self) -> int:
        """Get limit for database query."""
        return self.per_page