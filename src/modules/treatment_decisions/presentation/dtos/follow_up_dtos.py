from typing import Optional
from datetime import date, datetime
from pydantic import BaseModel, Field, field_validator
from decimal import Decimal

from src.modules.treatment_decisions.presentation.dtos.treatment_decision_dtos import PatientSummaryResponse


# ============ Shared Follow-up Responses ============

class FollowUpResponse(BaseModel):
    """Standard follow-up response DTO."""
    id: str
    patient_id: str
    patient: PatientSummaryResponse
    decision_id: str
    recorded_by: Optional[str]
    scheduled_date: date
    status: str
    visit_date: Optional[date]
    hba1c_new: Optional[Decimal]
    weight_new: Optional[Decimal]
    egfr_new: Optional[Decimal]
    bp_systolic_new: Optional[int]
    bp_diastolic_new: Optional[int]
    patient_status: Optional[str]
    adherence: Optional[str]
    adverse_events: Optional[str]
    patient_feedback: Optional[str]
    treatment_action: Optional[str]
    action_notes: Optional[str]
    created_at: datetime

    model_config = {
        'from_attributes': True
    }


# ============ Schedule Follow-up ============

class ScheduleFollowUpRequest(BaseModel):
    """DTO for scheduling a follow-up appointment."""
    patient_id: str = Field(..., min_length=1)
    decision_id: str = Field(..., min_length=1)
    scheduled_date: date = Field(..., description="Follow-up appointment date")

    @field_validator('scheduled_date')
    @classmethod
    def validate_future_date(cls, v):
        """Validate scheduled date is in the future."""
        if v < date.today():
            raise ValueError("scheduled_date must be in the future")
        return v


# ============ Complete Follow-up ============

class CompleteFollowUpRequest(BaseModel):
    """DTO for recording a completed follow-up visit."""
    visit_date: date = Field(..., description="Actual visit date")

    # New measurements
    hba1c_new: Optional[Decimal] = Field(None, ge=4.0, le=20.0)
    weight_new: Optional[Decimal] = Field(None, ge=30.0, le=300.0)
    egfr_new: Optional[Decimal] = Field(None, ge=0.0, le=150.0)
    bp_systolic_new: Optional[int] = Field(None, ge=70, le=250)
    bp_diastolic_new: Optional[int] = Field(None, ge=40, le=150)

    # Assessment
    patient_status: Optional[str] = Field(None, description="'improving', 'stable', or 'worsening'")
    adherence: Optional[str] = Field(None, description="'good', 'fair', or 'poor'")
    adverse_events: Optional[str] = None
    patient_feedback: Optional[str] = None

    # Treatment action
    treatment_action: Optional[str] = Field(None, description="'continue', 'adjust', or 'change'")
    action_notes: Optional[str] = Field(None, description="Required if treatment_action is 'adjust' or 'change'")

    @field_validator('patient_status')
    @classmethod
    def validate_patient_status(cls, v):
        """Validate patient status is valid."""
        if v is not None and v not in ['improving', 'stable', 'worsening']:
            raise ValueError("patient_status must be 'improving', 'stable', or 'worsening'")
        return v

    @field_validator('adherence')
    @classmethod
    def validate_adherence(cls, v):
        """Validate adherence is valid."""
        if v is not None and v not in ['good', 'fair', 'poor']:
            raise ValueError("adherence must be 'good', 'fair', or 'poor'")
        return v

    @field_validator('treatment_action')
    @classmethod
    def validate_treatment_action(cls, v):
        """Validate treatment action is valid."""
        if v is not None and v not in ['continue', 'adjust', 'change']:
            raise ValueError("treatment_action must be 'continue', 'adjust', or 'change'")
        return v

    @classmethod
    def model_validate(cls, value):
        """Validate that action_notes is provided when adjusting or changing treatment."""
        instance = super().model_validate(value)
        if instance.treatment_action in ['adjust', 'change'] and not instance.action_notes:
            raise ValueError("action_notes is required when treatment_action is 'adjust' or 'change'")
        return instance


# ============ Update Follow-up ============

class UpdateFollowUpRequest(BaseModel):
    """DTO for updating a scheduled follow-up."""
    scheduled_date: Optional[date] = None

    @field_validator('scheduled_date')
    @classmethod
    def validate_future_date(cls, v):
        """Validate scheduled date is in the future."""
        if v is not None and v < date.today():
            raise ValueError("scheduled_date must be in the future")
        return v


# ============ Cancel Follow-up ============

class CancelFollowUpRequest(BaseModel):
    """DTO for cancelling a follow-up."""
    follow_up_id: str = Field(..., min_length=1)


# ============ Get Single Follow-up ============

class GetFollowUpRequest(BaseModel):
    """DTO for getting a single follow-up by ID."""
    follow_up_id: str = Field(..., min_length=1)


# ============ Get Patient Follow-ups ============

class GetPatientFollowUpsRequest(BaseModel):
    """DTO for getting all follow-ups for a patient."""
    patient_id: str = Field(..., min_length=1)
    status: Optional[str] = Field(None, description="Filter by status: 'scheduled', 'completed', 'cancelled'")


# ============ Get Upcoming Follow-ups ============

class GetUpcomingFollowUpsRequest(BaseModel):
    """DTO for getting upcoming scheduled follow-ups."""
    page: int = Field(default=1, ge=1, description="Page number (starts at 1)")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page (max 100)")

    def get_offset(self) -> int:
        """Calculate database offset for pagination."""
        return (self.page - 1) * self.per_page

    def get_limit(self) -> int:
        """Get limit for database query."""
        return self.per_page