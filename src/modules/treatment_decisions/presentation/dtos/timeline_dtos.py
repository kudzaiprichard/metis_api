# src/modules/treatment_decisions/application/dtos/timeline_dtos.py
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel


# ============ Timeline Event ============

class TimelineEvent(BaseModel):
    """Individual timeline event."""
    type: str  # 'patient_created', 'prediction_generated', 'treatment_decision', etc.
    timestamp: datetime
    data: dict  # Event-specific data

    model_config = {
        'from_attributes': True
    }


# ============ Patient Summary for Timeline ============

class TimelinePatientSummary(BaseModel):
    """Minimal patient info for timeline."""
    id: str
    first_name: str
    last_name: str
    age: int
    gender: str

    model_config = {
        'from_attributes': True
    }


# ============ Patient Timeline Response ============

class PatientTimelineResponse(BaseModel):
    """Complete patient timeline response."""
    patient: TimelinePatientSummary
    timeline: List[TimelineEvent]
    total_events: int

    model_config = {
        'from_attributes': True
    }


# ============ Get Patient Timeline Request ============

class GetPatientTimelineRequest(BaseModel):
    """DTO for getting patient timeline."""
    patient_id: str
    limit: Optional[int] = None  # Optional limit on number of events