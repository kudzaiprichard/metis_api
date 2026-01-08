"""
DTOs for Online Learning endpoints.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from decimal import Decimal


# ============ Single Patient Outcome Input ============

class PatientOutcomeInput(BaseModel):
    """Single patient outcome for online learning."""

    # Patient medical data (21 base features)
    age: int = Field(..., ge=18, le=120)
    gender: str = Field(..., description="Must be 'Male' or 'Female'")
    ethnicity: str = Field(..., description="Must be 'Caucasian', 'African', 'Asian', 'Hispanic', or 'Other'")
    hba1c_baseline: Decimal = Field(..., ge=4.0, le=20.0)
    diabetes_duration: Decimal = Field(..., ge=0.0, le=50.0)
    fasting_glucose: Decimal = Field(..., ge=50.0, le=500.0)
    c_peptide: Decimal = Field(..., ge=0.0, le=10.0)
    egfr: Decimal = Field(..., ge=0.0, le=150.0)
    bmi: Decimal = Field(..., ge=10.0, le=80.0)
    bp_systolic: int = Field(..., ge=70, le=250)
    bp_diastolic: int = Field(..., ge=40, le=150)
    alt: Decimal = Field(..., ge=0.0, le=500.0)
    ldl: Decimal = Field(..., ge=0.0, le=500.0)
    hdl: Decimal = Field(..., ge=0.0, le=200.0)
    triglycerides: Decimal = Field(..., ge=0.0, le=1000.0)
    previous_prediabetes: bool
    hypertension: bool
    ckd: bool
    cvd: bool
    nafld: bool
    retinopathy: bool

    # Treatment and outcome
    treatment_given: str = Field(
        ...,
        description="Treatment given: 'Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', or 'Insulin'"
    )
    reward: Decimal = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Observed HbA1c reduction (reward)"
    )


# ============ Online Learning Request ============

class OnlineLearningRequest(BaseModel):
    """Request for online learning training."""
    outcomes: List[PatientOutcomeInput] = Field(
        ...,
        min_length=1,
        description="List of patient outcomes for training"
    )
    base_version: str = Field(
        ...,
        min_length=1,
        description="Base model version to train from (e.g., 'v1_0') - REQUIRED"
    )
    validate: bool = Field(
        default=True,
        description="Whether to validate performance before/after training"
    )
    disable_ewc: bool = Field(
        default=False,
        description="Disable Elastic Weight Consolidation (for testing only)"
    )
    epochs: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of training epochs (1-100)"
    )


# ============ Performance Metrics Response ============

class PerformanceMetrics(BaseModel):
    """Performance metrics before/after training."""
    avg_reward: float = Field(..., description="Average predicted reward")
    accuracy: float = Field(..., description="Treatment selection accuracy")
    diversity: int = Field(..., description="Number of unique treatments recommended")
    success_rate: float = Field(..., description="Proportion of successful outcomes")


# ============ Online Learning Response ============

class OnlineLearningResponse(BaseModel):
    """Response for online learning training."""
    success: bool
    version_number: Optional[str] = Field(None, description="New model version created")
    base_version: str
    outcomes_processed: int
    performance_before: Optional[PerformanceMetrics] = None
    performance_after: Optional[PerformanceMetrics] = None
    timestamp: str
    error: Optional[str] = None
    training_info: Optional[Dict] = Field(
        None,
        description="Training configuration (epochs, EWC status, etc.)"
    )


# ============ Training Status Response ============

class TrainingStatusResponse(BaseModel):
    """Response for training status check."""
    is_training: bool
    current_step: Optional[str] = None
    progress_percent: int
    started_at: Optional[str] = None
    estimated_completion: Optional[str] = None
    version_number: Optional[str] = None
    outcomes_count: int
    error: Optional[str] = None