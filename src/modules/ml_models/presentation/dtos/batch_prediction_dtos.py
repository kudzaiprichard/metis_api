"""
DTOs for Batch Prediction endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from decimal import Decimal


# ============ Single Batch Prediction Input ============

class BatchPredictionInput(BaseModel):
    """Single patient data for batch prediction."""
    id: str = Field(..., description="Unique identifier for this prediction")

    # 21 Base Features
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

    # Actual treatment (for comparison)
    actual_treatment: str = Field(
        ...,
        description="Actual treatment given: 'Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', or 'Insulin'"
    )


# ============ Batch Prediction Request ============

class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    predictions: List[BatchPredictionInput] = Field(
        ...,
        min_length=1,
        description="List of patient data for batch prediction"
    )
    model_version: str = Field(
        ...,
        description="Specific model version to use (e.g., 'v1_0', 'v1_2') - REQUIRED"
    )


# ============ Single Batch Prediction Result ============

class BatchPredictionResult(BaseModel):
    """Result for a single batch prediction."""
    id: str
    predicted_treatment: str
    actual_treatment: str
    is_correct: bool
    confidence_score: float = Field(..., description="Confidence percentage (0-100)")
    predicted_reduction: float = Field(..., description="Expected HbA1c reduction (%)")

    # Q-values for all treatments (optional, for debugging)
    all_q_values: Optional[dict] = Field(
        None,
        description="Q-values for all 5 treatments"
    )


# ============ Batch Prediction Response ============

class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    total_predictions: int
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float = Field(..., description="Accuracy percentage (0-100)")
    model_version_used: str
    results: List[BatchPredictionResult]


# ============ Batch Prediction Summary ============

class BatchPredictionSummary(BaseModel):
    """Summary statistics for batch predictions."""
    total_predictions: int
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float
    model_version_used: str

    # Per-treatment breakdown
    treatment_breakdown: dict = Field(
        ...,
        description="Accuracy breakdown per treatment type"
    )