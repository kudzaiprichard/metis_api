# src/modules/predictions/application/dtos/prediction_dtos.py
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from decimal import Decimal


# ============ Patient Summary for Predictions ============

class PatientSummaryResponse(BaseModel):
    """Minimal patient info for predictions."""
    id: str
    first_name: str
    last_name: str
    age: int
    gender: str

    model_config = {
        'from_attributes': True
    }


# ============ Shared Prediction Responses ============

class PredictionQValueResponse(BaseModel):
    """Q-value response DTO."""
    id: str
    treatment: str
    q_value: Decimal
    rank: int

    model_config = {
        'from_attributes': True
    }


class ExplanationFeatureResponse(BaseModel):
    """Explanation feature response DTO."""
    id: str
    feature_name: str
    scaled_value: Decimal
    raw_value: Decimal
    shap_value: Decimal
    rank: int
    interpretation: str
    reference_range: Optional[str]

    model_config = {
        'from_attributes': True
    }


class ExplanationAlternativeResponse(BaseModel):
    """Alternative treatment response DTO."""
    id: str
    rank: int
    treatment: str
    predicted_reduction: Decimal
    pros: str
    cons: str
    when_to_consider: str

    model_config = {
        'from_attributes': True
    }


class SafetyWarningResponse(BaseModel):
    """Safety warning response DTO."""
    id: str
    severity: str
    concern: str
    patient_factor: str
    mitigation: str

    model_config = {
        'from_attributes': True
    }


class PredictionExplanationResponse(BaseModel):
    """Prediction explanation response DTO."""
    id: str
    summary_text: str
    confidence_level: str
    clinical_priority: str
    why_this_treatment: str
    why_not_alternatives: str
    base_value: Decimal
    prediction_value: Decimal
    feature_interactions: Optional[str]
    features: List[ExplanationFeatureResponse] = []
    alternatives: List[ExplanationAlternativeResponse] = []
    created_at: datetime

    model_config = {
        'from_attributes': True
    }


class PredictionResponse(BaseModel):
    """Standard prediction response DTO."""
    id: str
    patient_id: str
    patient: PatientSummaryResponse
    model_version: str
    recommended_treatment: str
    treatment_index: int
    predicted_reduction: Decimal
    confidence_score: Decimal
    confidence_margin: Decimal
    created_at: datetime

    model_config = {
        'from_attributes': True
    }


class PredictionDetailResponse(BaseModel):
    """Detailed prediction with Q-values, explanation, and safety warnings."""
    id: str
    patient_id: str
    patient: PatientSummaryResponse
    model_version: str
    recommended_treatment: str
    treatment_index: int
    predicted_reduction: Decimal
    confidence_score: Decimal
    confidence_margin: Decimal
    q_values: List[PredictionQValueResponse] = []
    explanation: Optional[PredictionExplanationResponse] = None
    safety_warnings: List[SafetyWarningResponse] = []
    created_at: datetime

    model_config = {
        'from_attributes': True
    }


# ============ Generate Prediction ============

class GeneratePredictionRequest(BaseModel):
    """DTO for generating a new prediction."""
    patient_id: str = Field(..., min_length=1)


# ============ Get Single Prediction ============

class GetPredictionRequest(BaseModel):
    """DTO for getting a single prediction by ID."""
    prediction_id: str = Field(..., min_length=1)


# ============ Get Patient Predictions ============

class GetPatientPredictionsRequest(BaseModel):
    """DTO for getting all predictions for a patient."""
    patient_id: str = Field(..., min_length=1)
    limit: Optional[int] = Field(None, ge=1, le=100, description="Max number of predictions to return")


# ============ List Predictions (Pagination) ============

class ListPredictionsRequest(BaseModel):
    """DTO for listing predictions with pagination."""
    page: int = Field(default=1, ge=1, description="Page number (starts at 1)")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page (max 100)")
    patient_id: Optional[str] = None

    def get_offset(self) -> int:
        """Calculate database offset for pagination."""
        return (self.page - 1) * self.per_page

    def get_limit(self) -> int:
        """Get limit for database query."""
        return self.per_page