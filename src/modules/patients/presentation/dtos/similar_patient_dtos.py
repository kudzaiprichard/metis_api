"""
DTOs for similar patient search functionality.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, model_validator
from decimal import Decimal


# ============ Request DTOs ============

class FindSimilarPatientsRequest(BaseModel):
    """
    DTO for finding similar patients (tabular format).

    Accepts either patient_id (uses latest medical record) or
    medical_data_id (uses that specific record). If both are provided,
    medical_data_id takes priority.
    """
    patient_id: Optional[str] = Field(None, min_length=1, description="Patient ID to find similar cases for (uses latest medical record)")
    medical_data_id: Optional[str] = Field(None, min_length=1, description="Specific medical data record ID to use for similarity search")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of similar cases (1-20)")
    treatment_filter: Optional[str] = Field(
        None,
        description="Filter by treatment name (e.g., 'Metformin', 'GLP-1', 'SGLT-2')"
    )
    min_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0-1.0)"
    )

    @model_validator(mode='after')
    def validate_at_least_one_id(self):
        if not self.patient_id and not self.medical_data_id:
            raise ValueError("Either 'patient_id' or 'medical_data_id' must be provided")
        return self

    model_config = {
        'str_strip_whitespace': True
    }


class FindSimilarPatientsGraphRequest(BaseModel):
    """
    DTO for finding similar patients (graph format).

    Accepts either patient_id (uses latest medical record) or
    medical_data_id (uses that specific record). If both are provided,
    medical_data_id takes priority.
    """
    patient_id: Optional[str] = Field(None, min_length=1, description="Patient ID to find similar cases for (uses latest medical record)")
    medical_data_id: Optional[str] = Field(None, min_length=1, description="Specific medical data record ID to use for similarity search")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum number of similar cases (1-20)")
    treatment_filter: Optional[str] = Field(
        None,
        description="Filter by treatment name (e.g., 'Metformin', 'GLP-1', 'SGLT-2')"
    )
    min_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0-1.0)"
    )

    @model_validator(mode='after')
    def validate_at_least_one_id(self):
        if not self.patient_id and not self.medical_data_id:
            raise ValueError("Either 'patient_id' or 'medical_data_id' must be provided")
        return self

    model_config = {
        'str_strip_whitespace': True
    }


class GetSimilarPatientDetailRequest(BaseModel):
    """
    DTO for getting detailed information about a similar patient case from Neo4j.

    Used to retrieve complete historical patient data for cases returned
    from similarity search. These are synthetic patients from the Neo4j
    training dataset, not operational patients from PostgreSQL.
    """
    case_id: str = Field(
        ...,
        min_length=1,
        description="Patient ID from Neo4j (e.g., 'P000123')"
    )

    model_config = {
        'str_strip_whitespace': True
    }


# ============ Response DTOs ============

class PatientProfileResponse(BaseModel):
    """Patient profile information in similar case results."""
    age: int
    gender: str
    ethnicity: str
    hba1c_baseline: Decimal
    c_peptide: Decimal
    bmi: Decimal
    egfr: Decimal
    diabetes_duration: Decimal
    bp_systolic: int
    fasting_glucose: Decimal

    model_config = {
        'from_attributes': True
    }


class OutcomeResponse(BaseModel):
    """Treatment outcome information."""
    hba1c_reduction: Decimal
    hba1c_followup: Decimal
    time_to_target: str
    adverse_events: str
    outcome_category: str
    success: bool

    model_config = {
        'from_attributes': True
    }


class SimilarPatientCaseResponse(BaseModel):
    """
    Single similar patient case (tabular format).

    Used for list/table display in frontend.
    """
    case_id: str
    similarity_score: float
    clinical_similarity: float
    comorbidity_similarity: float
    profile: PatientProfileResponse
    comorbidities: List[str]
    treatment_given: str
    drug_class: str
    outcome: OutcomeResponse

    model_config = {
        'from_attributes': True
    }


class SimilarPatientsResponse(BaseModel):
    """
    Response containing list of similar patient cases (tabular format).
    """
    patient_id: str
    similar_cases: List[SimilarPatientCaseResponse]
    total_found: int
    filters_applied: dict

    model_config = {
        'from_attributes': True
    }


# ============ Graph Response DTOs ============

class GraphNodeStyleResponse(BaseModel):
    """Visual styling for graph nodes."""
    color: str
    size: str
    shape: str


class GraphNodeDataResponse(BaseModel):
    """Data payload for graph nodes."""
    pass  # Will be dynamic based on node type


class GraphNodeResponse(BaseModel):
    """Graph node representation."""
    id: str
    type: str
    label: str
    data: dict
    style: GraphNodeStyleResponse


class GraphEdgeStyleResponse(BaseModel):
    """Visual styling for graph edges."""
    width: int
    color: str


class GraphEdgeResponse(BaseModel):
    """Graph edge representation."""
    id: str
    source: str
    target: str
    type: str
    label: str
    data: dict
    style: GraphEdgeStyleResponse


class GraphMetadataResponse(BaseModel):
    """Metadata about the graph query."""
    query_patient: dict
    filters_applied: dict
    results_found: int
    similarity_range: Optional[dict] = None


class SimilarPatientsGraphResponse(BaseModel):
    """
    Response containing graph structure for visualization.

    Contains nodes (patients, treatments, outcomes) and edges (relationships)
    suitable for graph visualization libraries.
    """
    patient_id: str
    nodes: List[GraphNodeResponse]
    edges: List[GraphEdgeResponse]
    metadata: GraphMetadataResponse

    model_config = {
        'from_attributes': True
    }


# ============ Patient Detail Response DTOs ============

class DemographicsResponse(BaseModel):
    """Demographic information."""
    age: int
    gender: str
    ethnicity: str
    age_group: str

    model_config = {
        'from_attributes': True
    }


class ClinicalFeaturesResponse(BaseModel):
    """Complete clinical features (21 base features)."""
    hba1c_baseline: Decimal
    diabetes_duration: Decimal
    fasting_glucose: Decimal
    c_peptide: Decimal
    egfr: Decimal
    bmi: Decimal
    bp_systolic: int
    bp_diastolic: int
    alt: Decimal
    ldl: Decimal
    hdl: Decimal
    triglycerides: Decimal
    previous_prediabetes: bool

    model_config = {
        'from_attributes': True
    }


class ClinicalCategoriesResponse(BaseModel):
    """Clinical category classifications."""
    bmi_category: str
    hba1c_severity: str
    kidney_function: str

    model_config = {
        'from_attributes': True
    }


class TreatmentInfoResponse(BaseModel):
    """Treatment information."""
    drug_name: str
    drug_class: str
    cost_category: str
    evidence_level: str

    model_config = {
        'from_attributes': True
    }


class SimilarPatientDetailResponse(BaseModel):
    """
    Complete details for a similar patient case from Neo4j.

    Contains all clinical information, treatment, and outcome data
    for a historical patient case from the training dataset.
    """
    patient_id: str
    demographics: DemographicsResponse
    clinical_features: ClinicalFeaturesResponse
    clinical_categories: ClinicalCategoriesResponse
    comorbidities: List[str]
    treatment: Optional[TreatmentInfoResponse]
    outcome: Optional[OutcomeResponse]

    model_config = {
        'from_attributes': True
    }