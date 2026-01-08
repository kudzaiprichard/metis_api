"""
DTOs for ML Model Management endpoints.
"""

from typing import Optional, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field
from decimal import Decimal


# ============ Model Version Response ============

class ModelVersionResponse(BaseModel):
    """Response DTO for a single model version."""
    version_number: str
    model_file_path: str
    parent_version: Optional[str]
    trained_timestamp: str
    training_method: str
    training_info: Dict
    performance_metrics: Dict
    is_active: bool
    notes: Optional[str] = None

    model_config = {
        'from_attributes': True
    }


# ============ Model List Response ============

class ModelListResponse(BaseModel):
    """Response DTO for listing models."""
    total_versions: int
    active_version: Optional[str]
    latest_version: Optional[str]
    versions: List[ModelVersionResponse]


# ============ Active Model Response ============

class ActiveModelResponse(BaseModel):
    """Response DTO for active model info."""
    version_number: str
    trained_timestamp: str
    performance_metrics: Dict
    is_active: bool = True


# ============ Model Status Response ============

class ModelStatusResponse(BaseModel):
    """Response DTO for model manager status."""
    active_version: Optional[str]
    latest_version: Optional[str]
    total_versions: int
    available_versions: List[str]
    missing_versions: List[str]
    disk_space_used_mb: float


# ============ Model Comparison Response ============

class ModelComparisonResponse(BaseModel):
    """Response DTO for comparing two model versions."""
    version_1: str
    version_2: str
    metrics_v1: Dict
    metrics_v2: Dict
    differences: Dict


# ============ Model Lineage Response ============

class ModelLineageResponse(BaseModel):
    """Response DTO for model version lineage."""
    version: str
    lineage: List[str]
    depth: int


# ============ Activate Model Request ============

class ActivateModelRequest(BaseModel):
    """Request DTO for activating a model version."""
    version: str = Field(..., min_length=1, description="Version to activate (e.g., 'v1_2')")


# ============ Delete Model Request ============

class DeleteModelRequest(BaseModel):
    """Request DTO for deleting a model version."""
    version: str = Field(..., min_length=1)
    delete_files: bool = Field(default=True, description="Whether to delete model files from disk")


# ============ List Models Query Parameters ============

class ListModelsRequest(BaseModel):
    """Request DTO for listing models with sorting."""
    sort_by: str = Field(
        default='version',
        description="Sort key: 'version', 'date', 'avg_reward', 'accuracy'"
    )
    reverse: bool = Field(default=False, description="Reverse sort order")

    @classmethod
    def validate_sort_by(cls, v):
        """Validate sort_by field."""
        valid_sorts = ['version', 'date', 'avg_reward', 'accuracy']
        if v not in valid_sorts:
            raise ValueError(f"sort_by must be one of: {', '.join(valid_sorts)}")
        return v


# ============ Compare Models Request ============

class CompareModelsRequest(BaseModel):
    """Request DTO for comparing two model versions."""
    version_1: str = Field(..., min_length=1, description="First version to compare")
    version_2: str = Field(..., min_length=1, description="Second version to compare")