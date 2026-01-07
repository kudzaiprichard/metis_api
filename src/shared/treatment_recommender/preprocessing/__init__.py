"""
Preprocessing utilities for diabetes treatment recommendation.

This module provides:
- Patient feature preprocessing (encoding, scaling)
- Input validation for patient data
- Feature extraction from patient dictionaries
"""

from .feature_processor import PatientFeatureProcessor, create_feature_processor
from .validators import (
    validate_patient_data,
    validate_patient_field,
    validate_patient_batch,
    validate_treatment_outcome,
    validate_treatment_outcome_batch,
    REQUIRED_PATIENT_FIELDS,
    VALID_TREATMENTS,
    VALID_GENDERS,
    VALID_ETHNICITIES
)

__all__ = [
    # Feature Processing
    'PatientFeatureProcessor',
    'create_feature_processor',

    # Validation Functions
    'validate_patient_data',
    'validate_patient_field',
    'validate_patient_batch',
    'validate_treatment_outcome',
    'validate_treatment_outcome_batch',

    # Constants
    'REQUIRED_PATIENT_FIELDS',
    'VALID_TREATMENTS',
    'VALID_GENDERS',
    'VALID_ETHNICITIES',
]