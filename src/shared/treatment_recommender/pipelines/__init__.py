"""
Pipelines for diabetes treatment recommendation.

This module provides:
- Prediction pipeline (stateless, lazy-loading)
- Online learning pipeline (incremental training)
- Data transfer objects (DTOs)
- Shared constants and enums
"""

from ._base import (
    # DTOs
    TreatmentResult,
    TrainingResult,
    TrainingStatus,

    # Enums
    TrainingStep,
    SelectionMode,

    # Constants
    TREATMENT_NAMES,
    FEATURE_COUNT,

    # Utilities
    get_timestamp,
    calculate_confidence,
    format_percentage
)

from .prediction import (
    PredictionPipeline,
    create_prediction_pipeline
)

from .online_learning import (
    OnlineLearningPipeline,
    create_online_learning_pipeline
)

__all__ = [
    # DTOs
    'TreatmentResult',
    'TrainingResult',
    'TrainingStatus',

    # Enums
    'TrainingStep',
    'SelectionMode',

    # Constants
    'TREATMENT_NAMES',
    'FEATURE_COUNT',

    # Utilities
    'get_timestamp',
    'calculate_confidence',
    'format_percentage',

    # Pipelines
    'PredictionPipeline',
    'create_prediction_pipeline',
    'OnlineLearningPipeline',
    'create_online_learning_pipeline',
]