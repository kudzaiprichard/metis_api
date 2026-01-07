"""
Model management for diabetes treatment recommendation.

This module provides a unified ModelManager facade for all model operations:
- Stateless model loading (always fresh from disk)
- Version management and comparison
- Model registration and lifecycle
- Preprocessing component access

Primary Interface:
    ModelManager - Single entry point for all model operations

Usage:
    from treatment_recommender.registry import create_model_manager

    manager = create_model_manager(
        models_dir='artifacts',
        scaler_path='features/feature_scaler.pkl',
        metadata_path='features/preprocessing_metadata.json'
    )

    model = manager.get_active_model()
    processor = manager.get_feature_processor()
"""

# Primary Interface (USE THESE)
from .model_manager import (
    ModelManager,
    ModelManagerError,
    ModelNotFoundError,
    create_model_manager,
    get_model_manager,
    reset_global_manager,
    # Convenience functions
    load_active_model,
    load_latest_model,
    load_model_version,
    get_feature_processor_instance,
    list_available_versions,
    get_manager_status,
    validate_model_integrity
)

# Internal Components (advanced usage only)
from ._architecture import (
    NeuralTLearner,
    TreatmentSpecificNetwork,
    create_neural_t_learner
)

from ._loader import (
    ModelLoader,
    load_model,
    load_preprocessing_components
)

from ._metadata_manager import (
    ModelMetadataManager,
    create_metadata_manager
)

from ._model_registry import (
    ModelRegistry,
    create_model_registry,
    get_active_model_path,
    get_latest_model_path,
    list_all_versions,
    compare_model_versions
)

__all__ = [
    # Primary Interface
    'ModelManager',
    'ModelManagerError',
    'ModelNotFoundError',
    'create_model_manager',
    'get_model_manager',
    'reset_global_manager',

    # Convenience Functions
    'load_active_model',
    'load_latest_model',
    'load_model_version',
    'get_feature_processor_instance',
    'list_available_versions',
    'get_manager_status',
    'validate_model_integrity',

    # Advanced: Model Architecture
    'NeuralTLearner',
    'TreatmentSpecificNetwork',
    'create_neural_t_learner',

    # Advanced: Loading Utilities
    'ModelLoader',
    'load_model',
    'load_preprocessing_components',

    # Advanced: Version Management
    'ModelMetadataManager',
    'create_metadata_manager',
    'ModelRegistry',
    'create_model_registry',
    'get_active_model_path',
    'get_latest_model_path',
    'list_all_versions',
    'compare_model_versions',
]