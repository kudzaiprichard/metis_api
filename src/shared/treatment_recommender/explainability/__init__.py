"""
Explainability system for diabetes treatment recommendations.

This package provides:
- Feature attribution (SHAP values)
- Clinical context from graph database
- LLM-powered natural language explanations
- Structured JSON responses

Usage:
    from treatment_recommender.explainability import create_explainer
    from treatment_recommender.explainability.providers import create_gemini_provider
    from my_app.graph_db import MyGraphDatabase

    # Create LLM provider
    gemini = create_gemini_provider(
        api_key='YOUR_API_KEY',
        model_name='gemini-1.5-pro-latest'
    )

    # Create graph database
    graph_db = MyGraphDatabase()

    # Create explainer
    explainer = create_explainer(
        model=pipeline._model,
        feature_processor=processor,
        llm_provider=gemini,
        graph_db=graph_db
    )

    # Generate explanation
    explanation = explainer.explain(
        model_result=model_result,
        patient_data=patient_data
    )

    print(explanation.summary.one_sentence)
"""

# Core DTOs
from ._base import (
    # Main result DTO
    ExplanationResult,
    ExplanationSummary,
    ModelReasoning,
    FeatureImportance,
    ClinicalContext,
    SafetyChecks,
    AlternativeTreatments,
    ExplanationMetadata,

    # Component DTOs
    FeatureAttribution,
    KeyFactor,
    AlternativeTreatment,
    SafetyWarning,

    # Enums
    ConfidenceLevel,
    ClinicalPriority,
    SeverityLevel,

    # Interfaces (users must implement GraphDatabaseInterface)
    GraphDatabaseInterface,

    # Utilities
    get_confidence_level,
    determine_clinical_priority,
    generate_explanation_id,
    format_reference_range,

    # Constants
    TREATMENT_NAMES,
    DEFAULT_TOP_FEATURES,
    DEFAULT_SIMILAR_CASES,
)

# Feature attribution
from ._feature_attribution import (
    calculate_shap_values,
    extract_top_features,
    create_feature_attributions,
)

# LLM synthesizer
from ._llm_synthesizer import (
    LLMSynthesizer,
    create_llm_synthesizer,
)

# Main explainer
from .explainer import (
    TreatmentExplainer,
    create_explainer,
)

# LLM providers
from .providers import (
    BaseLLMProvider,
    GeminiProvider,
    create_gemini_provider,
)

__all__ = [
    # Main factory
    'create_explainer',
    'TreatmentExplainer',

    # Result DTOs
    'ExplanationResult',
    'ExplanationSummary',
    'ModelReasoning',
    'FeatureImportance',
    'ClinicalContext',
    'SafetyChecks',
    'AlternativeTreatments',
    'ExplanationMetadata',

    # Component DTOs
    'FeatureAttribution',
    'KeyFactor',
    'AlternativeTreatment',
    'SafetyWarning',

    # Enums
    'ConfidenceLevel',
    'ClinicalPriority',
    'SeverityLevel',

    # Interfaces
    'GraphDatabaseInterface',
    'BaseLLMProvider',

    # Feature attribution
    'calculate_shap_values',
    'extract_top_features',
    'create_feature_attributions',

    # LLM
    'LLMSynthesizer',
    'create_llm_synthesizer',
    'GeminiProvider',
    'create_gemini_provider',

    # Utilities
    'get_confidence_level',
    'determine_clinical_priority',
    'generate_explanation_id',
    'format_reference_range',

    # Constants
    'TREATMENT_NAMES',
    'DEFAULT_TOP_FEATURES',
    'DEFAULT_SIMILAR_CASES',
]

__version__ = '1.0.0'