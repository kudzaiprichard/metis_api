"""
Base components for explainability system.

This module provides:
- Data Transfer Objects (DTOs) for explanation results
- Abstract interface for graph database
- Constants and configuration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

# =============================================================================
# CONSTANTS
# =============================================================================

# Treatment names (consistent with main system)
TREATMENT_NAMES = ['Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', 'Insulin']


# Confidence levels
class ConfidenceLevel(Enum):
    """Confidence levels for treatment recommendations."""
    CRITICAL = "critical"  # <50%
    LOW = "low"  # 50-70%
    MODERATE = "moderate"  # 70-85%
    HIGH = "high"  # 85-95%
    VERY_HIGH = "very_high"  # >95%


class ClinicalPriority(Enum):
    """Clinical priority levels."""
    ROUTINE = "routine"
    STANDARD = "standard"
    URGENT = "urgent"
    CRITICAL = "critical"


class SeverityLevel(Enum):
    """Severity levels for warnings."""
    INFO = "info"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


# Default configuration
DEFAULT_TOP_FEATURES = 5
DEFAULT_SIMILAR_CASES = 5


# =============================================================================
# GRAPH DATABASE INTERFACE
# =============================================================================

class GraphDatabaseInterface(ABC):
    """
    Abstract interface for graph database integration.

    Users MUST implement this interface to provide clinical context
    for explainability. Without graph database, explanations cannot
    access guidelines, contraindications, or similar cases.

    Implementation Example:
        class Neo4jGraphDB(GraphDatabaseInterface):
            def __init__(self, uri, user, password):
                self.driver = GraphDatabase.driver(uri, auth=(user, password))

            def get_treatment_guidelines(self, treatment, patient_profile):
                with self.driver.session() as session:
                    result = session.run(
                        "MATCH (t:Treatment {name: $treatment})-[:HAS_GUIDELINE]->(g:Guideline) "
                        "WHERE g.hba1c_min <= $hba1c <= g.hba1c_max "
                        "RETURN g",
                        treatment=treatment,
                        hba1c=patient_profile['hba1c_baseline']
                    )
                    return [record['g'] for record in result]
    """

    @abstractmethod
    def get_treatment_guidelines(self,
                                 treatment: str,
                                 patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get clinical guidelines for specific treatment and patient profile.

        Args:
            treatment: Treatment name (e.g., "Insulin", "Metformin")
            patient_profile: Patient data dictionary

        Returns:
            Dictionary with:
            {
                "treatment": str,
                "indication": str,
                "guidelines": List[str],  # e.g., ["ADA 2024: ...", "AACE 2023: ..."]
                "dosing": str,
                "monitoring": str
            }

        Example:
            {
                "treatment": "Insulin",
                "indication": "HbA1c > 10% with beta cell failure",
                "guidelines": [
                    "ADA 2024: Insulin recommended for HbA1c >10% with symptoms",
                    "AACE 2023: Start basal insulin when beta cell function impaired"
                ],
                "dosing": "Start 10 units basal insulin or 0.2 units/kg",
                "monitoring": "Check fasting glucose daily, adjust by 2-4 units every 3 days"
            }
        """
        pass

    @abstractmethod
    def check_contraindications(self,
                                treatment: str,
                                patient_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for contraindications for specific treatment.

        Args:
            treatment: Treatment name
            patient_profile: Patient data dictionary

        Returns:
            List of contraindication dictionaries:
            [
                {
                    "condition": str,      # e.g., "eGFR < 30"
                    "severity": str,       # "critical", "warning", "caution"
                    "reason": str,         # Why it's contraindicated
                    "alternative": str     # Suggested alternative
                }
            ]

        Example:
            [
                {
                    "condition": "eGFR 25 mL/min/1.73m²",
                    "severity": "critical",
                    "reason": "Metformin contraindicated for eGFR <30 due to lactic acidosis risk",
                    "alternative": "Consider DPP-4 inhibitor or insulin"
                }
            ]
        """
        pass

    @abstractmethod
    def get_drug_interactions(self,
                              treatment: str,
                              comorbidities: List[str]) -> List[Dict[str, Any]]:
        """
        Get potential drug interactions based on comorbidities.

        Args:
            treatment: Treatment name
            comorbidities: List of comorbidity flags from patient
                          e.g., ["hypertension", "cvd"]

        Returns:
            List of interaction dictionaries:
            [
                {
                    "condition": str,
                    "interaction": str,
                    "recommendation": str,
                    "severity": str
                }
            ]

        Example:
            [
                {
                    "condition": "hypertension",
                    "interaction": "ACE inhibitors may enhance hypoglycemic effect of insulin",
                    "recommendation": "Monitor glucose more frequently",
                    "severity": "moderate"
                }
            ]
        """
        pass

    @abstractmethod
    def find_similar_cases(self,
                           patient_profile: Dict[str, Any],
                           limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar patient cases and their outcomes.

        Args:
            patient_profile: Patient data dictionary
            limit: Maximum number of similar cases to return

        Returns:
            List of similar case dictionaries:
            [
                {
                    "case_id": str,
                    "similarity_score": float,  # 0.0-1.0
                    "profile": Dict,            # Age, HbA1c, etc.
                    "treatment_given": str,
                    "outcome": {
                        "hba1c_reduction": float,
                        "time_to_target": str,
                        "adverse_events": str
                    }
                }
            ]

        Example:
            [
                {
                    "case_id": "P12345",
                    "similarity_score": 0.92,
                    "profile": {
                        "age": 60,
                        "hba1c_baseline": 11.2,
                        "c_peptide": 0.35,
                        "diabetes_duration": 20
                    },
                    "treatment_given": "Insulin",
                    "outcome": {
                        "hba1c_reduction": 4.2,
                        "time_to_target": "12 weeks",
                        "adverse_events": "None"
                    }
                }
            ]
        """
        pass

    @abstractmethod
    def get_background_data_for_shap(self, n_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Get random patient samples for SHAP background data.

        This method provides representative patient data that SHAP uses as a baseline
        to calculate feature importance. The data should be in RAW format (same as
        user input), NOT preprocessed - the feature processor will handle conversion.

        Args:
            n_samples: Number of random patients to retrieve (default: 100)
                      Recommended range: 50-200 patients

        Returns:
            List of patient dictionaries in same format as user input, containing:
            - age, gender, ethnicity
            - hba1c_baseline, diabetes_duration, c_peptide
            - fasting_glucose, egfr, bmi
            - bp_systolic, bp_diastolic, alt
            - ldl, hdl, triglycerides
            - previous_prediabetes
            - hypertension, ckd, cvd, nafld, retinopathy

        Example:
            [
                {
                    'age': 58,
                    'gender': 'Female',
                    'ethnicity': 'Caucasian',
                    'hba1c_baseline': 8.2,
                    'diabetes_duration': 5.0,
                    'c_peptide': 1.5,
                    'egfr': 75,
                    'bmi': 31.5,
                    ...
                    'hypertension': 1,
                    'ckd': 0,
                    'cvd': 0,
                    'nafld': 1,
                    'retinopathy': 0
                },
                ...
            ]

        Notes:
            - Data should be randomly sampled for diversity
            - Include all required features (21 total after preprocessing)
            - Gender/ethnicity should be strings (will be one-hot encoded)
            - Comorbidities should be 0/1 integers
            - This data represents "typical" patients for SHAP baseline comparison
        """
        pass

# =============================================================================
# DATA TRANSFER OBJECTS (DTOs)
# =============================================================================

@dataclass
class FeatureAttribution:
    """
    Single feature's attribution in the model decision.

    Attributes:
        feature: Feature name (e.g., "c_peptide")
        value: Patient's scaled value (z-score used by model)
        raw_value: Patient's raw/actual value (e.g., BMI 31.5, not -0.45)
        shap_value: SHAP attribution value
        importance_rank: Rank among all features (1 = most important)
        interpretation: Human-readable interpretation (uses raw_value)
        reference_range: Normal/expected range (optional)
    """
    feature: str
    value: float  # Scaled value (z-score)
    raw_value: float  # Raw value (actual patient value)
    shap_value: float
    importance_rank: int
    interpretation: str
    reference_range: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature": self.feature,
            "value": self.value,
            "raw_value": self.raw_value,
            "shap_value": self.shap_value,
            "importance_rank": self.importance_rank,
            "interpretation": self.interpretation,
            "reference_range": self.reference_range
        }


@dataclass
class KeyFactor:
    """
    Key factor driving the model's decision.

    Attributes:
        factor: Factor name (e.g., "Beta cell failure")
        evidence: Evidence supporting this factor
        impact: Clinical impact description
    """
    factor: str
    evidence: str
    impact: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factor": self.factor,
            "evidence": self.evidence,
            "impact": self.impact
        }


@dataclass
class AlternativeTreatment:
    """
    Alternative treatment option with pros/cons.

    Attributes:
        rank: Rank among alternatives (2 = second best)
        treatment: Treatment name
        predicted_reduction: Predicted HbA1c reduction
        pros: List of advantages
        cons: List of disadvantages
        when_to_consider: When to use this alternative
    """
    rank: int
    treatment: str
    predicted_reduction: float
    pros: List[str]
    cons: List[str]
    when_to_consider: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rank": self.rank,
            "treatment": self.treatment,
            "predicted_reduction": self.predicted_reduction,
            "pros": self.pros,
            "cons": self.cons,
            "when_to_consider": self.when_to_consider
        }


@dataclass
class SafetyWarning:
    """
    Safety warning or consideration.

    Attributes:
        severity: Severity level (info, caution, warning, critical)
        concern: What the concern is
        patient_factor: Relevant patient factor
        mitigation: How to mitigate the risk
    """
    severity: str
    concern: str
    patient_factor: str
    mitigation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "severity": self.severity,
            "concern": self.concern,
            "patient_factor": self.patient_factor,
            "mitigation": self.mitigation
        }


@dataclass
class ExplanationSummary:
    """
    High-level summary of the explanation.

    Attributes:
        primary_recommendation: Recommended treatment
        confidence_level: Confidence level (high, moderate, low)
        one_sentence: One-sentence explanation
        clinical_priority: Priority level for action
    """
    primary_recommendation: str
    confidence_level: str
    one_sentence: str
    clinical_priority: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_recommendation": self.primary_recommendation,
            "confidence_level": self.confidence_level,
            "one_sentence": self.one_sentence,
            "clinical_priority": self.clinical_priority
        }


@dataclass
class ModelReasoning:
    """
    Detailed model reasoning.

    Attributes:
        predicted_hba1c_reduction: Expected HbA1c reduction
        confidence_score: Model confidence (0-100)
        why_this_treatment: Natural language explanation
        key_factors: List of key factors driving decision
    """
    predicted_hba1c_reduction: float
    confidence_score: float
    why_this_treatment: str
    key_factors: List[KeyFactor]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predicted_hba1c_reduction": self.predicted_hba1c_reduction,
            "confidence_score": self.confidence_score,
            "why_this_treatment": self.why_this_treatment,
            "key_factors": [f.to_dict() for f in self.key_factors]
        }


@dataclass
class FeatureImportance:
    """
    Feature importance analysis.

    Attributes:
        top_features: List of top N most important features
        base_value: Model's baseline prediction
        prediction: Final prediction after features
        feature_interactions: Description of feature interactions
    """
    top_features: List[FeatureAttribution]
    base_value: float
    prediction: float
    feature_interactions: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "top_features": [f.to_dict() for f in self.top_features],
            "base_value": self.base_value,
            "prediction": self.prediction,
            "feature_interactions": self.feature_interactions
        }


@dataclass
class ClinicalContext:
    """
    Clinical context from graph database.

    Attributes:
        guideline_alignment: Alignment with clinical guidelines
        similar_cases: Similar patient cases and outcomes
        population_statistics: Population-level statistics
    """
    guideline_alignment: Dict[str, Any]
    similar_cases: Dict[str, Any]
    population_statistics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "guideline_alignment": self.guideline_alignment,
            "similar_cases": self.similar_cases,
            "population_statistics": self.population_statistics
        }


@dataclass
class SafetyChecks:
    """
    Safety checks and warnings.

    Attributes:
        contraindications: List of contraindications
        warnings: List of warnings
        monitoring_requirements: Required monitoring
        drug_interactions: Potential drug interactions
    """
    contraindications: List[str]
    warnings: List[SafetyWarning]
    monitoring_requirements: List[str]
    drug_interactions: List[Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contraindications": self.contraindications,
            "warnings": [w.to_dict() for w in self.warnings],
            "monitoring_requirements": self.monitoring_requirements,
            "drug_interactions": self.drug_interactions
        }


@dataclass
class AlternativeTreatments:
    """
    Alternative treatment options.

    Attributes:
        why_not_alternatives: Why alternatives weren't chosen
        alternatives: List of alternative treatments
    """
    why_not_alternatives: str
    alternatives: List[AlternativeTreatment]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "why_not_alternatives": self.why_not_alternatives,
            "alternatives": [a.to_dict() for a in self.alternatives]
        }


@dataclass
class ExplanationMetadata:
    """
    Metadata about the explanation generation.

    Attributes:
        explanation_id: Unique explanation ID
        timestamp: ISO timestamp
        model_version: Model version used
        shap_calculation_time_ms: SHAP calculation time
        graph_query_time_ms: Graph query time
        llm_generation_time_ms: LLM generation time
        total_time_ms: Total explanation time
        tokens_used: LLM tokens used (optional, tracked by provider)
    """
    explanation_id: str
    timestamp: str
    model_version: str
    shap_calculation_time_ms: int
    graph_query_time_ms: int
    llm_generation_time_ms: int
    total_time_ms: int
    tokens_used: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "shap_calculation_time_ms": self.shap_calculation_time_ms,
            "graph_query_time_ms": self.graph_query_time_ms,
            "llm_generation_time_ms": self.llm_generation_time_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_used": self.tokens_used
        }


@dataclass
class ExplanationResult:
    """
    Complete explanation result.

    This is the main DTO returned by the explainer.
    Contains all aspects of the explanation in a structured format.

    Attributes:
        summary: High-level summary
        model_reasoning: Detailed model reasoning
        feature_importance: Feature attribution analysis
        clinical_context: Clinical guidelines and evidence
        safety_checks: Safety warnings and monitoring
        alternatives: Alternative treatment options
        metadata: Explanation metadata

    Example:
        result = explainer.explain(model_result, patient_data)
        print(result.summary.one_sentence)
        print(result.model_reasoning.why_this_treatment)

        # Convert to JSON
        json_output = result.to_dict()
    """
    summary: ExplanationSummary
    model_reasoning: ModelReasoning
    feature_importance: FeatureImportance
    clinical_context: ClinicalContext
    safety_checks: SafetyChecks
    alternatives: AlternativeTreatments
    metadata: ExplanationMetadata

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all explanation components
        """
        return {
            "summary": self.summary.to_dict(),
            "model_reasoning": self.model_reasoning.to_dict(),
            "feature_importance": self.feature_importance.to_dict(),
            "clinical_context": self.clinical_context.to_dict(),
            "safety_checks": self.safety_checks.to_dict(),
            "alternatives": self.alternatives.to_dict(),
            "metadata": self.metadata.to_dict()
        }

    def to_json(self) -> str:
        """
        Convert to JSON string.

        Returns:
            JSON string representation
        """
        import json
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_confidence_level(confidence_score: float) -> str:
    """
    Convert confidence score to confidence level.

    Args:
        confidence_score: Confidence score (0-100)

    Returns:
        Confidence level string

    Example:
        level = get_confidence_level(92.5)
        # Returns: "high"
    """
    if confidence_score < 50:
        return ConfidenceLevel.CRITICAL.value
    elif confidence_score < 70:
        return ConfidenceLevel.LOW.value
    elif confidence_score < 85:
        return ConfidenceLevel.MODERATE.value
    elif confidence_score < 95:
        return ConfidenceLevel.HIGH.value
    else:
        return ConfidenceLevel.VERY_HIGH.value


def determine_clinical_priority(hba1c: float,
                                c_peptide: float,
                                contraindications: List[str]) -> str:
    """
    Determine clinical priority level.

    Args:
        hba1c: HbA1c baseline
        c_peptide: C-peptide level
        contraindications: List of contraindications

    Returns:
        Clinical priority level

    Example:
        priority = determine_clinical_priority(11.5, 0.4, [])
        # Returns: "urgent"
    """
    # Critical contraindication
    if any("CRITICAL" in c for c in contraindications):
        return ClinicalPriority.CRITICAL.value

    # Severe hyperglycemia
    if hba1c > 11.0:
        return ClinicalPriority.URGENT.value

    # Beta cell failure
    if c_peptide < 0.5:
        return ClinicalPriority.URGENT.value

    # Moderate hyperglycemia
    if hba1c > 9.0:
        return ClinicalPriority.STANDARD.value

    # Routine follow-up
    return ClinicalPriority.ROUTINE.value


def generate_explanation_id() -> str:
    """
    Generate unique explanation ID.

    Returns:
        Unique ID (format: exp_YYYYMMDD_HHMMSS_random)

    Example:
        exp_id = generate_explanation_id()
        # Returns: "exp_20250115_143022_a8f3c2"
    """
    import random
    import string

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

    return f"exp_{timestamp}_{random_suffix}"


def format_reference_range(feature_name: str, value: float) -> Optional[str]:
    """
    Get reference range for a feature.

    Args:
        feature_name: Feature name
        value: Current value

    Returns:
        Reference range string or None

    Example:
        ref = format_reference_range("c_peptide", 0.4)
        # Returns: "1.1-4.4 ng/mL"
    """
    reference_ranges = {
        'c_peptide': '1.1-4.4 ng/mL',
        'hba1c_baseline': '<7% (target)',
        'fasting_glucose': '70-100 mg/dL',
        'egfr': '>60 mL/min/1.73m²',
        'bmi': '18.5-24.9 kg/m²',
        'bp_systolic': '90-120 mmHg',
        'bp_diastolic': '60-80 mmHg',
        'ldl': '<100 mg/dL',
        'hdl': '>40 mg/dL (men), >50 mg/dL (women)',
        'triglycerides': '<150 mg/dL'
    }

    return reference_ranges.get(feature_name)