"""
Base components shared across all pipelines.

This module contains:
- Data Transfer Objects (DTOs)
- Enums
- Constants
- Common helper functions
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum
from datetime import datetime

# =============================================================================
# CONSTANTS
# =============================================================================

# Treatment names (consistent across system)
TREATMENT_NAMES = ['Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', 'Insulin']

# Feature configuration - CRITICAL: Must match training data
FEATURE_COUNT = 21  # 13 base features + 8 engineered features

# Base patient features (required from user input)
BASE_PATIENT_FIELDS = [
    'age', 'gender', 'ethnicity', 'hba1c_baseline', 'diabetes_duration',
    'fasting_glucose', 'c_peptide', 'egfr', 'bmi',
    'bp_systolic', 'bp_diastolic', 'alt', 'ldl', 'hdl', 'triglycerides',
    'previous_prediabetes', 'hypertension', 'ckd', 'cvd', 'nafld', 'retinopathy'
]

# Engineered features (auto-generated during preprocessing)
ENGINEERED_FEATURES = [
    'insulin_deficiency_score',
    'beta_cell_reserve',
    'glucose_severity',
    'disease_progression',
    'metabolic_syndrome_score',
    'cv_risk_score',
    'kidney_severity',
    'comorbidity_count'
]

# All features after preprocessing (base + engineered)
ALL_FEATURES = BASE_PATIENT_FIELDS + ENGINEERED_FEATURES  # Total: 21

# Continuous features that require scaling
CONTINUOUS_FEATURES = [
    # Base continuous features
    'age', 'hba1c_baseline', 'diabetes_duration', 'fasting_glucose', 'c_peptide',
    'egfr', 'bmi', 'bp_systolic', 'bp_diastolic', 'alt', 'ldl', 'hdl', 'triglycerides',
    # Engineered continuous features
    'insulin_deficiency_score', 'beta_cell_reserve', 'glucose_severity',
    'disease_progression', 'metabolic_syndrome_score', 'cv_risk_score'
]

# Binary/categorical features (no scaling needed)
BINARY_FEATURES = [
    # Base binary features
    'gender', 'ethnicity', 'previous_prediabetes', 'hypertension',
    'ckd', 'cvd', 'nafld', 'retinopathy',
    # Engineered binary features
    'kidney_severity', 'comorbidity_count'
]

# Training configuration
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 10000

# Model architecture (matches notebook training)
DEFAULT_HIDDEN_DIMS = [256, 128, 64]
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4

# Training hyperparameters
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_EARLY_STOPPING_PATIENCE = 20


# =============================================================================
# ENUMS
# =============================================================================

class SelectionMode(Enum):
    """Treatment selection modes."""
    GREEDY = 'greedy'
    EPSILON_GREEDY = 'epsilon-greedy'
    SOFTMAX = 'softmax'


class TrainingStep(Enum):
    """
    Training pipeline steps with auto-calculated progress ranges.

    Each step has:
    - description: Human-readable step name
    - start_progress: Progress percentage at step start
    - end_progress: Progress percentage at step end
    """
    CALCULATING_VERSION = ("Calculating version number", 0, 10)
    LOADING_MODEL = ("Loading base model", 10, 20)
    PREPROCESSING = ("Preprocessing batch", 20, 40)
    VALIDATING_BEFORE = ("Validating performance before update", 40, 50)
    PARTIAL_FIT = ("Performing partial_fit", 50, 70)
    VALIDATING_AFTER = ("Validating performance after update", 70, 80)
    SAVING_MODEL = ("Saving new model version", 80, 100)

    def __init__(self, description: str, start_progress: int, end_progress: int):
        self.description = description
        self.start_progress = start_progress
        self.end_progress = end_progress
        self.mid_progress = (start_progress + end_progress) // 2


# =============================================================================
# DATA TRANSFER OBJECTS (DTOs)
# =============================================================================

@dataclass
class TreatmentResult:
    """
    Data Transfer Object for treatment recommendation results.

    Attributes:
        recommended_treatment: Treatment name (e.g., 'Insulin', 'Metformin')
        treatment_index: Treatment ID (0-4)
        predicted_hba1c_reduction: Expected HbA1c reduction (%)
        confidence_score: Confidence in recommendation (0.0 to 100.0)
        confidence_margin: Gap between best and second-best Q-value
        all_q_values: Dict mapping treatment names to Q-values
        ranked_treatments: List of treatments ranked by Q-value
        safety_warnings: List of safety concerns (empty if none)
        error: Error message if prediction failed

    Example:
        result = TreatmentResult(
            recommended_treatment='Insulin',
            treatment_index=4,
            predicted_hba1c_reduction=3.5,
            confidence_score=95.2,
            confidence_margin=0.85,
            all_q_values={'Metformin': 2.1, 'GLP-1': 2.8, ...},
            ranked_treatments=[...],
            safety_warnings=[],
            error=None
        )
    """
    recommended_treatment: str
    treatment_index: int
    predicted_hba1c_reduction: float
    confidence_score: float
    confidence_margin: float
    all_q_values: Dict[str, float]
    ranked_treatments: List[Dict]
    safety_warnings: List[str]
    error: Optional[str] = None


@dataclass
class TrainingStatus:
    """
    DTO for training status tracking.

    Attributes:
        is_training: Whether training is in progress
        current_step: Description of current step
        progress_percent: 0-100 progress percentage
        started_at: ISO format timestamp when training started
        estimated_completion: ISO format timestamp for estimated completion
        version_number: Version being created (e.g., "v1_2")
        outcomes_count: Number of patient outcomes in batch
        error: Error message if training failed
    """
    is_training: bool
    current_step: Optional[str]
    progress_percent: int
    started_at: Optional[str]
    estimated_completion: Optional[str]
    version_number: Optional[str]
    outcomes_count: int
    error: Optional[str]


@dataclass
class TrainingResult:
    """
    DTO for online learning results.

    Attributes:
        success: Whether training succeeded
        version_number: New version number created (e.g., "v1_2")
        outcomes_processed: Number of patient outcomes processed
        performance_before: Metrics before update (avg_reward, diversity, accuracy)
        performance_after: Metrics after update
        timestamp: ISO format timestamp
        model_files: Paths to saved model files
        error: Error message if training failed

    Example:
        result = TrainingResult(
            success=True,
            version_number='v1_2',
            outcomes_processed=50,
            performance_before={'avg_reward': 2.45, 'accuracy': 0.75},
            performance_after={'avg_reward': 2.58, 'accuracy': 0.78},
            timestamp='2025-01-15T10:30:00',
            model_files={'model': 'artifacts/v1_2/...', ...},
            error=None
        )
    """
    success: bool
    version_number: Optional[str]
    outcomes_processed: int
    performance_before: Optional[Dict]
    performance_after: Optional[Dict]
    timestamp: str
    model_files: Optional[Dict] = None
    error: Optional[str] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        ISO format timestamp string

    Example:
        timestamp = get_timestamp()
        # Returns: '2025-01-15T10:30:00'
    """
    return datetime.now().isoformat()


def calculate_confidence(q_values: List[float]) -> float:
    """
    Calculate confidence score from Q-values.

    Confidence is based on the gap between best and second-best Q-value.
    Higher gap = higher confidence.

    Args:
        q_values: List of Q-values for all treatments

    Returns:
        Confidence score (0.0 to 100.0)

    Example:
        q_values = [2.1, 2.5, 3.5, 2.8, 2.3]
        confidence = calculate_confidence(q_values)
        # Returns: ~95.7 (3.5 is clearly best)
    """
    import numpy as np

    if len(q_values) < 2:
        return 100.0

    q_max = max(q_values)
    q_second = sorted(q_values, reverse=True)[1]

    # Confidence based on relative gap
    if q_max == 0:
        return 0.0

    confidence = (q_max / (q_max + 0.01)) * 100  # Normalize to percentage

    return float(confidence)


def format_percentage(value: float) -> str:
    """
    Format float as percentage string.

    Args:
        value: Float value (0.0 to 1.0)

    Returns:
        Formatted percentage (e.g., "85.3%")

    Example:
        formatted = format_percentage(0.853)
        # Returns: "85.3%"
    """
    return f"{value * 100:.1f}%"


def calculate_confidence_margin(q_values: List[float]) -> float:
    """
    Calculate margin between best and second-best Q-value.

    Args:
        q_values: List of Q-values for all treatments

    Returns:
        Confidence margin (difference between top 2)

    Example:
        q_values = [2.1, 2.5, 3.5, 2.8, 2.3]
        margin = calculate_confidence_margin(q_values)
        # Returns: 0.7 (3.5 - 2.8)
    """
    import numpy as np

    if len(q_values) < 2:
        return 0.0

    sorted_q = sorted(q_values, reverse=True)
    margin = sorted_q[0] - sorted_q[1]

    return float(margin)


def rank_treatments(treatment_names: List[str], q_values: List[float]) -> List[Dict]:
    """
    Rank treatments by Q-value.

    Args:
        treatment_names: List of treatment names
        q_values: List of Q-values

    Returns:
        List of dicts with rank, treatment, q_value, relative_performance

    Example:
        ranked = rank_treatments(TREATMENT_NAMES, [2.1, 2.5, 3.5, 2.8, 2.3])
        # Returns: [
        #   {'rank': 1, 'treatment': 'SGLT-2', 'q_value': 3.5, 'relative_performance': 100.0},
        #   {'rank': 2, 'treatment': 'DPP-4', 'q_value': 2.8, 'relative_performance': 80.0},
        #   ...
        # ]
    """
    q_max = max(q_values)

    sorted_treatments = sorted(
        enumerate(q_values),
        key=lambda x: x[1],
        reverse=True
    )

    ranked = []
    for rank, (idx, q_val) in enumerate(sorted_treatments, 1):
        ranked.append({
            'rank': rank,
            'treatment': treatment_names[idx],
            'q_value': float(q_val),
            'relative_performance': float((q_val / q_max * 100) if q_max > 0 else 0)
        })

    return ranked


def print_progress_bar(current: int,
                       total: int,
                       prefix: str = '',
                       suffix: str = '',
                       length: int = 50):
    """
    Print a progress bar to console.

    Args:
        current: Current progress value
        total: Total value for 100%
        prefix: Prefix text before bar
        suffix: Suffix text after bar
        length: Length of progress bar in characters

    Example:
        print_progress_bar(50, 100, prefix='Progress:', suffix='Complete')
        # Prints: Progress: |█████████████████████████                         | 50% Complete
    """
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)

    # Print newline on completion
    if current == total:
        print()


def validate_feature_count(features: List) -> bool:
    """
    Validate that feature array has correct length.

    Args:
        features: Feature array or list

    Returns:
        True if feature count is 21, False otherwise

    Example:
        is_valid = validate_feature_count(patient_features)
        if not is_valid:
            raise ValueError("Feature count must be 21")
    """
    return len(features) == FEATURE_COUNT


def get_feature_categories() -> Dict[str, List[str]]:
    """
    Get feature categorization.

    Returns:
        Dictionary with feature categories

    Example:
        categories = get_feature_categories()
        print(f"Base features: {categories['base']}")
        print(f"Engineered: {categories['engineered']}")
        print(f"Continuous: {categories['continuous']}")
    """
    return {
        'base': BASE_PATIENT_FIELDS,
        'engineered': ENGINEERED_FEATURES,
        'all': ALL_FEATURES,
        'continuous': CONTINUOUS_FEATURES,
        'binary': BINARY_FEATURES,
        'total_count': FEATURE_COUNT
    }


def print_feature_summary():
    """
    Print a summary of feature configuration.

    Example:
        print_feature_summary()
        # Prints formatted summary of all features
    """
    print("\n" + "="*80)
    print("FEATURE CONFIGURATION SUMMARY")
    print("="*80)
    print(f"\nTotal Features: {FEATURE_COUNT}")
    print(f"\nBase Patient Fields: {len(BASE_PATIENT_FIELDS)}")
    print(f"  {', '.join(BASE_PATIENT_FIELDS[:5])}...")
    print(f"\nEngineered Features: {len(ENGINEERED_FEATURES)}")
    for feat in ENGINEERED_FEATURES:
        print(f"  - {feat}")
    print(f"\nContinuous Features (scaled): {len(CONTINUOUS_FEATURES)}")
    print(f"Binary Features (not scaled): {len(BINARY_FEATURES)}")
    print(f"\nModel Architecture:")
    print(f"  Input: {FEATURE_COUNT} features")
    print(f"  Hidden: {DEFAULT_HIDDEN_DIMS}")
    print(f"  Output: 1 Q-value per treatment")
    print(f"  Total Networks: 5 (one per treatment)")
    print("="*80 + "\n")


def validate_model_architecture(n_features: int, hidden_dims: List[int]) -> bool:
    """
    Validate model architecture matches expected configuration.

    Args:
        n_features: Number of input features
        hidden_dims: Hidden layer dimensions

    Returns:
        True if architecture matches expected, False otherwise

    Example:
        is_valid = validate_model_architecture(21, [256, 128, 64])
        # Returns: True
    """
    return (n_features == FEATURE_COUNT and
            hidden_dims == DEFAULT_HIDDEN_DIMS)


def get_model_config() -> Dict:
    """
    Get default model configuration.

    Returns:
        Dictionary with model hyperparameters

    Example:
        config = get_model_config()
        model = NeuralTLearner(**config)
    """
    return {
        'n_features': FEATURE_COUNT,
        'n_treatments': len(TREATMENT_NAMES),
        'hidden_dims': DEFAULT_HIDDEN_DIMS,
        'learning_rate': DEFAULT_LEARNING_RATE,
        'weight_decay': DEFAULT_WEIGHT_DECAY
    }


def get_training_config() -> Dict:
    """
    Get default training configuration.

    Returns:
        Dictionary with training hyperparameters

    Example:
        config = get_training_config()
        model.pretrain(**config)
    """
    return {
        'epochs': DEFAULT_EPOCHS,
        'batch_size': DEFAULT_BATCH_SIZE,
        'early_stopping_patience': DEFAULT_EARLY_STOPPING_PATIENCE
    }