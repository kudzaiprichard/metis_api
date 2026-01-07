"""
Feature attribution using SHAP (SHapley Additive exPlanations).

This module calculates which features were most important in the model's
decision using mathematically rigorous SHAP values.

SHAP provides:
- Feature importance scores
- Direction of influence (positive/negative)
- Magnitude of contribution
- Additive explanations (sum to prediction difference from baseline)
"""

import numpy as np
import shap
from typing import List, Dict, Tuple, Optional
import warnings

import torch

from ._base import (
    FeatureAttribution,
    DEFAULT_TOP_FEATURES,
    format_reference_range
)


# =============================================================================
# SHAP CALCULATION
# =============================================================================

def calculate_shap_values(model,
                          patient_features: np.ndarray,
                          feature_names: List[str],
                          background_data: np.ndarray,
                          treatment_index: int = 0,
                          verbose: bool = False) -> Tuple[np.ndarray, float, float]:
    """
    Calculate SHAP values for a patient's features.

    SHAP (SHapley Additive exPlanations) explains the output of machine learning
    models by computing the contribution of each feature to the prediction.

    Args:
        model: Trained Neural T-Learner model
        patient_features: Feature array for patient (shape: (21,))
        feature_names: List of feature names (must match feature order)
        background_data: Background dataset for SHAP (REQUIRED, must not be None)
        treatment_index: Which treatment network to explain (0-4)
        verbose: If True, print detailed logs

    Returns:
        Tuple of (shap_values, base_value, prediction):
        - shap_values: Array of SHAP values for each feature (shape: (21,))
        - base_value: Baseline prediction (expected value)
        - prediction: Final prediction for this patient

    Raises:
        ValueError: If background_data is None

    Example:
        shap_values, base_value, prediction = calculate_shap_values(
            model=model,
            patient_features=features,
            feature_names=processor.get_feature_names(),
            background_data=background_samples,
            treatment_index=4  # Insulin
        )
        # shap_values: [-0.85, 0.72, 0.45, ...]
        # base_value: 2.1
        # prediction: 3.5
        # Sum: base_value + sum(shap_values) ≈ prediction
    """
    # Validate background_data is provided
    if background_data is None:
        raise ValueError(
            "background_data is REQUIRED for SHAP calculations. "
            "Background data should be loaded from the graph database during explainer initialization. "
            "Ensure your GraphDatabaseInterface implements get_background_data_for_shap()."
        )

    # Use CPU device
    device = 'cpu'

    if verbose:
        print("[FeatureAttribution] Calculating SHAP values...")
        print(f"[FeatureAttribution] Treatment index: {treatment_index}")
        print(f"[FeatureAttribution] Patient features shape: {patient_features.shape}")
        print(f"[FeatureAttribution] Background data shape: {background_data.shape}")
        print(f"[FeatureAttribution] Device: {device}")

    # Validate inputs
    if len(patient_features.shape) == 1:
        patient_features = patient_features.reshape(1, -1)

    if patient_features.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: {patient_features.shape[1]} features "
            f"but {len(feature_names)} feature names"
        )

    if background_data.shape[1] != patient_features.shape[1]:
        raise ValueError(
            f"Background data feature count ({background_data.shape[1]}) "
            f"does not match patient features ({patient_features.shape[1]})"
        )

    # Get the specific treatment network
    treatment_network = model.treatment_networks[treatment_index]

    # Convert background_data to PyTorch Tensor
    background_tensor = torch.FloatTensor(background_data).to(device)

    if verbose:
        print(f"[FeatureAttribution] Background tensor shape: {background_tensor.shape}")
        print(f"[FeatureAttribution] Using {background_data.shape[0]} background samples")

    # Suppress SHAP warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Create SHAP explainer
        # Use DeepExplainer for neural networks (faster than KernelExplainer)
        explainer = shap.DeepExplainer(
            treatment_network,
            data=background_tensor
        )

        # Convert patient features to tensor for SHAP calculation
        patient_tensor = torch.FloatTensor(patient_features).to(device)

        # Calculate SHAP values
        shap_values_raw = explainer.shap_values(patient_tensor)

        # Extract for single patient
        if isinstance(shap_values_raw, list):
            # Multi-output model (shouldn't happen for our case, but handle it)
            shap_values = shap_values_raw[0].flatten()
        else:
            shap_values = shap_values_raw.flatten()

    # Get base value (expected value from explainer)
    base_value = float(explainer.expected_value)

    # Get actual prediction
    treatment_network.eval()
    with torch.no_grad():
        x = torch.FloatTensor(patient_features).to(device)
        prediction = float(treatment_network(x).cpu().numpy().flatten()[0])

    if verbose:
        print(f"[FeatureAttribution] Base value: {base_value:.4f}")
        print(f"[FeatureAttribution] Prediction: {prediction:.4f}")
        print(f"[FeatureAttribution] SHAP sum: {shap_values.sum():.4f}")
        print(f"[FeatureAttribution] Difference: {abs(base_value + shap_values.sum() - prediction):.6f}")

    return shap_values, base_value, prediction


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_top_features(shap_values: np.ndarray,
                         feature_names: List[str],
                         patient_features: np.ndarray,
                         raw_values: Dict[str, float],
                         top_n: int = DEFAULT_TOP_FEATURES) -> List[Tuple[str, float, float, float]]:
    """
    Extract top N most important features by absolute SHAP value.

    Args:
        shap_values: Array of SHAP values (shape: (21,))
        feature_names: List of feature names
        patient_features: Patient's scaled feature values (z-scores)
        raw_values: Dictionary of raw (unscaled) feature values
        top_n: Number of top features to extract

    Returns:
        List of tuples: (feature_name, scaled_value, raw_value, shap_value)
        Sorted by absolute SHAP value (descending)

    Example:
        top_features = extract_top_features(
            shap_values=shap_values,
            feature_names=feature_names,
            patient_features=features,
            raw_values={'bmi': 31.5, 'hba1c_baseline': 8.2, ...},
            top_n=5
        )
        # Returns: [
        #   ("bmi", -0.45, 31.5, 1.088),
        #   ("hba1c_baseline", -3.01, 8.2, -0.603),
        #   ...
        # ]
    """
    # Get absolute SHAP values for ranking
    abs_shap = np.abs(shap_values)

    # Get indices of top N features
    top_indices = np.argsort(abs_shap)[::-1][:top_n]

    # Extract feature information
    top_features = []
    for idx in top_indices:
        feature_name = feature_names[idx]
        scaled_value = float(patient_features[idx])

        # Get raw value from dictionary, fallback to scaled if not found
        raw_value = raw_values.get(feature_name, scaled_value)

        shap_value = float(shap_values[idx])

        top_features.append((feature_name, scaled_value, raw_value, shap_value))

    return top_features


def create_feature_attributions(top_features: List[Tuple[str, float, float, float]],
                                treatment_name: str) -> List[FeatureAttribution]:
    """
    Create FeatureAttribution DTOs from top features.

    Args:
        top_features: List of (feature_name, scaled_value, raw_value, shap_value) tuples
        treatment_name: Name of treatment being explained

    Returns:
        List of FeatureAttribution DTOs with interpretations

    Example:
        attributions = create_feature_attributions(
            top_features=top_features,
            treatment_name="Insulin"
        )
        # Returns list of FeatureAttribution objects with interpretations
    """
    attributions = []

    for rank, (feature_name, scaled_value, raw_value, shap_value) in enumerate(top_features, 1):
        # Generate interpretation using RAW value
        interpretation = generate_feature_interpretation(
            feature_name=feature_name,
            raw_value=raw_value,
            shap_value=shap_value,
            treatment_name=treatment_name
        )

        # Get reference range
        reference_range = format_reference_range(feature_name, raw_value)

        # Create attribution
        attribution = FeatureAttribution(
            feature=feature_name,
            value=scaled_value,  # Keep scaled for technical users
            raw_value=raw_value,  # Add raw for interpretation
            shap_value=shap_value,
            importance_rank=rank,
            interpretation=interpretation,
            reference_range=reference_range
        )

        attributions.append(attribution)

    return attributions


# =============================================================================
# INTERPRETATION GENERATION
# =============================================================================

def generate_feature_interpretation(feature_name: str,
                                    raw_value: float,
                                    shap_value: float,
                                    treatment_name: str) -> str:
    """
    Generate human-readable interpretation of a feature's contribution.

    Args:
        feature_name: Name of the feature
        raw_value: Patient's RAW (unscaled) value for this feature
        shap_value: SHAP value (contribution to prediction)
        treatment_name: Name of treatment being explained

    Returns:
        Human-readable interpretation string

    Example:
        interp = generate_feature_interpretation(
            feature_name="c_peptide",
            raw_value=0.4,
            shap_value=-0.85,
            treatment_name="Insulin"
        )
        # Returns: "Very low C-peptide (0.4 ng/mL) strongly indicates beta cell failure"
    """
    # Feature-specific interpretations using RAW values
    interpretations = {
        'c_peptide': _interpret_c_peptide(raw_value, shap_value, treatment_name),
        'hba1c_baseline': _interpret_hba1c(raw_value, shap_value, treatment_name),
        'diabetes_duration': _interpret_duration(raw_value, shap_value, treatment_name),
        'insulin_deficiency_score': _interpret_insulin_deficiency(raw_value, shap_value, treatment_name),
        'beta_cell_reserve': _interpret_beta_cell_reserve(raw_value, shap_value, treatment_name),
        'glucose_severity': _interpret_glucose_severity(raw_value, shap_value, treatment_name),
        'egfr': _interpret_egfr(raw_value, shap_value, treatment_name),
        'bmi': _interpret_bmi(raw_value, shap_value, treatment_name),
        'age': _interpret_age(raw_value, shap_value, treatment_name),
        'fasting_glucose': _interpret_fasting_glucose(raw_value, shap_value, treatment_name),
        'ldl': _interpret_ldl(raw_value, shap_value, treatment_name),
        'hdl': _interpret_hdl(raw_value, shap_value, treatment_name),
        'triglycerides': _interpret_triglycerides(raw_value, shap_value, treatment_name),
    }

    # Get specific interpretation or generate generic one
    if feature_name in interpretations:
        return interpretations[feature_name]
    else:
        # Generic interpretation for other features
        direction = "supports" if shap_value > 0 else "indicates need for"
        return f"{feature_name.replace('_', ' ').title()} ({raw_value:.2f}) {direction} {treatment_name}"


# =============================================================================
# FEATURE-SPECIFIC INTERPRETATIONS (Using RAW values)
# =============================================================================

def _interpret_c_peptide(value: float, shap: float, treatment: str) -> str:
    """Interpret C-peptide contribution using RAW value."""
    if value < 0.6:
        severity = "critically low" if value < 0.5 else "very low"
        return f"{severity.capitalize()} C-peptide ({value:.2f} ng/mL) strongly indicates beta cell failure"
    elif value < 1.1:
        return f"Low C-peptide ({value:.2f} ng/mL) suggests reduced insulin production"
    else:
        return f"Normal C-peptide ({value:.2f} ng/mL) indicates preserved beta cell function"


def _interpret_hba1c(value: float, shap: float, treatment: str) -> str:
    """Interpret HbA1c contribution using RAW value."""
    if value > 10:
        return f"Severely elevated HbA1c ({value:.1f}%) requires aggressive treatment"
    elif value > 9:
        return f"Very high HbA1c ({value:.1f}%) indicates poor glycemic control"
    elif value > 8:
        return f"Elevated HbA1c ({value:.1f}%) above target requires intervention"
    elif value > 7:
        return f"HbA1c ({value:.1f}%) moderately elevated above target"
    else:
        return f"HbA1c ({value:.1f}%) at or near target"


def _interpret_duration(value: float, shap: float, treatment: str) -> str:
    """Interpret diabetes duration contribution using RAW value."""
    if value > 15:
        return f"Long disease duration ({value:.0f} years) suggests progressive beta cell decline"
    elif value > 10:
        return f"Moderate disease duration ({value:.0f} years) indicates established diabetes"
    elif value > 5:
        return f"Disease duration ({value:.0f} years) with ongoing progression"
    else:
        return f"Early diabetes ({value:.0f} years) with potential for good control"


def _interpret_insulin_deficiency(value: float, shap: float, treatment: str) -> str:
    """Interpret insulin deficiency score using RAW value."""
    if value > 2.5:
        return f"Very high insulin deficiency score ({value:.2f}) confirms need for exogenous insulin"
    elif value > 2.0:
        return f"High insulin deficiency score ({value:.2f}) indicates significant insulin lack"
    elif value > 1.5:
        return f"Moderate insulin deficiency score ({value:.2f})"
    else:
        return f"Low insulin deficiency score ({value:.2f}) suggests preserved function"


def _interpret_beta_cell_reserve(value: float, shap: float, treatment: str) -> str:
    """Interpret beta cell reserve using RAW value."""
    if value < 0.2:
        return f"Minimal beta cell reserve ({value:.2f}) remaining"
    elif value < 0.5:
        return f"Low beta cell reserve ({value:.2f}) indicates declining function"
    elif value < 1.0:
        return f"Moderate beta cell reserve ({value:.2f})"
    else:
        return f"Good beta cell reserve ({value:.2f}) indicates preserved function"


def _interpret_glucose_severity(value: float, shap: float, treatment: str) -> str:
    """Interpret glucose severity score using RAW value."""
    if value > 150:
        return f"Severe glucose dysregulation (score: {value:.1f})"
    elif value > 100:
        return f"Significant glucose dysregulation (score: {value:.1f})"
    else:
        return f"Moderate glucose dysregulation (score: {value:.1f})"


def _interpret_egfr(value: float, shap: float, treatment: str) -> str:
    """Interpret eGFR contribution using RAW value."""
    if value < 30:
        return f"Severely reduced kidney function (eGFR {value:.0f} mL/min/1.73m²) limits treatment options"
    elif value < 60:
        return f"Reduced kidney function (eGFR {value:.0f} mL/min/1.73m²) requires careful medication selection"
    elif value < 90:
        return f"Mildly reduced kidney function (eGFR {value:.0f} mL/min/1.73m²)"
    else:
        return f"Normal kidney function (eGFR {value:.0f} mL/min/1.73m²)"


def _interpret_bmi(value: float, shap: float, treatment: str) -> str:
    """Interpret BMI contribution using RAW value."""
    if value > 35:
        return f"Class II obesity (BMI {value:.1f} kg/m²) favors weight-loss medications"
    elif value > 30:
        return f"Obesity (BMI {value:.1f} kg/m²) supports weight-loss treatments"
    elif value > 25:
        return f"Overweight (BMI {value:.1f} kg/m²)"
    else:
        return f"Normal weight (BMI {value:.1f} kg/m²)"


def _interpret_age(value: float, shap: float, treatment: str) -> str:
    """Interpret age contribution using RAW value."""
    if value > 75:
        return f"Elderly patient (age {value:.0f} years) requires careful treatment selection"
    elif value > 65:
        return f"Older adult (age {value:.0f} years) with age-related considerations"
    else:
        return f"Age {value:.0f} years with standard treatment options"


def _interpret_fasting_glucose(value: float, shap: float, treatment: str) -> str:
    """Interpret fasting glucose using RAW value."""
    if value > 180:
        return f"Severely elevated fasting glucose ({value:.0f} mg/dL)"
    elif value > 140:
        return f"Elevated fasting glucose ({value:.0f} mg/dL) indicates poor control"
    elif value > 100:
        return f"Mildly elevated fasting glucose ({value:.0f} mg/dL)"
    else:
        return f"Normal fasting glucose ({value:.0f} mg/dL)"


def _interpret_ldl(value: float, shap: float, treatment: str) -> str:
    """Interpret LDL cholesterol using RAW value."""
    if value > 160:
        return f"Very high LDL cholesterol ({value:.0f} mg/dL)"
    elif value > 130:
        return f"High LDL cholesterol ({value:.0f} mg/dL)"
    elif value > 100:
        return f"Borderline high LDL cholesterol ({value:.0f} mg/dL)"
    else:
        return f"Optimal LDL cholesterol ({value:.0f} mg/dL)"


def _interpret_hdl(value: float, shap: float, treatment: str) -> str:
    """Interpret HDL cholesterol using RAW value."""
    if value < 40:
        return f"Low HDL cholesterol ({value:.0f} mg/dL) increases cardiovascular risk"
    elif value < 60:
        return f"Moderate HDL cholesterol ({value:.0f} mg/dL)"
    else:
        return f"High HDL cholesterol ({value:.0f} mg/dL) provides cardiovascular protection"


def _interpret_triglycerides(value: float, shap: float, treatment: str) -> str:
    """Interpret triglycerides using RAW value."""
    if value > 500:
        return f"Very high triglycerides ({value:.0f} mg/dL) indicates severe hypertriglyceridemia"
    elif value > 200:
        return f"High triglycerides ({value:.0f} mg/dL)"
    elif value > 150:
        return f"Borderline high triglycerides ({value:.0f} mg/dL)"
    else:
        return f"Normal triglycerides ({value:.0f} mg/dL)"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_shap_additivity(shap_values: np.ndarray,
                             base_value: float,
                             prediction: float,
                             tolerance: float = 0.01) -> bool:
    """
    Validate that SHAP values are additive (sum to prediction difference).

    SHAP property: base_value + sum(shap_values) ≈ prediction

    Args:
        shap_values: Array of SHAP values
        base_value: Baseline prediction
        prediction: Actual prediction
        tolerance: Acceptable difference threshold

    Returns:
        True if additive property holds, False otherwise

    Example:
        is_valid = validate_shap_additivity(shap_values, 2.1, 3.5)
        # Checks if 2.1 + sum(shap_values) ≈ 3.5
    """
    reconstructed = base_value + shap_values.sum()
    difference = abs(reconstructed - prediction)

    return difference < tolerance


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def calculate_shap_batch(model,
                         patient_features_batch: np.ndarray,
                         feature_names: List[str],
                         background_data: np.ndarray,
                         treatment_index: int = 0,
                         verbose: bool = False) -> List[Tuple[np.ndarray, float, float]]:
    """
    Calculate SHAP values for multiple patients at once.

    Args:
        model: Trained model
        patient_features_batch: Feature matrix (n_patients, 21)
        feature_names: List of feature names
        background_data: Background dataset for SHAP (REQUIRED)
        treatment_index: Which treatment to explain
        verbose: Print progress

    Returns:
        List of (shap_values, base_value, prediction) for each patient

    Raises:
        ValueError: If background_data is None

    Example:
        results = calculate_shap_batch(
            model,
            features_batch,
            feature_names,
            background_data
        )
        for shap_vals, base, pred in results:
            print(f"Prediction: {pred}, Base: {base}")
    """
    if background_data is None:
        raise ValueError("background_data is REQUIRED for SHAP calculations")

    results = []

    for i, patient_features in enumerate(patient_features_batch):
        if verbose:
            print(f"[FeatureAttribution] Processing patient {i + 1}/{len(patient_features_batch)}")

        shap_vals, base_val, pred = calculate_shap_values(
            model=model,
            patient_features=patient_features,
            feature_names=feature_names,
            background_data=background_data,
            treatment_index=treatment_index,
            verbose=False
        )

        results.append((shap_vals, base_val, pred))

    return results