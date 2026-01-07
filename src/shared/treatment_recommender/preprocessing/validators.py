"""
Input validation utilities for diabetes treatment recommendation.

This module provides comprehensive validation for:
- Patient data fields (base features only)
- Treatment outcomes
- Batch inputs
- Data type checking
- Range validation

Note: Engineered features are auto-generated and NOT validated in user input.
"""

from typing import Dict, List, Tuple, Optional, Any

# =============================================================================
# CONSTANTS
# =============================================================================

# Required fields from user (base features only - NO engineered features)
REQUIRED_PATIENT_FIELDS = [
    'age', 'gender', 'ethnicity', 'hba1c_baseline', 'diabetes_duration',
    'fasting_glucose', 'c_peptide', 'egfr', 'bmi',
    'bp_systolic', 'bp_diastolic', 'alt', 'ldl', 'hdl', 'triglycerides',
    'previous_prediabetes', 'hypertension', 'ckd', 'cvd', 'nafld', 'retinopathy'
]

# Engineered features (auto-generated - should NOT be in user input)
ENGINEERED_FIELDS = [
    'insulin_deficiency_score',
    'beta_cell_reserve',
    'glucose_severity',
    'disease_progression',
    'metabolic_syndrome_score',
    'cv_risk_score',
    'kidney_severity',
    'comorbidity_count'
]

# All fields after preprocessing (base + engineered)
ALL_PATIENT_FIELDS = REQUIRED_PATIENT_FIELDS + ENGINEERED_FIELDS  # Total: 21

VALID_TREATMENTS = ['Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', 'Insulin']

VALID_GENDERS = ['Male', 'Female']

VALID_ETHNICITIES = ['African', 'Asian', 'Caucasian', 'Hispanic', 'Other']

# Feature ranges for validation
FEATURE_RANGES = {
    'age': (18, 120),
    'hba1c_baseline': (4.0, 20.0),
    'diabetes_duration': (0, 50),
    'fasting_glucose': (50, 500),
    'c_peptide': (0, 5.0),
    'egfr': (5, 150),
    'bmi': (10, 80),
    'bp_systolic': (70, 250),
    'bp_diastolic': (40, 150),
    'alt': (0, 500),
    'ldl': (0, 500),
    'hdl': (0, 200),
    'triglycerides': (0, 1000)
}

BINARY_FIELDS = [
    'previous_prediabetes', 'hypertension', 'ckd',
    'cvd', 'nafld', 'retinopathy'
]


# =============================================================================
# PATIENT DATA VALIDATION
# =============================================================================

def validate_patient_field(field_name: str,
                           field_value: Any,
                           field_type: str = 'continuous') -> Tuple[bool, Optional[str]]:
    """
    Validate a single patient field.

    Args:
        field_name: Name of the field
        field_value: Value to validate
        field_type: Type of field ('continuous', 'binary', 'categorical')

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        is_valid, error = validate_patient_field('age', 58, 'continuous')
        if not is_valid:
            print(f"Validation error: {error}")
    """
    # Check if None
    if field_value is None:
        return False, f"{field_name} cannot be None"

    # Validate by type
    if field_type == 'continuous':
        # Check numeric
        try:
            value = float(field_value)
        except (TypeError, ValueError):
            return False, f"{field_name} must be numeric, got {type(field_value).__name__}"

        # Check range
        if field_name in FEATURE_RANGES:
            min_val, max_val = FEATURE_RANGES[field_name]
            if value < min_val or value > max_val:
                return False, f"{field_name} must be between {min_val} and {max_val}, got {value}"

    elif field_type == 'binary':
        # Check 0 or 1
        if field_value not in [0, 1]:
            return False, f"{field_name} must be 0 or 1, got {field_value}"

    elif field_type == 'categorical':
        # Validate categorical values
        if field_name == 'gender':
            if field_value not in VALID_GENDERS:
                return False, f"Gender must be one of {VALID_GENDERS}, got '{field_value}'"

        elif field_name == 'ethnicity':
            if field_value not in VALID_ETHNICITIES:
                return False, f"Ethnicity must be one of {VALID_ETHNICITIES}, got '{field_value}'"

    return True, None


def validate_patient_data(patient_dict: Dict,
                          strict: bool = True) -> Tuple[bool, Optional[str]]:
    """
    Validate complete patient data dictionary.

    This validates ONLY base features. Engineered features should NOT be present
    in user input as they are auto-generated.

    Args:
        patient_dict: Patient data dictionary
        strict: If True, require all base fields. If False, allow missing fields

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        patient = {'age': 58, 'gender': 'Female', ...}
        is_valid, error = validate_patient_data(patient)
        if not is_valid:
            raise ValueError(error)
    """
    # Check if dict
    if not isinstance(patient_dict, dict):
        return False, f"Patient data must be dictionary, got {type(patient_dict).__name__}"

    # Check for engineered features (should NOT be in user input)
    has_engineered = []
    for field in ENGINEERED_FIELDS:
        if field in patient_dict:
            has_engineered.append(field)

    if has_engineered:
        return False, (
            f"Patient data should not contain engineered features (auto-generated): "
            f"{', '.join(has_engineered)}. Please remove these fields."
        )

    # Check required base fields
    if strict:
        missing_fields = []
        for field in REQUIRED_PATIENT_FIELDS:
            if field not in patient_dict:
                missing_fields.append(field)

        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"

    # Validate each field
    for field_name, field_value in patient_dict.items():
        # Skip unknown fields
        if field_name not in REQUIRED_PATIENT_FIELDS:
            continue

        # Determine field type
        if field_name in BINARY_FIELDS:
            field_type = 'binary'
        elif field_name in ['gender', 'ethnicity']:
            field_type = 'categorical'
        else:
            field_type = 'continuous'

        # Validate
        is_valid, error = validate_patient_field(field_name, field_value, field_type)
        if not is_valid:
            return False, error

    return True, None


def validate_patient_batch(patients: List[Dict],
                           min_batch_size: int = 1,
                           max_batch_size: int = 10000) -> Tuple[bool, Optional[str]]:
    """
    Validate a batch of patient records.

    Args:
        patients: List of patient dictionaries
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        batch = [patient1, patient2, patient3]
        is_valid, error = validate_patient_batch(batch)
    """
    # Check if list
    if not isinstance(patients, list):
        return False, f"Batch must be a list, got {type(patients).__name__}"

    # Check batch size
    if len(patients) < min_batch_size:
        return False, f"Batch size must be at least {min_batch_size}, got {len(patients)}"

    if len(patients) > max_batch_size:
        return False, f"Batch size must be at most {max_batch_size}, got {len(patients)}"

    # Validate each patient
    for i, patient in enumerate(patients):
        is_valid, error = validate_patient_data(patient, strict=True)
        if not is_valid:
            return False, f"Patient {i + 1}: {error}"

    return True, None


# =============================================================================
# TREATMENT OUTCOME VALIDATION
# =============================================================================

def validate_treatment_outcome(outcome: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate a treatment outcome record for online learning.

    Args:
        outcome: Dictionary with keys: 'patient', 'treatment_given', 'reward'

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        outcome = {
            'patient': patient_dict,
            'treatment_given': 'Insulin',
            'reward': 3.5
        }
        is_valid, error = validate_treatment_outcome(outcome)
    """
    # Check required fields
    required = ['patient', 'treatment_given', 'reward']
    for field in required:
        if field not in outcome:
            return False, f"Missing required field: {field}"

    # Validate patient data
    is_valid, error = validate_patient_data(outcome['patient'], strict=True)
    if not is_valid:
        return False, f"Invalid patient data: {error}"

    # Validate treatment
    treatment = outcome['treatment_given']
    if treatment not in VALID_TREATMENTS:
        return False, f"Invalid treatment '{treatment}'. Valid: {VALID_TREATMENTS}"

    # Validate reward (HbA1c reduction)
    try:
        reward = float(outcome['reward'])
    except (TypeError, ValueError):
        return False, f"Reward must be numeric, got {type(outcome['reward']).__name__}"

    # Reasonable range for HbA1c reduction
    if reward < -5 or reward > 10:
        return False, f"Reward (HbA1c reduction) must be between -5 and 10, got {reward}"

    return True, None


def validate_treatment_outcome_batch(outcomes: List[Dict],
                                     min_batch_size: int = 1,
                                     max_batch_size: int = 10000) -> Tuple[bool, Optional[str]]:
    """
    Validate a batch of treatment outcomes.

    Args:
        outcomes: List of outcome dictionaries
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        outcomes = [outcome1, outcome2, outcome3]
        is_valid, error = validate_treatment_outcome_batch(outcomes)
    """
    # Check if list
    if not isinstance(outcomes, list):
        return False, f"Outcomes must be a list, got {type(outcomes).__name__}"

    # Check batch size
    if len(outcomes) < min_batch_size:
        return False, f"Batch size must be at least {min_batch_size}, got {len(outcomes)}"

    if len(outcomes) > max_batch_size:
        return False, f"Batch size must be at most {max_batch_size}, got {len(outcomes)}"

    # Validate each outcome
    for i, outcome in enumerate(outcomes):
        is_valid, error = validate_treatment_outcome(outcome)
        if not is_valid:
            return False, f"Outcome {i + 1}: {error}"

    return True, None


# =============================================================================
# SAFETY VALIDATION
# =============================================================================

def validate_safety_constraints(patient_dict: Dict,
                                treatment: str) -> List[str]:
    """
    Check for safety contraindications.

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment name

    Returns:
        List of safety warnings (empty if no concerns)

    Example:
        warnings = validate_safety_constraints(patient, 'Metformin')
        if warnings:
            print(f"Safety concerns: {warnings}")
    """
    warnings = []

    egfr = patient_dict.get('egfr', 100)
    age = patient_dict.get('age', 50)

    # Metformin contraindications
    if treatment == 'Metformin':
        if egfr < 30:
            warnings.append("CRITICAL: Metformin contraindicated for eGFR < 30")
        elif egfr < 45:
            warnings.append("WARNING: Use Metformin with caution for eGFR 30-45")

    # SGLT-2 reduced efficacy
    if treatment == 'SGLT-2' and egfr < 30:
        warnings.append("WARNING: SGLT-2 inhibitor less effective with eGFR < 30")

    # Insulin in elderly
    if treatment == 'Insulin' and age > 75:
        warnings.append("CAUTION: Insulin in elderly - Monitor for hypoglycemia")

    return warnings


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_missing_fields(patient_dict: Dict) -> List[str]:
    """
    Get list of missing required base fields.

    Args:
        patient_dict: Patient data dictionary

    Returns:
        List of missing field names

    Example:
        missing = get_missing_fields(patient_dict)
        if missing:
            print(f"Missing: {missing}")
    """
    missing = []
    for field in REQUIRED_PATIENT_FIELDS:
        if field not in patient_dict:
            missing.append(field)
    return missing


def get_invalid_fields(patient_dict: Dict) -> Dict[str, str]:
    """
    Get dictionary of invalid fields and their errors.

    Args:
        patient_dict: Patient data dictionary

    Returns:
        Dictionary mapping field names to error messages

    Example:
        invalid = get_invalid_fields(patient_dict)
        for field, error in invalid.items():
            print(f"{field}: {error}")
    """
    invalid = {}

    for field_name, field_value in patient_dict.items():
        if field_name not in REQUIRED_PATIENT_FIELDS:
            continue

        # Determine type
        if field_name in BINARY_FIELDS:
            field_type = 'binary'
        elif field_name in ['gender', 'ethnicity']:
            field_type = 'categorical'
        else:
            field_type = 'continuous'

        # Validate
        is_valid, error = validate_patient_field(field_name, field_value, field_type)
        if not is_valid:
            invalid[field_name] = error

    return invalid


def check_for_engineered_features(patient_dict: Dict) -> List[str]:
    """
    Check if patient dictionary contains engineered features (should not).

    Args:
        patient_dict: Patient data dictionary

    Returns:
        List of engineered feature names found in input

    Example:
        found = check_for_engineered_features(patient_dict)
        if found:
            print(f"Remove these auto-generated fields: {found}")
    """
    found = []
    for field in ENGINEERED_FIELDS:
        if field in patient_dict:
            found.append(field)
    return found


def validate_feature_completeness(patient_dict: Dict) -> Tuple[bool, str]:
    """
    Comprehensive validation check.

    Checks for:
    1. All required base fields present
    2. No engineered features in input
    3. All field values valid

    Args:
        patient_dict: Patient data dictionary

    Returns:
        Tuple of (is_valid, summary_message)

    Example:
        is_valid, msg = validate_feature_completeness(patient_dict)
        print(msg)
    """
    # Check for engineered features
    engineered = check_for_engineered_features(patient_dict)
    if engineered:
        return False, f"Contains auto-generated features (remove these): {', '.join(engineered)}"

    # Check for missing fields
    missing = get_missing_fields(patient_dict)
    if missing:
        return False, f"Missing {len(missing)} required fields: {', '.join(missing[:5])}..."

    # Check for invalid fields
    invalid = get_invalid_fields(patient_dict)
    if invalid:
        error_summary = "; ".join([f"{k}: {v}" for k, v in list(invalid.items())[:3]])
        return False, f"Invalid field values: {error_summary}"

    return True, f"✓ All {len(REQUIRED_PATIENT_FIELDS)} base fields valid"


def get_field_info(field_name: str) -> Dict[str, Any]:
    """
    Get detailed information about a field.

    Args:
        field_name: Name of the field

    Returns:
        Dictionary with field information

    Example:
        info = get_field_info('age')
        print(f"Type: {info['type']}")
        print(f"Range: {info['range']}")
    """
    info = {
        'name': field_name,
        'is_required': field_name in REQUIRED_PATIENT_FIELDS,
        'is_engineered': field_name in ENGINEERED_FIELDS,
        'type': None,
        'range': None,
        'valid_values': None
    }

    # Determine type
    if field_name in BINARY_FIELDS:
        info['type'] = 'binary'
        info['valid_values'] = [0, 1]
    elif field_name == 'gender':
        info['type'] = 'categorical'
        info['valid_values'] = VALID_GENDERS
    elif field_name == 'ethnicity':
        info['type'] = 'categorical'
        info['valid_values'] = VALID_ETHNICITIES
    elif field_name in FEATURE_RANGES:
        info['type'] = 'continuous'
        info['range'] = FEATURE_RANGES[field_name]
    else:
        info['type'] = 'continuous'
        info['range'] = None

    return info


def print_validation_summary(patient_dict: Dict):
    """
    Print a detailed validation summary.

    Args:
        patient_dict: Patient data dictionary

    Example:
        print_validation_summary(patient_dict)
    """
    print("\n" + "=" * 80)
    print("PATIENT DATA VALIDATION SUMMARY")
    print("=" * 80)

    # Overall validation
    is_valid, msg = validate_feature_completeness(patient_dict)
    print(f"\nOverall Status: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"Message: {msg}")

    # Field counts
    print(f"\nField Counts:")
    print(f"  Required base fields: {len(REQUIRED_PATIENT_FIELDS)}")
    print(f"  Fields in input: {len(patient_dict)}")

    # Missing fields
    missing = get_missing_fields(patient_dict)
    if missing:
        print(f"\n✗ Missing Fields ({len(missing)}):")
        for field in missing[:10]:
            print(f"  - {field}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
    else:
        print(f"\n✓ All required fields present")

    # Engineered features
    engineered = check_for_engineered_features(patient_dict)
    if engineered:
        print(f"\n✗ Engineered Features Found (should not be in input):")
        for field in engineered:
            print(f"  - {field}")
    else:
        print(f"\n✓ No engineered features in input")

    # Invalid fields
    invalid = get_invalid_fields(patient_dict)
    if invalid:
        print(f"\n✗ Invalid Field Values ({len(invalid)}):")
        for field, error in list(invalid.items())[:5]:
            print(f"  - {field}: {error}")
        if len(invalid) > 5:
            print(f"  ... and {len(invalid) - 5} more")
    else:
        print(f"\n✓ All field values valid")

    print("=" * 80 + "\n")


def create_sample_patient() -> Dict:
    """
    Create a sample patient dictionary with valid values.

    Returns:
        Sample patient dictionary

    Example:
        sample = create_sample_patient()
        is_valid, _ = validate_patient_data(sample)
        # is_valid will be True
    """
    return {
        'age': 58,
        'gender': 'Female',
        'ethnicity': 'Caucasian',
        'hba1c_baseline': 8.2,
        'diabetes_duration': 5.0,
        'fasting_glucose': 165,
        'c_peptide': 1.5,
        'egfr': 75,
        'bmi': 31.5,
        'bp_systolic': 135,
        'bp_diastolic': 85,
        'alt': 28,
        'ldl': 120,
        'hdl': 45,
        'triglycerides': 160,
        'previous_prediabetes': 1,
        'hypertension': 1,
        'ckd': 0,
        'cvd': 0,
        'nafld': 1,
        'retinopathy': 0
    }


def validate_and_clean_patient_data(patient_dict: Dict) -> Tuple[bool, Dict, List[str]]:
    """
    Validate patient data and return cleaned version with warnings.

    Args:
        patient_dict: Raw patient data dictionary

    Returns:
        Tuple of (is_valid, cleaned_dict, warnings)

    Example:
        is_valid, cleaned, warnings = validate_and_clean_patient_data(raw_patient)
        if is_valid:
            features = processor.process_patient(cleaned)
        else:
            for warning in warnings:
                print(f"Warning: {warning}")
    """
    warnings = []
    cleaned = {}

    # Remove engineered features if present
    engineered_found = check_for_engineered_features(patient_dict)
    if engineered_found:
        warnings.append(f"Removed auto-generated features: {', '.join(engineered_found)}")

    # Copy only valid base fields
    for field in REQUIRED_PATIENT_FIELDS:
        if field in patient_dict:
            cleaned[field] = patient_dict[field]

    # Validate cleaned data
    is_valid, error = validate_patient_data(cleaned, strict=True)
    if not is_valid:
        warnings.append(error)

    return is_valid, cleaned, warnings


def get_validation_template() -> Dict:
    """
    Get a template showing all required fields with example values.

    Returns:
        Dictionary template for patient data

    Example:
        template = get_validation_template()
        print(json.dumps(template, indent=2))
    """
    template = {}

    for field in REQUIRED_PATIENT_FIELDS:
        info = get_field_info(field)

        if info['type'] == 'binary':
            template[field] = 0
        elif info['type'] == 'categorical' and info['valid_values']:
            template[field] = info['valid_values'][0]
        elif info['range']:
            min_val, max_val = info['range']
            template[field] = (min_val + max_val) / 2
        else:
            template[field] = 0

    return template


def explain_field_requirements():
    """
    Print detailed explanation of all field requirements.

    Example:
        explain_field_requirements()
    """
    print("\n" + "=" * 80)
    print("PATIENT DATA FIELD REQUIREMENTS")
    print("=" * 80)

    print(f"\nTotal Required Fields: {len(REQUIRED_PATIENT_FIELDS)}")
    print(f"Auto-Generated Fields: {len(ENGINEERED_FIELDS)} (do not include in input)")

    print("\n" + "-" * 80)
    print("BASE PATIENT FIELDS (Required in Input)")
    print("-" * 80)

    # Group by category
    continuous = []
    binary = []
    categorical = []

    for field in REQUIRED_PATIENT_FIELDS:
        info = get_field_info(field)
        if info['type'] == 'binary':
            binary.append(field)
        elif info['type'] == 'categorical':
            categorical.append(field)
        else:
            continuous.append(field)

    print(f"\nCategorical Fields ({len(categorical)}):")
    for field in categorical:
        info = get_field_info(field)
        print(f"  - {field}: {info['valid_values']}")

    print(f"\nContinuous Fields ({len(continuous)}):")
    for field in continuous:
        info = get_field_info(field)
        if info['range']:
            print(f"  - {field}: {info['range'][0]} to {info['range'][1]}")
        else:
            print(f"  - {field}: numeric value")

    print(f"\nBinary Fields ({len(binary)}):")
    for field in binary:
        print(f"  - {field}: 0 or 1")

    print("\n" + "-" * 80)
    print("ENGINEERED FIELDS (Auto-Generated - Do NOT Include)")
    print("-" * 80)
    for i, field in enumerate(ENGINEERED_FIELDS, 1):
        print(f"  {i}. {field}")

    print("\n" + "=" * 80)
    print("NOTE: Engineered features are automatically created during preprocessing.")
    print("      Only provide the base patient fields listed above.")
    print("=" * 80 + "\n")


def quick_validate(patient_dict: Dict) -> bool:
    """
    Quick validation check (returns boolean only).

    Args:
        patient_dict: Patient data dictionary

    Returns:
        True if valid, False otherwise

    Example:
        if quick_validate(patient):
            result = pipeline.predict(patient)
    """
    is_valid, _ = validate_patient_data(patient_dict, strict=True)
    return is_valid


def get_field_count_summary() -> Dict[str, int]:
    """
    Get summary of field counts.

    Returns:
        Dictionary with field count breakdown

    Example:
        summary = get_field_count_summary()
        print(f"Total features: {summary['total']}")
        print(f"Required from user: {summary['required']}")
    """
    return {
        'required_base_fields': len(REQUIRED_PATIENT_FIELDS),
        'engineered_fields': len(ENGINEERED_FIELDS),
        'total_features': len(ALL_PATIENT_FIELDS),
        'continuous_features': len([f for f in REQUIRED_PATIENT_FIELDS
                                    if f not in BINARY_FIELDS and f not in ['gender', 'ethnicity']]),
        'binary_features': len(BINARY_FIELDS),
        'categorical_features': 2  # gender, ethnicity
    }


def compare_patient_dicts(patient1: Dict, patient2: Dict) -> Dict:
    """
    Compare two patient dictionaries for differences.

    Args:
        patient1: First patient dictionary
        patient2: Second patient dictionary

    Returns:
        Dictionary with comparison results

    Example:
        diff = compare_patient_dicts(patient_old, patient_new)
        print(f"Changed fields: {diff['changed']}")
    """
    comparison = {
        'fields_only_in_1': [],
        'fields_only_in_2': [],
        'changed_values': {},
        'unchanged_count': 0
    }

    all_fields = set(patient1.keys()) | set(patient2.keys())

    for field in all_fields:
        if field not in patient1:
            comparison['fields_only_in_2'].append(field)
        elif field not in patient2:
            comparison['fields_only_in_1'].append(field)
        elif patient1[field] != patient2[field]:
            comparison['changed_values'][field] = {
                'old': patient1[field],
                'new': patient2[field]
            }
        else:
            comparison['unchanged_count'] += 1

    return comparison