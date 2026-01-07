"""
Safety validation utilities for diabetes treatment recommendation.

This module provides:
- Treatment-specific safety checks (Metformin, SGLT-2, Insulin)
- Contraindication detection
- Clinical consideration warnings
- Comprehensive safety evaluation
"""

from typing import List, Dict, Optional

# =============================================================================
# CONSTANTS
# =============================================================================

# eGFR thresholds (mL/min/1.73m²)
EGFR_CRITICAL_THRESHOLD = 30  # Below this: Metformin contraindicated
EGFR_CAUTION_THRESHOLD = 45  # Below this: Metformin with caution

# Age thresholds
ELDERLY_AGE_THRESHOLD = 75  # Above this: Insulin requires extra monitoring

# HbA1c thresholds
SEVERE_DIABETES_THRESHOLD = 11.0  # Above this: Consider insulin


# =============================================================================
# TREATMENT-SPECIFIC SAFETY CHECKS
# =============================================================================

def check_metformin_safety(patient_dict: Dict) -> List[str]:
    """
    Check Metformin safety based on eGFR.

    Metformin contraindications:
    - eGFR < 30: Contraindicated (risk of lactic acidosis)
    - eGFR 30-45: Use with caution

    Args:
        patient_dict: Patient data dictionary

    Returns:
        List of safety warnings (empty if safe)

    Example:
        warnings = check_metformin_safety({'egfr': 25, ...})
        # Returns: ["CRITICAL: Metformin contraindicated for eGFR < 30"]
    """
    warnings = []
    egfr = patient_dict.get('egfr', 100)

    if egfr < EGFR_CRITICAL_THRESHOLD:
        warnings.append(
            f"CRITICAL: Metformin contraindicated for eGFR < {EGFR_CRITICAL_THRESHOLD} "
            f"(current: {egfr:.1f}). High risk of lactic acidosis."
        )
    elif egfr < EGFR_CAUTION_THRESHOLD:
        warnings.append(
            f"WARNING: Use Metformin with caution for eGFR {EGFR_CRITICAL_THRESHOLD}-{EGFR_CAUTION_THRESHOLD} "
            f"(current: {egfr:.1f}). Consider dose reduction."
        )

    return warnings


def check_sglt2_safety(patient_dict: Dict) -> List[str]:
    """
    Check SGLT-2 inhibitor safety based on eGFR.

    SGLT-2 considerations:
    - eGFR < 30: Reduced efficacy
    - Still can be used for cardiovascular benefits

    Args:
        patient_dict: Patient data dictionary

    Returns:
        List of safety warnings

    Example:
        warnings = check_sglt2_safety({'egfr': 25, ...})
        # Returns: ["WARNING: SGLT-2 inhibitor less effective with eGFR < 30"]
    """
    warnings = []
    egfr = patient_dict.get('egfr', 100)

    if egfr < EGFR_CRITICAL_THRESHOLD:
        warnings.append(
            f"WARNING: SGLT-2 inhibitor less effective with eGFR < {EGFR_CRITICAL_THRESHOLD} "
            f"(current: {egfr:.1f}). May still provide cardiovascular benefits."
        )

    return warnings


def check_insulin_safety(patient_dict: Dict) -> List[str]:
    """
    Check Insulin safety based on age.

    Insulin considerations:
    - Age > 75: Increased risk of hypoglycemia
    - Requires careful monitoring

    Args:
        patient_dict: Patient data dictionary

    Returns:
        List of safety warnings

    Example:
        warnings = check_insulin_safety({'age': 80, ...})
        # Returns: ["CAUTION: Insulin in elderly (age 80) - Monitor for hypoglycemia"]
    """
    warnings = []
    age = patient_dict.get('age', 50)

    if age > ELDERLY_AGE_THRESHOLD:
        warnings.append(
            f"CAUTION: Insulin in elderly (age {age}) - "
            f"Increased risk of hypoglycemia. Monitor blood glucose closely."
        )

    return warnings


def validate_safety_constraints(patient_dict: Dict,
                                treatment: str) -> List[str]:
    """
    Check for safety contraindications for specific treatment.

    This is the main safety validation function called by prediction pipeline.

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

    # Treatment-specific checks
    if treatment == 'Metformin':
        warnings.extend(check_metformin_safety(patient_dict))

    elif treatment == 'SGLT-2':
        warnings.extend(check_sglt2_safety(patient_dict))

    elif treatment == 'Insulin':
        warnings.extend(check_insulin_safety(patient_dict))

    return warnings


# =============================================================================
# COMPREHENSIVE SAFETY CHECKS
# =============================================================================

def perform_comprehensive_safety_check(patient_dict: Dict,
                                       treatment: str) -> Dict[str, List[str]]:
    """
    Perform comprehensive safety evaluation.

    Returns warnings for:
    - Treatment-specific contraindications
    - Cardiovascular considerations
    - Renal considerations
    - Age-related considerations

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment

    Returns:
        Dictionary categorizing warnings by type

    Example:
        safety = perform_comprehensive_safety_check(patient, 'Metformin')
        print(f"Contraindications: {safety['contraindications']}")
        print(f"Considerations: {safety['clinical_considerations']}")
    """
    result = {
        'contraindications': [],
        'clinical_considerations': [],
        'cardiovascular_notes': [],
        'renal_notes': [],
        'age_notes': []
    }

    # Treatment-specific contraindications
    contraindications = validate_safety_constraints(patient_dict, treatment)
    result['contraindications'].extend(contraindications)

    # Cardiovascular considerations
    cv_notes = check_cardiovascular_considerations(patient_dict, treatment)
    result['cardiovascular_notes'].extend(cv_notes)

    # Renal considerations
    renal_notes = check_renal_considerations(patient_dict, treatment)
    result['renal_notes'].extend(renal_notes)

    # Age considerations
    age_notes = check_age_considerations(patient_dict, treatment)
    result['age_notes'].extend(age_notes)

    # Combine all into clinical considerations
    result['clinical_considerations'] = (
            cv_notes + renal_notes + age_notes
    )

    return result


def get_all_safety_warnings(patient_dict: Dict,
                            treatment: str) -> List[str]:
    """
    Get all safety warnings as a flat list.

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment

    Returns:
        List of all safety warnings

    Example:
        all_warnings = get_all_safety_warnings(patient, 'Metformin')
        for warning in all_warnings:
            print(f"- {warning}")
    """
    safety = perform_comprehensive_safety_check(patient_dict, treatment)

    all_warnings = []
    all_warnings.extend(safety['contraindications'])
    all_warnings.extend(safety['clinical_considerations'])

    return all_warnings


# =============================================================================
# CLINICAL CONSIDERATIONS
# =============================================================================

def check_cardiovascular_considerations(patient_dict: Dict,
                                        treatment: str) -> List[str]:
    """
    Check cardiovascular-related considerations.

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment

    Returns:
        List of cardiovascular considerations

    Example:
        cv_notes = check_cardiovascular_considerations(patient, 'DPP-4')
        # May return: ["NOTE: Patient has CVD. GLP-1 or SGLT-2 may provide additional CV benefits"]
    """
    notes = []
    cvd = patient_dict.get('cvd', 0)

    # CVD present but not using cardioprotective agent
    if cvd == 1 and treatment not in ['GLP-1', 'SGLT-2']:
        notes.append(
            f"NOTE: Patient has cardiovascular disease. "
            f"GLP-1 or SGLT-2 inhibitors may provide additional cardiovascular benefits. "
            f"Currently recommending: {treatment}"
        )

    return notes


def check_renal_considerations(patient_dict: Dict,
                               treatment: str) -> List[str]:
    """
    Check renal-related considerations.

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment

    Returns:
        List of renal considerations
    """
    notes = []
    ckd = patient_dict.get('ckd', 0)
    egfr = patient_dict.get('egfr', 100)

    # CKD present
    if ckd == 1:
        notes.append(
            f"NOTE: Patient has chronic kidney disease (eGFR: {egfr:.1f}). "
            f"Careful medication dosing required."
        )

        # Specific treatment notes for CKD
        if treatment == 'Metformin' and egfr < EGFR_CAUTION_THRESHOLD:
            notes.append(
                f"Consider alternative to Metformin given CKD and eGFR {egfr:.1f}"
            )

        if treatment == 'SGLT-2' and egfr < 45:
            notes.append(
                f"SGLT-2 efficacy may be reduced with eGFR {egfr:.1f}, "
                f"but cardiovascular and renal benefits may still apply"
            )

    return notes


def check_age_considerations(patient_dict: Dict,
                             treatment: str) -> List[str]:
    """
    Check age-related considerations.

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment

    Returns:
        List of age-related considerations
    """
    notes = []
    age = patient_dict.get('age', 50)

    # Elderly patients
    if age > ELDERLY_AGE_THRESHOLD:
        notes.append(
            f"NOTE: Elderly patient (age {age}). "
            f"Simplified treatment regimens and hypoglycemia prevention are priorities."
        )

        # Specific treatment notes for elderly
        if treatment == 'Insulin':
            notes.append(
                f"Insulin in elderly requires careful dose titration and frequent monitoring"
            )

        if treatment in ['SGLT-2', 'GLP-1']:
            notes.append(
                f"Monitor for dehydration and volume depletion in elderly patients on {treatment}"
            )

    return notes


# =============================================================================
# ADDITIONAL SAFETY UTILITIES
# =============================================================================

def check_severe_diabetes(patient_dict: Dict) -> bool:
    """
    Check if patient has severe uncontrolled diabetes.

    Args:
        patient_dict: Patient data

    Returns:
        True if HbA1c > 11.0 (severe)

    Example:
        is_severe = check_severe_diabetes({'hba1c_baseline': 12.5, ...})
        # Returns: True
    """
    hba1c = patient_dict.get('hba1c_baseline', 7.0)
    return hba1c > SEVERE_DIABETES_THRESHOLD


def check_hypoglycemia_risk(patient_dict: Dict, treatment: str) -> bool:
    """
    Check if patient has elevated hypoglycemia risk.

    Risk factors:
    - Elderly (age > 75)
    - Low eGFR
    - Insulin therapy

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment

    Returns:
        True if hypoglycemia risk is elevated

    Example:
        at_risk = check_hypoglycemia_risk({'age': 80, ...}, 'Insulin')
        # Returns: True
    """
    age = patient_dict.get('age', 50)
    egfr = patient_dict.get('egfr', 100)

    # High risk if elderly on insulin
    if age > ELDERLY_AGE_THRESHOLD and treatment == 'Insulin':
        return True

    # High risk if low eGFR
    if egfr < EGFR_CRITICAL_THRESHOLD:
        return True

    return False


def get_treatment_precautions(treatment: str) -> List[str]:
    """
    Get general precautions for a treatment.

    Args:
        treatment: Treatment name

    Returns:
        List of general precautions

    Example:
        precautions = get_treatment_precautions('Metformin')
        # Returns: ["Monitor renal function regularly", ...]
    """
    precautions_map = {
        'Metformin': [
            "Monitor renal function regularly (every 3-6 months)",
            "Take with meals to reduce GI side effects",
            "Watch for signs of lactic acidosis (rare but serious)"
        ],
        'GLP-1': [
            "Monitor for gastrointestinal side effects (nausea, vomiting)",
            "Start with low dose and titrate gradually",
            "Educate on injection technique",
            "Monitor for pancreatitis symptoms"
        ],
        'SGLT-2': [
            "Monitor for genital yeast infections",
            "Ensure adequate hydration",
            "Watch for signs of diabetic ketoacidosis",
            "Monitor renal function regularly"
        ],
        'DPP-4': [
            "Generally well-tolerated with few side effects",
            "Monitor for joint pain (rare)",
            "No dose adjustment needed for most patients"
        ],
        'Insulin': [
            "Educate on injection technique and timing",
            "Monitor blood glucose frequently",
            "Watch for hypoglycemia symptoms",
            "Rotate injection sites to prevent lipodystrophy"
        ]
    }

    return precautions_map.get(treatment, [])


def check_obesity_considerations(patient_dict: Dict, treatment: str) -> List[str]:
    """
    Check obesity-related considerations.

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment

    Returns:
        List of obesity-related notes

    Example:
        notes = check_obesity_considerations({'bmi': 38, ...}, 'Metformin')
        # Returns: ["NOTE: Patient is obese (BMI 38.0). GLP-1 may provide weight loss benefits"]
    """
    notes = []
    bmi = patient_dict.get('bmi', 25)

    # Obesity (BMI >= 30)
    if bmi >= 30:
        if treatment not in ['GLP-1', 'SGLT-2']:
            notes.append(
                f"NOTE: Patient is obese (BMI {bmi:.1f}). "
                f"GLP-1 or SGLT-2 inhibitors may provide additional weight loss benefits."
            )
        else:
            notes.append(
                f"NOTE: {treatment} is appropriate for obese patients and may aid weight loss."
            )

    # Severe obesity (BMI >= 35)
    if bmi >= 35:
        notes.append(
            f"NOTE: Patient has severe obesity (BMI {bmi:.1f}). "
            f"Consider multidisciplinary approach including nutrition and lifestyle counseling."
        )

    return notes


def check_nafld_considerations(patient_dict: Dict, treatment: str) -> List[str]:
    """
    Check NAFLD (non-alcoholic fatty liver disease) considerations.

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment

    Returns:
        List of NAFLD-related notes

    Example:
        notes = check_nafld_considerations({'nafld': 1, ...}, 'DPP-4')
        # Returns: ["NOTE: Patient has NAFLD. GLP-1 or SGLT-2 may improve liver outcomes"]
    """
    notes = []
    nafld = patient_dict.get('nafld', 0)
    alt = patient_dict.get('alt', 30)

    if nafld == 1:
        if treatment not in ['GLP-1', 'SGLT-2']:
            notes.append(
                f"NOTE: Patient has NAFLD. "
                f"GLP-1 or SGLT-2 inhibitors may provide additional liver health benefits."
            )

        if alt > 40:
            notes.append(
                f"NOTE: Elevated ALT ({alt:.0f} U/L) with NAFLD. "
                f"Monitor liver function regularly."
            )

    return notes


def check_polypharmacy_risk(patient_dict: Dict) -> List[str]:
    """
    Check for polypharmacy concerns.

    Args:
        patient_dict: Patient data

    Returns:
        List of polypharmacy-related notes

    Example:
        notes = check_polypharmacy_risk({
            'hypertension': 1, 'cvd': 1, 'ckd': 1, ...
        })
    """
    notes = []

    # Count comorbidities (proxy for polypharmacy)
    comorbidity_count = sum([
        patient_dict.get('hypertension', 0),
        patient_dict.get('ckd', 0),
        patient_dict.get('cvd', 0),
        patient_dict.get('nafld', 0),
        patient_dict.get('retinopathy', 0)
    ])

    if comorbidity_count >= 3:
        notes.append(
            f"NOTE: Patient has {comorbidity_count} comorbidities. "
            f"Consider polypharmacy and drug interaction risks. "
            f"Simplified regimen may improve adherence."
        )

    return notes


def generate_safety_report(patient_dict: Dict,
                           treatment: str) -> Dict[str, any]:
    """
    Generate comprehensive safety report.

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment

    Returns:
        Comprehensive safety report with all checks

    Example:
        report = generate_safety_report(patient, 'Metformin')
        print(f"Contraindications: {report['contraindications']}")
        print(f"Risk level: {report['overall_risk_level']}")
    """
    # Get comprehensive safety check
    safety = perform_comprehensive_safety_check(patient_dict, treatment)

    # Additional considerations
    obesity_notes = check_obesity_considerations(patient_dict, treatment)
    nafld_notes = check_nafld_considerations(patient_dict, treatment)
    polypharmacy_notes = check_polypharmacy_risk(patient_dict)
    precautions = get_treatment_precautions(treatment)

    # Risk assessment
    has_contraindications = len(safety['contraindications']) > 0
    has_critical = any('CRITICAL' in w for w in safety['contraindications'])
    hypoglycemia_risk = check_hypoglycemia_risk(patient_dict, treatment)
    severe_diabetes = check_severe_diabetes(patient_dict)

    # Determine overall risk level
    if has_critical:
        risk_level = 'CRITICAL'
    elif has_contraindications:
        risk_level = 'HIGH'
    elif hypoglycemia_risk or len(safety['clinical_considerations']) > 2:
        risk_level = 'MODERATE'
    else:
        risk_level = 'LOW'

    report = {
        'treatment': treatment,
        'overall_risk_level': risk_level,
        'contraindications': safety['contraindications'],
        'cardiovascular_notes': safety['cardiovascular_notes'],
        'renal_notes': safety['renal_notes'],
        'age_notes': safety['age_notes'],
        'obesity_notes': obesity_notes,
        'nafld_notes': nafld_notes,
        'polypharmacy_notes': polypharmacy_notes,
        'general_precautions': precautions,
        'clinical_flags': {
            'has_contraindications': has_contraindications,
            'has_critical_contraindications': has_critical,
            'hypoglycemia_risk': hypoglycemia_risk,
            'severe_diabetes': severe_diabetes
        },
        'patient_summary': {
            'age': patient_dict.get('age'),
            'egfr': patient_dict.get('egfr'),
            'bmi': patient_dict.get('bmi'),
            'hba1c_baseline': patient_dict.get('hba1c_baseline')
        }
    }

    return report


def format_safety_warnings(warnings: List[str],
                           max_length: int = 100) -> List[str]:
    """
    Format safety warnings for display.

    Args:
        warnings: List of warning strings
        max_length: Maximum length per warning

    Returns:
        Formatted warnings
    """
    formatted = []

    for warning in warnings:
        if len(warning) > max_length:
            # Truncate and add ellipsis
            formatted.append(warning[:max_length - 3] + '...')
        else:
            formatted.append(warning)

    return formatted


def prioritize_warnings(warnings: List[str]) -> List[str]:
    """
    Prioritize warnings (critical first, then warnings, then notes).

    Args:
        warnings: List of warning strings

    Returns:
        Sorted list with critical warnings first

    Example:
        warnings = ["NOTE: ...", "CRITICAL: ...", "WARNING: ..."]
        prioritized = prioritize_warnings(warnings)
        # Returns: ["CRITICAL: ...", "WARNING: ...", "NOTE: ..."]
    """
    critical = [w for w in warnings if w.startswith('CRITICAL')]
    warnings_level = [w for w in warnings if w.startswith('WARNING')]
    caution = [w for w in warnings if w.startswith('CAUTION')]
    notes = [w for w in warnings if w.startswith('NOTE')]
    other = [w for w in warnings if not any(w.startswith(p) for p in ['CRITICAL', 'WARNING', 'CAUTION', 'NOTE'])]

    return critical + warnings_level + caution + notes + other


def count_warnings_by_severity(warnings: List[str]) -> Dict[str, int]:
    """
    Count warnings by severity level.

    Args:
        warnings: List of warning strings

    Returns:
        Dictionary with counts per severity

    Example:
        counts = count_warnings_by_severity(warnings)
        # Returns: {'critical': 1, 'warning': 2, 'caution': 1, 'note': 3}
    """
    counts = {
        'critical': 0,
        'warning': 0,
        'caution': 0,
        'note': 0,
        'other': 0
    }

    for warning in warnings:
        if warning.startswith('CRITICAL'):
            counts['critical'] += 1
        elif warning.startswith('WARNING'):
            counts['warning'] += 1
        elif warning.startswith('CAUTION'):
            counts['caution'] += 1
        elif warning.startswith('NOTE'):
            counts['note'] += 1
        else:
            counts['other'] += 1

    return counts


def is_treatment_safe(patient_dict: Dict, treatment: str) -> bool:
    """
    Simple boolean check if treatment is safe (no critical contraindications).

    Args:
        patient_dict: Patient data
        treatment: Proposed treatment

    Returns:
        True if no critical contraindications, False otherwise

    Example:
        safe = is_treatment_safe({'egfr': 25, ...}, 'Metformin')
        # Returns: False (eGFR too low for Metformin)
    """
    warnings = validate_safety_constraints(patient_dict, treatment)
    has_critical = any('CRITICAL' in w for w in warnings)

    return not has_critical