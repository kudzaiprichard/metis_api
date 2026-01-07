"""
Utility functions for diabetes treatment recommendation.

This module provides:
- Performance metrics (R2, RMSE, MAE, reward metrics)
- Safety validation (eGFR checks, contraindications)
- Treatment evaluation utilities
"""

from .metrics import (
    # Regression metrics
    calculate_r2_score,
    calculate_rmse,
    calculate_mae,
    calculate_mse,

    # Reward metrics
    calculate_avg_reward,
    calculate_success_rate,
    calculate_treatment_diversity,
    calculate_treatment_distribution,

    # Comprehensive evaluation
    calculate_all_metrics,
    calculate_per_treatment_metrics,
    compare_metrics,
    format_metrics,

    # Statistical measures
    calculate_confidence_interval,
    calculate_standard_error
)

from .safety import (
    # Safety checks
    check_metformin_safety,
    check_sglt2_safety,
    check_insulin_safety,
    validate_safety_constraints,

    # Comprehensive safety
    perform_comprehensive_safety_check,
    get_all_safety_warnings,

    # Clinical considerations
    check_cardiovascular_considerations,
    check_renal_considerations,
    check_age_considerations,

    # Constants
    EGFR_CRITICAL_THRESHOLD,
    EGFR_CAUTION_THRESHOLD,
    ELDERLY_AGE_THRESHOLD
)

__all__ = [
    # Regression metrics
    'calculate_r2_score',
    'calculate_rmse',
    'calculate_mae',
    'calculate_mse',

    # Reward metrics
    'calculate_avg_reward',
    'calculate_success_rate',
    'calculate_treatment_diversity',
    'calculate_treatment_distribution',

    # Comprehensive evaluation
    'calculate_all_metrics',
    'calculate_per_treatment_metrics',
    'compare_metrics',
    'format_metrics',

    # Statistical measures
    'calculate_confidence_interval',
    'calculate_standard_error',

    # Safety checks
    'check_metformin_safety',
    'check_sglt2_safety',
    'check_insulin_safety',
    'validate_safety_constraints',
    'perform_comprehensive_safety_check',
    'get_all_safety_warnings',

    # Clinical considerations
    'check_cardiovascular_considerations',
    'check_renal_considerations',
    'check_age_considerations',

    # Constants
    'EGFR_CRITICAL_THRESHOLD',
    'EGFR_CAUTION_THRESHOLD',
    'ELDERLY_AGE_THRESHOLD',
]