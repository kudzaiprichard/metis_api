"""
Performance metrics calculation for diabetes treatment recommendation.

This module provides:
- Regression metrics (R2, RMSE, MAE, MSE)
- Reward metrics (average reward, success rate)
- Treatment diversity metrics
- Performance comparison utilities
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination) score.

    R² measures how well predictions approximate the true values.
    1.0 = perfect prediction, 0.0 = as good as mean baseline, <0 = worse than baseline

    Args:
        y_true: True values (actual HbA1c reductions)
        y_pred: Predicted values (Q-values)

    Returns:
        R² score

    Example:
        y_true = np.array([2.5, 3.0, 2.1])
        y_pred = np.array([2.4, 2.9, 2.2])
        r2 = calculate_r2_score(y_true, y_pred)
        # Returns: ~0.98
    """
    return float(r2_score(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE measures average prediction error (in same units as target).
    Lower is better. Penalizes large errors more than MAE.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value

    Example:
        rmse = calculate_rmse(y_true, y_pred)
        # Returns: ~0.15 (average error of 0.15% HbA1c)
    """
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    MAE measures average absolute prediction error.
    Lower is better. More interpretable than RMSE.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value

    Example:
        mae = calculate_mae(y_true, y_pred)
        # Returns: ~0.12 (average error of 0.12% HbA1c)
    """
    return float(mean_absolute_error(y_true, y_pred))


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MSE value
    """
    return float(mean_squared_error(y_true, y_pred))


# =============================================================================
# REWARD METRICS
# =============================================================================

def calculate_avg_reward(rewards: np.ndarray) -> float:
    """
    Calculate average reward (HbA1c reduction).

    Args:
        rewards: Array of HbA1c reductions

    Returns:
        Average reward

    Example:
        rewards = np.array([2.5, 3.0, 2.1, 2.8])
        avg = calculate_avg_reward(rewards)
        # Returns: 2.6
    """
    return float(np.mean(rewards))


def calculate_success_rate(rewards: np.ndarray,
                           threshold: float = 1.5) -> float:
    """
    Calculate success rate (proportion achieving meaningful reduction).

    Success is defined as HbA1c reduction >= threshold.
    Default threshold: 1.5% (clinically meaningful)

    Args:
        rewards: Array of HbA1c reductions
        threshold: Minimum reduction to count as success

    Returns:
        Success rate (0.0 to 1.0)

    Example:
        rewards = np.array([2.5, 3.0, 1.2, 2.8])
        success_rate = calculate_success_rate(rewards, threshold=1.5)
        # Returns: 0.75 (3 out of 4 achieved >= 1.5%)
    """
    successes = np.sum(rewards >= threshold)
    total = len(rewards)
    return float(successes / total) if total > 0 else 0.0


def calculate_treatment_diversity(recommendations: np.ndarray) -> int:
    """
    Calculate treatment diversity (number of unique treatments used).

    Higher diversity indicates the model personalizes recommendations.
    Maximum diversity: 5 (all treatments used)

    Args:
        recommendations: Array of treatment indices (0-4)

    Returns:
        Number of unique treatments

    Example:
        recommendations = np.array([0, 1, 2, 1, 3, 0, 1])
        diversity = calculate_treatment_diversity(recommendations)
        # Returns: 4 (used Metformin, GLP-1, SGLT-2, DPP-4)
    """
    return int(len(np.unique(recommendations)))


def calculate_treatment_distribution(recommendations: np.ndarray,
                                     n_treatments: int = 5) -> np.ndarray:
    """
    Calculate distribution of treatment recommendations.

    Args:
        recommendations: Array of treatment indices
        n_treatments: Total number of treatments

    Returns:
        Array of counts for each treatment

    Example:
        recommendations = np.array([0, 1, 2, 1, 4, 0, 1])
        distribution = calculate_treatment_distribution(recommendations)
        # Returns: [2, 3, 1, 0, 1] (counts for each treatment)
    """
    return np.bincount(recommendations, minlength=n_treatments)


# =============================================================================
# COMPREHENSIVE METRICS
# =============================================================================

def calculate_all_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          recommendations: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate all available metrics at once.

    Args:
        y_true: True HbA1c reductions
        y_pred: Predicted HbA1c reductions (Q-values)
        recommendations: Optional treatment recommendations

    Returns:
        Dictionary with all metrics

    Example:
        metrics = calculate_all_metrics(y_true, y_pred, recommendations)
        print(f"R²: {metrics['r2']:.3f}")
        print(f"RMSE: {metrics['rmse']:.3f}")
        print(f"Avg reward: {metrics['avg_reward']:.3f}")
    """
    metrics = {
        # Regression metrics
        'r2': calculate_r2_score(y_true, y_pred),
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'mse': calculate_mse(y_true, y_pred),

        # Reward metrics
        'avg_reward': calculate_avg_reward(y_pred),
        'success_rate': calculate_success_rate(y_pred),
    }

    # Add treatment metrics if recommendations provided
    if recommendations is not None:
        metrics['diversity'] = calculate_treatment_diversity(recommendations)
        metrics['accuracy'] = float((recommendations == np.arange(len(recommendations)) % 5).mean())

    return metrics


def calculate_per_treatment_metrics(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    treatments: np.ndarray,
                                    treatment_names: List[str]) -> Dict[str, Dict]:
    """
    Calculate metrics separately for each treatment.

    Args:
        y_true: True HbA1c reductions
        y_pred: Predicted HbA1c reductions
        treatments: Treatment assignments (0-4)
        treatment_names: List of treatment names

    Returns:
        Dictionary mapping treatment names to their metrics

    Example:
        per_treatment = calculate_per_treatment_metrics(
            y_true, y_pred, treatments,
            ['Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', 'Insulin']
        )

        print(f"Metformin R²: {per_treatment['Metformin']['r2']:.3f}")
        print(f"Insulin RMSE: {per_treatment['Insulin']['rmse']:.3f}")
    """
    per_treatment = {}

    for treatment_id, treatment_name in enumerate(treatment_names):
        mask = (treatments == treatment_id)

        if mask.sum() > 0:
            y_true_treatment = y_true[mask]
            y_pred_treatment = y_pred[mask]

            per_treatment[treatment_name] = {
                'r2': calculate_r2_score(y_true_treatment, y_pred_treatment),
                'rmse': calculate_rmse(y_true_treatment, y_pred_treatment),
                'mae': calculate_mae(y_true_treatment, y_pred_treatment),
                'avg_reward': calculate_avg_reward(y_pred_treatment),
                'success_rate': calculate_success_rate(y_pred_treatment),
                'sample_count': int(mask.sum())
            }
        else:
            per_treatment[treatment_name] = {
                'r2': 0.0,
                'rmse': 0.0,
                'mae': 0.0,
                'avg_reward': 0.0,
                'success_rate': 0.0,
                'sample_count': 0
            }

    return per_treatment


# =============================================================================
# PERFORMANCE COMPARISON
# =============================================================================

def compare_metrics(metrics_before: Dict[str, float],
                    metrics_after: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Compare metrics before and after model update.

    Args:
        metrics_before: Metrics before update
        metrics_after: Metrics after update

    Returns:
        Dictionary with comparison for each metric

    Example:
        comparison = compare_metrics(metrics_before, metrics_after)
        print(f"R² change: {comparison['r2']['change']:+.3f}")
        print(f"Avg reward improved by: {comparison['avg_reward']['change_percent']:.1f}%")
    """
    comparison = {}

    for metric_name in ['r2', 'rmse', 'mae', 'avg_reward', 'success_rate', 'diversity']:
        before = metrics_before.get(metric_name, 0.0)
        after = metrics_after.get(metric_name, 0.0)

        change = after - before
        change_percent = (change / before * 100) if before != 0 else 0.0

        comparison[metric_name] = {
            'before': float(before),
            'after': float(after),
            'change': float(change),
            'change_percent': float(change_percent)
        }

    return comparison


def format_metrics(metrics: Dict[str, float],
                   precision: int = 3,
                   as_percentage: bool = False) -> Dict[str, str]:
    """
    Format metrics for display.

    Args:
        metrics: Dictionary of metric values
        precision: Number of decimal places
        as_percentage: If True, format as percentages (for rates)

    Returns:
        Dictionary with formatted metric strings

    Example:
        metrics = {'r2': 0.856, 'avg_reward': 2.453}
        formatted = format_metrics(metrics, precision=2)
        # Returns: {'r2': '0.86', 'avg_reward': '2.45'}
    """
    formatted = {}

    for key, value in metrics.items():
        if as_percentage and key in ['success_rate', 'accuracy']:
            formatted[key] = f"{value * 100:.{precision - 2}f}%"
        else:
            formatted[key] = f"{value:.{precision}f}"

    return formatted


# =============================================================================
# STATISTICAL MEASURES
# =============================================================================

def calculate_confidence_interval(data: np.ndarray,
                                  confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for mean.

    Args:
        data: Array of values
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)

    Example:
        rewards = np.array([2.5, 3.0, 2.1, 2.8, 2.6])
        lower, upper = calculate_confidence_interval(rewards)
        print(f"95% CI: [{lower:.2f}, {upper:.2f}]")
    """
    from scipy import stats

    mean = np.mean(data)
    se = calculate_standard_error(data)

    # Calculate critical value
    alpha = 1 - confidence
    df = len(data) - 1
    critical_value = stats.t.ppf(1 - alpha / 2, df)

    margin = critical_value * se

    return float(mean - margin), float(mean + margin)


def calculate_standard_error(data: np.ndarray) -> float:
    """
    Calculate standard error of the mean.

    Args:
        data: Array of values

    Returns:
        Standard error

    Example:
        se = calculate_standard_error(rewards)
        # Returns: ~0.15
    """
    return float(np.std(data, ddof=1) / np.sqrt(len(data)))


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def generate_performance_summary(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 recommendations: np.ndarray,
                                 treatment_names: List[str]) -> Dict:
    """
    Generate comprehensive performance summary.

    Args:
        y_true: True HbA1c reductions
        y_pred: Predicted HbA1c reductions
        recommendations: Treatment recommendations
        treatment_names: List of treatment names

    Returns:
        Dictionary with comprehensive performance summary

    Example:
        summary = generate_performance_summary(
            y_true, y_pred, recommendations, treatment_names
        )

        print(f"Overall metrics: {summary['overall_metrics']}")
        print(f"Per-treatment: {summary['per_treatment_metrics']}")
    """
    # Get treatments from recommendations
    treatments = recommendations

    summary = {
        'overall_metrics': calculate_all_metrics(y_true, y_pred, recommendations),
        'per_treatment_metrics': calculate_per_treatment_metrics(
            y_true, y_pred, treatments, treatment_names
        ),
        'treatment_distribution': calculate_treatment_distribution(recommendations).tolist(),
        'sample_count': len(y_true),
        'diversity': calculate_treatment_diversity(recommendations)
    }

    # Add confidence intervals
    lower, upper = calculate_confidence_interval(y_pred)
    summary['reward_95ci'] = {
        'lower': float(lower),
        'upper': float(upper)
    }

    return summary


def calculate_improvement_metrics(baseline_metrics: Dict,
                                  current_metrics: Dict) -> Dict:
    """
    Calculate improvement from baseline to current.

    Args:
        baseline_metrics: Baseline performance metrics
        current_metrics: Current performance metrics

    Returns:
        Dictionary with improvement metrics

    Example:
        improvement = calculate_improvement_metrics(baseline, current)
        print(f"Avg reward improved: {improvement['avg_reward_improved']}")
        print(f"Improvement: {improvement['avg_reward_improvement_pct']:.1f}%")
    """
    improvement = {}

    for metric in ['avg_reward', 'success_rate', 'r2', 'diversity']:
        baseline_val = baseline_metrics.get(metric, 0.0)
        current_val = current_metrics.get(metric, 0.0)

        improved = current_val > baseline_val
        change = current_val - baseline_val
        change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0.0

        improvement[f'{metric}_improved'] = improved
        improvement[f'{metric}_improvement'] = float(change)
        improvement[f'{metric}_improvement_pct'] = float(change_pct)

    return improvement