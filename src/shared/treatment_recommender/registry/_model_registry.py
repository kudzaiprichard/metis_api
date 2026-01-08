"""
Model registry for version management and discovery.

This module provides:
- Active model discovery
- Version listing and filtering
- Model comparison
- Registry operations (activate, deactivate, delete)
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime


class ModelRegistry:
    """
    Central registry for managing model versions.

    This class provides:
    1. Version discovery - Find all available model versions
    2. Active model management - Get/set active version
    3. Model comparison - Compare performance across versions
    4. Version filtering - Filter by date, performance, etc.

    The registry operates on the centralized model_metadata.json file
    and provides a clean interface for model management operations.

    Usage:
        # Create registry
        registry = ModelRegistry()

        # Get active version
        active = registry.get_active_version()
        print(f"Active: {active}")

        # List all versions
        versions = registry.list_versions()
        for v in versions:
            print(f"{v['version_number']}: {v['performance_metrics']}")

        # Compare versions
        comparison = registry.compare_versions('v1_0', 'v1_2')
        print(f"Performance change: {comparison}")

        # Activate a version
        registry.activate_version('v1_2')
    """

    def __init__(self,
                 models_dir: str = 'artifacts',
                 verbose: bool = False):
        """
        Initialize model registry.

        Args:
            models_dir: Base directory for artifacts
            verbose: If True, print detailed logs
        """
        self.models_dir = models_dir
        self.verbose = verbose

        # Initialize metadata manager
        from ..registry import create_metadata_manager
        self.metadata_manager = create_metadata_manager(
            metadata_file_path=None,
            models_dir=self.models_dir
        )

        # Initialize if metadata file doesn't exist
        if not os.path.exists(self.metadata_manager.metadata_file_path):
            self.metadata_manager.initialize_metadata_file()

        if self.verbose:
            print(f"[ModelRegistry] Initialized")
            print(f"[ModelRegistry] Models directory: {models_dir}")

    # =========================================================================
    # VERSION DISCOVERY
    # =========================================================================

    def get_active_version(self) -> Optional[str]:
        """
        Get the currently active version.

        Returns:
            Active version number (e.g., "v1_2") or None if no active version

        Example:
            active = registry.get_active_version()
            # Returns: "v1_0"
        """
        version = self.metadata_manager.get_active_version()

        if self.verbose:
            print(f"[ModelRegistry] Active version: {version}")

        return version

    def get_latest_version(self) -> Optional[str]:
        """
        Get the latest version (most recently created).

        Returns:
            Latest version number or None if no versions

        Example:
            latest = registry.get_latest_version()
            # Returns: "v1_2"
        """
        version = self.metadata_manager.get_latest_version()

        if self.verbose:
            print(f"[ModelRegistry] Latest version: {version}")

        return version

    def get_version_details(self, version_number: str) -> Optional[Dict]:
        """
        Get detailed information for a specific version.

        Args:
            version_number: Version to retrieve (e.g., "v1_2")

        Returns:
            Version details dictionary or None if not found

        Example:
            details = registry.get_version_details('v1_2')
            print(f"Metrics: {details['performance_metrics']}")
            print(f"Trained: {details['trained_timestamp']}")
        """
        details = self.metadata_manager.get_version_details(version_number)

        if self.verbose and details:
            print(f"[ModelRegistry] Version {version_number} found")
            print(f"[ModelRegistry] Performance: {details.get('performance_metrics')}")

        return details

    def list_versions(self,
                      sort_by: str = 'version',
                      reverse: bool = False) -> List[Dict]:
        """
        List all available model versions.

        Args:
            sort_by: Sort key ('version', 'date', 'avg_reward', 'accuracy')
            reverse: If True, reverse sort order

        Returns:
            List of version dictionaries

        Example:
            # Get all versions sorted by performance
            versions = registry.list_versions(sort_by='avg_reward', reverse=True)

            for v in versions:
                print(f"{v['version_number']}: {v['performance_metrics']['avg_reward']}")
        """
        versions = self.metadata_manager.get_all_versions()

        # Sort versions
        if sort_by == 'version':
            versions.sort(key=lambda v: v['version_number'], reverse=reverse)
        elif sort_by == 'date':
            versions.sort(
                key=lambda v: v.get('trained_timestamp', ''),
                reverse=reverse
            )
        elif sort_by == 'avg_reward':
            versions.sort(
                key=lambda v: v.get('performance_metrics', {}).get('avg_reward', 0),
                reverse=reverse
            )
        elif sort_by == 'accuracy':
            versions.sort(
                key=lambda v: v.get('performance_metrics', {}).get('accuracy', 0),
                reverse=reverse
            )

        if self.verbose:
            print(f"[ModelRegistry] Found {len(versions)} versions")

        return versions

    def get_version_count(self) -> int:
        """
        Get total number of versions.

        Returns:
            Count of versions

        Example:
            count = registry.get_version_count()
            # Returns: 3
        """
        count = self.metadata_manager.get_total_versions()

        if self.verbose:
            print(f"[ModelRegistry] Total versions: {count}")

        return count

    def version_exists(self, version_number: str) -> bool:
        """
        Check if a version exists.

        Args:
            version_number: Version to check

        Returns:
            True if version exists, False otherwise

        Example:
            exists = registry.version_exists('v1_5')
            # Returns: False
        """
        details = self.metadata_manager.get_version_details(version_number)
        return details is not None

    # =========================================================================
    # VERSION FILTERING
    # =========================================================================

    def filter_versions_by_date(self,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> List[Dict]:
        """
        Filter versions by training date.

        Args:
            start_date: Start date (ISO format: "2025-01-15")
            end_date: End date (ISO format: "2025-01-20")

        Returns:
            List of versions within date range

        Example:
            versions = registry.filter_versions_by_date(
                start_date='2025-01-15',
                end_date='2025-01-20'
            )
        """
        all_versions = self.metadata_manager.get_all_versions()
        filtered = []

        for version in all_versions:
            timestamp = version.get('trained_timestamp', '')

            # Parse date from timestamp (format: "2025-01-15 10:30:00")
            date_str = timestamp.split(' ')[0] if ' ' in timestamp else timestamp

            # Check date range
            include = True
            if start_date and date_str < start_date:
                include = False
            if end_date and date_str > end_date:
                include = False

            if include:
                filtered.append(version)

        if self.verbose:
            print(f"[ModelRegistry] Filtered {len(filtered)}/{len(all_versions)} versions by date")

        return filtered

    def filter_versions_by_performance(self,
                                       metric: str = 'avg_reward',
                                       min_value: Optional[float] = None,
                                       max_value: Optional[float] = None) -> List[Dict]:
        """
        Filter versions by performance metric.

        Args:
            metric: Metric name ('avg_reward', 'accuracy', 'success_rate')
            min_value: Minimum metric value (inclusive)
            max_value: Maximum metric value (inclusive)

        Returns:
            List of versions matching criteria

        Example:
            # Get versions with avg_reward >= 2.5
            high_performers = registry.filter_versions_by_performance(
                metric='avg_reward',
                min_value=2.5
            )
        """
        all_versions = self.metadata_manager.get_all_versions()
        filtered = []

        for version in all_versions:
            metrics = version.get('performance_metrics', {})
            value = metrics.get(metric)

            if value is None:
                continue

            # Check range
            include = True
            if min_value is not None and value < min_value:
                include = False
            if max_value is not None and value > max_value:
                include = False

            if include:
                filtered.append(version)

        if self.verbose:
            print(f"[ModelRegistry] Filtered {len(filtered)}/{len(all_versions)} versions by {metric}")

        return filtered

    def get_best_version(self, metric: str = 'avg_reward') -> Optional[Dict]:
        """
        Get best-performing version by metric.

        Args:
            metric: Metric to optimize ('avg_reward', 'accuracy', etc.)

        Returns:
            Best version details or None if no versions

        Example:
            best = registry.get_best_version(metric='avg_reward')
            print(f"Best version: {best['version_number']}")
            print(f"Avg reward: {best['performance_metrics']['avg_reward']}")
        """
        versions = self.list_versions(sort_by=metric, reverse=True)

        if len(versions) == 0:
            return None

        best = versions[0]

        if self.verbose:
            metric_value = best.get('performance_metrics', {}).get(metric, 0)
            print(f"[ModelRegistry] Best version by {metric}: {best['version_number']} ({metric_value})")

        return best

    # =========================================================================
    # VERSION COMPARISON
    # =========================================================================

    def compare_versions(self,
                         version1: str,
                         version2: str) -> Dict:
        """
        Compare performance between two versions.

        Args:
            version1: First version number
            version2: Second version number

        Returns:
            Comparison dictionary with metrics differences

        Example:
            comparison = registry.compare_versions('v1_0', 'v1_2')

            print(f"Avg reward change: {comparison['avg_reward_diff']:.3f}")
            print(f"Accuracy change: {comparison['accuracy_diff']:.3f}")
        """
        comparison = self.metadata_manager.get_performance_comparison(version1, version2)

        if self.verbose:
            print(f"[ModelRegistry] Comparing {version1} vs {version2}")
            print(f"[ModelRegistry] Avg reward diff: {comparison['avg_reward_diff']:.3f}")
            print(f"[ModelRegistry] Accuracy diff: {comparison['accuracy_diff']:.3f}")

        return comparison

    def compare_to_baseline(self,
                            version: str,
                            baseline: str = 'v1_0') -> Dict:
        """
        Compare a version to baseline (default: v1_0).

        Args:
            version: Version to compare
            baseline: Baseline version (default: 'v1_0')

        Returns:
            Comparison dictionary

        Example:
            improvement = registry.compare_to_baseline('v1_2')
            print(f"Improvement over baseline: {improvement['avg_reward_diff']:.3f}")
        """
        return self.compare_versions(baseline, version)

    def get_version_lineage(self, version_number: str) -> List[str]:
        """
        Get the lineage of a version (trace back to root).

        Args:
            version_number: Version to trace

        Returns:
            List of version numbers from root to specified version

        Example:
            lineage = registry.get_version_lineage('v1_3')
            # Returns: ['v1_0', 'v1_1', 'v1_2', 'v1_3']
        """
        lineage = self.metadata_manager.get_version_lineage(version_number)

        if self.verbose:
            print(f"[ModelRegistry] Lineage for {version_number}: {' -> '.join(lineage)}")

        return lineage

    # =========================================================================
    # ACTIVE MODEL MANAGEMENT
    # =========================================================================

    def activate_version(self, version_number: str) -> bool:
        """
        Set a version as active (and deactivate all others).

        Args:
            version_number: Version to activate

        Returns:
            True if successful, False if version not found

        Example:
            success = registry.activate_version('v1_2')
            if success:
                print("Version v1_2 is now active")
        """
        success = self.metadata_manager.set_active_version(version_number)

        if self.verbose:
            if success:
                print(f"[ModelRegistry] Activated version: {version_number}")
            else:
                print(f"[ModelRegistry] Failed to activate version: {version_number}")

        return success

    def deactivate_all_versions(self) -> None:
        """
        Deactivate all versions (no active version).

        This is useful when you want to temporarily disable the model
        or when testing without an active version.

        Example:
            registry.deactivate_all_versions()
        """
        # Get all versions and deactivate them
        versions = self.metadata_manager.get_all_versions()

        for version in versions:
            version['is_active'] = False

        # Update metadata
        metadata = self.metadata_manager._load_metadata()
        metadata['versions'] = versions
        metadata['active_version'] = None
        metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata_manager._save_metadata(metadata)

        if self.verbose:
            print("[ModelRegistry] All versions deactivated")

    # =========================================================================
    # MODEL FILE PATHS
    # =========================================================================

    def get_model_path(self, version_number: str) -> Optional[str]:
        """
        Get the model file path for a specific version.

        Args:
            version_number: Version number

        Returns:
            Path to model .pth file or None if not found

        Example:
            path = registry.get_model_path('v1_2')
            # Returns: "artifacts/v1_2/production/neural_t_learner.pth"
        """
        path = self.metadata_manager.get_model_file_path(version_number)

        if self.verbose:
            print(f"[ModelRegistry] Model path for {version_number}: {path}")

        return path

    def get_active_model_path(self) -> Optional[str]:
        """
        Get the model file path for the active version.

        Returns:
            Path to active model or None if no active version

        Example:
            path = registry.get_active_model_path()
            # Use this path to load the active model
        """
        active_version = self.get_active_version()

        if active_version is None:
            if self.verbose:
                print("[ModelRegistry] No active version")
            return None

        return self.get_model_path(active_version)

    def get_latest_model_path(self) -> Optional[str]:
        """
        Get the model file path for the latest version.

        Returns:
            Path to latest model or None if no versions

        Example:
            path = registry.get_latest_model_path()
        """
        latest_version = self.get_latest_version()

        if latest_version is None:
            if self.verbose:
                print("[ModelRegistry] No versions available")
            return None

        return self.get_model_path(latest_version)

    def get_shared_components(self) -> Dict[str, str]:
        """
        Get paths to shared preprocessing components.

        Returns:
            Dictionary with scaler and metadata paths

        Example:
            components = registry.get_shared_components()
            scaler_path = components['feature_scaler']
            metadata_path = components['preprocessing_metadata']
        """
        components = self.metadata_manager.get_shared_components()

        if self.verbose:
            print(f"[ModelRegistry] Shared components: {components}")

        return components

    # =========================================================================
    # VERSION INFORMATION
    # =========================================================================

    def get_version_summary(self, version_number: str) -> Optional[Dict]:
        """
        Get a concise summary of a version.

        Args:
            version_number: Version to summarize

        Returns:
            Summary dictionary with key information

        Example:
            summary = registry.get_version_summary('v1_2')
            print(f"Version: {summary['version']}")
            print(f"Active: {summary['is_active']}")
            print(f"Performance: {summary['performance']}")
        """
        details = self.get_version_details(version_number)

        if details is None:
            return None

        metrics = details.get('performance_metrics', {})

        summary = {
            'version': version_number,
            'is_active': details.get('is_active', False),
            'parent': details.get('parent_version'),
            'created': details.get('trained_timestamp'),
            'training_method': details.get('training_method'),
            'performance': {
                'avg_reward': metrics.get('avg_reward', 0),
                'accuracy': metrics.get('accuracy', 0),
                'diversity': metrics.get('diversity', 0)
            },
            'training_info': details.get('training_info', {}),
            'model_path': details.get('model_file_path')
        }

        return summary

    def print_version_info(self, version_number: str) -> None:
        """
        Print detailed version information to console.

        Args:
            version_number: Version to display

        Example:
            registry.print_version_info('v1_2')
        """
        details = self.get_version_details(version_number)

        if details is None:
            print(f"Version {version_number} not found")
            return

        print("\n" + "=" * 80)
        print(f"VERSION: {version_number}")
        print("=" * 80)

        print(f"\nStatus:")
        print(f"  Active: {details.get('is_active', False)}")
        print(f"  Parent: {details.get('parent_version', 'None')}")
        print(f"  Created: {details.get('trained_timestamp')}")
        print(f"  Method: {details.get('training_method')}")

        print(f"\nPerformance Metrics:")
        metrics = details.get('performance_metrics', {})
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print(f"\nTraining Info:")
        training_info = details.get('training_info', {})
        for key, value in training_info.items():
            print(f"  {key}: {value}")

        print(f"\nModel Path:")
        print(f"  {details.get('model_file_path')}")

        if details.get('notes'):
            print(f"\nNotes:")
            print(f"  {details['notes']}")

        print("=" * 80 + "\n")

    def print_all_versions(self) -> None:
        """
        Print summary of all versions.

        Example:
            registry.print_all_versions()
        """
        versions = self.list_versions(sort_by='version')

        print("\n" + "=" * 80)
        print("ALL MODEL VERSIONS")
        print("=" * 80)

        if len(versions) == 0:
            print("No versions found")
            print("=" * 80 + "\n")
            return

        print(f"\nTotal versions: {len(versions)}")
        print(f"Active version: {self.get_active_version()}")
        print(f"Latest version: {self.get_latest_version()}")

        print(f"\n{'Version':<12} {'Active':<8} {'Avg Reward':<12} {'Accuracy':<12} {'Created'}")
        print("-" * 80)

        for v in versions:
            version_num = v['version_number']
            is_active = "✓" if v.get('is_active', False) else ""
            metrics = v.get('performance_metrics', {})
            avg_reward = metrics.get('avg_reward', 0)
            accuracy = metrics.get('accuracy', 0)
            created = v.get('trained_timestamp', '')

            print(f"{version_num:<12} {is_active:<8} {avg_reward:<12.4f} {accuracy:<12.4f} {created}")

        print("=" * 80 + "\n")

    # =========================================================================
    # VERSION DELETION
    # =========================================================================

    def delete_version(self, version_number: str,
                       delete_files: bool = False) -> bool:
        """
        Delete a version from registry.

        Args:
            version_number: Version to delete
            delete_files: If True, also delete model files from disk

        Returns:
            True if deleted, False if failed

        Note:
            Cannot delete active version. Deactivate first.

        Example:
            # Delete from metadata only
            registry.delete_version('v1_1')

            # Delete metadata AND files
            registry.delete_version('v1_1', delete_files=True)
        """
        # Check if active
        if self.get_active_version() == version_number:
            if self.verbose:
                print(f"[ModelRegistry] Cannot delete active version {version_number}")
            return False

        # Delete from metadata
        success = self.metadata_manager.delete_version(version_number)

        if not success:
            if self.verbose:
                print(f"[ModelRegistry] Version {version_number} not found")
            return False

        # Delete files if requested
        if delete_files:
            version_dir = os.path.join(self.models_dir, version_number)
            if os.path.exists(version_dir):
                import shutil
                shutil.rmtree(version_dir)
                if self.verbose:
                    print(f"[ModelRegistry] Deleted files: {version_dir}")

        if self.verbose:
            print(f"[ModelRegistry] Deleted version: {version_number}")

        return True


# =============================================================================
# STANDALONE HELPER FUNCTIONS
# =============================================================================

def create_model_registry(models_dir: str = 'artifacts',
                          verbose: bool = False) -> ModelRegistry:
    """
    Factory function to create ModelRegistry instance.

    Args:
        models_dir: Base directory for artifacts
        verbose: Enable detailed logging

    Returns:
        ModelRegistry instance

    Example:
        registry = create_model_registry()
        active = registry.get_active_version()
    """
    return ModelRegistry(models_dir=models_dir, verbose=verbose)


def get_active_model_path(models_dir: str = 'artifacts') -> Optional[str]:
    """
    Quick helper to get active model path.

    Args:
        models_dir: Base directory for artifacts

    Returns:
        Path to active model or None

    Example:
        path = get_active_model_path()
        if path:
            model = load_model(path)
    """
    registry = ModelRegistry(models_dir=models_dir, verbose=False)
    return registry.get_active_model_path()


def get_latest_model_path(models_dir: str = 'artifacts') -> Optional[str]:
    """
    Quick helper to get latest model path.

    Args:
        models_dir: Base directory for artifacts

    Returns:
        Path to latest model or None

    Example:
        path = get_latest_model_path()
    """
    registry = ModelRegistry(models_dir=models_dir, verbose=False)
    return registry.get_latest_model_path()


def list_all_versions(models_dir: str = 'artifacts',
                      sort_by: str = 'version') -> List[Dict]:
    """
    Quick helper to list all versions.

    Args:
        models_dir: Base directory for artifacts
        sort_by: Sort key

    Returns:
        List of version dictionaries

    Example:
        versions = list_all_versions(sort_by='avg_reward')
        for v in versions:
            print(v['version_number'])
    """
    registry = ModelRegistry(models_dir=models_dir, verbose=False)
    return registry.list_versions(sort_by=sort_by)


def compare_model_versions(version1: str,
                           version2: str,
                           models_dir: str = 'artifacts') -> Dict:
    """
    Quick helper to compare two versions.

    Args:
        version1: First version
        version2: Second version
        models_dir: Base directory for artifacts

    Returns:
        Comparison dictionary

    Example:
        comparison = compare_model_versions('v1_0', 'v1_2')
        print(f"Improvement: {comparison['avg_reward_diff']:.3f}")
    """
    registry = ModelRegistry(models_dir=models_dir, verbose=False)
    return registry.compare_versions(version1, version2)