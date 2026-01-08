"""
Model version metadata management for diabetes treatment recommendation.

This module provides:
- Centralized version tracking in model_metadata.json
- CRUD operations for model versions
- Performance tracking
- Version comparison
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


class ModelMetadataManager:
    """
    Interface for managing centralized model_metadata.json.

    Provides CRUD-like operations for model version tracking.

    Metadata Structure:
    {
        "metadata_version": "1.0",
        "last_updated": "2025-01-15 10:30:00",
        "active_version": "v1_0",
        "latest_version": "v1_2",
        "total_versions": 3,
        "shared_components": {
            "feature_scaler": "features/feature_scaler.pkl",
            "preprocessing_metadata": "features/preprocessing_metadata.json"
        },
        "versions": [
            {
                "version_number": "v1_0",
                "parent_version": null,
                "is_active": false,
                "trained_timestamp": "2025-01-15 10:00:00",
                "training_method": "initial_training",
                "model_info": {
                    "model_name": "NeuralTLearner",
                    "architecture": "18-128-64-1",
                    "n_treatments": 5
                },
                "performance_metrics": {
                    "avg_reward": 2.45,
                    "diversity": 5,
                    "accuracy": 0.75
                },
                "training_info": {
                    "dataset_size": 99115,
                    "training_samples": 79292,
                    "training_time_seconds": 120.5
                },
                "model_file_path": "models/v1_0/production/neural_t_learner.pth",
                "notes": "Base model trained on complete initial dataset"
            }
        ]
    }

    Usage:
        manager = ModelMetadataManager(
            metadata_file_path='models/model_metadata.json',
            models_dir='models'
        )

        # Get active version
        active = manager.get_active_version()

        # Add new version
        manager.add_version(
            version_number='v1_1',
            parent_version='v1_0',
            performance_metrics={'avg_reward': 2.55, ...},
            training_info={'corrections_used': 50, ...}
        )

        # Set active
        manager.set_active_version('v1_1')
    """

    # Default configuration
    DEFAULT_VERSION_NUMBER = "v1_0"

    # Base model performance (from training)
    BASE_MODEL_METRICS = {
        "avg_reward": 2.45,
        "diversity": 5,
        "accuracy": 0.75,
        "success_rate": 0.82
    }

    BASE_MODEL_TRAINING_INFO = {
        "dataset_size": 99115,
        "training_samples": 79292,
        "training_time_seconds": 120.5
    }

    BASE_MODEL_NOTES = """
Base Neural T-Learner model trained on complete initial dataset.
- Architecture: 18 -> 128 -> 64 -> 1 (per treatment)
- 5 separate networks (one per treatment)
- Trained with MSE loss, Adam optimizer
- All subsequent versions use online learning (partial_fit)
"""

    def __init__(self,
                 metadata_file_path: Optional[str] = None,
                 models_dir: str = 'artifacts'):
        """
        Initialize metadata manager.

        Args:
            metadata_file_path: Path to model_metadata.json
                               If None, uses 'artifacts/model_metadata.json'
            models_dir: Root directory for model versions (e.g., 'models', 'artifacts')
        """
        if metadata_file_path is None:
            # Use models_dir to construct path
            self.metadata_file_path = os.path.join(models_dir, 'model_metadata.json')
        else:
            self.metadata_file_path = metadata_file_path

        self.models_dir = models_dir

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.metadata_file_path), exist_ok=True)

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from JSON file."""
        if not os.path.exists(self.metadata_file_path):
            # Return empty metadata structure
            return {
                'metadata_version': '1.0',
                'last_updated': None,
                'active_version': None,
                'latest_version': None,
                'total_versions': 0,
                'shared_components': {
                    'feature_scaler': 'features/feature_scaler.pkl',
                    'preprocessing_metadata': 'features/preprocessing_metadata.json'
                },
                'versions': []
            }

        with open(self.metadata_file_path, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to JSON file."""
        with open(self.metadata_file_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    # =========================================================================
    # INITIALIZATION METHODS
    # =========================================================================

    def initialize_metadata_file(self) -> None:
        """
        Create model_metadata.json file if it doesn't exist.
        Initializes with base model (v1_0) entry.
        """
        if os.path.exists(self.metadata_file_path):
            return

        # Default metadata with base model
        default_metadata = {
            'metadata_version': '1.0',
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'active_version': 'v1_0',
            'latest_version': 'v1_0',
            'total_versions': 1,
            'shared_components': {
                'feature_scaler': 'features/feature_scaler.pkl',
                'preprocessing_metadata': 'features/preprocessing_metadata.json'
            },
            'versions': [
                {
                    'version_number': 'v1_0',
                    'parent_version': None,
                    'is_active': True,
                    'trained_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'training_method': 'initial_training',
                    'model_info': {
                        'model_name': 'NeuralTLearner',
                        'model_type': 'Multi-Treatment Neural Network',
                        'architecture': '18-128-64-1',
                        'n_treatments': 5,
                        'treatment_names': ['Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', 'Insulin']
                    },
                    'performance_metrics': self.BASE_MODEL_METRICS,
                    'training_info': self.BASE_MODEL_TRAINING_INFO,
                    'model_file_path': f'{self.models_dir}/v1_0/production/neural_t_learner.pth',
                    'notes': self.BASE_MODEL_NOTES
                }
            ]
        }

        # Write metadata file
        with open(self.metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(default_metadata, f, indent=2, ensure_ascii=False)

    def verify_model_files(self) -> None:
        """
        Verify all required model files exist.

        Directory structure expected:
        {models_dir}/
          v1_0/
            production/
              neural_t_learner.pth
          features/  (SHARED)
            feature_scaler.pkl
            preprocessing_metadata.json

        Raises:
            FileNotFoundError: If any required file is missing
        """
        metadata = self._load_metadata()

        missing_files = []

        # Check model files for each version
        for version in metadata.get('versions', []):
            model_path = version.get('model_file_path')
            if model_path and not os.path.exists(model_path):
                missing_files.append(model_path)

        # Check shared components
        shared = metadata.get('shared_components', {})
        for component, path in shared.items():
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            error_msg = f"Missing required files: {', '.join(missing_files)}"
            raise FileNotFoundError(error_msg)

    # =========================================================================
    # READ OPERATIONS
    # =========================================================================

    def get_latest_version(self) -> Optional[str]:
        """
        Get the latest version number.

        Returns:
            Latest version number (e.g., "v1_2") or None if no versions
        """
        metadata = self._load_metadata()
        return metadata.get('latest_version')

    def get_active_version(self) -> Optional[str]:
        """
        Get the currently active version number.

        Returns:
            Active version number (e.g., "v1_0") or None if no active version
        """
        metadata = self._load_metadata()
        return metadata.get('active_version')

    def get_version_details(self, version_number: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific version.

        Args:
            version_number: Version to retrieve (e.g., "v1_2")

        Returns:
            Version details dict or None if not found
        """
        metadata = self._load_metadata()

        for version in metadata.get('versions', []):
            if version['version_number'] == version_number:
                return version

        return None

    def get_all_versions(self) -> List[Dict[str, Any]]:
        """
        Get all version records.

        Returns:
            List of version dictionaries
        """
        metadata = self._load_metadata()
        return metadata.get('versions', [])

    def get_total_versions(self) -> int:
        """
        Get total number of versions.

        Returns:
            Count of versions
        """
        metadata = self._load_metadata()
        return metadata.get('total_versions', 0)

    def get_version_history(self) -> List[str]:
        """
        Get chronological list of version numbers.

        Returns:
            List of version numbers in order
        """
        metadata = self._load_metadata()
        return [v['version_number'] for v in metadata.get('versions', [])]

    def get_shared_components(self) -> Dict[str, str]:
        """
        Get shared preprocessing components paths.

        Returns:
            Dict with component paths
        """
        metadata = self._load_metadata()
        return metadata.get('shared_components', {
            'feature_scaler': 'features/feature_scaler.pkl',
            'preprocessing_metadata': 'features/preprocessing_metadata.json'
        })

    def get_model_file_path(self, version_number: str) -> Optional[str]:
        """
        Get the model file path for a specific version.

        Args:
            version_number: Version to retrieve path for

        Returns:
            Model file path or None if version not found
        """
        version_details = self.get_version_details(version_number)
        if version_details:
            return version_details.get('model_file_path')
        return None

    # =========================================================================
    # CREATE OPERATIONS
    # =========================================================================

    def add_version(self,
                    version_number: str,
                    parent_version: str,
                    performance_metrics: Dict[str, float],
                    training_info: Dict[str, Any],
                    is_active: bool = False,
                    notes: str = None) -> Dict[str, Any]:
        """
        Add a new version to metadata.

        Args:
            version_number: New version number (e.g., "v1_3")
            parent_version: Parent version number
            performance_metrics: Dict with avg_reward, diversity, accuracy, etc.
            training_info: Dict with training details
            is_active: Whether this version is active
            notes: Optional notes about this version

        Returns:
            The newly created version dict

        Example:
            manager.add_version(
                version_number='v1_1',
                parent_version='v1_0',
                performance_metrics={'avg_reward': 2.55, 'diversity': 5, 'accuracy': 0.78},
                training_info={'corrections_used': 50, 'training_time_seconds': 2.5},
                notes='Online learning with 50 patient outcomes'
            )
        """
        metadata = self._load_metadata()

        # Check if version already exists
        if any(v['version_number'] == version_number for v in metadata['versions']):
            raise ValueError(f"Version {version_number} already exists")

        # Create new version entry
        new_version = {
            'version_number': version_number,
            'parent_version': parent_version,
            'is_active': is_active,
            'trained_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'training_method': 'partial_fit',
            'model_info': {
                'model_name': 'NeuralTLearner',
                'model_type': 'Multi-Treatment Neural Network',
                'architecture': '18-128-64-1',
                'n_treatments': 5,
                'treatment_names': ['Metformin', 'GLP-1', 'SGLT-2', 'DPP-4', 'Insulin']
            },
            'performance_metrics': {
                'avg_reward': round(performance_metrics.get('avg_reward', 0), 4),
                'diversity': int(performance_metrics.get('diversity', 0)),
                'accuracy': round(performance_metrics.get('accuracy', 0), 4),
                'success_rate': round(performance_metrics.get('success_rate', 0), 4)
            },
            'training_info': training_info,
            'model_file_path': f"{self.models_dir}/{version_number}/production/neural_t_learner.pth",
            'notes': notes or f'Online learning with {training_info.get("corrections_used", 0)} patient outcomes'
        }

        # Add to versions list
        metadata['versions'].append(new_version)

        # Update top-level fields
        metadata['latest_version'] = version_number
        metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata['total_versions'] = len(metadata['versions'])

        # Ensure shared_components exists
        if 'shared_components' not in metadata:
            metadata['shared_components'] = {
                'feature_scaler': 'features/feature_scaler.pkl',
                'preprocessing_metadata': 'features/preprocessing_metadata.json'
            }

        # Save updated metadata
        self._save_metadata(metadata)

        return new_version

    # =========================================================================
    # UPDATE OPERATIONS
    # =========================================================================

    def set_active_version(self, version_number: str) -> bool:
        """
        Set a version as active (and deactivate all others).

        Args:
            version_number: Version to activate

        Returns:
            True if successful, False if version not found
        """
        metadata = self._load_metadata()

        version_found = False

        # Deactivate all versions and activate the target
        for version in metadata['versions']:
            if version['version_number'] == version_number:
                version['is_active'] = True
                version_found = True
            else:
                version['is_active'] = False

        if not version_found:
            return False

        # Update active_version field
        metadata['active_version'] = version_number
        metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self._save_metadata(metadata)
        return True

    def update_version_notes(self, version_number: str, notes: str) -> bool:
        """
        Update notes for a specific version.

        Args:
            version_number: Version to update
            notes: New notes text

        Returns:
            True if successful, False if version not found
        """
        metadata = self._load_metadata()

        for version in metadata['versions']:
            if version['version_number'] == version_number:
                version['notes'] = notes
                metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self._save_metadata(metadata)
                return True

        return False

    # =========================================================================
    # DELETE OPERATIONS
    # =========================================================================

    def delete_version(self, version_number: str) -> bool:
        """
        Delete a version from metadata.
        WARNING: This only removes the metadata entry, not the actual files.

        Args:
            version_number: Version to delete

        Returns:
            True if deleted, False if not found or is active version
        """
        metadata = self._load_metadata()

        # Cannot delete active version
        if metadata.get('active_version') == version_number:
            raise ValueError(f"Cannot delete active version {version_number}")

        # Find and remove version
        original_count = len(metadata['versions'])
        metadata['versions'] = [v for v in metadata['versions'] if v['version_number'] != version_number]

        if len(metadata['versions']) == original_count:
            return False  # Version not found

        # Update metadata
        metadata['total_versions'] = len(metadata['versions'])
        metadata['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Update latest_version if we deleted it
        if metadata.get('latest_version') == version_number:
            if metadata['versions']:
                metadata['latest_version'] = metadata['versions'][-1]['version_number']
            else:
                metadata['latest_version'] = None

        self._save_metadata(metadata)
        return True

    # =========================================================================
    # UTILITY OPERATIONS
    # =========================================================================

    def calculate_next_version_number(self) -> str:
        """
        Calculate the next version number based on latest version.

        Returns:
            Next version number (e.g., "v1_3")
        """
        latest = self.get_latest_version()

        if latest is None:
            return 'v1_0'

        # Parse version: "v1_2" -> major=1, minor=2
        version_parts = latest.replace('v', '').split('_')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        # Increment minor version
        return f"v{major}_{minor + 1}"

    def get_version_lineage(self, version_number: str) -> List[str]:
        """
        Get the lineage of a version (trace back to root).

        Args:
            version_number: Version to trace

        Returns:
            List of version numbers from root to specified version
        """
        metadata = self._load_metadata()
        lineage = []

        current = version_number
        while current:
            lineage.insert(0, current)

            # Find parent
            version_data = next((v for v in metadata['versions'] if v['version_number'] == current), None)
            if not version_data or not version_data.get('parent_version'):
                break

            current = version_data['parent_version']

        return lineage

    def get_performance_comparison(self, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare performance metrics between two versions.

        Args:
            version1: First version
            version2: Second version

        Returns:
            Dict with comparison data
        """
        v1_data = self.get_version_details(version1)
        v2_data = self.get_version_details(version2)

        if not v1_data or not v2_data:
            raise ValueError("One or both versions not found")

        v1_metrics = v1_data['performance_metrics']
        v2_metrics = v2_data['performance_metrics']

        return {
            'version1': version1,
            'version2': version2,
            'avg_reward_diff': v2_metrics.get('avg_reward', 0) - v1_metrics.get('avg_reward', 0),
            'accuracy_diff': v2_metrics.get('accuracy', 0) - v1_metrics.get('accuracy', 0),
            'diversity_diff': v2_metrics.get('diversity', 0) - v1_metrics.get('diversity', 0),
            'v1_metrics': v1_metrics,
            'v2_metrics': v2_metrics
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_metadata_manager(metadata_file_path: Optional[str] = None,
                           models_dir: str = 'artifacts') -> ModelMetadataManager:
    """
    Create ModelMetadataManager instance.

    Args:
        metadata_file_path: Path to model_metadata.json
                           If None, uses 'artifacts/model_metadata.json'
        models_dir: Root directory for model versions (e.g., 'models', 'artifacts')

    Returns:
        ModelMetadataManager instance

    Example:
        manager = create_metadata_manager(
            metadata_file_path='models/model_metadata.json',
            models_dir='models'
        )
        active_version = manager.get_active_version()
    """
    return ModelMetadataManager(
        metadata_file_path=metadata_file_path,
        models_dir=models_dir
    )