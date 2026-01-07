"""
Stateless Model Manager - Central facade for all model operations.

This module provides a unified interface for:
- Loading models (always fresh from disk)
- Version management
- Model registration
- Preprocessing components (cached)
"""

import os
import json
import shutil
import tempfile
from typing import Optional, Dict, List
from datetime import datetime

from ._loader import ModelLoader
from ._architecture import NeuralTLearner
from ._metadata_manager import ModelMetadataManager
from ._model_registry import ModelRegistry
from ..preprocessing import PatientFeatureProcessor, create_feature_processor

class ModelManagerError(Exception):
    """Base exception for ModelManager errors."""
    pass


class ModelNotFoundError(ModelManagerError):
    """Raised when requested model version doesn't exist."""
    pass


class ModelManager:
    """
    Stateless facade for all model operations.

    Key Features:
    - No model caching - always loads fresh from disk
    - Automatic directory/metadata initialization
    - Centralized version management
    - Thread-safe operations
    - Clear error messages

    Usage:
        # Initialize
        manager = ModelManager(
            models_dir='artifacts',
            scaler_path='features/feature_scaler.pkl',
            metadata_path='features/preprocessing_metadata.json',
            auto_activate=False,
            verbose=False
        )

        # Load models (fresh from disk every time)
        model = manager.get_active_model()
        model = manager.get_latest_model()
        model = manager.get_model_by_version('v1_2')

        # Manage versions
        manager.activate_version('v1_3')
        versions = manager.list_versions(sort_by='performance')
        comparison = manager.compare_versions('v1_0', 'v1_2')

        # Register new models (from training pipeline)
        new_version = manager.register_new_version(
            model=trained_model,
            performance_metrics={'avg_reward': 2.58, 'accuracy': 0.78},
            training_info={'outcomes_processed': 50}
        )

        # Get preprocessing (cached)
        processor = manager.get_feature_processor()
    """

    def __init__(self,
                 models_dir: str = 'artifacts',
                 scaler_path: str = 'features/feature_scaler.pkl',
                 metadata_path: str = 'features/preprocessing_metadata.json',
                 auto_activate: bool = False,
                 verbose: bool = False):
        """
        Initialize ModelManager.

        This performs full initialization:
        - Creates missing directories
        - Initializes or validates model_metadata.json
        - Validates preprocessing files exist
        - Sets up internal components

        Args:
            models_dir: Root directory for model versions (e.g., 'artifacts')
            scaler_path: Path to feature scaler pickle file
            metadata_path: Path to preprocessing metadata JSON
            auto_activate: If True, newly registered models become active automatically
            verbose: If True, print detailed logs

        Raises:
            FileNotFoundError: If scaler or metadata files don't exist
            ModelManagerError: If initialization fails
        """
        # Store configuration
        self.models_dir = models_dir
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.auto_activate = auto_activate
        self.verbose = verbose

        if self.verbose:
            print("\n" + "=" * 80)
            print("INITIALIZING MODEL MANAGER")
            print("=" * 80)

        # Step 1: Create directories
        self._create_directories()

        # Step 2: Initialize or validate metadata file
        self.metadata_file = os.path.join(self.models_dir, 'model_metadata.json')
        self._initialize_metadata_file()

        # Step 3: Validate preprocessing files
        self._validate_preprocessing_files()

        # Step 4: Initialize internal components
        self._initialize_components()

        # Step 5: Initialize processor cache (empty)
        self._processor: Optional[PatientFeatureProcessor] = None

        # Step 6: Log summary
        if self.verbose:
            self._log_initialization_summary()
            print("=" * 80 + "\n")
        else:
            print(f"[ModelManager] Initialized - {self.get_version_count()} versions available\n")

    # =========================================================================
    # INITIALIZATION HELPERS
    # =========================================================================

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.models_dir,
            os.path.join(self.models_dir, 'storage'),
            os.path.dirname(self.scaler_path),
            os.path.dirname(self.metadata_path)
        ]

        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                if self.verbose:
                    print(f"[ModelManager] Created directory: {directory}")

    def _initialize_metadata_file(self):
        """Initialize or validate model_metadata.json."""
        if not os.path.exists(self.metadata_file):
            if self.verbose:
                print(f"[ModelManager] Creating metadata file: {self.metadata_file}")

            # Check if base model v1_0 exists
            base_model_path = os.path.join(
                self.models_dir,
                'v1_0',
                'production',
                'neural_t_learner.pth'
            )

            # Base model MUST exist
            if not os.path.exists(base_model_path):
                raise FileNotFoundError(
                    f"Base model not found at expected location: {base_model_path}\n"
                    f"Please ensure the pre-trained model exists at:\n"
                    f"  {base_model_path}\n\n"
                    f"Expected structure:\n"
                    f"  models/\n"
                    f"    └── v1_0/\n"
                    f"        └── production/\n"
                    f"            └── neural_t_learner.pth"
                )

            # Create metadata WITH v1_0 registered
            initial_metadata = {
                'metadata_version': '1.0',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'active_version': 'v1_0',
                'latest_version': 'v1_0',
                'total_versions': 1,
                'shared_components': {
                    'feature_scaler': self.scaler_path,
                    'preprocessing_metadata': self.metadata_path
                },
                'versions': [
                    {
                        'version_number': 'v1_0',
                        'model_file_path': base_model_path,
                        'parent_version': None,
                        'trained_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'training_method': 'supervised',
                        'training_info': {
                            'dataset_size': 99115,
                            'training_samples': 79292,
                            'validation_samples': 19823,
                            'epochs': 100,
                            'batch_size': 512,
                            'note': 'Initial pre-trained base model'
                        },
                        'performance_metrics': {
                            'avg_reward': 2.45,
                            'accuracy': 0.82,
                            'rmse': 0.45,
                            'mae': 0.32,
                            'r2': 0.82,
                            'diversity': 5,
                            'success_rate': 0.82
                        },
                        'is_active': True
                    }
                ]
            }

            if self.verbose:
                print(f"[ModelManager] Base model v1_0 found at: {base_model_path}")
                print(f"[ModelManager] Registered v1_0 as active version")

            self._write_metadata_atomic(initial_metadata)
        else:
            # Validate existing metadata
            if self.verbose:
                print(f"[ModelManager] Validating metadata file: {self.metadata_file}")

            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Check required fields
                required_fields = ['metadata_version', 'versions', 'shared_components']
                for field in required_fields:
                    if field not in metadata:
                        raise ValueError(f"Missing required field: {field}")

                if self.verbose:
                    print(f"[ModelManager] Metadata valid - {len(metadata.get('versions', []))} versions found")

            except (json.JSONDecodeError, ValueError) as e:
                # Backup corrupted file
                backup_path = f"{self.metadata_file}.corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(self.metadata_file, backup_path)

                print(f"[ModelManager] WARNING: Corrupted metadata detected")
                print(f"[ModelManager] Backup saved to: {backup_path}")
                print(f"[ModelManager] Creating fresh metadata file")

                # Recursively call to create fresh file
                os.remove(self.metadata_file)
                self._initialize_metadata_file()

    def _validate_preprocessing_files(self):
        """Validate that preprocessing files exist."""
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Feature scaler not found: {self.scaler_path}\n"
                f"This file is required for model initialization.\n"
                f"Please ensure it exists before creating ModelManager."
            )

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Preprocessing metadata not found: {self.metadata_path}\n"
                f"This file is required for model initialization.\n"
                f"Please ensure it exists before creating ModelManager."
            )

        if self.verbose:
            print(f"[ModelManager] Preprocessing files validated")
            print(f"[ModelManager]   Scaler: {self.scaler_path}")
            print(f"[ModelManager]   Metadata: {self.metadata_path}")

    def _initialize_components(self):
        """Initialize internal components."""
        self._metadata_manager = ModelMetadataManager(
            metadata_file_path=self.metadata_file,
            models_dir=self.models_dir
        )

        self._loader = ModelLoader(verbose=False)

        self._registry = ModelRegistry(
            models_dir=self.models_dir,
            verbose=False
        )

        if self.verbose:
            print(f"[ModelManager] Internal components initialized")

    def _log_initialization_summary(self):
        """Log initialization summary."""
        active = self.get_active_version()
        latest = self.get_latest_version()
        total = self.get_version_count()

        print(f"\n[ModelManager] Initialization Summary:")
        print(f"  Models directory: {self.models_dir}")
        print(f"  Metadata file: {self.metadata_file}")
        print(f"  Total versions: {total}")
        print(f"  Active version: {active if active else 'None (will use latest)'}")
        print(f"  Latest version: {latest if latest else 'None'}")
        print(f"  Auto-activate: {self.auto_activate}")

        if total == 0:
            print(f"\n[ModelManager] WARNING: No models found")
            print(f"[ModelManager] Register a model using training pipeline to begin")

    # =========================================================================
    # PUBLIC API: MODEL ACCESS (ALWAYS FRESH FROM DISK)
    # =========================================================================

    def get_active_model(self) -> NeuralTLearner:
        """
        Get the currently active model (fresh from disk).

        Returns:
            Loaded NeuralTLearner instance

        Raises:
            ModelNotFoundError: If active model file doesn't exist

        Note:
            - If no active version set, auto-falls back to latest (with warning)
            - Every call loads fresh from disk (no caching)

        Example:
            model = manager.get_active_model()
            result = model.predict_q_values(features)
        """
        if self.verbose:
            print("[ModelManager] Loading active model...")

        # Get active version from metadata
        active_version = self._metadata_manager.get_active_version()

        # Fallback to latest if no active version
        if active_version is None:
            latest_version = self._metadata_manager.get_latest_version()

            if latest_version is None:
                raise ModelNotFoundError(
                    "No models available.\n"
                    "Register a model using training pipeline first."
                )

            print(f"[ModelManager] WARNING: No active version set")
            print(f"[ModelManager] Auto-using latest: {latest_version}")
            active_version = latest_version

        # Load model from disk
        return self._load_model_by_version(active_version)

    def get_latest_model(self) -> NeuralTLearner:
        """
        Get the most recently created model (fresh from disk).

        Returns:
            Loaded NeuralTLearner instance

        Raises:
            ModelNotFoundError: If no models exist

        Note:
            - Independent of active status
            - Useful for "always use newest" pattern

        Example:
            model = manager.get_latest_model()
        """
        if self.verbose:
            print("[ModelManager] Loading latest model...")

        latest_version = self._metadata_manager.get_latest_version()

        if latest_version is None:
            raise ModelNotFoundError(
                "No models available.\n"
                "Register a model using training pipeline first."
            )

        return self._load_model_by_version(latest_version)

    def get_model_by_version(self, version: str) -> NeuralTLearner:
        """
        Load specific model version (fresh from disk).

        Args:
            version: Version number (e.g., 'v1_2')

        Returns:
            Loaded NeuralTLearner instance

        Raises:
            ModelNotFoundError: If version doesn't exist

        Example:
            # Compare two versions
            model_old = manager.get_model_by_version('v1_0')
            model_new = manager.get_model_by_version('v1_2')
        """
        if self.verbose:
            print(f"[ModelManager] Loading model version: {version}")

        return self._load_model_by_version(version)

    def _load_model_by_version(self, version: str) -> NeuralTLearner:
        """
        Internal method to load model from disk.

        Args:
            version: Version number

        Returns:
            Loaded model

        Raises:
            ModelNotFoundError: If model file doesn't exist
        """
        # Get model path from metadata
        model_path = self._metadata_manager.get_model_file_path(version)

        if model_path is None:
            available = [v['version_number'] for v in self._metadata_manager.get_all_versions()]
            raise ModelNotFoundError(
                f"Model version '{version}' not found in metadata.\n"
                f"Available versions: {', '.join(available)}\n"
                f"Use manager.list_versions() to see all versions."
            )

        # Check file exists
        if not os.path.exists(model_path):
            raise ModelNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Version '{version}' exists in metadata but file is missing.\n"
                f"Use manager.validate_integrity() to check all versions."
            )

        # Load model using loader
        try:
            model = self._loader.load_model(
                model_path=model_path,
                n_features=21,
                n_treatments=5,
                hidden_dims=[256, 128, 64],
                learning_rate=0.001,
                weight_decay=1e-4,
                device='cpu'
            )

            if self.verbose:
                print(f"[ModelManager] Model loaded: {version} from {model_path}")

            return model

        except Exception as e:
            raise ModelManagerError(
                f"Failed to load model {version} from {model_path}\n"
                f"Error: {str(e)}"
            )

    # =========================================================================
    # PUBLIC API: VERSION MANAGEMENT
    # =========================================================================

    def get_active_version(self) -> Optional[str]:
        """
        Get the active version number (no model loading).

        Returns:
            Active version string (e.g., 'v1_2') or None

        Example:
            active = manager.get_active_version()
            print(f"Active: {active}")
        """
        return self._metadata_manager.get_active_version()

    def get_latest_version(self) -> Optional[str]:
        """
        Get the latest version number (no model loading).

        Returns:
            Latest version string or None

        Example:
            latest = manager.get_latest_version()
        """
        return self._metadata_manager.get_latest_version()

    def get_version_count(self) -> int:
        """
        Get total number of versions.

        Returns:
            Count of versions

        Example:
            count = manager.get_version_count()
        """
        return self._metadata_manager.get_total_versions()

    def list_versions(self,
                      sort_by: str = 'version',
                      reverse: bool = False) -> List[Dict]:
        """
        List all model versions (no model loading).

        Args:
            sort_by: Sort key ('version', 'date', 'avg_reward', 'accuracy')
            reverse: If True, reverse sort order

        Returns:
            List of version info dictionaries

        Example:
            # Get all versions sorted by performance
            versions = manager.list_versions(sort_by='avg_reward', reverse=True)

            for v in versions:
                print(f"{v['version_number']}: {v['performance_metrics']}")
        """
        return self._registry.list_versions(sort_by=sort_by, reverse=reverse)

    def get_model_info(self, version: str) -> Optional[Dict]:
        """
        Get detailed info for a version (no model loading).

        Args:
            version: Version number

        Returns:
            Version info dict or None if not found

        Example:
            info = manager.get_model_info('v1_2')
            print(f"Performance: {info['performance_metrics']}")
            print(f"Trained: {info['trained_timestamp']}")
        """
        return self._metadata_manager.get_version_details(version)

    def activate_version(self, version: str) -> bool:
        """
        Set a version as active.

        Args:
            version: Version to activate

        Returns:
            True if successful

        Raises:
            ModelNotFoundError: If version doesn't exist

        Note:
            - Does not load model, just updates metadata
            - Next get_active_model() call will load newly activated version

        Example:
            manager.activate_version('v1_3')
            model = manager.get_active_model()  # Loads v1_3
        """
        # Validate version exists
        if not self._metadata_manager.get_version_details(version):
            available = [v['version_number'] for v in self._metadata_manager.get_all_versions()]
            raise ModelNotFoundError(
                f"Cannot activate version '{version}' - not found.\n"
                f"Available versions: {', '.join(available)}"
            )

        # Update metadata
        success = self._metadata_manager.set_active_version(version)

        if success and self.verbose:
            print(f"[ModelManager] Activated version: {version}")

        return success

    def compare_versions(self, v1: str, v2: str) -> Dict:
        """
        Compare performance between two versions.

        Args:
            v1: First version
            v2: Second version

        Returns:
            Comparison dict with metric differences

        Example:
            comparison = manager.compare_versions('v1_0', 'v1_2')
            print(f"Improvement: {comparison['avg_reward_diff']:.3f}")
        """
        return self._metadata_manager.get_performance_comparison(v1, v2)

    def get_version_lineage(self, version: str) -> List[str]:
        """
        Get version ancestry (trace back to root).

        Args:
            version: Version to trace

        Returns:
            List of version numbers from root to specified version

        Example:
            lineage = manager.get_version_lineage('v1_3')
            # Returns: ['v1_0', 'v1_1', 'v1_2', 'v1_3']
        """
        return self._metadata_manager.get_version_lineage(version)

    # =========================================================================
    # PUBLIC API: MODEL REGISTRATION
    # =========================================================================

    def register_new_version(self,
                             model: NeuralTLearner,
                             performance_metrics: Dict,
                             training_info: Dict,
                             parent_version: Optional[str] = None,
                             notes: Optional[str] = None) -> str:
        """
        Register a new model version (used by training pipeline).

        This method:
        1. Calculates next version number
        2. Creates version directory
        3. Saves model to disk
        4. Updates metadata
        5. Optionally activates (if auto_activate=True)

        Args:
            model: Trained NeuralTLearner instance
            performance_metrics: Dict with avg_reward, accuracy, diversity, etc.
            training_info: Dict with outcomes_processed, training_time_seconds, etc.
            parent_version: Parent version (if None, uses latest)
            notes: Optional notes about this version

        Returns:
            New version number (e.g., 'v1_3')

        Example:
            new_version = manager.register_new_version(
                model=trained_model,
                performance_metrics={'avg_reward': 2.58, 'accuracy': 0.78},
                training_info={'outcomes_processed': 50, 'training_time_seconds': 12.5},
                parent_version='v1_2'
            )
            # Returns: 'v1_3'
        """
        if self.verbose:
            print("\n[ModelManager] Registering new model version...")

        # Step 1: Calculate next version number
        next_version = self._metadata_manager.calculate_next_version_number()

        if self.verbose:
            print(f"[ModelManager] New version: {next_version}")

        # Step 2: Create version directory
        version_dir = os.path.join(self.models_dir, next_version)
        production_dir = os.path.join(version_dir, 'production')
        os.makedirs(production_dir, exist_ok=True)

        if self.verbose:
            print(f"[ModelManager] Created directory: {production_dir}")

        # Step 3: Save model
        model_path = os.path.join(production_dir, 'neural_t_learner.pth')

        try:
            model.save_models(model_path)

            if self.verbose:
                print(f"[ModelManager] Model saved: {model_path}")

        except Exception as e:
            # Cleanup on failure
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
            raise ModelManagerError(f"Failed to save model: {str(e)}")

        # Step 4: Determine parent version
        if parent_version is None:
            parent_version = self._metadata_manager.get_latest_version()

        # Step 5: Update metadata
        try:
            self._metadata_manager.add_version(
                version_number=next_version,
                parent_version=parent_version,
                performance_metrics=performance_metrics,
                training_info=training_info,
                is_active=False,  # Will activate separately if needed
                notes=notes
            )

            if self.verbose:
                print(f"[ModelManager] Metadata updated")

        except Exception as e:
            # Cleanup on failure
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)
            raise ModelManagerError(f"Failed to update metadata: {str(e)}")

        # Step 6: Auto-activate if configured
        if self.auto_activate:
            self.activate_version(next_version)

            if self.verbose:
                print(f"[ModelManager] Auto-activated: {next_version}")

        if self.verbose:
            print(f"[ModelManager] Registration complete: {next_version}\n")
        else:
            print(f"[ModelManager] Registered new version: {next_version}\n")

        return next_version

    def delete_version(self,
                       version: str,
                       delete_files: bool = True) -> bool:
        """
        Delete a model version.

        Args:
            version: Version to delete
            delete_files: If True, also delete model files from disk

        Returns:
            True if successful

        Raises:
            ModelManagerError: If trying to delete active version

        Note:
            Cannot delete active version. Activate different one first.

        Example:
            # Delete metadata and files
            manager.delete_version('v1_1', delete_files=True)
        """
        # Check if active
        if self.get_active_version() == version:
            raise ModelManagerError(
                f"Cannot delete active version '{version}'.\n"
                f"Activate a different version first using activate_version()."
            )

        # Delete from metadata
        success = self._metadata_manager.delete_version(version)

        if not success:
            if self.verbose:
                print(f"[ModelManager] Version {version} not found")
            return False

        # Delete files if requested
        if delete_files:
            version_dir = os.path.join(self.models_dir, version)
            if os.path.exists(version_dir):
                shutil.rmtree(version_dir)

                if self.verbose:
                    print(f"[ModelManager] Deleted files: {version_dir}")

        if self.verbose:
            print(f"[ModelManager] Deleted version: {version}")

        return True

    # =========================================================================
    # PUBLIC API: PREPROCESSING (CACHED)
    # =========================================================================

    def get_feature_processor(self) -> PatientFeatureProcessor:
        """
        Get feature processor (cached after first access).

        Returns:
            PatientFeatureProcessor instance

        Note:
            - Processor is cached (not reloaded every time)
            - Use reload_preprocessing_components() to refresh

        Example:
            processor = manager.get_feature_processor()
            features = processor.process_patient(patient_dict)
        """
        if self._processor is None:
            if self.verbose:
                print("[ModelManager] Loading feature processor (first access)...")

            self._processor = create_feature_processor(
                scaler_path=self.scaler_path,
                metadata_path=self.metadata_path,
                verbose=False
            )

            if self.verbose:
                print("[ModelManager] Processor cached for future use")

        return self._processor

    def reload_preprocessing_components(self):
        """
        Force reload preprocessing components from disk.

        Use this if scaler/metadata files are updated externally.

        Example:
            # After updating scaler file
            manager.reload_preprocessing_components()
            processor = manager.get_feature_processor()  # Fresh processor
        """
        if self.verbose:
            print("[ModelManager] Reloading preprocessing components...")

        self._processor = None
        self._processor = self.get_feature_processor()

        if self.verbose:
            print("[ModelManager] Preprocessing components reloaded")

    # =========================================================================
    # PUBLIC API: DIAGNOSTICS & VALIDATION
    # =========================================================================

    def get_status(self) -> Dict:
        """
        Get current ModelManager status.

        Returns:
            Status dict with:
            - active_version
            - latest_version
            - total_versions
            - available_versions (list)
            - missing_versions (list)
            - disk_space_used (MB)

        Example:
            status = manager.get_status()
            print(f"Active: {status['active_version']}")
            print(f"Total: {status['total_versions']}")
        """
        # Read metadata
        versions = self._metadata_manager.get_all_versions()

        # Check which versions have files
        available = []
        missing = []
        total_size = 0

        for v in versions:
            version_num = v['version_number']
            model_path = v.get('model_file_path')

            if model_path and os.path.exists(model_path):
                available.append(version_num)
                # Get file size
                total_size += os.path.getsize(model_path)
            else:
                missing.append(version_num)

        return {
            'active_version': self.get_active_version(),
            'latest_version': self.get_latest_version(),
            'total_versions': len(versions),
            'available_versions': available,
            'missing_versions': missing,
            'disk_space_used_mb': round(total_size / (1024 * 1024), 2)
        }

    def validate_integrity(self) -> List[str]:
        """
        Validate integrity of all versions.

        Returns:
            List of issues found (empty if all good)

        Example:
            issues = manager.validate_integrity()
            if issues:
                for issue in issues:
                    print(f"Issue: {issue}")
        """
        issues = []

        versions = self._metadata_manager.get_all_versions()

        for v in versions:
            version_num = v['version_number']
            model_path = v.get('model_file_path')

            # Check model file exists
            if not model_path:
                issues.append(f"Version {version_num}: No model_file_path in metadata")
                continue

            if not os.path.exists(model_path):
                issues.append(f"Version {version_num}: Model file missing at {model_path}")
                continue

            # Check directory structure
            version_dir = os.path.join(self.models_dir, version_num)
            if not os.path.exists(version_dir):
                issues.append(f"Version {version_num}: Directory missing at {version_dir}")

        # Check for orphaned directories
        if os.path.exists(self.models_dir):
            for item in os.listdir(self.models_dir):
                item_path = os.path.join(self.models_dir, item)
                if os.path.isdir(item_path) and item.startswith('v'):
                    # Check if in metadata
                    version_nums = [v['version_number'] for v in versions]
                    if item not in version_nums and item != 'storage':
                        issues.append(f"Orphaned directory (not in metadata): {item_path}")

        return issues

    def print_status(self):
        """
        Print formatted status report.

        Example:
            manager.print_status()
        """
        status = self.get_status()

        print("\n" + "=" * 80)
        print("MODEL MANAGER STATUS")
        print("=" * 80)
        print(f"\nActive Version: {status['active_version'] or 'None'}")
        print(f"Latest Version: {status['latest_version'] or 'None'}")
        print(f"Total Versions: {status['total_versions']}")
        print(f"Disk Space Used: {status['disk_space_used_mb']} MB")

        if status['missing_versions']:
            print(f"\n⚠️  Missing Files ({len(status['missing_versions'])}):")
            for v in status['missing_versions']:
                print(f"  - {v}")

        print("\nAvailable Versions:")
        for v in status['available_versions']:
            marker = " (ACTIVE)" if v == status['active_version'] else ""
            print(f"  - {v}{marker}")

        print("=" * 80 + "\n")

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _write_metadata_atomic(self, metadata: Dict):
        """
        Write metadata file atomically (thread-safe).

        Args:
            metadata: Metadata dict to write
        """
        # Write to temporary file
        temp_fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.metadata_file),
            suffix='.tmp'
        )

        try:
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Atomic rename (overwrites existing file)
            if os.name == 'nt':  # Windows
                if os.path.exists(self.metadata_file):
                    os.replace(temp_path, self.metadata_file)
                else:
                    os.rename(temp_path, self.metadata_file)
            else:  # Unix/Linux/Mac
                os.rename(temp_path, self.metadata_file)

        except Exception as e:
            # Cleanup temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise ModelManagerError(f"Failed to write metadata: {str(e)}")

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_model_manager(models_dir: str = 'artifacts',
                         scaler_path: str = 'features/feature_scaler.pkl',
                         metadata_path: str = 'features/preprocessing_metadata.json',
                         auto_activate: bool = False,
                         verbose: bool = False) -> ModelManager:
    """
    Factory function to create ModelManager instance.

    Args:
        models_dir: Root directory for model versions
        scaler_path: Path to feature scaler
        metadata_path: Path to preprocessing metadata
        auto_activate: Auto-activate newly registered models
        verbose: Enable detailed logging

    Returns:
        Configured ModelManager instance

    Example:
        manager = create_model_manager(
            models_dir='artifacts',
            scaler_path='features/feature_scaler.pkl',
            metadata_path='features/preprocessing_metadata.json'
        )

        model = manager.get_active_model()
    """
    return ModelManager(
        models_dir=models_dir,
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        auto_activate=auto_activate,
        verbose=verbose
    )

# Singleton instance for convenience
_global_manager: Optional[ModelManager] = None

def get_model_manager(models_dir: str = 'artifacts',
                      scaler_path: str = 'features/feature_scaler.pkl',
                      metadata_path: str = 'features/preprocessing_metadata.json',
                      auto_activate: bool = False,
                      verbose: bool = False) -> ModelManager:
    """
    Get or create global ModelManager instance (singleton pattern).

    This provides a convenient shared instance for most use cases.
    First call creates the manager, subsequent calls return same instance.

    Args:
        models_dir: Root directory for model versions (only used on first call)
        scaler_path: Path to feature scaler (only used on first call)
        metadata_path: Path to preprocessing metadata (only used on first call)
        auto_activate: Auto-activate newly registered models (only used on first call)
        verbose: Enable detailed logging (only used on first call)

    Returns:
        Shared ModelManager instance

    Note:
        - Parameters only used on first call (when creating instance)
        - To use different paths, create separate ModelManager instances

    Example:
        # First call - creates instance
        manager = get_model_manager()

        # Subsequent calls - returns same instance
        manager2 = get_model_manager()  # Same as manager

        # Both share same registry
        model = manager.get_active_model()
        model2 = manager2.get_active_model()  # Same model
    """
    global _global_manager

    if _global_manager is None:
        _global_manager = ModelManager(
            models_dir=models_dir,
            scaler_path=scaler_path,
            metadata_path=metadata_path,
            auto_activate=auto_activate,
            verbose=verbose
        )

    return _global_manager

def reset_global_manager():
    """
    Reset the global manager singleton.

    Useful for testing or when you need to reinitialize with different config.

    Example:
        # Use default config
        manager1 = get_model_manager()

        # Reset and use different config
        reset_global_manager()
        manager2 = get_model_manager(models_dir='artifacts_dev')
    """
    global _global_manager
    _global_manager = None

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_active_model(models_dir: str = 'artifacts',
                      scaler_path: str = 'features/feature_scaler.pkl',
                      metadata_path: str = 'features/preprocessing_metadata.json') -> NeuralTLearner:
    """
    Quick helper to load active model without creating manager.

    Args:
        models_dir: Models directory
        scaler_path: Scaler path
        metadata_path: Metadata path

    Returns:
        Active model instance

    Example:
        model = load_active_model()
        q_values = model.predict_q_values(features)
    """
    manager = get_model_manager(
        models_dir=models_dir,
        scaler_path=scaler_path,
        metadata_path=metadata_path
    )
    return manager.get_active_model()

def load_latest_model(models_dir: str = 'artifacts',
                      scaler_path: str = 'features/feature_scaler.pkl',
                      metadata_path: str = 'features/preprocessing_metadata.json') -> NeuralTLearner:
    """
    Quick helper to load latest model without creating manager.

    Args:
        models_dir: Models directory
        scaler_path: Scaler path
        metadata_path: Metadata path

    Returns:
        Latest model instance

    Example:
        model = load_latest_model()
    """
    manager = get_model_manager(
        models_dir=models_dir,
        scaler_path=scaler_path,
        metadata_path=metadata_path
    )
    return manager.get_latest_model()

def load_model_version(version: str,
                       models_dir: str = 'artifacts',
                       scaler_path: str = 'features/feature_scaler.pkl',
                       metadata_path: str = 'features/preprocessing_metadata.json') -> NeuralTLearner:
    """
    Quick helper to load specific model version.

    Args:
        version: Version number (e.g., 'v1_2')
        models_dir: Models directory
        scaler_path: Scaler path
        metadata_path: Metadata path

    Returns:
        Model instance for specified version

    Example:
        model = load_model_version('v1_2')
    """
    manager = get_model_manager(
        models_dir=models_dir,
        scaler_path=scaler_path,
        metadata_path=metadata_path
    )
    return manager.get_model_by_version(version)

def get_feature_processor_instance(scaler_path: str = 'features/feature_scaler.pkl',
                                   metadata_path: str = 'features/preprocessing_metadata.json') -> PatientFeatureProcessor:
    """
    Quick helper to get feature processor.

    Args:
        scaler_path: Scaler path
        metadata_path: Metadata path

    Returns:
        Feature processor instance (cached)

    Example:
        processor = get_feature_processor_instance()
        features = processor.process_patient(patient_dict)
    """
    manager = get_model_manager(
        scaler_path=scaler_path,
        metadata_path=metadata_path
    )
    return manager.get_feature_processor()

def list_available_versions(models_dir: str = 'artifacts',
                            sort_by: str = 'version') -> List[Dict]:
    """
    Quick helper to list all versions.

    Args:
        models_dir: Models directory
        sort_by: Sort key ('version', 'date', 'avg_reward', 'accuracy')

    Returns:
        List of version info dicts

    Example:
        versions = list_available_versions(sort_by='avg_reward')
        for v in versions:
            print(f"{v['version_number']}: {v['performance_metrics']['avg_reward']}")
    """
    manager = get_model_manager(models_dir=models_dir)
    return manager.list_versions(sort_by=sort_by)

def get_manager_status(models_dir: str = 'artifacts') -> Dict:
    """
    Quick helper to get manager status.

    Args:
        models_dir: Models directory

    Returns:
        Status dict

    Example:
        status = get_manager_status()
        print(f"Active: {status['active_version']}")
        print(f"Total: {status['total_versions']}")
    """
    manager = get_model_manager(models_dir=models_dir)
    return manager.get_status()

def validate_model_integrity(models_dir: str = 'artifacts') -> List[str]:
    """
    Quick helper to validate all models.

    Args:
        models_dir: Models directory

    Returns:
        List of issues (empty if all good)

    Example:
        issues = validate_model_integrity()
        if issues:
            print("Issues found:")
            for issue in issues:
                print(f"  - {issue}")
    """
    manager = get_model_manager(models_dir=models_dir)
    return manager.validate_integrity()