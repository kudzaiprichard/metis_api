"""
Model loading utilities for diabetes treatment recommendation.

This module provides:
- Load PyTorch models (.pth files)
- Load preprocessing components (scaler, metadata)
- Validate model files
- Safe file I/O with error handling
"""

import os
import pickle
import json
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict

from ._architecture import NeuralTLearner, create_neural_t_learner


class ModelLoader:
    """
    Utility class for loading models and preprocessing components.

    This class provides safe, validated loading of:
    - Trained PyTorch models (.pth files)
    - Feature scalers (pickle files)
    - Metadata (JSON files)

    All methods include proper error handling and file validation.

    Usage:
        loader = ModelLoader(verbose=True)

        # Load model
        model = loader.load_model(
            'artifacts/v1_0/production/neural_t_learner.pth'
        )

        # Load preprocessing components
        scaler, metadata = loader.load_preprocessing_components(
            scaler_path='features/feature_scaler.pkl',
            metadata_path='features/preprocessing_metadata.json'
        )

        # Load everything
        model, scaler, metadata = loader.load_all_components(
            model_path='artifacts/v1_0/production/neural_t_learner.pth',
            scaler_path='features/feature_scaler.pkl',
            metadata_path='features/preprocessing_metadata.json'
        )
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize ModelLoader.

        Args:
            verbose: If True, print detailed loading logs
        """
        self.verbose = verbose

        if self.verbose:
            print("[ModelLoader] Initialized")

    def validate_file_path(self,
                           file_path: str,
                           file_description: str = "File") -> bool:
        """
        Validate that a file path exists and is accessible.

        Args:
            file_path: Path to file
            file_description: Description for error messages

        Returns:
            True if file is valid

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file isn't readable
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"{file_description} not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"File not readable: {file_path}")

        if self.verbose:
            size_kb = path.stat().st_size / 1024
            print(f"[ModelLoader] {file_description} validated: {file_path} ({size_kb:.1f} KB)")

        return True

    def load_model(self,
                   model_path: str,
                   n_features: int = 21,
                   n_treatments: int = 5,
                   hidden_dims: list = [256, 128, 64],
                   learning_rate: float = 0.001,
                   weight_decay: float = 1e-4,
                   device: Optional[str] = None) -> NeuralTLearner:
        """
        Load a trained Neural T-Learner model.

        Args:
            model_path: Path to .pth file
            n_features: Number of input features (21 with engineered features)
            n_treatments: Number of treatments
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            weight_decay: L2 regularization coefficient
            device: 'cpu' or 'cuda' (auto-detect if None)

        Returns:
            Loaded NeuralTLearner instance

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If loading fails

        Example:
            loader = ModelLoader()
            model = loader.load_model('artifacts/v1_0/production/neural_t_learner.pth')
        """
        # Validate file
        self.validate_file_path(model_path, "Model file")

        if self.verbose:
            print(f"[ModelLoader] Loading model from: {model_path}")

        try:
            # Create model architecture
            model = create_neural_t_learner(
                n_features=n_features,
                n_treatments=n_treatments,
                hidden_dims=hidden_dims,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device
            )

            # Load weights
            model.load_models(model_path)

            if self.verbose:
                print(f"[ModelLoader] Model loaded successfully")
                print(f"[ModelLoader] Device: {model.device}")
                print(f"[ModelLoader] Features: {model.n_features}")
                print(f"[ModelLoader] Treatments: {model.n_treatments}")
                print(f"[ModelLoader] Architecture: {n_features} -> {' -> '.join(map(str, hidden_dims))} -> 1")

            return model

        except Exception as e:
            raise Exception(f"Failed to load model from {model_path}: {str(e)}")

    def load_scaler(self, scaler_path: str):
        """
        Load feature scaler from pickle file.

        Args:
            scaler_path: Path to scaler pickle file

        Returns:
            Loaded StandardScaler object

        Raises:
            FileNotFoundError: If scaler file doesn't exist
            Exception: If loading fails
        """
        self.validate_file_path(scaler_path, "Feature scaler")

        if self.verbose:
            print(f"[ModelLoader] Loading scaler from: {scaler_path}")

        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            if self.verbose:
                print(f"[ModelLoader] Scaler loaded successfully")
                if hasattr(scaler, 'n_features_in_'):
                    print(f"[ModelLoader] Scaler expects {scaler.n_features_in_} features")

            return scaler

        except Exception as e:
            raise Exception(f"Failed to load scaler from {scaler_path}: {str(e)}")

    def load_metadata(self, metadata_path: str) -> Dict:
        """
        Load preprocessing metadata from JSON file.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            Metadata dictionary

        Raises:
            FileNotFoundError: If metadata file doesn't exist
            Exception: If loading fails
        """
        self.validate_file_path(metadata_path, "Preprocessing metadata")

        if self.verbose:
            print(f"[ModelLoader] Loading metadata from: {metadata_path}")

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            if self.verbose:
                print(f"[ModelLoader] Metadata loaded successfully")

                # Validate expected structure
                expected_keys = ['feature_cols', 'continuous_features', 'binary_features']
                for key in expected_keys:
                    if key not in metadata:
                        print(f"[ModelLoader] WARNING: Missing key '{key}' in metadata")

                if 'feature_cols' in metadata:
                    print(f"[ModelLoader] Features: {len(metadata['feature_cols'])}")

                    # Verify feature count
                    if len(metadata['feature_cols']) != 21:
                        print(f"[ModelLoader] WARNING: Expected 21 features, found {len(metadata['feature_cols'])}")

                if 'continuous_features' in metadata:
                    print(f"[ModelLoader] Continuous features: {len(metadata['continuous_features'])}")

                if 'binary_features' in metadata:
                    print(f"[ModelLoader] Binary features: {len(metadata['binary_features'])}")

            return metadata

        except Exception as e:
            raise Exception(f"Failed to load metadata from {metadata_path}: {str(e)}")

    def load_preprocessing_components(self,
                                      scaler_path: str,
                                      metadata_path: str) -> Tuple:
        """
        Load both scaler and metadata.

        Args:
            scaler_path: Path to scaler pickle file
            metadata_path: Path to metadata JSON file

        Returns:
            Tuple of (scaler, metadata)

        Example:
            scaler, metadata = loader.load_preprocessing_components(
                scaler_path='features/feature_scaler.pkl',
                metadata_path='features/preprocessing_metadata.json'
            )
        """
        if self.verbose:
            print("[ModelLoader] Loading preprocessing components...")

        scaler = self.load_scaler(scaler_path)
        metadata = self.load_metadata(metadata_path)

        if self.verbose:
            print("[ModelLoader] All preprocessing components loaded")

        return scaler, metadata

    def load_all_components(self,
                            model_path: str,
                            scaler_path: str,
                            metadata_path: str,
                            n_features: int = 21,
                            n_treatments: int = 5,
                            hidden_dims: list = [256, 128, 64],
                            learning_rate: float = 0.001,
                            weight_decay: float = 1e-4,
                            device: Optional[str] = None) -> Tuple:
        """
        Load model and preprocessing components together.

        Args:
            model_path: Path to model .pth file
            scaler_path: Path to scaler pickle file
            metadata_path: Path to metadata JSON file
            n_features: Number of input features (21 with engineered features)
            n_treatments: Number of treatments
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            weight_decay: L2 regularization coefficient
            device: 'cpu' or 'cuda'

        Returns:
            Tuple of (model, scaler, metadata)

        Example:
            model, scaler, metadata = loader.load_all_components(
                model_path='artifacts/v1_0/production/neural_t_learner.pth',
                scaler_path='features/feature_scaler.pkl',
                metadata_path='features/preprocessing_metadata.json'
            )
        """
        if self.verbose:
            print("[ModelLoader] Loading all components...")

        # Load model
        model = self.load_model(
            model_path=model_path,
            n_features=n_features,
            n_treatments=n_treatments,
            hidden_dims=hidden_dims,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device
        )

        # Load preprocessing
        scaler, metadata = self.load_preprocessing_components(
            scaler_path=scaler_path,
            metadata_path=metadata_path
        )

        # Validate compatibility
        if self.verbose:
            print("[ModelLoader] Validating component compatibility...")
            self._validate_compatibility(model, scaler, metadata)

        if self.verbose:
            print("[ModelLoader] All components loaded successfully")

        return model, scaler, metadata

    def _validate_compatibility(self, model, scaler, metadata):
        """
        Validate that model, scaler, and metadata are compatible.

        Args:
            model: Loaded NeuralTLearner
            scaler: Loaded StandardScaler
            metadata: Loaded metadata dict
        """
        warnings = []

        # Check feature count
        model_features = model.n_features
        metadata_features = len(metadata.get('feature_cols', []))

        if model_features != metadata_features:
            warnings.append(
                f"Feature count mismatch: model expects {model_features}, "
                f"metadata has {metadata_features}"
            )

        if model_features != 21:
            warnings.append(
                f"Model expects {model_features} features, expected 21"
            )

        # Check scaler
        if hasattr(scaler, 'n_features_in_'):
            scaler_features = len(metadata.get('continuous_features', []))
            if scaler.n_features_in_ != scaler_features:
                warnings.append(
                    f"Scaler feature mismatch: scaler has {scaler.n_features_in_}, "
                    f"metadata continuous features: {scaler_features}"
                )

        # Print warnings
        if warnings:
            print("[ModelLoader] WARNING: Compatibility issues detected:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("[ModelLoader] ✓ All components compatible")

    def get_file_info(self, file_path: str) -> Dict:
        """
        Get information about a file.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file information
        """
        path = Path(file_path)

        if not path.exists():
            return {
                'path': str(path.absolute()),
                'exists': False,
                'size_bytes': 0,
                'size_kb': 0.0,
                'size_mb': 0.0
            }

        size_bytes = path.stat().st_size

        return {
            'path': str(path.absolute()),
            'exists': True,
            'size_bytes': size_bytes,
            'size_kb': size_bytes / 1024,
            'size_mb': size_bytes / (1024 * 1024)
        }

    def print_component_summary(self,
                                model_path: str,
                                scaler_path: str,
                                metadata_path: str):
        """
        Print summary of all component files.

        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
            metadata_path: Path to metadata file

        Example:
            loader.print_component_summary(
                'artifacts/v1_0/production/neural_t_learner.pth',
                'features/feature_scaler.pkl',
                'features/preprocessing_metadata.json'
            )
        """
        print("\n" + "=" * 80)
        print("MODEL COMPONENT SUMMARY")
        print("=" * 80)

        # Model file
        model_info = self.get_file_info(model_path)
        print(f"\n1. Model File:")
        print(f"   Path: {model_path}")
        print(f"   Exists: {'✓' if model_info['exists'] else '✗'}")
        if model_info['exists']:
            print(f"   Size: {model_info['size_mb']:.2f} MB")

        # Scaler file
        scaler_info = self.get_file_info(scaler_path)
        print(f"\n2. Feature Scaler:")
        print(f"   Path: {scaler_path}")
        print(f"   Exists: {'✓' if scaler_info['exists'] else '✗'}")
        if scaler_info['exists']:
            print(f"   Size: {scaler_info['size_kb']:.2f} KB")

        # Metadata file
        metadata_info = self.get_file_info(metadata_path)
        print(f"\n3. Preprocessing Metadata:")
        print(f"   Path: {metadata_path}")
        print(f"   Exists: {'✓' if metadata_info['exists'] else '✗'}")
        if metadata_info['exists']:
            print(f"   Size: {metadata_info['size_kb']:.2f} KB")

        # Overall status
        all_exist = all([
            model_info['exists'],
            scaler_info['exists'],
            metadata_info['exists']
        ])

        print(f"\nOverall Status: {'✓ All files present' if all_exist else '✗ Missing files'}")
        print("=" * 80 + "\n")


# =============================================================================
# STANDALONE HELPER FUNCTIONS
# =============================================================================

def load_model(model_path: str,
               n_features: int = 21,
               n_treatments: int = 5,
               hidden_dims: list = [256, 128, 64],
               learning_rate: float = 0.001,
               weight_decay: float = 1e-4,
               device: Optional[str] = None,
               verbose: bool = False) -> NeuralTLearner:
    """
    Quick helper to load a model without creating ModelLoader instance.

    Args:
        model_path: Path to model .pth file
        n_features: Number of input features (21 with engineered features)
        n_treatments: Number of treatments
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        weight_decay: L2 regularization coefficient
        device: 'cpu' or 'cuda'
        verbose: Print loading info

    Returns:
        Loaded NeuralTLearner instance

    Example:
        from treatment_recommender.models import load_model
        model = load_model('artifacts/v1_0/production/neural_t_learner.pth')
    """
    loader = ModelLoader(verbose=verbose)
    return loader.load_model(
        model_path=model_path,
        n_features=n_features,
        n_treatments=n_treatments,
        hidden_dims=hidden_dims,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )


def load_preprocessing_components(scaler_path: str,
                                  metadata_path: str,
                                  verbose: bool = False) -> Tuple:
    """
    Quick helper to load preprocessing components.

    Args:
        scaler_path: Path to scaler pickle file
        metadata_path: Path to metadata JSON file
        verbose: Print loading info

    Returns:
        Tuple of (scaler, metadata)

    Example:
        from treatment_recommender.models import load_preprocessing_components
        scaler, metadata = load_preprocessing_components(
            scaler_path='features/feature_scaler.pkl',
            metadata_path='features/preprocessing_metadata.json'
        )
    """
    loader = ModelLoader(verbose=verbose)
    return loader.load_preprocessing_components(scaler_path, metadata_path)


def validate_model_files(model_path: str,
                         scaler_path: str,
                         metadata_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that all required model files exist.

    Args:
        model_path: Path to model file
        scaler_path: Path to scaler file
        metadata_path: Path to metadata file

    Returns:
        Tuple of (all_valid, error_message)

    Example:
        is_valid, error = validate_model_files(
            model_path='artifacts/v1_0/production/neural_t_learner.pth',
            scaler_path='features/feature_scaler.pkl',
            metadata_path='features/preprocessing_metadata.json'
        )
        if not is_valid:
            print(f"Validation failed: {error}")
    """
    loader = ModelLoader(verbose=False)

    try:
        loader.validate_file_path(model_path, "Model file")
        loader.validate_file_path(scaler_path, "Scaler file")
        loader.validate_file_path(metadata_path, "Metadata file")
        return True, None
    except Exception as e:
        return False, str(e)


def load_model_with_auto_config(model_path: str,
                                verbose: bool = False) -> NeuralTLearner:
    """
    Load model with automatic configuration detection from checkpoint.

    This function tries to extract configuration from the saved checkpoint
    to ensure the model architecture matches exactly.

    Args:
        model_path: Path to model .pth file
        verbose: Print loading info

    Returns:
        Loaded NeuralTLearner instance

    Example:
        model = load_model_with_auto_config('artifacts/v1_0/production/neural_t_learner.pth')
    """
    if verbose:
        print(f"[ModelLoader] Auto-detecting model configuration from: {model_path}")

    # Load checkpoint to inspect
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Extract configuration from checkpoint
        n_features = checkpoint.get('n_features', 21)
        n_treatments = checkpoint.get('n_treatments', 5)

        if verbose:
            print(f"[ModelLoader] Detected: n_features={n_features}, n_treatments={n_treatments}")

        # Create model with detected config
        model = create_neural_t_learner(
            n_features=n_features,
            n_treatments=n_treatments,
            hidden_dims=[256, 128, 64],
            learning_rate=0.001,
            weight_decay=1e-4,
            device='cpu'
        )

        # Load weights
        model.load_models(model_path)

        if verbose:
            print(f"[ModelLoader] Model loaded successfully with auto-detected config")

        return model

    except Exception as e:
        raise Exception(f"Failed to auto-load model: {str(e)}")


def check_model_compatibility(model_path: str,
                              scaler_path: str,
                              metadata_path: str,
                              verbose: bool = True) -> Dict:
    """
    Check compatibility between model, scaler, and metadata.

    Args:
        model_path: Path to model file
        scaler_path: Path to scaler file
        metadata_path: Path to metadata file
        verbose: Print compatibility report

    Returns:
        Dictionary with compatibility check results

    Example:
        compat = check_model_compatibility(
            'artifacts/v1_0/production/neural_t_learner.pth',
            'features/feature_scaler.pkl',
            'features/preprocessing_metadata.json'
        )
        if not compat['compatible']:
            print(f"Issues: {compat['issues']}")
    """
    loader = ModelLoader(verbose=False)

    try:
        # Load all components
        model, scaler, metadata = loader.load_all_components(
            model_path=model_path,
            scaler_path=scaler_path,
            metadata_path=metadata_path
        )

        issues = []

        # Check feature counts
        model_features = model.n_features
        metadata_features = len(metadata.get('feature_cols', []))

        if model_features != 21:
            issues.append(f"Model expects {model_features} features, should be 21")

        if metadata_features != 21:
            issues.append(f"Metadata has {metadata_features} features, should be 21")

        if model_features != metadata_features:
            issues.append(f"Model/metadata mismatch: {model_features} vs {metadata_features}")

        # Check scaler
        continuous_features = metadata.get('continuous_features', [])
        if hasattr(scaler, 'n_features_in_'):
            if scaler.n_features_in_ != len(continuous_features):
                issues.append(
                    f"Scaler mismatch: scaler has {scaler.n_features_in_}, "
                    f"continuous features: {len(continuous_features)}"
                )

        # Build result
        result = {
            'compatible': len(issues) == 0,
            'model_features': model_features,
            'metadata_features': metadata_features,
            'continuous_features': len(continuous_features),
            'scaler_features': scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None,
            'issues': issues
        }

        if verbose:
            print("\n" + "=" * 80)
            print("COMPATIBILITY CHECK")
            print("=" * 80)
            print(f"\nModel features: {result['model_features']}")
            print(f"Metadata features: {result['metadata_features']}")
            print(f"Continuous features: {result['continuous_features']}")
            print(f"Scaler features: {result['scaler_features']}")

            if result['compatible']:
                print(f"\n✓ All components compatible")
            else:
                print(f"\n✗ Compatibility issues found:")
                for issue in result['issues']:
                    print(f"  - {issue}")
            print("=" * 80 + "\n")

        return result

    except Exception as e:
        return {
            'compatible': False,
            'error': str(e),
            'issues': [str(e)]
        }