"""
Stateless prediction pipeline for diabetes treatment recommendation.

Key Features:
- No singleton pattern - each instance is independent
- Lazy loading - model loaded on first predict() call
- No cached state - can reload artifacts dynamically
- Thread-safe - separate instances per thread
- Multi-model support - test multiple versions simultaneously
- Uses ModelManager for all model operations
"""

import warnings
import numpy as np
from typing import Optional, List

from ._base import (
    TreatmentResult,
    SelectionMode,
    TREATMENT_NAMES,
    get_timestamp,
    calculate_confidence,
    calculate_confidence_margin,
    rank_treatments
)
from ..registry import ModelManager


class PredictionPipeline:
    """
    Stateless prediction pipeline for diabetes treatment recommendation.

    This pipeline is designed to be:
    1. Stateless - no cached state between predictions
    2. Reusable - can create multiple instances with different models
    3. Thread-safe - each instance operates independently
    4. Hot-swappable - switch model versions dynamically

    Usage:
        # Create pipeline with ModelManager
        pipeline = PredictionPipeline(
            model_manager=manager,
            version='v1_2',  # Optional: specific version
            verbose=False
        )

        # Predict
        result = pipeline.predict(patient_dict)

        # Test multiple versions simultaneously
        pipeline_v1 = PredictionPipeline(model_manager=manager, version='v1_0')
        pipeline_v2 = PredictionPipeline(model_manager=manager, version='v1_2')

        result_v1 = pipeline_v1.predict(patient)
        result_v2 = pipeline_v2.predict(patient)
    """

    def __init__(self,
                 model_manager: ModelManager,
                 version: Optional[str] = None,
                 verbose: bool = False):
        """
        Initialize prediction pipeline.

        Args:
            model_manager: ModelManager instance for all model operations
            version: Specific version to use (e.g., 'v1_2')
                    If None, uses active version
            verbose: If True, print detailed logs
        """
        self.manager = model_manager
        self.version = version
        self.verbose = verbose

        # Lazy-loaded model (loaded on first predict call)
        self._model = None
        self._model_name = None

        if self.verbose:
            version_str = f"version {version}" if version else "active version"
            print(f"[PredictionPipeline] Initialized with {version_str}")
            print(f"[PredictionPipeline] Model will be loaded on first prediction")

    def _load_model(self):
        """
        Lazy load model from ModelManager.

        This method is called automatically on first predict() call.
        Model is loaded into memory and cached for subsequent predictions.

        Returns:
            Loaded model object
        """
        if self.verbose:
            version_str = f"version {self.version}" if self.version else "active version"
            print(f"[PredictionPipeline] Loading {version_str}...")

        # Load model from manager
        if self.version:
            model = self.manager.get_model_by_version(self.version)
        else:
            model = self.manager.get_active_model()

        self._model_name = 'NeuralTLearner'

        if self.verbose:
            loaded_version = self.version if self.version else self.manager.get_active_version()
            print(f"[PredictionPipeline] Model loaded: {loaded_version}")

        return model

    def reload_model(self):
        """
        Manually reload model from disk.

        Useful when you want to refresh the model without creating a new pipeline instance.

        Example:
            pipeline = PredictionPipeline(model_manager=manager)
            result1 = pipeline.predict(patient)  # Uses active version

            # Activate new version
            manager.activate_version('v1_3')
            pipeline.reload_model()  # Reload from disk

            result2 = pipeline.predict(patient)  # Uses v1_3
        """
        if self.verbose:
            print("[PredictionPipeline] Reloading model...")

        self._model = None
        self._model = self._load_model()

        if self.verbose:
            print("[PredictionPipeline] Model reloaded successfully")

    def switch_version(self, version: str):
        """
        Switch to a different model version.

        Args:
            version: Version to switch to (e.g., 'v1_3')

        Example:
            pipeline = PredictionPipeline(model_manager=manager, version='v1_0')
            result1 = pipeline.predict(patient)  # Uses v1_0

            pipeline.switch_version('v1_2')
            result2 = pipeline.predict(patient)  # Uses v1_2
        """
        if self.verbose:
            print(f"[PredictionPipeline] Switching from {self.version} to {version}")

        self.version = version
        self._model = None  # Force reload on next predict

        if self.verbose:
            print(f"[PredictionPipeline] Version switched to {version}")

    def predict(self,
                patient_dict: dict,
                mode: str = 'greedy',
                epsilon: float = 0.1,
                temperature: float = 0.3,
                verbose: Optional[bool] = None) -> TreatmentResult:
        """
        Predict treatment recommendation for a patient.

        Pipeline execution:
        1. Lazy load model (first call only)
        2. Preprocess patient data (encode, scale)
        3. Predict Q-values for all 5 treatments
        4. Select best treatment (greedy/epsilon-greedy/softmax)
        5. Check safety constraints
        6. Calculate confidence scores
        7. Return TreatmentResult DTO

        Args:
            patient_dict: Patient data dictionary
            mode: Selection mode ('greedy', 'epsilon-greedy', 'softmax')
            epsilon: Exploration probability for epsilon-greedy
            temperature: Temperature for softmax (lower = more confident)
            verbose: Override instance verbose setting

        Returns:
            TreatmentResult DTO with recommendation details

        Example:
            patient = {
                'age': 58, 'gender': 'Female', 'ethnicity': 'Caucasian',
                'hba1c_baseline': 8.2, 'egfr': 75, 'bmi': 31.5,
                ...
            }

            result = pipeline.predict(patient)

            print(f"Recommended: {result.recommended_treatment}")
            print(f"Expected reduction: {result.predicted_hba1c_reduction:.2f}%")
            print(f"Confidence: {result.confidence_score:.1f}%")
        """
        # Use instance verbose if not overridden
        show_verbose = verbose if verbose is not None else self.verbose

        if not show_verbose:
            print("[PredictionPipeline] Starting treatment recommendation...")
        else:
            print("\n" + "=" * 80)
            print("[PredictionPipeline] ===== TREATMENT RECOMMENDATION PIPELINE =====")
            print("=" * 80)

        try:
            # Step 1: Lazy load model if not already loaded
            if self._model is None:
                self._model = self._load_model()

            # Step 2: Preprocess patient (get processor from manager)
            if show_verbose:
                print("[PredictionPipeline] Step 1/4: Preprocessing patient data")

            processor = self.manager.get_feature_processor()
            features = processor.process_patient(patient_dict)

            if show_verbose:
                print(f"[PredictionPipeline] Features shape: {features.shape}")

            # Step 3: Predict Q-values
            if show_verbose:
                print("[PredictionPipeline] Step 2/4: Predicting Q-values")

            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q_values = self._model.predict_q_values(features).flatten()

            if show_verbose:
                print(f"[PredictionPipeline] Q-values: {q_values}")

            # Step 4: Select treatment
            if show_verbose:
                print("[PredictionPipeline] Step 3/4: Selecting treatment")

            treatment_idx = self._model.select_treatment(
                features,
                mode=mode,
                epsilon=epsilon,
                temperature=temperature
            )

            treatment_name = TREATMENT_NAMES[treatment_idx]

            if show_verbose:
                print(f"[PredictionPipeline] Selected: {treatment_name} (index: {treatment_idx})")

            # Step 5: Safety checks
            if show_verbose:
                print("[PredictionPipeline] Step 4/4: Performing safety checks")

            safety_warnings = self._check_safety(patient_dict, treatment_name)

            if show_verbose and safety_warnings:
                print(f"[PredictionPipeline] Safety warnings: {safety_warnings}")

            # Step 6: Calculate confidence
            confidence_score = calculate_confidence(q_values.tolist())
            confidence_margin = calculate_confidence_margin(q_values.tolist())

            # Step 7: Rank treatments
            ranked = rank_treatments(TREATMENT_NAMES, q_values.tolist())

            # Step 8: Build result DTO
            result = TreatmentResult(
                recommended_treatment=treatment_name,
                treatment_index=int(treatment_idx),
                predicted_hba1c_reduction=float(q_values[treatment_idx]),
                confidence_score=float(confidence_score),
                confidence_margin=float(confidence_margin),
                all_q_values={
                    TREATMENT_NAMES[i]: float(q_values[i])
                    for i in range(len(TREATMENT_NAMES))
                },
                ranked_treatments=ranked,
                safety_warnings=safety_warnings
            )

            if show_verbose:
                print("\n" + "=" * 80)
                print("[PredictionPipeline] ===== RECOMMENDATION RESULT =====")
                print("=" * 80)
                print(f"Recommended: {result.recommended_treatment}")
                print(f"Expected HbA1c reduction: {result.predicted_hba1c_reduction:.2f}%")
                print(f"Confidence: {result.confidence_score:.1f}%")
                print(f"Safety warnings: {len(result.safety_warnings)}")
                print("=" * 80 + "\n")
            else:
                print(f"[PredictionPipeline] Recommendation: {result.recommended_treatment} "
                      f"(expected reduction: {result.predicted_hba1c_reduction:.2f}%)\n")

            return result

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"

            if show_verbose:
                print(f"[PredictionPipeline] Error: {error_msg}")
            else:
                print(f"[PredictionPipeline] Prediction failed: {error_msg}\n")

            return TreatmentResult(
                recommended_treatment='ERROR',
                treatment_index=-1,
                predicted_hba1c_reduction=0.0,
                confidence_score=0.0,
                confidence_margin=0.0,
                all_q_values={},
                ranked_treatments=[],
                safety_warnings=[],
                error=error_msg
            )

    def _check_safety(self, patient_dict: dict, treatment: str) -> List[str]:
        """
        Check for safety contraindications.

        Args:
            patient_dict: Patient data
            treatment: Proposed treatment name

        Returns:
            List of safety warnings (empty if no concerns)
        """
        from ..utils import validate_safety_constraints

        return validate_safety_constraints(patient_dict, treatment)

    def predict_batch(self,
                      patients: List[dict],
                      mode: str = 'greedy',
                      verbose: Optional[bool] = None) -> List[TreatmentResult]:
        """
        Predict multiple patients in batch.

        Args:
            patients: List of patient dictionaries
            mode: Selection mode
            verbose: Override instance verbose setting

        Returns:
            List of TreatmentResult DTOs

        Example:
            patients = [patient1, patient2, patient3]
            results = pipeline.predict_batch(patients)

            for i, result in enumerate(results, 1):
                print(f"Patient {i}: {result.recommended_treatment}")
        """
        show_verbose = verbose if verbose is not None else self.verbose

        if not show_verbose:
            print(f"[PredictionPipeline] Starting batch prediction for {len(patients)} patients...")
        else:
            print(f"\n{'=' * 80}")
            print(f"[PredictionPipeline] ===== BATCH PREDICTION =====")
            print(f"{'=' * 80}")
            print(f"Total patients: {len(patients)}")

        results = []

        for i, patient in enumerate(patients, 1):
            if show_verbose:
                print(f"\n[PredictionPipeline] Processing patient {i}/{len(patients)}")

            try:
                result = self.predict(
                    patient_dict=patient,
                    mode=mode,
                    verbose=False  # Don't show verbose for each patient
                )
                results.append(result)

                if show_verbose:
                    print(f"Patient {i}: {result.recommended_treatment}")

            except Exception as e:
                if show_verbose:
                    print(f"Patient {i}: Error - {str(e)}")

                results.append(TreatmentResult(
                    recommended_treatment='ERROR',
                    treatment_index=-1,
                    predicted_hba1c_reduction=0.0,
                    confidence_score=0.0,
                    confidence_margin=0.0,
                    all_q_values={},
                    ranked_treatments=[],
                    safety_warnings=[],
                    error=str(e)
                ))

        # Summary
        successful = sum(1 for r in results if r.error is None)
        treatment_counts = {}
        for r in results:
            if r.error is None:
                treatment_counts[r.recommended_treatment] = treatment_counts.get(r.recommended_treatment, 0) + 1

        if show_verbose:
            print(f"\n{'=' * 80}")
            print("[PredictionPipeline] ===== BATCH SUMMARY =====")
            print(f"{'=' * 80}")
            print(f"Processed: {successful}/{len(patients)}")
            print(f"Treatment distribution:")
            for treatment, count in treatment_counts.items():
                print(f"  {treatment}: {count}")
            print(f"{'=' * 80}\n")
        else:
            print(f"[PredictionPipeline] Batch complete: {successful}/{len(patients)} successful\n")

        return results

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            dict: Model information including type, version, etc.
        """
        # Lazy load if needed
        if self._model is None:
            self._model = self._load_model()

        # Get version info from manager
        current_version = self.version if self.version else self.manager.get_active_version()
        version_info = self.manager.get_model_info(current_version) if current_version else {}

        return {
            'model_name': self._model_name,
            'model_type': 'NeuralTLearner',
            'version': current_version,
            'n_features': self._model.n_features,
            'n_treatments': self._model.n_treatments,
            'device': self._model.device,
            'performance_metrics': version_info.get('performance_metrics', {}),
            'trained_timestamp': version_info.get('trained_timestamp', None)
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_prediction_pipeline(model_manager: ModelManager,
                               version: Optional[str] = None,
                               verbose: bool = False) -> PredictionPipeline:
    """
    Factory function to create a new prediction pipeline instance.

    Args:
        model_manager: ModelManager instance for all model operations
        version: Specific version to use (e.g., 'v1_2')
                If None, uses active version
        verbose: Enable detailed logging

    Returns:
        New PredictionPipeline instance

    Example:
        from treatment_recommender.registry import create_model_manager

        manager = create_model_manager()

        # Use active version
        pipeline = create_prediction_pipeline(model_manager=manager)

        # Or use specific version
        pipeline_v2 = create_prediction_pipeline(
            model_manager=manager,
            version='v1_2'
        )

        result = pipeline.predict(patient)
    """
    return PredictionPipeline(
        model_manager=model_manager,
        version=version,
        verbose=verbose
    )