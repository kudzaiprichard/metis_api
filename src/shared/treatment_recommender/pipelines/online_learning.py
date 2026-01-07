"""
Stateless online learning pipeline for diabetes treatment recommendation.

Key Features:
- No singleton pattern - each instance is independent
- Stateless - no cached training history
- Creates new model versions (doesn't overwrite)
- Supports incremental learning with partial_fit and EWC
- Thread-safe training operations
- Uses ModelManager for all model operations
"""

import os
import json
import warnings
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List

from ._base import (
    TrainingResult,
    TrainingStatus,
    TrainingStep,
    get_timestamp
)
from ..registry import ModelManager


class OnlineLearningPipeline:
    """
    Stateless online learning pipeline for incremental model updates.

    This pipeline is designed to be:
    1. Stateless - no cached training history or state
    2. Version-safe - creates new versions instead of overwriting
    3. Independent - multiple instances can train simultaneously
    4. Thread-safe - separate instances per training job
    5. EWC-enabled - uses Elastic Weight Consolidation for catastrophic forgetting prevention

    Pipeline Flow:
    1. Load base model from ModelManager
    2. Preprocess patient outcomes
    3. Validate performance before update (optional)
    4. Perform partial_fit on new data (with EWC)
    5. Validate performance after update (optional)
    6. Register new version via ModelManager

    Usage:
        pipeline = OnlineLearningPipeline(
            model_manager=manager,
            verbose=False
        )

        outcomes = [
            {'patient': patient_dict1, 'treatment_given': 'Insulin', 'reward': 3.5},
            {'patient': patient_dict2, 'treatment_given': 'Metformin', 'reward': 2.1},
        ]

        result = pipeline.partial_fit(
            outcomes=outcomes,
            base_version='v1_0',
            validate=True
        )

        if result.success:
            print(f"New version created: {result.version_number}")
    """

    def __init__(self,
                 model_manager: ModelManager,
                 verbose: bool = False):
        """
        Initialize online learning pipeline.

        Args:
            model_manager: ModelManager instance for all model operations
            verbose: If True, print detailed logs
        """
        self.manager = model_manager
        self.verbose = verbose

        os.makedirs('storage', exist_ok=True)

        self._status = TrainingStatus(
            is_training=False,
            current_step=None,
            progress_percent=0,
            started_at=None,
            estimated_completion=None,
            version_number=None,
            outcomes_count=0,
            error=None
        )

        if self.verbose:
            print("[OnlineLearningPipeline] Initialized with ModelManager")
        else:
            print("[OnlineLearningPipeline] Initialization complete\n")

    def _update_progress(self,
                         step: TrainingStep,
                         stage: str,
                         error: str = None):
        """
        Update progress for a training step.

        Args:
            step: TrainingStep enum
            stage: 'start', 'complete', or 'error'
            error: Error message if stage is 'error'
        """
        if stage == 'start':
            progress = step.start_progress
            status = step.description
        elif stage == 'complete':
            progress = step.end_progress
            status = f"{step.description} completed"
        elif stage == 'error':
            progress = step.start_progress
            status = error
        else:
            progress = step.mid_progress
            status = step.description

        self._status.current_step = status
        self._status.progress_percent = progress

        if error:
            self._status.error = error
            self._status.is_training = False

        if self._status.is_training and self._status.started_at:
            elapsed = (datetime.now() - datetime.fromisoformat(self._status.started_at)).total_seconds()
            if progress > 0:
                total_estimated = (elapsed / progress) * 100
                remaining = total_estimated - elapsed
                estimated_completion = datetime.now().timestamp() + remaining
                self._status.estimated_completion = datetime.fromtimestamp(estimated_completion).isoformat()

        try:
            status_dict = {
                'is_training': self._status.is_training,
                'current_step': self._status.current_step,
                'progress_percent': self._status.progress_percent,
                'started_at': self._status.started_at,
                'estimated_completion': self._status.estimated_completion,
                'version_number': self._status.version_number,
                'outcomes_count': self._status.outcomes_count,
                'error': self._status.error
            }
            with open('storage/training_status.json', 'w') as f:
                json.dump(status_dict, f, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"[OnlineLearningPipeline] Warning: Could not write status: {str(e)}")

        if self.verbose:
            print(f"[OnlineLearningPipeline] [{progress}%] {status}")

    def get_status(self) -> TrainingStatus:
        """
        Get current training status.

        Returns:
            TrainingStatus DTO with current progress
        """
        return self._status

    def _preprocess_outcomes(self, outcomes: List[Dict]) -> tuple:
        """
        Preprocess batch of patient outcomes.

        Args:
            outcomes: List of dicts with keys: 'patient', 'treatment_given', 'reward'

        Returns:
            Tuple of (X_features, T_treatments, Y_rewards, successful_count)
        """
        if self.verbose:
            print(f"[OnlineLearningPipeline] Preprocessing {len(outcomes)} patient outcomes")

        processor = self.manager.get_feature_processor()

        X_features = []
        T_treatments = []
        Y_rewards = []
        failed = 0

        for i, outcome in enumerate(outcomes):
            try:
                if not all(k in outcome for k in ['patient', 'treatment_given', 'reward']):
                    raise ValueError("Outcome missing required fields")

                features = processor.process_patient(outcome['patient'])
                treatment_id = processor.encode_treatment(outcome['treatment_given'])
                reward = float(outcome['reward'])

                X_features.append(features)
                T_treatments.append(treatment_id)
                Y_rewards.append(reward)

            except Exception as e:
                failed += 1
                if self.verbose:
                    print(f"[OnlineLearningPipeline] Outcome {i + 1} failed: {str(e)[:50]}")

        if len(X_features) == 0:
            raise ValueError("No outcomes could be processed successfully")

        if self.verbose:
            print(f"[OnlineLearningPipeline] Processed: {len(X_features)}/{len(outcomes)}")

        X = np.vstack(X_features)
        T = np.array(T_treatments)
        Y = np.array(Y_rewards)

        return X, T, Y, len(X_features)

    def _validate_performance(self, model, X, T, Y) -> Dict:
        """
        Validate model performance on batch.

        Args:
            model: NeuralTLearner instance
            X: Features (n, 21)
            T: Treatments (n,)
            Y: Rewards (n,)

        Returns:
            Dictionary with performance metrics
        """
        if self.verbose:
            print(f"[OnlineLearningPipeline] Validating performance on {len(X)} samples")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            q_values = model.predict_q_values(X)

            recommendations = []
            for i in range(len(X)):
                treatment = model.select_treatment(X[i], mode='greedy')
                recommendations.append(treatment)
            recommendations = np.array(recommendations)

            predicted_rewards = []
            for i in range(len(X)):
                rec_treatment = recommendations[i]
                actual_treatment = T[i]

                if rec_treatment == actual_treatment:
                    predicted_rewards.append(Y[i])
                else:
                    predicted_rewards.append(q_values[i, rec_treatment])

            predicted_rewards = np.array(predicted_rewards)

            metrics = {
                'avg_reward': float(np.mean(predicted_rewards)),
                'diversity': int(len(np.unique(recommendations))),
                'accuracy': float((recommendations == T).mean()),
                'success_rate': float((predicted_rewards >= 1.5).mean())
            }

        if self.verbose:
            print(f"[OnlineLearningPipeline] Validation metrics: {metrics}")

        return metrics

    def partial_fit(self,
                    outcomes: List[Dict],
                    base_version: Optional[str] = None,
                    validate: bool = True,
                    disable_ewc: bool = False,
                    epochs: int = 1) -> TrainingResult:
        """
        Perform online learning on a batch of patient outcomes with EWC.

        This method uses the model's partial_fit() which automatically:
        - Updates only relevant treatment networks
        - Applies EWC penalty to prevent catastrophic forgetting (unless disabled)
        - Uses lower learning rate for stability (0.0005)
        - Protects important weights learned during initial training

        Args:
            outcomes: List of outcome dicts with keys:
                     'patient': patient_dict
                     'treatment_given': treatment name (e.g., 'Insulin')
                     'reward': observed HbA1c reduction
            base_version: Base version to update from (e.g., "v1_0")
                         If None, uses active version
            validate: If True, validate performance before/after
            disable_ewc: If True, disable EWC for this training session (default: False)
                        Use for testing or when you want unrestricted learning
            epochs: Number of times to iterate over the training data (default: 1)
                   Higher epochs = more learning but risk of overfitting

        Returns:
            TrainingResult DTO with training results

        Example:
            outcomes = [
                {
                    'patient': {'age': 58, 'gender': 'Female', ...},
                    'treatment_given': 'Insulin',
                    'reward': 3.5
                },
                {
                    'patient': {'age': 45, 'gender': 'Male', ...},
                    'treatment_given': 'Metformin',
                    'reward': 2.1
                }
            ]

            # Normal training with EWC
            result = pipeline.partial_fit(
                outcomes,
                base_version="v1_0"
            )

            # Training with EWC disabled and multiple epochs (for testing)
            result = pipeline.partial_fit(
                outcomes,
                base_version="v1_0",
                disable_ewc=True,
                epochs=20
            )

            if result.success:
                print(f"New version: {result.version_number}")
        """
        self._status = TrainingStatus(
            is_training=True,
            current_step=None,
            progress_percent=0,
            started_at=get_timestamp(),
            estimated_completion=None,
            version_number=None,
            outcomes_count=len(outcomes),
            error=None
        )

        if not self.verbose:
            print(f"[OnlineLearningPipeline] Starting online learning on {len(outcomes)} patient outcomes...")
        else:
            print("\n" + "=" * 80)
            print("[OnlineLearningPipeline] ===== ONLINE LEARNING WITH EWC =====")
            print("=" * 80)
            print(f"Batch size: {len(outcomes)}")
            print(f"Epochs: {epochs}")

        try:
            self._update_progress(TrainingStep.CALCULATING_VERSION, 'start')

            if base_version is None:
                base_version = self.manager.get_active_version()
                if base_version is None:
                    base_version = self.manager.get_latest_version()
                if self.verbose:
                    print(f"[OnlineLearningPipeline] Using base version: {base_version}")

            self._update_progress(TrainingStep.CALCULATING_VERSION, 'complete')

            self._update_progress(TrainingStep.LOADING_MODEL, 'start')

            if base_version:
                model = self.manager.get_model_by_version(base_version)
            else:
                model = self.manager.get_active_model()

            if self.verbose:
                print(f"[OnlineLearningPipeline] Base model loaded: {base_version}")

            # ========== CRITICAL FIX: DISABLE EWC AFTER MODEL LOAD ==========
            if disable_ewc:
                if self.verbose:
                    print(f"[OnlineLearningPipeline] Disabling EWC for this training session...")

                # Store original values for logging
                original_lambda = model.ewc_lambda
                original_enabled = model.ewc_enabled.copy()

                # Disable EWC
                model.ewc_lambda = 0
                model.ewc_enabled = {t: False for t in range(5)}

                if self.verbose:
                    print(f"[OnlineLearningPipeline] EWC lambda: {original_lambda} -> {model.ewc_lambda}")
                    print(f"[OnlineLearningPipeline] EWC enabled: {original_enabled} -> {model.ewc_enabled}")
                    print(f"[OnlineLearningPipeline] Unrestricted learning enabled")
                else:
                    print(f"⚠️  EWC DISABLED for testing\n")
            # ================================================================

            self._update_progress(TrainingStep.LOADING_MODEL, 'complete')

            self._update_progress(TrainingStep.PREPROCESSING, 'start')
            X, T, Y, count = self._preprocess_outcomes(outcomes)
            self._update_progress(TrainingStep.PREPROCESSING, 'complete')

            before = {}
            if validate:
                self._update_progress(TrainingStep.VALIDATING_BEFORE, 'start')
                before = self._validate_performance(model, X, T, Y)
                if self.verbose:
                    print(f"[OnlineLearningPipeline] Before: avg_reward={before['avg_reward']:.3f}, "
                          f"accuracy={before['accuracy']:.3f}")
                self._update_progress(TrainingStep.VALIDATING_BEFORE, 'complete')

            self._update_progress(TrainingStep.PARTIAL_FIT, 'start')

            # ========== TRAINING LOOP WITH EPOCHS ==========
            if self.verbose:
                print(f"[OnlineLearningPipeline] Training for {epochs} epoch(s)...")
            elif epochs > 1:
                print(f"Training for {epochs} epochs...")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                losses = None
                for epoch in range(epochs):
                    losses = model.partial_fit(X, T, Y)

                    # Print progress every 5 epochs (or on last epoch)
                    if self.verbose and (epoch % 5 == 4 or epoch == epochs - 1):
                        avg_loss = np.mean(list(losses.values()))
                        print(f"  Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.3f}")
                    elif not self.verbose and epochs > 1 and (epoch % 5 == 4 or epoch == epochs - 1):
                        avg_loss = np.mean(list(losses.values()))
                        print(f"  Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.3f}")
            # ===============================================

            if self.verbose:
                ewc_status = "disabled" if disable_ewc else "enabled"
                print(f"[OnlineLearningPipeline] Partial fit completed (EWC {ewc_status})")
                print(f"[OnlineLearningPipeline] Final treatment losses: {losses}")
            elif epochs > 1:
                print()  # Empty line after training progress

            self._update_progress(TrainingStep.PARTIAL_FIT, 'complete')

            after = {}
            if validate:
                self._update_progress(TrainingStep.VALIDATING_AFTER, 'start')
                after = self._validate_performance(model, X, T, Y)
                if self.verbose:
                    print(f"[OnlineLearningPipeline] After: avg_reward={after['avg_reward']:.3f}, "
                          f"accuracy={after['accuracy']:.3f}")

                    if before:
                        reward_change = after['avg_reward'] - before['avg_reward']
                        acc_change = after['accuracy'] - before['accuracy']
                        print(f"[OnlineLearningPipeline] Change: reward={reward_change:+.3f}, "
                              f"accuracy={acc_change:+.3f}")
                self._update_progress(TrainingStep.VALIDATING_AFTER, 'complete')

            self._update_progress(TrainingStep.SAVING_MODEL, 'start')

            training_info = {
                'corrections_used': len(outcomes),
                'outcomes_processed': count,
                'epochs': epochs,
                'training_time_seconds': 0.0,
                'ewc_enabled': not disable_ewc
            }

            # Updated notes to reflect EWC status and epochs
            ewc_note = " (EWC disabled)" if disable_ewc else " with EWC"
            epoch_note = f", {epochs} epoch(s)" if epochs > 1 else ""
            notes = f'Online learning{ewc_note}: {len(outcomes)} patient outcomes{epoch_note}'

            new_version = self.manager.register_new_version(
                model=model,
                performance_metrics=after if validate else {'avg_reward': 0.0, 'accuracy': 0.0, 'diversity': 0,
                                                            'success_rate': 0.0},
                training_info=training_info,
                parent_version=base_version,
                notes=notes
            )

            self._status.version_number = new_version

            if self.verbose:
                print(f"[OnlineLearningPipeline] New version registered: {new_version}")

            self._update_progress(TrainingStep.SAVING_MODEL, 'complete')

            self._status.is_training = False

            model_info = self.manager.get_model_info(new_version)
            files = {
                'model_file_path': model_info.get('model_file_path'),
                'metadata_path': self.manager.metadata_file
            }

            result = TrainingResult(
                success=True,
                version_number=new_version,
                outcomes_processed=count,
                performance_before=before if validate else None,
                performance_after=after if validate else None,
                timestamp=get_timestamp(),
                model_files=files
            )

            if self.verbose:
                print("\n" + "=" * 80)
                print("[OnlineLearningPipeline] ===== TRAINING COMPLETE =====")
                print("=" * 80 + "\n")
            else:
                print(f"[OnlineLearningPipeline] Training complete: Version {new_version} created\n")

            return result

        except Exception as e:
            error_msg = str(e)

            if self.verbose:
                print(f"[OnlineLearningPipeline] Training failed: {error_msg}\n")
            else:
                print(f"[OnlineLearningPipeline] Training failed: {error_msg}\n")

            self._update_progress(TrainingStep.PARTIAL_FIT, 'error', error=error_msg)

            return TrainingResult(
                success=False,
                version_number=self._status.version_number,
                outcomes_processed=0,
                performance_before=None,
                performance_after=None,
                timestamp=get_timestamp(),
                error=error_msg
            )

def create_online_learning_pipeline(model_manager: ModelManager,
                                    verbose: bool = False) -> OnlineLearningPipeline:
    """
    Factory function to create a new online learning pipeline instance.

    Args:
        model_manager: ModelManager instance for all model operations
        verbose: Enable detailed logging

    Returns:
        New OnlineLearningPipeline instance with EWC support

    Example:
        from treatment_recommender.registry import create_model_manager

        manager = create_model_manager()

        pipeline = create_online_learning_pipeline(
            model_manager=manager,
            verbose=False
        )

        outcomes = [...]
        result = pipeline.partial_fit(outcomes)
    """
    return OnlineLearningPipeline(
        model_manager=model_manager,
        verbose=verbose
    )