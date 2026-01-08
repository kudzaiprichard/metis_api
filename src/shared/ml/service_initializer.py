"""
ML Service Initializer - Stateless design for reliable version selection.
"""
from typing import List, Dict, Optional

from src.shared.data.neo4j.neo4j_manager import get_neo4j_manager
from src.shared.treatment_recommender.explainability import create_gemini_provider, create_explainer
from src.shared.treatment_recommender.pipelines import create_prediction_pipeline, create_online_learning_pipeline
from src.shared.treatment_recommender.registry import create_model_manager


class MLServiceManager:
    """
    Stateless ML service manager - loads models per-request for guaranteed version accuracy.

    Design Philosophy:
    - Active model cached for default/fast path
    - Specific versions loaded fresh per-request (no cache)
    - Perfect for A/B testing and online learning validation
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # Core components (always available)
            self.model_manager = None
            self._app = None
            self._ml_config = None
            self._gemini_provider = None
            self._graph_db = None

            # Active model cache ONLY (for fast default predictions)
            self._active_pipeline = None
            self._active_explainer = None
            self._active_version = None

            self._initialized = True

    def initialize(self, app):
        """
        Initialize ML services from Flask app config.

        Args:
            app: Flask application instance

        Raises:
            RuntimeError: If Neo4j is not available (critical dependency)
        """
        self._app = app
        ml_config = app.config['ML_CONFIG']
        self._ml_config = ml_config
        gemini_config = app.config['GEMINI_CONFIG']

        # ============================================
        # Initialize Model Manager (always required)
        # ============================================
        app.logger.info("Initializing ML Model Manager...")
        self.model_manager = create_model_manager(
            models_dir=ml_config['models_dir'],
            scaler_path=ml_config['scaler_path'],
            metadata_path=ml_config['preprocessing_metadata_path'],
            auto_activate=ml_config['auto_activate'],
            verbose=ml_config['verbose']
        )

        status = self.model_manager.get_status()
        self._active_version = status['active_version']
        app.logger.info(f"Active Model Version: {self._active_version}")
        app.logger.info(f"Available Versions: {', '.join(status['available_versions'])}")

        # ============================================
        # Initialize Shared Resources (reused across versions)
        # ============================================
        app.logger.info("Initializing shared ML resources...")

        # Gemini LLM provider (shared)
        self._gemini_provider = create_gemini_provider(
            api_key=gemini_config['api_key'],
            model_name=gemini_config['model'],
            timeout=100
        )

        # Get shared Neo4j from Neo4jManager (already initialized and verified)
        neo4j_manager = get_neo4j_manager()

        if not neo4j_manager.is_available():
            raise RuntimeError(
                "CRITICAL: Neo4j Manager is not available. "
                "ML services cannot initialize without Neo4j. "
                "This should not happen if application startup succeeded."
            )

        self._graph_db = neo4j_manager.get_database()
        app.logger.info("ML Service using shared Neo4j connection")

        # ============================================
        # Load Active Model (cached for fast default path)
        # ============================================
        app.logger.info(f"Loading active model pipeline ({self._active_version})...")
        self._active_pipeline = create_prediction_pipeline(
            model_manager=self.model_manager,
            version=None,  # Uses active
            verbose=ml_config['verbose']
        )

        app.logger.info("Loading active model explainer...")
        model = self.model_manager.get_active_model()
        processor = self.model_manager.get_feature_processor()

        self._active_explainer = create_explainer(
            model=model,
            feature_processor=processor,
            llm_provider=self._gemini_provider,
            graph_db=self._graph_db
        )

        app.logger.info("ML Services initialized successfully (stateless mode)")

    # ============================================
    # ONLINE LEARNING
    # ============================================

    def perform_online_learning(self,
                                outcomes: List[Dict],
                                base_version: str,  # REQUIRED - no default
                                validate: bool = True,
                                disable_ewc: bool = False,
                                epochs: int = 1):
        """
        Perform online learning on patient outcomes to create new model version.

        Process:
        1. Load specified base model version
        2. Train on patient outcomes using partial_fit with EWC
        3. Validate performance before/after (optional)
        4. Register new model version
        5. Return training results

        Note: Does NOT auto-activate the new version. Use switch_active_version()
        manually to promote the new model to production.

        Args:
            outcomes: List of outcome dicts with keys:
                     'patient': patient_dict (21 base features)
                     'treatment_given': treatment name (e.g., 'Insulin')
                     'reward': observed HbA1c reduction
            base_version: Base version to train from (e.g., 'v1_0') - REQUIRED
            validate: If True, validate performance before/after training
            disable_ewc: If True, disable EWC for unrestricted learning (default: False)
                        Use for testing or experimentation
            epochs: Number of training epochs (default: 1)
                   Higher values = more learning but risk of overfitting

        Returns:
            TrainingResult with:
            - success: bool
            - version_number: str (new version if successful)
            - outcomes_processed: int
            - performance_before: dict (if validate=True)
            - performance_after: dict (if validate=True)
            - timestamp: str
            - error: str (if failed)
            - model_files: dict (paths to saved files)

        Raises:
            RuntimeError: If ML services not initialized
            ValueError: If base_version doesn't exist

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

            # Train with EWC (recommended)
            result = ml_service.perform_online_learning(
                outcomes=outcomes,
                base_version='v1_0',
                validate=True
            )

            if result.success:
                print(f"New version: {result.version_number}")
                # Manually activate when ready
                ml_service.switch_active_version(result.version_number)

            # Train without EWC (testing)
            result = ml_service.perform_online_learning(
                outcomes=outcomes,
                base_version='v1_0',
                disable_ewc=True,
                epochs=20
            )
        """
        if self.model_manager is None:
            raise RuntimeError("ML services not initialized. Call initialize() first.")

        # Validate base_version exists (REQUIRED)
        available = [v['version_number'] for v in self.model_manager.list_versions()]
        if base_version not in available:
            raise ValueError(
                f"Base version '{base_version}' not found. "
                f"Available versions: {', '.join(available)}"
            )

        if self._app:
            self._app.logger.info(
                f"Starting online learning: {len(outcomes)} outcomes, "
                f"base={base_version}, epochs={epochs}, ewc={'off' if disable_ewc else 'on'}"
            )

        # Create online learning pipeline
        pipeline = create_online_learning_pipeline(
            model_manager=self.model_manager,
            verbose=False
        )

        # Perform training
        result = pipeline.partial_fit(
            outcomes=outcomes,
            base_version=base_version,
            validate=validate,
            disable_ewc=disable_ewc,
            epochs=epochs
        )

        if result.success:
            if self._app:
                self._app.logger.info(
                    f"Online learning complete: New version {result.version_number} created. "
                    f"Use switch_active_version() to activate."
                )
        else:
            if self._app:
                self._app.logger.error(f"Online learning failed: {result.error}")

        return result

    # ============================================
    # PRIMARY PREDICTION METHODS
    # ============================================

    def predict_with_active_model(self, patient_features, include_explanation=True):
        """
        Predict using the currently active model (FAST PATH - uses cache).

        Use this for:
        - Production predictions
        - Default recommendations
        - When you want the "current best" model

        Args:
            patient_features: Patient data dict (21 base features)
            include_explanation: Whether to generate explanation

        Returns:
            dict: {
                'prediction': TreatmentResult,
                'explanation': ExplanationResult or None,
                'model_version_used': str
            }
        """
        if self._active_pipeline is None:
            raise RuntimeError("ML services not initialized. Call initialize() first.")

        # Use cached active model (fast)
        prediction_result = self._active_pipeline.predict(patient_features)

        explanation_result = None
        if include_explanation:
            explanation_result = self._active_explainer.explain(
                model_result=prediction_result,
                patient_data=patient_features
            )

        return {
            'prediction': prediction_result,
            'explanation': explanation_result,
            'model_version_used': self._active_version
        }

    def predict_with_specific_version(self, patient_features, version, include_explanation=True):
        """
        Predict using a SPECIFIC model version (STATELESS - no cache).

        Use this for:
        - A/B testing different model versions
        - Validating newly trained models
        - Comparing performance across versions
        - Online learning validation

        IMPORTANT: This loads the model fresh every time to guarantee version accuracy.
        Slightly slower but 100% reliable for critical version selection.

        Args:
            patient_features: Patient data dict (21 base features)
            version: Exact model version to use (e.g., 'v1_3')
            include_explanation: Whether to generate explanation

        Returns:
            dict: {
                'prediction': TreatmentResult,
                'explanation': ExplanationResult or None,
                'model_version_used': str
            }

        Raises:
            ValueError: If version doesn't exist
        """
        if self.model_manager is None:
            raise RuntimeError("ML services not initialized. Call initialize() first.")

        # Validate version exists
        available = [v['version_number'] for v in self.model_manager.list_versions()]
        if version not in available:
            raise ValueError(
                f"Model version '{version}' not found. "
                f"Available versions: {', '.join(available)}"
            )

        if self._app:
            self._app.logger.info(f"Loading model version {version} for prediction (stateless)")

        # Load fresh pipeline for this version (NO CACHE)
        pipeline = create_prediction_pipeline(
            model_manager=self.model_manager,
            version=version,
            verbose=False
        )

        # Make prediction
        prediction_result = pipeline.predict(patient_features)

        # Generate explanation if requested
        explanation_result = None
        if include_explanation:
            # Load fresh explainer for this version (NO CACHE)
            model = self.model_manager.get_model_by_version(version)
            processor = self.model_manager.get_feature_processor()

            explainer = create_explainer(
                model=model,
                feature_processor=processor,
                llm_provider=self._gemini_provider,
                graph_db=self._graph_db
            )

            explanation_result = explainer.explain(
                model_result=prediction_result,
                patient_data=patient_features
            )

        if self._app:
            self._app.logger.info(f"Completed prediction with version {version}")

        return {
            'prediction': prediction_result,
            'explanation': explanation_result,
            'model_version_used': version
        }

    # ============================================
    # VERSION MANAGEMENT
    # ============================================

    def switch_active_version(self, version):
        """
        Switch the active model version and reload active cache.

        Use this when:
        - Promoting a new model to production
        - Rolling back to a previous version
        - Changing default prediction model

        Args:
            version: Version to activate (e.g., 'v1_3')

        Returns:
            dict: Status of the switch operation
        """
        if self.model_manager is None:
            raise RuntimeError("ML services not initialized.")

        # Validate version exists
        available = [v['version_number'] for v in self.model_manager.list_versions()]
        if version not in available:
            raise ValueError(f"Version '{version}' not found. Available: {', '.join(available)}")

        old_version = self._active_version

        if self._app:
            self._app.logger.info(f"Switching active version: {old_version} to {version}")

        # Update registry
        self.model_manager.activate_version(version)
        self._active_version = version

        # Reload active cache
        if self._app:
            self._app.logger.info("Reloading active model cache...")

        self._active_pipeline = create_prediction_pipeline(
            model_manager=self.model_manager,
            version=None,  # Uses new active
            verbose=self._ml_config.get('verbose', False)
        )

        model = self.model_manager.get_active_model()
        processor = self.model_manager.get_feature_processor()

        self._active_explainer = create_explainer(
            model=model,
            feature_processor=processor,
            llm_provider=self._gemini_provider,
            graph_db=self._graph_db
        )

        if self._app:
            self._app.logger.info(f"Successfully switched to version {version}")

        # Get performance comparison
        old_info = self.model_manager.get_model_info(old_version)
        new_info = self.model_manager.get_model_info(version)

        return {
            'previous_version': old_version,
            'current_version': version,
            'status': 'switched successfully',
            'performance_comparison': {
                'old': old_info.get('performance_metrics', {}) if old_info else {},
                'new': new_info.get('performance_metrics', {}) if new_info else {}
            }
        }

    # ============================================
    # STATUS & UTILITIES
    # ============================================

    def get_current_status(self):
        """Get comprehensive status of ML services"""
        if self.model_manager is None:
            return {'status': 'not_initialized'}

        registry_status = self.model_manager.get_status()
        version_info = self.model_manager.get_model_info(self._active_version)

        return {
            'status': 'initialized',
            'mode': 'stateless',
            'active_version': self._active_version,
            'active_version_cached': True,
            'available_versions': registry_status['available_versions'],
            'total_versions': registry_status['total_versions'],
            'performance_metrics': version_info.get('performance_metrics', {}) if version_info else {},
            'version_details': version_info if version_info else {}
        }

    def list_all_versions(self, sort_by='version'):
        """List all available model versions"""
        if self.model_manager is None:
            raise RuntimeError("ML services not initialized.")
        return self.model_manager.list_versions(sort_by=sort_by, reverse=True)

    def compare_versions(self, version1, version2):
        """Compare performance between two versions"""
        if self.model_manager is None:
            raise RuntimeError("ML services not initialized.")
        return self.model_manager.compare_versions(version1, version2)

    def get_model_manager(self):
        """Get model manager instance"""
        if self.model_manager is None:
            raise RuntimeError("ML services not initialized.")
        return self.model_manager


def get_ml_service():
    """Get ML service manager singleton"""
    return MLServiceManager()