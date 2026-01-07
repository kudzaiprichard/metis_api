"""
Main explainability orchestrator.

This module coordinates feature attribution, graph database queries,
and LLM synthesis to produce complete clinical explanations.
"""

import time
from typing import Dict, Any

from ._base import (
    GraphDatabaseInterface,
    ExplanationResult,
    FeatureAttribution,
    DEFAULT_TOP_FEATURES,
)
from .providers._base import BaseLLMProvider
from ._feature_attribution import (
    calculate_shap_values,
    extract_top_features,
    create_feature_attributions,
)
from ._llm_synthesizer import LLMSynthesizer, create_llm_synthesizer


# =============================================================================
# TREATMENT EXPLAINER
# =============================================================================

class TreatmentExplainer:
    """
    Main explainability orchestrator for diabetes treatment recommendations.

    Coordinates three components:
    1. Feature Attribution (SHAP) - Which features drove the decision?
    2. Graph Database - What does clinical evidence say?
    3. LLM Synthesis - How to explain this to a doctor?

    Usage:
        explainer = TreatmentExplainer(
            model=trained_model,
            feature_processor=processor,
            graph_db=my_graph_db,
            llm_synthesizer=synthesizer
        )

        explanation = explainer.explain(
            model_result=prediction_result,
            patient_data=patient_dict
        )

        print(explanation.summary.one_sentence)
        print(explanation.to_json())
    """

    def __init__(self,
                 model: Any,
                 feature_processor: Any,
                 graph_db: GraphDatabaseInterface,
                 llm_synthesizer: LLMSynthesizer,
                 top_features: int = DEFAULT_TOP_FEATURES,
                 n_background_samples: int = 100,
                 verbose: bool = False):
        """
        Initialize treatment explainer.

        Args:
            model: Trained NeuralTLearner model
            feature_processor: PatientFeatureProcessor instance
            graph_db: Graph database implementation (REQUIRED)
            llm_synthesizer: LLM synthesizer instance
            top_features: Number of top features to extract (default: 5)
            n_background_samples: Number of patients for SHAP background (default: 100)
            verbose: If True, print detailed logs

        Raises:
            ValueError: If graph_db is None (graph database is REQUIRED)
        """
        # Validate required components
        if graph_db is None:
            raise ValueError(
                "Graph database is REQUIRED for clinical explanations. "
                "Please implement GraphDatabaseInterface and provide it here. "
                "See explainability/graph_interface.py for implementation guide."
            )

        self.model = model
        self.feature_processor = feature_processor
        self.graph_db = graph_db
        self.llm_synthesizer = llm_synthesizer
        self.top_features = top_features
        self.verbose = verbose

        # Get feature names from processor
        self.feature_names = feature_processor.get_feature_names()

        if self.verbose:
            print("[TreatmentExplainer] Initialized")
            print(f"[TreatmentExplainer] Model: {type(model).__name__}")
            print(f"[TreatmentExplainer] Features: {len(self.feature_names)}")
            print(f"[TreatmentExplainer] Graph DB: {type(graph_db).__name__}")
            print(f"[TreatmentExplainer] LLM: {llm_synthesizer.llm_provider.provider_name}")

        # Load background data from graph database for SHAP
        if self.verbose:
            print(f"[TreatmentExplainer] Loading {n_background_samples} background samples from graph database...")

        try:
            # Get raw patient dictionaries from graph database
            background_patients = self.graph_db.get_background_data_for_shap(n_background_samples)

            if not background_patients:
                raise ValueError("Graph database returned no background data")

            if self.verbose:
                print(f"[TreatmentExplainer] Retrieved {len(background_patients)} patients from database")
                print(f"[TreatmentExplainer] Processing background data through feature processor...")

            # Process through feature processor to get numpy arrays
            self.background_data = self.feature_processor.process_batch(background_patients)

            if self.verbose:
                print(f"[TreatmentExplainer] Background data shape: {self.background_data.shape}")
                print(f"[TreatmentExplainer] Background data ready for SHAP calculations")

        except Exception as e:
            raise ValueError(
                f"Failed to load background data from graph database: {str(e)}. "
                "Ensure your GraphDatabaseInterface implements get_background_data_for_shap()."
            )

    def explain(self,
                model_result: Any,
                patient_data: Dict[str, Any]) -> ExplanationResult:
        """
        Generate complete explanation for a treatment recommendation.

        Pipeline:
        1. Calculate SHAP values for feature attribution (using graph DB background data)
        2. Query graph database for clinical context
        3. Synthesize explanation via LLM
        4. Return structured ExplanationResult

        Args:
            model_result: TreatmentResult from prediction pipeline
            patient_data: Original patient data dictionary

        Returns:
            ExplanationResult with complete structured explanation

        Example:
            # Step 1: Get prediction
            result = pipeline.predict(patient_data)

            # Step 2: Get explanation (background data loaded automatically)
            explanation = explainer.explain(
                model_result=result,
                patient_data=patient_data
            )

            # Step 3: Use explanation
            print(explanation.summary.one_sentence)
            print(f"Top factor: {explanation.model_reasoning.key_factors[0].factor}")

            # Export to JSON
            json_output = explanation.to_json()
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("[TreatmentExplainer] STARTING EXPLANATION GENERATION")
            print("=" * 80)
            print(f"Treatment: {model_result.recommended_treatment}")
            print(f"Predicted Reduction: {model_result.predicted_hba1c_reduction:.2f}%")
            print(f"Confidence: {model_result.confidence_score:.1f}%")

        total_start = time.time()

        # Step 1: Feature Attribution (SHAP) - uses self.background_data automatically
        if self.verbose:
            print("\n[TreatmentExplainer] Step 1/3: Calculating SHAP values...")

        shap_start = time.time()
        shap_data = self._calculate_feature_attribution(
            model_result=model_result,
            patient_data=patient_data
        )
        shap_time = int((time.time() - shap_start) * 1000)
        shap_data['calculation_time_ms'] = shap_time

        if self.verbose:
            print(f"[TreatmentExplainer] SHAP completed ({shap_time}ms)")
            print(f"[TreatmentExplainer] Top feature: {shap_data['top_features'][0].feature}")

        # Step 2: Query Graph Database
        if self.verbose:
            print("\n[TreatmentExplainer] Step 2/3: Querying graph database...")

        graph_start = time.time()
        graph_insights = self._query_graph_database(
            model_result=model_result,
            patient_data=patient_data
        )
        graph_time = int((time.time() - graph_start) * 1000)
        graph_insights['query_time_ms'] = graph_time

        if self.verbose:
            print(f"[TreatmentExplainer] Graph query completed ({graph_time}ms)")
            guidelines_count = len(graph_insights.get('guidelines', {}).get('guidelines', []))
            print(f"[TreatmentExplainer] Guidelines found: {guidelines_count}")

        # Step 3: LLM Synthesis
        if self.verbose:
            print("\n[TreatmentExplainer] Step 3/3: Synthesizing explanation via LLM...")

        explanation = self.llm_synthesizer.synthesize(
            model_result=model_result,
            shap_data=shap_data,
            graph_insights=graph_insights,
            patient_data=patient_data,
            model_version=getattr(self.model, '_model_version', 'unknown')
        )

        total_time = int((time.time() - total_start) * 1000)

        if self.verbose:
            print(f"\n[TreatmentExplainer] EXPLANATION COMPLETE")
            print(f"[TreatmentExplainer] Total time: {total_time}ms")
            print(f"[TreatmentExplainer]   - SHAP: {shap_time}ms")
            print(f"[TreatmentExplainer]   - Graph: {graph_time}ms")
            print(f"[TreatmentExplainer]   - LLM: {explanation.metadata.llm_generation_time_ms}ms")
            print("=" * 80 + "\n")

        return explanation

    def _calculate_feature_attribution(self,
                                       model_result: Any,
                                       patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate SHAP values and extract top features with both scaled and raw values.

        Uses self.background_data loaded from graph database during initialization.

        Args:
            model_result: Model prediction result
            patient_data: Patient data

        Returns:
            Dictionary with SHAP data:
            {
                'top_features': List[FeatureAttribution],
                'base_value': float,
                'prediction': float,
                'shap_values': np.ndarray
            }
        """
        # Preprocess patient data to get SCALED features for SHAP
        patient_features = self.feature_processor.process_patient(patient_data)

        # Extract RAW feature values for interpretation
        raw_values = self._extract_raw_feature_values(patient_data)

        # Get treatment index
        treatment_index = model_result.treatment_index

        # Calculate SHAP values using background data from graph database
        shap_values, base_value, prediction = calculate_shap_values(
            model=self.model,
            patient_features=patient_features,
            feature_names=self.feature_names,
            background_data=self.background_data,  # From graph DB
            treatment_index=treatment_index,
            verbose=self.verbose
        )

        # Extract top N features WITH BOTH scaled and raw values
        top_features_raw = extract_top_features(
            shap_values=shap_values,
            feature_names=self.feature_names,
            patient_features=patient_features,  # Scaled values
            raw_values=raw_values,  # Raw values
            top_n=self.top_features
        )

        # Create FeatureAttribution DTOs
        top_features = create_feature_attributions(
            top_features=top_features_raw,
            treatment_name=model_result.recommended_treatment
        )

        return {
            'top_features': top_features,
            'base_value': base_value,
            'prediction': prediction,
            'shap_values': shap_values
        }

    def _extract_raw_feature_values(self, patient_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract raw (unscaled) feature values from patient data.

        This maps original patient data to feature names that match what the model uses.
        These raw values are used for human-readable interpretations.

        Args:
            patient_data: Original patient data dictionary

        Returns:
            Dictionary mapping feature names to raw values

        Example:
            raw_values = {
                'age': 58.0,
                'bmi': 31.5,  # Actual BMI, not z-score
                'hba1c_baseline': 8.2,
                ...
            }
        """
        # Direct mappings from patient data
        raw_values = {
            # Demographics
            'age': float(patient_data.get('age', 0)),

            # Lab values
            'hba1c_baseline': float(patient_data.get('hba1c_baseline', 0)),
            'c_peptide': float(patient_data.get('c_peptide', 0)),
            'fasting_glucose': float(patient_data.get('fasting_glucose', 0)),
            'egfr': float(patient_data.get('egfr', 0)),

            # Physical measurements
            'bmi': float(patient_data.get('bmi', 0)),
            'bp_systolic': float(patient_data.get('bp_systolic', 0)),
            'bp_diastolic': float(patient_data.get('bp_diastolic', 0)),

            # Other labs
            'alt': float(patient_data.get('alt', 0)),
            'ldl': float(patient_data.get('ldl', 0)),
            'hdl': float(patient_data.get('hdl', 0)),
            'triglycerides': float(patient_data.get('triglycerides', 0)),

            # Disease history
            'diabetes_duration': float(patient_data.get('diabetes_duration', 0)),
            'previous_prediabetes': float(patient_data.get('previous_prediabetes', 0)),

            # Comorbidities (binary)
            'hypertension': float(patient_data.get('hypertension', 0)),
            'ckd': float(patient_data.get('ckd', 0)),
            'cvd': float(patient_data.get('cvd', 0)),
            'nafld': float(patient_data.get('nafld', 0)),
            'retinopathy': float(patient_data.get('retinopathy', 0)),
        }

        # Handle engineered features
        # These are computed features that may not be in raw patient data
        # We need to approximate them or calculate them from raw values

        # Glucose severity (approximation based on HbA1c)
        hba1c = patient_data.get('hba1c_baseline', 7.0)
        raw_values['glucose_severity'] = (hba1c - 7.0) * 20  # Rough approximation

        # Beta cell reserve (approximation based on C-peptide)
        c_peptide = patient_data.get('c_peptide', 1.0)
        raw_values['beta_cell_reserve'] = c_peptide / 2.0  # Rough approximation

        # Insulin deficiency score (approximation)
        raw_values['insulin_deficiency_score'] = max(0, (10 - hba1c) / 2.0 + (2.0 - c_peptide))

        # One-hot encoded features (gender, ethnicity)
        # These should match the scaled values as they're already 0/1
        gender = patient_data.get('gender', 'Male')
        if gender == 'Female':
            raw_values['gender_Female'] = 1.0
            raw_values['gender_Male'] = 0.0
        else:
            raw_values['gender_Female'] = 0.0
            raw_values['gender_Male'] = 1.0

        ethnicity = patient_data.get('ethnicity', 'Caucasian')
        raw_values['ethnicity_African_American'] = 1.0 if ethnicity == 'African American' else 0.0
        raw_values['ethnicity_Asian'] = 1.0 if ethnicity == 'Asian' else 0.0
        raw_values['ethnicity_Caucasian'] = 1.0 if ethnicity == 'Caucasian' else 0.0
        raw_values['ethnicity_Hispanic'] = 1.0 if ethnicity == 'Hispanic' else 0.0

        return raw_values

    def _query_graph_database(self,
                              model_result: Any,
                              patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query graph database for clinical context.

        Args:
            model_result: Model prediction result
            patient_data: Patient data

        Returns:
            Dictionary with graph insights:
            {
                'guidelines': Dict,
                'contraindications': List[Dict],
                'drug_interactions': List[Dict],
                'similar_cases': Dict
            }
        """
        treatment = model_result.recommended_treatment

        # Extract comorbidities
        comorbidities = []
        if patient_data.get('hypertension'):
            comorbidities.append('hypertension')
        if patient_data.get('ckd'):
            comorbidities.append('ckd')
        if patient_data.get('cvd'):
            comorbidities.append('cvd')
        if patient_data.get('nafld'):
            comorbidities.append('nafld')
        if patient_data.get('retinopathy'):
            comorbidities.append('retinopathy')

        # Query 1: Treatment Guidelines
        if self.verbose:
            print("[TreatmentExplainer] Querying treatment guidelines...")

        guidelines = self.graph_db.get_treatment_guidelines(
            treatment=treatment,
            patient_profile=patient_data
        )

        # Query 2: Contraindications
        if self.verbose:
            print("[TreatmentExplainer] Checking contraindications...")

        contraindications = self.graph_db.check_contraindications(
            treatment=treatment,
            patient_profile=patient_data
        )

        # Query 3: Drug Interactions
        if self.verbose:
            print("[TreatmentExplainer] Checking drug interactions...")

        drug_interactions = self.graph_db.get_drug_interactions(
            treatment=treatment,
            comorbidities=comorbidities
        )

        # Query 4: Similar Cases
        if self.verbose:
            print("[TreatmentExplainer] Finding similar patient cases...")

        similar_cases_raw = self.graph_db.find_similar_cases(
            patient_profile=patient_data,
            limit=5
        )

        # Format similar cases
        similar_cases = self._format_similar_cases(similar_cases_raw)

        return {
            'guidelines': guidelines,
            'contraindications': contraindications,
            'drug_interactions': drug_interactions,
            'similar_cases': similar_cases
        }

    def _format_similar_cases(self, cases: list) -> Dict[str, Any]:
        """
        Format similar cases from graph database.

        Args:
            cases: List of similar case dictionaries

        Returns:
            Formatted similar cases dictionary
        """
        if not cases:
            return {
                'found': 0,
                'average_outcome': {},
                'notable_case': {}
            }

        # Calculate average outcomes
        total_reduction = 0
        success_count = 0

        for case in cases:
            outcome = case.get('outcome', {})
            reduction = outcome.get('hba1c_reduction', 0)
            total_reduction += reduction
            if reduction >= 1.5:
                success_count += 1

        avg_reduction = total_reduction / len(cases) if cases else 0
        success_rate = success_count / len(cases) if cases else 0

        # Get notable case (highest similarity)
        notable_case = None
        if cases:
            notable = cases[0]  # First case (highest similarity)
            notable_case = {
                'patient': f"Similar {notable['profile']['age']}yo with HbA1c {notable['profile']['hba1c_baseline']:.1f}%, C-peptide {notable['profile']['c_peptide']:.2f}",
                'treatment': notable['treatment_given'],
                'outcome': f"HbA1c reduced to target in {notable['outcome'].get('time_to_target', 'N/A')}"
            }

        return {
            'found': len(cases),
            'average_outcome': {
                'hba1c_reduction': avg_reduction,
                'success_rate': success_rate,
                'time_to_target': '10-14 weeks'  # Could be calculated from cases
            },
            'notable_case': notable_case
        }

    def explain_batch(self,
                      model_results: list,
                      patient_data_list: list) -> list:
        """
        Generate explanations for multiple predictions.

        Background data from graph database is used automatically for all patients.

        Args:
            model_results: List of TreatmentResult objects
            patient_data_list: List of patient data dictionaries

        Returns:
            List of ExplanationResult objects

        Example:
            results = pipeline.predict_batch(patients)
            explanations = explainer.explain_batch(results, patients)

            for explanation in explanations:
                print(explanation.summary.one_sentence)
        """
        if len(model_results) != len(patient_data_list):
            raise ValueError("Number of results must match number of patient data entries")

        explanations = []

        for i, (result, patient) in enumerate(zip(model_results, patient_data_list)):
            if self.verbose:
                print(f"\n[TreatmentExplainer] Explaining {i + 1}/{len(model_results)}...")

            explanation = self.explain(
                model_result=result,
                patient_data=patient
            )

            explanations.append(explanation)

        return explanations

    def get_quick_summary(self,
                          model_result: Any,
                          patient_data: Dict[str, Any]) -> str:
        """
        Get a quick one-sentence summary without full explanation.

        Useful when you need fast feedback without waiting for LLM.
        Uses background data from graph database for SHAP calculation.

        Args:
            model_result: Model prediction result
            patient_data: Patient data

        Returns:
            One-sentence summary string

        Example:
            summary = explainer.get_quick_summary(result, patient)
            print(summary)
            # "Insulin recommended due to low C-peptide (0.4 ng/mL) and high HbA1c (11.5%)"
        """
        # Calculate SHAP quickly using background data
        patient_features = self.feature_processor.process_patient(patient_data)

        # Extract raw values for interpretation
        raw_values = self._extract_raw_feature_values(patient_data)

        shap_values, _, _ = calculate_shap_values(
            model=self.model,
            patient_features=patient_features,
            feature_names=self.feature_names,
            background_data=self.background_data,  # From graph DB
            treatment_index=model_result.treatment_index,
            verbose=False
        )

        # Get top 2 features WITH RAW VALUES
        top_features_raw = extract_top_features(
            shap_values=shap_values,
            feature_names=self.feature_names,
            patient_features=patient_features,
            raw_values=raw_values,  # Add raw values
            top_n=2
        )

        # Build simple summary using RAW values
        feat1_name, _, feat1_raw, _ = top_features_raw[0]  # Unpack 4-tuple
        feat2_name, _, feat2_raw, _ = top_features_raw[1]  # Unpack 4-tuple

        summary = (
            f"{model_result.recommended_treatment} recommended due to "
            f"{feat1_name.replace('_', ' ')} ({feat1_raw:.1f}) and "
            f"{feat2_name.replace('_', ' ')} ({feat2_raw:.1f})"
        )

        return summary


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_explainer(model: Any,
                     feature_processor: Any,
                     llm_provider: BaseLLMProvider,
                     graph_db: GraphDatabaseInterface,
                     top_features: int = DEFAULT_TOP_FEATURES,
                     temperature: float = 0.3,
                     n_background_samples: int = 100,
                     verbose: bool = False) -> TreatmentExplainer:
    """
    Factory function to create treatment explainer.

    This is the main entry point for creating explainability.

    Args:
        model: Trained NeuralTLearner model
        feature_processor: PatientFeatureProcessor instance
        llm_provider: LLM provider (Gemini, GPT, Claude)
        graph_db: Graph database implementation (REQUIRED)
        top_features: Number of top features to extract
        temperature: LLM temperature (0.0-1.0)
        n_background_samples: Number of patients for SHAP background (default: 100)
        verbose: Enable detailed logging

    Returns:
        Configured TreatmentExplainer instance

    Example:
        from treatment_recommender.pipelines import create_prediction_pipeline
        from treatment_recommender.preprocessing import create_feature_processor
        from treatment_recommender.explainability import create_explainer
        from treatment_recommender.explainability.providers import create_gemini_provider
        from my_app.graph_db import MyGraphDatabase

        # Setup
        processor = create_feature_processor()
        pipeline = create_prediction_pipeline(model_path='...', feature_processor=processor)
        gemini = create_gemini_provider(
            api_key='YOUR_KEY',
            model_name='gemini-1.5-pro-latest'
        )
        graph_db = MyGraphDatabase()

        # Create explainer (background data loaded automatically from graph_db)
        explainer = create_explainer(
            model=pipeline._model,
            feature_processor=processor,
            llm_provider=gemini,
            graph_db=graph_db
        )

        # Get prediction
        result = pipeline.predict(patient_data)

        # Get explanation (uses background data automatically)
        explanation = explainer.explain(
            model_result=result,
            patient_data=patient_data
        )

        print(explanation.summary.one_sentence)
    """
    # Validate graph_db is provided
    if graph_db is None:
        raise ValueError(
            "Graph database is REQUIRED for explainability. "
            "Please implement GraphDatabaseInterface. "
            "See explainability/graph_interface.py for guide."
        )

    # Create LLM synthesizer
    llm_synthesizer = create_llm_synthesizer(
        llm_provider=llm_provider,
        temperature=temperature,
        verbose=verbose
    )

    # Create and return explainer (background data loaded automatically)
    return TreatmentExplainer(
        model=model,
        feature_processor=feature_processor,
        graph_db=graph_db,
        llm_synthesizer=llm_synthesizer,
        top_features=top_features,
        n_background_samples=n_background_samples,
        verbose=verbose
    )