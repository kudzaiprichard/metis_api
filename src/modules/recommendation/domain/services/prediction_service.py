"""
Prediction service for generating AI-powered treatment recommendations.
"""

from decimal import Decimal

from src.modules.recommendation.domain.models.prediction import Prediction
from src.modules.recommendation.domain.models.prediction_q_value import PredictionQValue
from src.modules.recommendation.domain.models.prediction_explanation import PredictionExplanation
from src.modules.recommendation.domain.models.explanation_feature import ExplanationFeature
from src.modules.recommendation.domain.models.explanation_alternative import ExplanationAlternative
from src.modules.recommendation.domain.models.safety_warning import SafetyWarning
from src.modules.recommendation.domain.models.enums import (
    Treatment,
    ConfidenceLevel,
    ClinicalPriority,
    SafetySeverity
)
from src.modules.recommendation.presentation.dtos.prediction_dtos import (
    GeneratePredictionRequest,
    PredictionDetailResponse,
    PatientSummaryResponse
)
from src.modules.recommendation.domain.repositories.prediction_repository import PredictionRepository
from src.modules.recommendation.domain.repositories.prediction_q_value_repository import PredictionQValueRepository
from src.modules.recommendation.domain.repositories.prediction_explanation_repository import PredictionExplanationRepository
from src.modules.recommendation.domain.repositories.explanation_feature_repository import ExplanationFeatureRepository
from src.modules.recommendation.domain.repositories.explanation_alternative_repository import ExplanationAlternativeRepository
from src.modules.recommendation.domain.repositories.safety_warning_repository import SafetyWarningRepository

from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.patient_medical_data_repository import PatientMedicalDataRepository

from src.shared.exceptions.exceptions import NotFoundException, ValidationException, ConflictException
from src.shared.response.error_detail import ErrorDetail
from src.shared.ml.service_initializer import get_ml_service


class PredictionService:
    """
    Service for generating AI-powered treatment recommendations.
    Each prediction is tied to a specific medical data snapshot.
    Uses ACTIVE MODEL ONLY for production predictions.
    """

    def __init__(self):
        self.prediction_repository = PredictionRepository()
        self.q_value_repository = PredictionQValueRepository()
        self.explanation_repository = PredictionExplanationRepository()
        self.feature_repository = ExplanationFeatureRepository()
        self.alternative_repository = ExplanationAlternativeRepository()
        self.safety_warning_repository = SafetyWarningRepository()
        self.patient_repository = PatientRepository()
        self.medical_data_repository = PatientMedicalDataRepository()

        self.ml_service = get_ml_service()

    # ==========================================================================
    # ENUM MAPPING HELPERS
    # ==========================================================================

    def _normalize_treatment_name(self, treatment_str: str) -> str:
        """Normalize treatment name to match Treatment enum."""
        normalized = treatment_str.upper().strip().replace('-', '').replace(' ', '')

        if 'GLP' in normalized:
            return 'GLP1'
        elif 'SGLT' in normalized:
            return 'SGLT2'
        elif 'DPP' in normalized:
            return 'DPP4'
        elif 'INSULIN' in normalized:
            return 'INSULIN'
        elif 'METFORMIN' in normalized:
            return 'METFORMIN'

        return normalized

    def _map_severity_to_enum(self, severity_str: str) -> SafetySeverity:
        """Map severity string to SafetySeverity enum with fallback."""
        severity_map = {
            'HIGH': SafetySeverity.CRITICAL,
            'CRITICAL': SafetySeverity.CRITICAL,
            'SEVERE': SafetySeverity.CRITICAL,
            'MODERATE': SafetySeverity.WARNING,
            'MEDIUM': SafetySeverity.WARNING,
            'WARNING': SafetySeverity.WARNING,
            'LOW': SafetySeverity.CAUTION,
            'MILD': SafetySeverity.CAUTION,
            'MINOR': SafetySeverity.CAUTION,
            'CAUTION': SafetySeverity.CAUTION,
            'INFO': SafetySeverity.INFO,
            'INFORMATIONAL': SafetySeverity.INFO
        }
        severity_upper = severity_str.upper() if severity_str else 'WARNING'
        return severity_map.get(severity_upper, SafetySeverity.WARNING)

    def _map_confidence_to_enum(self, confidence_str: str) -> ConfidenceLevel:
        """Map confidence string to ConfidenceLevel enum with fallback."""
        confidence_map = {
            'CRITICAL': ConfidenceLevel.CRITICAL,
            'VERY_LOW': ConfidenceLevel.CRITICAL,
            'LOW': ConfidenceLevel.LOW,
            'MODERATE': ConfidenceLevel.MODERATE,
            'MEDIUM': ConfidenceLevel.MODERATE,
            'HIGH': ConfidenceLevel.HIGH,
            'VERY_HIGH': ConfidenceLevel.VERY_HIGH,
            'VERYHIGH': ConfidenceLevel.VERY_HIGH
        }
        confidence_upper = confidence_str.upper().replace(' ', '_').replace('-', '_') if confidence_str else 'MODERATE'
        return confidence_map.get(confidence_upper, ConfidenceLevel.MODERATE)

    def _map_priority_to_enum(self, priority_str: str) -> ClinicalPriority:
        """Map priority string to ClinicalPriority enum with fallback."""
        priority_map = {
            'ROUTINE': ClinicalPriority.ROUTINE,
            'LOW': ClinicalPriority.ROUTINE,
            'STANDARD': ClinicalPriority.STANDARD,
            'NORMAL': ClinicalPriority.STANDARD,
            'MODERATE': ClinicalPriority.STANDARD,
            'URGENT': ClinicalPriority.URGENT,
            'HIGH': ClinicalPriority.URGENT,
            'CRITICAL': ClinicalPriority.CRITICAL,
            'EMERGENCY': ClinicalPriority.CRITICAL
        }
        priority_upper = priority_str.upper() if priority_str else 'STANDARD'
        return priority_map.get(priority_upper, ClinicalPriority.STANDARD)

    # ==========================================================================
    # PATIENT SUMMARY HELPER
    # ==========================================================================

    def _build_patient_summary(self, patient, medical_data) -> PatientSummaryResponse:
        """Build patient summary from patient and medical data entities."""
        return PatientSummaryResponse(
            id=patient.id,
            first_name=patient.first_name,
            last_name=patient.last_name,
            age=medical_data.age,
            gender=medical_data.gender.value
        )

    # ==========================================================================
    # GENERATE PREDICTION
    # ==========================================================================

    def generate_prediction(self, request: GeneratePredictionRequest,
                            created_by_user_id: str) -> PredictionDetailResponse:
        """
        Generate AI prediction for a specific medical data snapshot.

        Args:
            request: GeneratePredictionRequest DTO with medical_data_id
            created_by_user_id: ID of the user generating the prediction

        Returns:
            PredictionDetailResponse DTO with full prediction details

        Raises:
            NotFoundException: If medical data or patient not found
            ConflictException: If prediction already exists for this medical data
            ValidationException: If ML prediction fails
        """
        # ============================================
        # 1. VALIDATE MEDICAL DATA EXISTS
        # ============================================
        medical_data = self.medical_data_repository.find_by_id(request.medical_data_id)
        if not medical_data:
            error = ErrorDetail(
                title="Medical Data Not Found",
                code="MEDICAL_DATA_NOT_FOUND",
                status=404,
                details=[f"Medical data with ID {request.medical_data_id} does not exist"]
            )
            raise NotFoundException(
                message="The medical data record you're trying to predict for doesn't exist",
                error_detail=error
            )

        # ============================================
        # 2. CHECK PREDICTION DOESN'T ALREADY EXIST
        # ============================================
        if self.prediction_repository.exists_for_medical_data(request.medical_data_id):
            error = ErrorDetail(
                title="Prediction Already Exists",
                code="PREDICTION_EXISTS",
                status=409,
                details=["A prediction has already been generated for this medical data record"]
            )
            raise ConflictException(
                message="Prediction already exists for this medical data",
                error_detail=error
            )

        # ============================================
        # 3. VALIDATE PATIENT EXISTS
        # ============================================
        patient = self.patient_repository.find_by_id(medical_data.patient_id)
        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {medical_data.patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient linked to this medical data doesn't exist",
                error_detail=error
            )

        # ============================================
        # 4. EXTRACT PATIENT FEATURES (21 base features)
        # ============================================
        patient_features = {
            'age': medical_data.age,
            'gender': medical_data.gender.value,
            'ethnicity': medical_data.ethnicity.value,
            'hba1c_baseline': float(medical_data.hba1c_baseline),
            'diabetes_duration': float(medical_data.diabetes_duration),
            'fasting_glucose': float(medical_data.fasting_glucose),
            'c_peptide': float(medical_data.c_peptide),
            'egfr': float(medical_data.egfr),
            'bmi': float(medical_data.bmi),
            'bp_systolic': medical_data.bp_systolic,
            'bp_diastolic': medical_data.bp_diastolic,
            'alt': float(medical_data.alt),
            'ldl': float(medical_data.ldl),
            'hdl': float(medical_data.hdl),
            'triglycerides': float(medical_data.triglycerides),
            'previous_prediabetes': medical_data.previous_prediabetes,
            'hypertension': medical_data.hypertension,
            'ckd': medical_data.ckd,
            'cvd': medical_data.cvd,
            'nafld': medical_data.nafld,
            'retinopathy': medical_data.retinopathy
        }

        # ============================================
        # 5. GENERATE PREDICTION USING ACTIVE MODEL
        # ============================================
        try:
            ml_result = self.ml_service.predict_with_active_model(
                patient_features=patient_features,
                include_explanation=True
            )
        except Exception as e:
            error = ErrorDetail(
                title="Prediction Error",
                code="PREDICTION_FAILED",
                status=500,
                details=[f"ML prediction failed: {str(e)}"]
            )
            raise ValidationException(
                message="Failed to generate prediction. Please try again.",
                error_detail=error
            )

        # ============================================
        # 6. EXTRACT ML RESULTS
        # ============================================
        prediction_result = ml_result['prediction']
        explanation_result = ml_result['explanation']
        model_version_used = ml_result['model_version_used']

        # ============================================
        # 7. CONVERT ML RESULTS TO DB STRUCTURE
        # ============================================
        ai_result = self._convert_ml_result_to_dict(
            prediction_result,
            explanation_result,
            model_version_used
        )

        # ============================================
        # 8. CREATE PREDICTION RECORD
        # ============================================
        prediction = Prediction(
            medical_data_id=request.medical_data_id,
            created_by=created_by_user_id,
            model_version=ai_result['model_version'],
            recommended_treatment=Treatment[self._normalize_treatment_name(ai_result['recommended_treatment'])],
            treatment_index=ai_result['treatment_index'],
            predicted_reduction=Decimal(str(ai_result['predicted_reduction'])),
            confidence_score=Decimal(str(ai_result['confidence_score'])),
            confidence_margin=Decimal(str(ai_result['confidence_margin']))
        )
        saved_prediction = self.prediction_repository.create(prediction)

        # ============================================
        # 9. CREATE Q-VALUES (all 5 treatments)
        # ============================================
        q_value_records = []
        for qv in ai_result['q_values']:
            q_value = PredictionQValue(
                prediction_id=saved_prediction.id,
                treatment=Treatment[self._normalize_treatment_name(qv['treatment'])],
                q_value=Decimal(str(qv['q_value'])),
                rank=qv['rank']
            )
            q_value_records.append(q_value)
        self.q_value_repository.create_many(q_value_records)

        # ============================================
        # 10. CREATE EXPLANATION
        # ============================================
        if ai_result.get('explanation'):
            exp_data = ai_result['explanation']

            explanation = PredictionExplanation(
                prediction_id=saved_prediction.id,
                summary_text=exp_data['summary_text'],
                confidence_level=self._map_confidence_to_enum(exp_data['confidence_level']),
                clinical_priority=self._map_priority_to_enum(exp_data['clinical_priority']),
                why_this_treatment=exp_data['why_this_treatment'],
                why_not_alternatives=exp_data['why_not_alternatives'],
                base_value=Decimal(str(exp_data['base_value'])),
                prediction_value=Decimal(str(exp_data['prediction_value'])),
                feature_interactions=exp_data.get('feature_interactions')
            )
            saved_explanation = self.explanation_repository.create(explanation)

            # Create top features
            feature_records = []
            for feat in exp_data.get('top_features', []):
                feature = ExplanationFeature(
                    explanation_id=saved_explanation.id,
                    feature_name=feat['feature_name'],
                    scaled_value=Decimal(str(feat['scaled_value'])),
                    raw_value=Decimal(str(feat['raw_value'])),
                    shap_value=Decimal(str(feat['shap_value'])),
                    rank=feat['rank'],
                    interpretation=feat['interpretation'],
                    reference_range=feat.get('reference_range')
                )
                feature_records.append(feature)
            if feature_records:
                self.feature_repository.create_many(feature_records)

            # Create alternatives
            alternative_records = []
            for alt in exp_data.get('alternatives', []):
                alternative = ExplanationAlternative(
                    explanation_id=saved_explanation.id,
                    rank=alt['rank'],
                    treatment=Treatment[self._normalize_treatment_name(alt['treatment'])],
                    predicted_reduction=Decimal(str(alt['predicted_reduction'])),
                    pros=alt['pros'],
                    cons=alt['cons'],
                    when_to_consider=alt['when_to_consider']
                )
                alternative_records.append(alternative)
            if alternative_records:
                self.alternative_repository.create_many(alternative_records)

        # ============================================
        # 11. CREATE SAFETY WARNINGS
        # ============================================
        warning_records = []
        for warn in ai_result.get('safety_warnings', []):
            warning = SafetyWarning(
                prediction_id=saved_prediction.id,
                severity=self._map_severity_to_enum(warn['severity']),
                concern=warn['concern'],
                patient_factor=warn['patient_factor'],
                mitigation=warn['mitigation'],
                reason=warn.get('reason')
            )
            warning_records.append(warning)
        if warning_records:
            self.safety_warning_repository.create_many(warning_records)

        # ============================================
        # 12. BUILD AND RETURN RESPONSE
        # ============================================
        patient_summary = self._build_patient_summary(patient, medical_data)

        response_dict = saved_prediction.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return PredictionDetailResponse(**response_dict)

    # ==========================================================================
    # ML RESULT CONVERSION
    # ==========================================================================

    def _convert_ml_result_to_dict(self, prediction_result, explanation_result, model_version):
        """Convert ML module results to database-compatible dictionary structure."""
        # Extract Q-values and create ranked list
        q_values = []
        for treatment, q_value in prediction_result.all_q_values.items():
            rank = next(
                (item['rank'] for item in prediction_result.ranked_treatments
                 if item['treatment'] == treatment),
                999
            )
            q_values.append({
                'treatment': self._normalize_treatment_name(treatment),
                'q_value': q_value,
                'rank': rank
            })

        # Build explanation structure
        explanation_dict = None
        if explanation_result:
            top_features = []
            for feat in explanation_result.feature_importance.top_features[:5]:
                top_features.append({
                    'feature_name': feat.feature,
                    'scaled_value': feat.value,
                    'raw_value': feat.raw_value,
                    'shap_value': feat.shap_value,
                    'rank': feat.importance_rank,
                    'interpretation': feat.interpretation,
                    'reference_range': feat.reference_range
                })

            alternatives = []
            for alt in explanation_result.alternatives.alternatives[:3]:
                alternatives.append({
                    'rank': alt.rank,
                    'treatment': self._normalize_treatment_name(alt.treatment),
                    'predicted_reduction': alt.predicted_reduction,
                    'pros': ', '.join(alt.pros) if isinstance(alt.pros, list) else alt.pros,
                    'cons': ', '.join(alt.cons) if isinstance(alt.cons, list) else alt.cons,
                    'when_to_consider': alt.when_to_consider
                })

            feature_interactions = None
            if explanation_result.model_reasoning.key_factors:
                feature_interactions = explanation_result.model_reasoning.key_factors[0].evidence

            explanation_dict = {
                'summary_text': explanation_result.summary.one_sentence,
                'confidence_level': explanation_result.summary.confidence_level,
                'clinical_priority': explanation_result.summary.clinical_priority,
                'why_this_treatment': explanation_result.model_reasoning.why_this_treatment,
                'why_not_alternatives': explanation_result.alternatives.why_not_alternatives,
                'base_value': explanation_result.feature_importance.base_value,
                'prediction_value': prediction_result.predicted_hba1c_reduction,
                'feature_interactions': feature_interactions,
                'top_features': top_features,
                'alternatives': alternatives
            }

        # Build safety warnings
        safety_warnings = []
        if explanation_result and explanation_result.safety_checks.warnings:
            for warning in explanation_result.safety_checks.warnings:
                safety_warnings.append({
                    'severity': warning.severity,
                    'concern': warning.concern,
                    'patient_factor': warning.patient_factor,
                    'mitigation': warning.mitigation,
                    'reason': warning.reason
                })

        return {
            'model_version': model_version,
            'recommended_treatment': self._normalize_treatment_name(prediction_result.recommended_treatment),
            'treatment_index': prediction_result.treatment_index,
            'predicted_reduction': prediction_result.predicted_hba1c_reduction,
            'confidence_score': prediction_result.confidence_score,
            'confidence_margin': prediction_result.confidence_margin,
            'q_values': q_values,
            'explanation': explanation_dict,
            'safety_warnings': safety_warnings
        }