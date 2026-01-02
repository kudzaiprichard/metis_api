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
from src.modules.recommendation.domain.repositories.explanation_alternative_repository import \
    ExplanationAlternativeRepository
from src.modules.recommendation.domain.repositories.safety_warning_repository import SafetyWarningRepository

from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.patient_medical_data_repository import PatientMedicalDataRepository

from src.shared.exceptions.exceptions import NotFoundException
from src.shared.response.error_detail import ErrorDetail


class PredictionService:
    """
    Service for generating AI recommendation.
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

    def _build_patient_summary(self, patient_id: str) -> PatientSummaryResponse:
        """
        Build patient summary for prediction responses.

        Args:
            patient_id: Patient ID

        Returns:
            PatientSummaryResponse DTO
        """
        patient = self.patient_repository.find_by_id(patient_id)
        medical_data = self.medical_data_repository.find_by_patient_id(patient_id)

        return PatientSummaryResponse(
            id=patient.id,
            first_name=patient.first_name,
            last_name=patient.last_name,
            age=medical_data.age,
            gender=medical_data.gender.value
        )

    def generate_prediction(self, request: GeneratePredictionRequest,
                            created_by_user_id: str) -> PredictionDetailResponse:
        """
        Generate AI prediction for a patient.

        Args:
            request: GeneratePredictionRequest DTO
            created_by_user_id: ID of the user generating the prediction

        Returns:
            PredictionDetailResponse DTO with full prediction details

        Raises:
            NotFoundException: If patient not found or medical data missing
        """
        # 1. Validate patient exists
        patient = self.patient_repository.find_by_id(request.patient_id)
        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {request.patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're trying to generate prediction for doesn't exist",
                error_detail=error
            )

        # 2. Validate medical data exists
        medical_data = self.medical_data_repository.find_by_patient_id(request.patient_id)
        if not medical_data:
            error = ErrorDetail(
                title="Medical Data Not Found",
                code="MEDICAL_DATA_NOT_FOUND",
                status=404,
                details=[f"Medical data for patient ID {request.patient_id} does not exist"]
            )
            raise NotFoundException(
                message="Patient must have medical data before generating prediction",
                error_detail=error
            )

        # 3. Extract patient features (21 base features)
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

        # TODO: Pass patient_features to AI model and get prediction results
        # ai_result = ai_model.predict(patient_features)
        # For now, using mock data structure

        # Mock AI model response structure (replace with actual AI call)
        ai_result = {
            'model_version': 'v1.0.0',  # TODO: Get from active model
            'recommended_treatment': 'SGLT2',  # TODO: Get from model
            'treatment_index': 2,  # TODO: Get from model (0-4)
            'predicted_reduction': 2.5,  # TODO: Get from model
            'confidence_score': 85.5,  # TODO: Get from model
            'confidence_margin': 12.3,  # TODO: Get from model
            'q_values': [  # TODO: Get all 5 Q-values from model
                {'treatment': 'Metformin', 'q_value': 1.8, 'rank': 3},
                {'treatment': 'GLP-1', 'q_value': 2.3, 'rank': 2},
                {'treatment': 'SGLT-2', 'q_value': 2.5, 'rank': 1},
                {'treatment': 'DPP-4', 'q_value': 1.5, 'rank': 4},
                {'treatment': 'Insulin', 'q_value': 1.2, 'rank': 5}
            ],
            'explanation': {  # TODO: Get from SHAP/LLM explanation
                'summary_text': 'SGLT-2 inhibitor recommended based on patient profile',
                'confidence_level': 'high',
                'clinical_priority': 'standard',
                'why_this_treatment': 'Patient has good kidney function (eGFR 75) and would benefit from SGLT-2 cardiovascular protection.',
                'why_not_alternatives': 'GLP-1 was close second but SGLT-2 better for CKD prevention.',
                'base_value': 6.5,
                'prediction_value': 5.7,
                'feature_interactions': 'eGFR and HbA1c interaction suggests SGLT-2 efficacy',
                'top_features': [  # TODO: Get top 5 SHAP features
                    {
                        'feature_name': 'egfr',
                        'scaled_value': 0.5,
                        'raw_value': 75.0,
                        'shap_value': 0.35,
                        'rank': 1,
                        'interpretation': 'Good kidney function supports SGLT-2 use',
                        'reference_range': '60-120 mL/min/1.73m²'
                    },
                    {
                        'feature_name': 'hba1c_baseline',
                        'scaled_value': 1.2,
                        'raw_value': 8.2,
                        'shap_value': 0.28,
                        'rank': 2,
                        'interpretation': 'Elevated HbA1c indicates need for effective treatment',
                        'reference_range': '4.0-5.6%'
                    },
                    {
                        'feature_name': 'bmi',
                        'scaled_value': 0.8,
                        'raw_value': 31.5,
                        'shap_value': 0.22,
                        'rank': 3,
                        'interpretation': 'Overweight status benefits from SGLT-2 weight loss effect',
                        'reference_range': '18.5-24.9 kg/m²'
                    },
                    {
                        'feature_name': 'cvd',
                        'scaled_value': 0.0,
                        'raw_value': 0.0,
                        'shap_value': 0.15,
                        'rank': 4,
                        'interpretation': 'No CVD but SGLT-2 provides cardiovascular protection',
                        'reference_range': 'N/A'
                    },
                    {
                        'feature_name': 'age',
                        'scaled_value': 0.3,
                        'raw_value': 58.0,
                        'shap_value': 0.12,
                        'rank': 5,
                        'interpretation': 'Age appropriate for SGLT-2 therapy',
                        'reference_range': '18-120 years'
                    }
                ],
                'alternatives': [  # TODO: Get alternatives with pros/cons
                    {
                        'rank': 2,
                        'treatment': 'GLP-1',
                        'predicted_reduction': 2.3,
                        'pros': 'Weight loss, cardiovascular benefits',
                        'cons': 'Injection required, GI side effects',
                        'when_to_consider': 'If patient prefers injectable or needs more weight loss'
                    },
                    {
                        'rank': 3,
                        'treatment': 'Metformin',
                        'predicted_reduction': 1.8,
                        'pros': 'First-line, low cost, oral',
                        'cons': 'Lower efficacy at this HbA1c level',
                        'when_to_consider': 'If cost is primary concern'
                    }
                ]
            },
            'safety_warnings': [  # TODO: Get from safety checker
                {
                    'severity': 'caution',
                    'concern': 'Monitor for genital infections',
                    'patient_factor': 'SGLT-2 therapy',
                    'mitigation': 'Educate on hygiene, monitor symptoms'
                }
            ]
        }

        # 4. Create Prediction record
        prediction = Prediction(
            patient_id=request.patient_id,
            created_by=created_by_user_id,
            model_version=ai_result['model_version'],
            recommended_treatment=Treatment[ai_result['recommended_treatment'].upper().replace('-', '')],
            treatment_index=ai_result['treatment_index'],
            predicted_reduction=Decimal(str(ai_result['predicted_reduction'])),
            confidence_score=Decimal(str(ai_result['confidence_score'])),
            confidence_margin=Decimal(str(ai_result['confidence_margin']))
        )
        saved_prediction = self.prediction_repository.create(prediction)

        # 5. Create Q-values (all 5 treatments)
        q_value_records = []
        for qv in ai_result['q_values']:
            q_value = PredictionQValue(
                prediction_id=saved_prediction.id,
                treatment=Treatment[qv['treatment'].upper().replace('-', '')],
                q_value=Decimal(str(qv['q_value'])),
                rank=qv['rank']
            )
            q_value_records.append(q_value)
        saved_q_values = self.q_value_repository.create_many(q_value_records)

        # 6. Create Explanation (if provided by AI)
        saved_explanation = None
        saved_features = []
        saved_alternatives = []

        if ai_result.get('explanation'):
            exp_data = ai_result['explanation']

            # Create explanation record
            explanation = PredictionExplanation(
                prediction_id=saved_prediction.id,
                summary_text=exp_data['summary_text'],
                confidence_level=ConfidenceLevel[exp_data['confidence_level'].upper()],
                clinical_priority=ClinicalPriority[exp_data['clinical_priority'].upper()],
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
                saved_features = self.feature_repository.create_many(feature_records)

            # Create alternatives
            alternative_records = []
            for alt in exp_data.get('alternatives', []):
                alternative = ExplanationAlternative(
                    explanation_id=saved_explanation.id,
                    rank=alt['rank'],
                    treatment=Treatment[alt['treatment'].upper().replace('-', '')],
                    predicted_reduction=Decimal(str(alt['predicted_reduction'])),
                    pros=alt['pros'],
                    cons=alt['cons'],
                    when_to_consider=alt['when_to_consider']
                )
                alternative_records.append(alternative)
            if alternative_records:
                saved_alternatives = self.alternative_repository.create_many(alternative_records)

        # 7. Create Safety Warnings
        saved_warnings = []
        warning_records = []
        for warn in ai_result.get('safety_warnings', []):
            warning = SafetyWarning(
                prediction_id=saved_prediction.id,
                severity=SafetySeverity[warn['severity'].upper()],
                concern=warn['concern'],
                patient_factor=warn['patient_factor'],
                mitigation=warn['mitigation']
            )
            warning_records.append(warning)
        if warning_records:
            saved_warnings = self.safety_warning_repository.create_many(warning_records)

        # 8. Build patient summary
        patient_summary = self._build_patient_summary(request.patient_id)

        # 9. Build and return detailed response with patient info
        response_dict = saved_prediction.to_dict()
        response_dict['patient'] = patient_summary.model_dump()

        return PredictionDetailResponse(**response_dict)