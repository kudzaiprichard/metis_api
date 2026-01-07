"""
LLM-powered explanation synthesis.

This module orchestrates the creation of natural language explanations
by combining model insights, SHAP values, and graph database context
into structured prompts for LLM providers.
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from ._base import (
    ExplanationResult,
    ExplanationSummary,
    ModelReasoning,
    FeatureImportance,
    ClinicalContext,
    SafetyChecks,
    AlternativeTreatments,
    ExplanationMetadata,
    KeyFactor,
    AlternativeTreatment,
    SafetyWarning,
    FeatureAttribution,
    get_confidence_level,
    determine_clinical_priority,
    generate_explanation_id,
    TREATMENT_NAMES
)
from .providers._base import BaseLLMProvider


# =============================================================================
# LLM SYNTHESIZER
# =============================================================================

class LLMSynthesizer:
    """
    Synthesizes clinical explanations using LLM providers.

    Takes structured data (model outputs, SHAP, graph DB) and generates
    natural language explanations through LLM APIs.

    Usage:
        synthesizer = LLMSynthesizer(
            llm_provider=gemini_provider,
            temperature=0.3
        )

        explanation = synthesizer.synthesize(
            model_result=model_result,
            shap_data=shap_data,
            graph_insights=graph_insights,
            patient_data=patient_data
        )
    """

    def __init__(self,
                 llm_provider: BaseLLMProvider,
                 temperature: float = 0.3,
                 verbose: bool = False):
        """
        Initialize LLM synthesizer.

        Args:
            llm_provider: LLM provider implementation (Gemini, GPT, etc.)
            temperature: Sampling temperature (0.0-1.0, lower = more deterministic)
            verbose: If True, print detailed logs
        """
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.verbose = verbose

        if self.verbose:
            print(f"[LLMSynthesizer] Initialized with provider: {llm_provider.provider_name}")
            print(f"[LLMSynthesizer] Temperature: {temperature}")

    def synthesize(self,
                   model_result: Any,
                   shap_data: Dict[str, Any],
                   graph_insights: Dict[str, Any],
                   patient_data: Dict[str, Any],
                   model_version: str = "unknown") -> ExplanationResult:
        """
        Synthesize complete explanation from all data sources.

        Args:
            model_result: TreatmentResult from prediction pipeline
            shap_data: SHAP values and feature attributions
            graph_insights: Clinical context from graph database
            patient_data: Original patient data
            model_version: Model version identifier

        Returns:
            ExplanationResult with complete structured explanation

        Example:
            explanation = synthesizer.synthesize(
                model_result=prediction_result,
                shap_data={'top_features': [...], 'base_value': 2.1, ...},
                graph_insights={'guidelines': {...}, 'similar_cases': [...]},
                patient_data={'age': 60, 'hba1c_baseline': 11.5, ...}
            )
        """
        if self.verbose:
            print("[LLMSynthesizer] Starting synthesis...")

        start_time = datetime.now()
        explanation_id = generate_explanation_id()

        # Build comprehensive prompt
        prompt = self._build_prompt(
            model_result=model_result,
            shap_data=shap_data,
            graph_insights=graph_insights,
            patient_data=patient_data
        )

        if self.verbose:
            print(f"[LLMSynthesizer] Prompt length: {len(prompt)} characters")
            print(f"[LLMSynthesizer] Calling LLM provider: {self.llm_provider.provider_name}")

        # Generate explanation via LLM
        llm_start = datetime.now()
        raw_response = self.llm_provider.generate_explanation(
            prompt=prompt,
            temperature=self.temperature
        )
        llm_time = (datetime.now() - llm_start).total_seconds() * 1000

        if self.verbose:
            print(f"[LLMSynthesizer] LLM response received ({llm_time:.0f}ms)")
            print(f"[LLMSynthesizer] Response length: {len(raw_response)} characters")

        # Parse LLM response
        parsed_response = self.llm_provider.parse_structured_response(raw_response)

        # Build ExplanationResult
        explanation = self._build_explanation_result(
            parsed_response=parsed_response,
            model_result=model_result,
            shap_data=shap_data,
            graph_insights=graph_insights,
            patient_data=patient_data,
            explanation_id=explanation_id,
            model_version=model_version,
            llm_time_ms=int(llm_time)
        )

        total_time = (datetime.now() - start_time).total_seconds() * 1000
        explanation.metadata.total_time_ms = int(total_time)

        if self.verbose:
            print(f"[LLMSynthesizer] Synthesis complete ({total_time:.0f}ms total)")

        return explanation

    def _build_prompt(self,
                      model_result: Any,
                      shap_data: Dict[str, Any],
                      graph_insights: Dict[str, Any],
                      patient_data: Dict[str, Any]) -> str:
        """
        Build comprehensive prompt for LLM.

        Combines all data sources into a structured prompt that guides
        the LLM to generate clinically relevant explanations.

        Args:
            model_result: Model prediction result
            shap_data: SHAP feature attributions
            graph_insights: Clinical context from graph DB
            patient_data: Patient information

        Returns:
            Formatted prompt string
        """
        # Extract key information
        recommended_treatment = model_result.recommended_treatment
        predicted_reduction = model_result.predicted_hba1c_reduction
        confidence_score = model_result.confidence_score
        all_q_values = model_result.all_q_values
        ranked_treatments = model_result.ranked_treatments

        # Build prompt sections
        # Build prompt sections
        prompt = f"""You are a clinical decision support AI explaining diabetes treatment recommendations with STRICT adherence to output format.
# PATIENT PROFILE
Age: {patient_data.get('age')} years
Gender: {patient_data.get('gender')}
HbA1c: {patient_data.get('hba1c_baseline')}%
C-peptide: {patient_data.get('c_peptide')} ng/mL
Diabetes Duration: {patient_data.get('diabetes_duration')} years
eGFR: {patient_data.get('egfr')} mL/min/1.73m²
BMI: {patient_data.get('bmi')} kg/m²

Comorbidities:
- Hypertension: {'Yes' if patient_data.get('hypertension') else 'No'}
- CKD: {'Yes' if patient_data.get('ckd') else 'No'}
- CVD: {'Yes' if patient_data.get('cvd') else 'No'}
- NAFLD: {'Yes' if patient_data.get('nafld') else 'No'}
- Retinopathy: {'Yes' if patient_data.get('retinopathy') else 'No'}

# MODEL RECOMMENDATION
Recommended Treatment: {recommended_treatment}
Predicted HbA1c Reduction: {predicted_reduction:.2f}%
Confidence Score: {confidence_score:.1f}%

All Treatment Predictions (Q-values):
{self._format_q_values(all_q_values)}

# FEATURE IMPORTANCE (SHAP Analysis)
These features had the strongest influence on the model's decision:

{self._format_shap_features(shap_data['top_features'])}

# CLINICAL GUIDELINES
{self._format_guidelines(graph_insights.get('guidelines', {}))}

# SIMILAR PATIENT CASES
{self._format_similar_cases(graph_insights.get('similar_cases', {}))}

# CONTRAINDICATIONS & SAFETY
{self._format_contraindications(graph_insights.get('contraindications', []))}

{self._format_drug_interactions(graph_insights.get('drug_interactions', []))}

# YOUR TASK
Generate a comprehensive clinical explanation following the EXACT JSON structure below.

CRITICAL RULES - VIOLATIONS WILL CAUSE SYSTEM FAILURE

**JSON FORMATTING (NON-NEGOTIABLE):**
1. Output ONLY valid JSON - no text before or after
2. NO markdown fences (```json or ```)
3. NO trailing commas before }} or ]
4. Use ONLY double quotes (") for strings
5. Escape internal quotes with backslash (\")
6. Close all arrays and objects properly
7. NO comments in JSON

**FIELD REQUIREMENTS (MANDATORY):**
1. "patient_factor" - MUST be a SPECIFIC patient value from their profile
   ✓ CORRECT: "Stage 4 CKD (eGFR 23)", "BMI 32.9", "HbA1c 12.0%", "Age 30"
   ✗ WRONG: "N/A", "Patient characteristic", "Clinical factor"

2. "alternatives" - MUST contain EXACTLY 3 treatments (ranks 2, 3, 4)
   - Include the next 3 best options after the recommended treatment
   - Each must have complete pros, cons, and when_to_consider

3. "treatment" field - Use ONLY these EXACT names (case-sensitive):
   - METFORMIN
   - GLP-1
   - SGLT-2
   - DPP-4
   - INSULIN

4. "severity" field - Use ONLY these values:
   - low
   - moderate
   - high

5. "clinical_priority" - Use ONLY these values:
   - routine
   - standard
   - urgent
   - critical

REQUIRED JSON STRUCTURE

{{
  "summary": {{
    "one_sentence": "Single comprehensive sentence explaining the recommendation",
    "clinical_priority": "routine|standard|urgent|critical"
  }},
  "model_reasoning": {{
    "why_this_treatment": "2-3 sentences explaining why this specific treatment was chosen based on patient factors",
    "key_factors": [
      {{
        "factor": "Specific clinical factor name",
        "evidence": "Supporting evidence from patient data",
        "impact": "How this impacts treatment choice"
      }},
      {{
        "factor": "Another clinical factor",
        "evidence": "More supporting evidence",
        "impact": "Treatment impact"
      }},
      {{
        "factor": "Third clinical factor",
        "evidence": "Additional evidence",
        "impact": "Further impact"
      }}
    ]
  }},
  "feature_importance_explanation": {{
    "feature_interactions": "One detailed paragraph explaining how the top features work together to support this recommendation. Connect multiple features and explain their combined clinical significance."
  }},
  "clinical_context": {{
    "guideline_alignment": {{
      "aligned": true,
      "explanation": "Detailed explanation of how this recommendation aligns with clinical guidelines"
    }},
    "notable_case": {{
      "description": "Description of a similar case from the data",
      "outcome": "What happened to that patient"
    }}
  }},
  "safety_considerations": {{
    "warnings": [
      {{
        "severity": "low|moderate|high",
        "concern": "Clear description of the safety concern",
        "patient_factor": "SPECIFIC patient value (e.g., 'eGFR 23', 'BMI 32.9', 'Age 30', 'HbA1c 12.0%')",
        "reason": "Why this is a concern for THIS SPECIFIC PATIENT with THEIR specific values",
        "mitigation": "Specific clinical action to address this concern"
      }},
      {{
        "severity": "low|moderate|high",
        "concern": "Another safety concern",
        "patient_factor": "ANOTHER specific patient value from their profile",
        "reason": "Patient-specific reasoning with their exact values",
        "mitigation": "Another mitigation strategy"
      }}
    ],
    "monitoring_requirements": [
      "Specific monitoring action with frequency",
      "Another monitoring requirement with details",
      "Third monitoring parameter to track"
    ]
  }},
  "alternatives_explanation": {{
    "why_not_alternatives": "2-3 sentences clearly explaining why the other top alternatives were NOT chosen despite their potential benefits. Reference specific patient factors that made them less suitable.",
    "alternatives": [
      {{
        "rank": 2,
        "treatment": "METFORMIN|GLP-1|SGLT-2|DPP-4|INSULIN",
        "predicted_reduction": 2.45,
        "pros": ["Specific advantage 1", "Specific advantage 2", "Specific advantage 3"],
        "cons": ["Specific disadvantage 1", "Specific disadvantage 2"],
        "when_to_consider": "Clear clinical scenario when this would be the better choice"
      }},
      {{
        "rank": 3,
        "treatment": "METFORMIN|GLP-1|SGLT-2|DPP-4|INSULIN",
        "predicted_reduction": 1.58,
        "pros": ["Advantage 1", "Advantage 2", "Advantage 3"],
        "cons": ["Disadvantage 1", "Disadvantage 2"],
        "when_to_consider": "When to use this option instead"
      }},
      {{
        "rank": 4,
        "treatment": "METFORMIN|GLP-1|SGLT-2|DPP-4|INSULIN",
        "predicted_reduction": 0.36,
        "pros": ["Advantage 1", "Advantage 2"],
        "cons": ["Disadvantage 1", "Disadvantage 2"],
        "when_to_consider": "Specific scenario for this treatment"
      }}
    ]
  }}
}}

VALIDATION CHECKLIST - VERIFY BEFORE RESPONDING

Before submitting, verify:
☐ JSON is valid (no trailing commas, all brackets closed)
☐ No markdown fences or extra text
☐ All "patient_factor" fields contain SPECIFIC patient values (NOT "N/A")
☐ Exactly 3 alternatives provided (ranks 2, 3, 4)
☐ All treatment names use EXACT spellings (METFORMIN, GLP-1, SGLT-2, DPP-4, INSULIN)
☐ All severity values are: low, moderate, or high
☐ clinical_priority is: routine, standard, urgent, or critical
☐ All "pros" and "cons" are arrays with multiple items
☐ All "reason" fields explain WHY for THIS SPECIFIC patient
☐ monitoring_requirements has at least 2-3 items

Generate the response now. Output ONLY the JSON object, nothing else.
"""

        return prompt

    # ============================================
    # METHOD 1: Simple Patient Factor Mapping
    # ============================================

    def _map_patient_factor(self,
                            concern: str,
                            reason: str,
                            patient_data: Dict[str, Any]) -> str:
        """
        Simple keyword-based mapping to patient factors.
        Checks both concern and reason for keywords, then uses patient_data as fallback.

        Args:
            concern: Warning concern text (e.g., "Renal Dosing Required")
            reason: Warning reason text (e.g., "Patient has Stage 4 CKD (eGFR 23)")
            patient_data: Patient data dictionary with clinical values

        Returns:
            Appropriate patient factor string (e.g., "eGFR 23 mL/min")

        Examples:
            >>> _map_patient_factor("Renal dosing", "eGFR is 23", {"egfr": 23})
            "eGFR 23 mL/min"

            >>> _map_patient_factor("Weight concern", "Patient obese", {"bmi": 32.9})
            "BMI 32.9 (Obesity)"
        """
        # Combine both fields for better keyword detection (case-insensitive)
        text = f"{concern} {reason}".lower()

        # ============================================
        # Renal/Kidney Function
        # ============================================
        if any(kw in text for kw in ['egfr', 'renal', 'kidney', 'ckd', 'stage']):
            egfr = patient_data.get('egfr')
            if egfr:
                if egfr < 30:
                    return f"Stage 4-5 CKD (eGFR {egfr} mL/min)"
                elif egfr < 60:
                    return f"Stage 3 CKD (eGFR {egfr} mL/min)"
                return f"eGFR {egfr} mL/min"
            return "Renal impairment"

        # ============================================
        # BMI/Weight/Obesity
        # ============================================
        if any(kw in text for kw in ['bmi', 'obesity', 'obese', 'weight', 'overweight']):
            bmi = patient_data.get('bmi')
            if bmi:
                if bmi >= 40:
                    return f"BMI {bmi} (Class III Obesity)"
                elif bmi >= 35:
                    return f"BMI {bmi} (Class II Obesity)"
                elif bmi >= 30:
                    return f"BMI {bmi} (Class I Obesity)"
                elif bmi >= 25:
                    return f"BMI {bmi} (Overweight)"
                return f"BMI {bmi}"
            return "Weight concern"

        # ============================================
        # Age
        # ============================================
        if any(kw in text for kw in ['age', 'young', 'elderly', 'older']):
            age = patient_data.get('age')
            return f"Age {age} years" if age else "Age consideration"

        # ============================================
        # HbA1c/Glucose Control
        # ============================================
        if any(kw in text for kw in ['hba1c', 'a1c', 'glucose', 'glycemic', 'hyperglycemia']):
            hba1c = patient_data.get('hba1c_baseline')
            if hba1c:
                if hba1c >= 10:
                    return f"HbA1c {hba1c}% (Severely uncontrolled)"
                elif hba1c >= 8:
                    return f"HbA1c {hba1c}% (Uncontrolled)"
                return f"HbA1c {hba1c}%"
            return "Glucose control"

        # ============================================
        # Blood Pressure/Hypertension
        # ============================================
        if any(kw in text for kw in ['blood pressure', 'hypertension', 'bp', 'systolic']):
            bp_sys = patient_data.get('bp_systolic')
            bp_dia = patient_data.get('bp_diastolic')
            if bp_sys and bp_dia:
                return f"BP {bp_sys}/{bp_dia} mmHg"
            elif patient_data.get('hypertension'):
                return "History of hypertension"
            return "Blood pressure concern"

        # ============================================
        # C-peptide/Beta Cell Function
        # ============================================
        if any(kw in text for kw in ['c-peptide', 'c peptide', 'beta cell', 'beta-cell']):
            c_pep = patient_data.get('c_peptide')
            if c_pep:
                status = "preserved" if c_pep >= 1.1 else "reduced"
                return f"C-peptide {c_pep} ng/mL ({status})"
            return "Beta-cell function"

        # ============================================
        # Liver/NAFLD
        # ============================================
        if any(kw in text for kw in ['nafld', 'liver', 'hepatic', 'fatty liver', 'alt']):
            alt = patient_data.get('alt')
            if alt:
                status = "elevated" if alt > 40 else "normal"
                return f"ALT {alt} U/L ({status})"
            elif patient_data.get('nafld'):
                return "NAFLD present"
            return "Liver function"

        # ============================================
        # Cardiovascular Disease
        # ============================================
        if any(kw in text for kw in ['cvd', 'cardiovascular', 'heart', 'cardiac']):
            if patient_data.get('cvd'):
                return "History of CVD"
            return "CV risk factors"

        # ============================================
        # HDL Cholesterol
        # ============================================
        if 'hdl' in text and 'ldl' not in text:
            hdl = patient_data.get('hdl')
            if hdl:
                status = "low" if hdl < 40 else "normal"
                return f"HDL {hdl} mg/dL ({status})"
            return "HDL cholesterol"

        # ============================================
        # LDL Cholesterol
        # ============================================
        if 'ldl' in text:
            ldl = patient_data.get('ldl')
            if ldl:
                if ldl >= 160:
                    status = "very high"
                elif ldl >= 130:
                    status = "high"
                else:
                    status = "acceptable"
                return f"LDL {ldl} mg/dL ({status})"
            return "LDL cholesterol"

        # ============================================
        # Triglycerides
        # ============================================
        if 'triglyceride' in text:
            trig = patient_data.get('triglycerides')
            if trig:
                status = "high" if trig >= 200 else "normal"
                return f"Triglycerides {trig} mg/dL ({status})"
            return "Triglycerides"

        # ============================================
        # Diabetes Duration
        # ============================================
        if any(kw in text for kw in ['duration', 'years with diabetes', 'diabetes for']):
            duration = patient_data.get('diabetes_duration')
            return f"Diabetes duration {duration} years" if duration else "Disease duration"

        # ============================================
        # Specific Conditions
        # ============================================
        if 'pancreatitis' in text:
            return "History of pancreatitis"

        if any(kw in text for kw in ['thyroid', 'mtc', 'medullary']):
            return "Thyroid cancer risk"

        if 'retinopathy' in text:
            return "Diabetic retinopathy" if patient_data.get('retinopathy') else "Eye complications"

        # ============================================
        # Default Fallback
        # ============================================
        return "General clinical consideration"

    def _format_q_values(self, q_values: Dict[str, float]) -> str:
        """Format Q-values for prompt."""
        lines = []
        for treatment, q_value in q_values.items():
            lines.append(f"  - {treatment}: {q_value:.2f}% reduction")
        return "\n".join(lines)

    def _format_shap_features(self, top_features: List[FeatureAttribution]) -> str:
        """Format SHAP features for prompt - use RAW values only."""
        lines = []
        for feat in top_features:
            lines.append(
                f"{feat.importance_rank}. {feat.feature} = {feat.raw_value:.2f}\n"  # Use raw_value
                f"   SHAP value: {feat.shap_value:+.3f}\n"
                f"   {feat.interpretation}"
            )
        return "\n\n".join(lines)

    def _format_guidelines(self, guidelines: Dict[str, Any]) -> str:
        """Format clinical guidelines for prompt."""
        if not guidelines:
            return "No specific guidelines available."

        lines = [f"Treatment: {guidelines.get('treatment', 'N/A')}"]
        lines.append(f"Indication: {guidelines.get('indication', 'N/A')}")
        lines.append("\nGuideline Sources:")
        for guideline in guidelines.get('guidelines', []):
            lines.append(f"  - {guideline}")
        lines.append(f"\nDosing: {guidelines.get('dosing', 'N/A')}")
        lines.append(f"Monitoring: {guidelines.get('monitoring', 'N/A')}")

        return "\n".join(lines)

    def _format_similar_cases(self, similar_cases: Dict[str, Any]) -> str:
        """Format similar cases for prompt."""
        if not similar_cases or not similar_cases.get('found'):
            return "No similar cases available."

        lines = [f"Found {similar_cases.get('found', 0)} similar cases"]

        avg_outcome = similar_cases.get('average_outcome', {})
        if avg_outcome:
            lines.append(f"\nAverage Outcomes:")
            lines.append(f"  - HbA1c Reduction: {avg_outcome.get('hba1c_reduction', 0):.1f}%")
            lines.append(f"  - Success Rate: {avg_outcome.get('success_rate', 0):.0%}")
            lines.append(f"  - Time to Target: {avg_outcome.get('time_to_target', 'N/A')}")

        notable = similar_cases.get('notable_case', {})
        if notable:
            lines.append(f"\nNotable Case:")
            lines.append(f"  Patient: {notable.get('patient', 'N/A')}")
            lines.append(f"  Treatment: {notable.get('treatment', 'N/A')}")
            lines.append(f"  Outcome: {notable.get('outcome', 'N/A')}")

        return "\n".join(lines)

    def _format_contraindications(self, contraindications: List[Dict]) -> str:
        """Format contraindications for prompt."""
        if not contraindications:
            return "No contraindications found."

        lines = ["Contraindications:"]
        for contra in contraindications:
            lines.append(
                f"  - [{contra.get('severity', 'unknown').upper()}] "
                f"{contra.get('condition', 'N/A')}: {contra.get('reason', 'N/A')}"
            )

        return "\n".join(lines)

    def _format_drug_interactions(self, interactions: List[Dict]) -> str:
        """Format drug interactions for prompt."""
        if not interactions:
            return "No significant drug interactions."

        lines = ["Drug Interactions:"]
        for interaction in interactions:
            lines.append(
                f"  - {interaction.get('condition', 'N/A')}: "
                f"{interaction.get('interaction', 'N/A')} "
                f"({interaction.get('severity', 'unknown')} severity)"
            )

        return "\n".join(lines)

    def _build_explanation_result(self,
                                  parsed_response: Dict[str, Any],
                                  model_result: Any,
                                  shap_data: Dict[str, Any],
                                  graph_insights: Dict[str, Any],
                                  patient_data: Dict[str, Any],
                                  explanation_id: str,
                                  model_version: str,
                                  llm_time_ms: int) -> ExplanationResult:
        """
        Build ExplanationResult from parsed LLM response with simple patient_factor mapping.

        Args:
            parsed_response: Parsed JSON from LLM
            model_result: Original model result (TreatmentResult)
            shap_data: SHAP feature attributions and values
            graph_insights: Clinical context from graph database
            patient_data: Original patient data dictionary
            explanation_id: Unique explanation ID
            model_version: Model version string
            llm_time_ms: LLM generation time in milliseconds

        Returns:
            Complete ExplanationResult DTO with all components
        """
        # ============================================
        # 1. Extract Summary
        # ============================================
        summary_data = parsed_response.get('summary', {})
        confidence_level = get_confidence_level(model_result.confidence_score)
        clinical_priority = determine_clinical_priority(
            hba1c=patient_data.get('hba1c_baseline', 7.0),
            c_peptide=patient_data.get('c_peptide', 1.5),
            contraindications=[str(c) for c in graph_insights.get('contraindications', [])]
        )

        summary = ExplanationSummary(
            primary_recommendation=model_result.recommended_treatment,
            confidence_level=confidence_level,
            one_sentence=summary_data.get('one_sentence', ''),
            clinical_priority=clinical_priority
        )

        # ============================================
        # 2. Extract Model Reasoning
        # ============================================
        reasoning_data = parsed_response.get('model_reasoning', {})
        key_factors = [
            KeyFactor(
                factor=f.get('factor', ''),
                evidence=f.get('evidence', ''),
                impact=f.get('impact', '')
            )
            for f in reasoning_data.get('key_factors', [])[:5]
        ]

        model_reasoning = ModelReasoning(
            predicted_hba1c_reduction=model_result.predicted_hba1c_reduction,
            confidence_score=model_result.confidence_score,
            why_this_treatment=reasoning_data.get('why_this_treatment', ''),
            key_factors=key_factors
        )

        # ============================================
        # 3. Build Feature Importance
        # ============================================
        feature_importance = FeatureImportance(
            top_features=shap_data['top_features'],
            base_value=shap_data['base_value'],
            prediction=shap_data['prediction'],
            feature_interactions=parsed_response.get('feature_importance_explanation', {}).get('feature_interactions')
        )

        # ============================================
        # 4. Build Clinical Context
        # ============================================
        clinical_context_data = parsed_response.get('clinical_context', {})
        clinical_context = ClinicalContext(
            guideline_alignment=clinical_context_data.get('guideline_alignment', {}),
            similar_cases={
                'found': graph_insights.get('similar_cases', {}).get('found', 0),
                'average_outcome': graph_insights.get('similar_cases', {}).get('average_outcome', {}),
                'notable_case': clinical_context_data.get('notable_case', {})
            },
            population_statistics=graph_insights.get('population_statistics')
        )

        # ============================================
        # 5. Build Safety Checks with Simple Patient Factor Mapping
        # ============================================
        safety_data = parsed_response.get('safety_considerations', {})
        warnings = []

        # Process LLM-generated warnings
        for w in safety_data.get('warnings', []):
            patient_factor = w.get('patient_factor', 'N/A')
            concern = w.get('concern', '')
            reason = w.get('reason', '')

            # If LLM returned N/A or empty, use our simple mapping
            if patient_factor in ['N/A', 'n/a', '', None]:
                patient_factor = self._map_patient_factor(
                    concern=concern,
                    reason=reason,
                    patient_data=patient_data
                )

            warnings.append(
                SafetyWarning(
                    severity=w.get('severity', 'info'),
                    concern=concern,
                    patient_factor=patient_factor,  # Now properly mapped!
                    mitigation=w.get('mitigation', ''),
                    reason=reason if reason else f"Clinical consideration for {concern}"
                )
            )

        # Process contraindications from graph database
        contraindications_from_graph = graph_insights.get('contraindications', [])

        for contra in contraindications_from_graph:
            if isinstance(contra, dict):
                # Dict contraindication with details
                condition = contra.get('condition', 'Unknown contraindication')
                reason = contra.get('reason', f"Contraindicated due to {condition.lower()}")
                severity = contra.get('severity', 'critical')
                mitigation = contra.get('alternative', 'Consult physician before use')

                # Use simple mapping for patient_factor
                patient_factor = self._map_patient_factor(
                    concern=condition,
                    reason=reason,
                    patient_data=patient_data
                )

                warnings.append(
                    SafetyWarning(
                        severity=severity,
                        concern=condition,
                        patient_factor=patient_factor,
                        mitigation=mitigation,
                        reason=reason
                    )
                )

            elif isinstance(contra, str):
                # String contraindication - simple format
                patient_factor = self._map_patient_factor(
                    concern=contra,
                    reason='',
                    patient_data=patient_data
                )

                warnings.append(
                    SafetyWarning(
                        severity='critical',
                        concern=contra,
                        patient_factor=patient_factor,
                        mitigation='Consult physician before use',
                        reason=f"Contraindicated due to {contra.lower()}"
                    )
                )

        # Build SafetyChecks object
        safety_checks = SafetyChecks(
            contraindications=[str(c) for c in contraindications_from_graph],
            warnings=warnings,  # Now includes both LLM warnings and graph DB contraindications
            monitoring_requirements=safety_data.get('monitoring_requirements', []),
            drug_interactions=graph_insights.get('drug_interactions', [])
        )

        # ============================================
        # 6. Build Alternatives with Array Conversion
        # ============================================
        alternatives_data = parsed_response.get('alternatives_explanation', {})
        alternatives = []

        for alt in alternatives_data.get('alternatives', [])[:3]:
            # Ensure pros/cons are lists, not strings
            pros = alt.get('pros', [])
            if isinstance(pros, str):
                # Split comma-separated string into list
                pros = [p.strip() for p in pros.split(',') if p.strip()]

            cons = alt.get('cons', [])
            if isinstance(cons, str):
                # Split comma-separated string into list
                cons = [c.strip() for c in cons.split(',') if c.strip()]

            alternatives.append(
                AlternativeTreatment(
                    rank=alt.get('rank', 0),
                    treatment=alt.get('treatment', ''),
                    predicted_reduction=alt.get('predicted_reduction', 0.0),
                    pros=pros,
                    cons=cons,
                    when_to_consider=alt.get('when_to_consider', '')
                )
            )

        alternative_treatments = AlternativeTreatments(
            why_not_alternatives=alternatives_data.get('why_not_alternatives', ''),
            alternatives=alternatives
        )

        # ============================================
        # 7. Build Metadata
        # ============================================
        metadata = ExplanationMetadata(
            explanation_id=explanation_id,
            timestamp=datetime.now().isoformat(),
            model_version=model_version,
            shap_calculation_time_ms=shap_data.get('calculation_time_ms', 0),
            graph_query_time_ms=graph_insights.get('query_time_ms', 0),
            llm_generation_time_ms=llm_time_ms,
            total_time_ms=0,  # Will be set by caller
            tokens_used=parsed_response.get('metadata', {}).get('tokens_used')
        )

        # ============================================
        # 8. Create and Return Complete ExplanationResult
        # ============================================
        return ExplanationResult(
            summary=summary,
            model_reasoning=model_reasoning,
            feature_importance=feature_importance,
            clinical_context=clinical_context,
            safety_checks=safety_checks,
            alternatives=alternative_treatments,
            metadata=metadata
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_llm_synthesizer(llm_provider: BaseLLMProvider,
                           temperature: float = 0.3,
                           verbose: bool = False) -> LLMSynthesizer:
    """
    Factory function to create LLM synthesizer.

    Args:
        llm_provider: LLM provider implementation
        temperature: Sampling temperature (0.0-1.0)
        verbose: Enable detailed logging

    Returns:
        Configured LLMSynthesizer instance

    Example:
        from treatment_recommender.explainability.providers import create_gemini_provider

        gemini = create_gemini_provider(
            api_key='YOUR_KEY',
            model_name='gemini-1.5-pro-latest'
        )
        synthesizer = create_llm_synthesizer(gemini, temperature=0.3)
    """
    return LLMSynthesizer(
        llm_provider=llm_provider,
        temperature=temperature,
        verbose=verbose
    )