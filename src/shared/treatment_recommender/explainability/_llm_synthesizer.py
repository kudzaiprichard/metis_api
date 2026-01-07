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
        prompt = f"""
You are a clinical decision support AI explaining why a diabetes treatment recommendation was made.
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
Generate a comprehensive clinical explanation in the EXACT JSON format specified below.

**CRITICAL JSON FORMATTING RULES:**
1. Respond ONLY with valid JSON - no explanatory text before or after
2. Do NOT use markdown code fences (no ```json or ```)
3. Do NOT include trailing commas before }} or ]
4. All string values must use double quotes (")
5. Escape any quotes inside strings with backslash (\")
6. Ensure all arrays and objects are properly closed
7. Do NOT include comments in the JSON

**REQUIRED JSON STRUCTURE:**

{{
  "summary": {{
    "one_sentence": "Single sentence explaining the recommendation",
    "clinical_priority": "routine|standard|urgent|critical"
  }},
  "model_reasoning": {{
    "why_this_treatment": "2-3 sentences explaining the decision",
    "key_factors": [
      {{
        "factor": "Factor name",
        "evidence": "Supporting evidence",
        "impact": "Impact description"
      }}
    ]
  }},
  "feature_importance_explanation": {{
    "feature_interactions": "1 paragraph explaining how features work together"
  }},
  "clinical_context": {{
    "guideline_alignment": {{
      "aligned": true,
      "explanation": "Explanation of alignment"
    }},
    "notable_case": {{
      "description": "Case description",
      "outcome": "Patient outcome"
    }}
  }},
  "safety_considerations": {{
    "warnings": [
      {{
        "severity": "low|moderate|high",
        "concern": "Warning description",
        "patient_factor": "Relevant patient factor",
        "mitigation": "How to address"
      }}
    ],
    "monitoring_requirements": [
      "Monitoring action 1",
      "Monitoring action 2"
    ]
  }},
  "alternatives_explanation": {{
    "why_not_alternatives": "2-3 sentences explaining why other treatments weren't chosen",
    "alternatives": [
      {{
        "rank": 2,
        "treatment": "Treatment name",
        "predicted_reduction": 0.0,
        "pros": ["Pro 1", "Pro 2"],
        "cons": ["Con 1", "Con 2"],
        "when_to_consider": "When to use this instead"
      }}
    ]
  }}
}}

Be concise, clinical, and evidence-based. Respond with ONLY the JSON object, nothing else.
"""

        return prompt

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
        Build ExplanationResult from parsed LLM response.

        Args:
            parsed_response: Parsed JSON from LLM
            model_result: Original model result
            shap_data: SHAP data
            graph_insights: Graph database insights
            patient_data: Patient data
            explanation_id: Unique explanation ID
            model_version: Model version
            llm_time_ms: LLM generation time

        Returns:
            Complete ExplanationResult DTO
        """
        # Extract summary
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

        # Extract model reasoning
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

        # Build feature importance
        feature_importance = FeatureImportance(
            top_features=shap_data['top_features'],
            base_value=shap_data['base_value'],
            prediction=shap_data['prediction'],
            feature_interactions=parsed_response.get('feature_importance_explanation', {}).get('feature_interactions')
        )

        # Build clinical context
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

        # Build safety checks
        safety_data = parsed_response.get('safety_considerations', {})
        warnings = [
            SafetyWarning(
                severity=w.get('severity', 'info'),
                concern=w.get('concern', ''),
                patient_factor=w.get('patient_factor', ''),
                mitigation=w.get('mitigation', '')
            )
            for w in safety_data.get('warnings', [])
        ]

        safety_checks = SafetyChecks(
            contraindications=[str(c) for c in graph_insights.get('contraindications', [])],
            warnings=warnings,
            monitoring_requirements=safety_data.get('monitoring_requirements', []),
            drug_interactions=graph_insights.get('drug_interactions', [])
        )

        # Build alternatives with string-to-list conversion for pros/cons
        alternatives_data = parsed_response.get('alternatives_explanation', {})
        alternatives = []

        for alt in alternatives_data.get('alternatives', [])[:3]:
            # Ensure pros/cons are lists, not strings
            pros = alt.get('pros', [])
            if isinstance(pros, str):
                pros = [pros]

            cons = alt.get('cons', [])
            if isinstance(cons, str):
                cons = [cons]

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

        # Build metadata
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

        # Create and return result
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