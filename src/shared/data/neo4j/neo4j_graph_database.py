"""
Neo4j Graph Database for ML Explainability in Flask.
"""

from neo4j import GraphDatabase
import pandas as pd
from typing import Dict, List, Any
import logging

from src.shared.treatment_recommender.explainability import GraphDatabaseInterface

logger = logging.getLogger(__name__)


class Neo4jGraphDatabase(GraphDatabaseInterface):
    """
    Neo4j implementation of GraphDatabaseInterface for explainability.

    Provides:
    - Clinical guidelines retrieval
    - Contraindication checking
    - Drug interaction warnings
    - Similar patient case finding
    - Background data for SHAP analysis

    This implementation is Flask-aware and handles connection lifecycle properly.
    """

    def __init__(self, uri: str, username: str, password: str):
        """
        Initialize Neo4j connection parameters.

        Args:
            uri: Neo4j URI (e.g., 'neo4j://localhost:7687')
            username: Database username
            password: Database password
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self._connected = False

    def connect(self) -> bool:
        """
        Establish connection to Neo4j.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            self._connected = True
            logger.info("Connected to Neo4j for explainability")
            return True
        except Exception:
            # Don't log here - let caller handle error messaging
            self._connected = False
            return False

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self._connected = False
            logger.info("Neo4j connection closed")

    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self._connected and self.driver is not None

    # =========================================================================
    # INTERFACE IMPLEMENTATION - Required Methods
    # =========================================================================

    def get_treatment_guidelines(self,
                                 treatment: str,
                                 patient_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get clinical guidelines for specific treatment and patient profile.

        Args:
            treatment: Treatment name (e.g., "Insulin", "Metformin")
            patient_profile: Patient data dictionary

        Returns:
            Dictionary with guidelines, dosing, monitoring
        """
        if not self.is_connected():
            logger.warning("Neo4j not connected, returning fallback guidelines")
            return self._get_fallback_guidelines(treatment)

        hba1c = patient_profile.get('hba1c_baseline', 7.0)

        try:
            with self.driver.session() as session:
                query = """
                MATCH (t:Treatment {drug_name: $treatment})-[:HAS_GUIDELINE]->(g:Guideline)
                WHERE g.hba1c_min <= $hba1c <= g.hba1c_max
                RETURN g.source AS source, 
                       g.year AS year,
                       g.text AS text, 
                       g.dosing AS dosing,
                       g.monitoring AS monitoring,
                       g.indication AS indication
                ORDER BY g.year DESC
                LIMIT 5
                """

                result = session.run(query, treatment=treatment, hba1c=hba1c)

                guidelines = []
                dosing = None
                monitoring = None
                indication = None

                for record in result:
                    guidelines.append(
                        f"{record['source']} {record['year']}: {record['text']}"
                    )
                    if not dosing:
                        dosing = record['dosing']
                    if not monitoring:
                        monitoring = record['monitoring']
                    if not indication:
                        indication = record['indication']

                # Fallback if no guidelines found
                if not guidelines:
                    return self._get_fallback_guidelines(treatment)

                return {
                    "treatment": treatment,
                    "indication": indication or f"HbA1c {hba1c:.1f}%",
                    "guidelines": guidelines,
                    "dosing": dosing,
                    "monitoring": monitoring
                }
        except Exception as e:
            logger.error(f"Error getting guidelines: {e}")
            return self._get_fallback_guidelines(treatment)

    def get_background_data_for_shap(self, n_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Get random patient samples for SHAP background data.

        Returns raw patient dictionaries (NOT preprocessed features).
        The feature processor will handle preprocessing.

        Args:
            n_samples: Number of random patients to retrieve

        Returns:
            List of patient dictionaries in same format as user input
        """
        if not self.is_connected():
            logger.warning("Neo4j not connected, returning empty background data")
            return []

        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:Patient)
                RETURN p.age AS age,
                       p.gender AS gender,
                       p.ethnicity AS ethnicity,
                       p.hba1c_baseline AS hba1c_baseline,
                       p.diabetes_duration AS diabetes_duration,
                       p.fasting_glucose AS fasting_glucose,
                       p.c_peptide AS c_peptide,
                       p.egfr AS egfr,
                       p.bmi AS bmi,
                       p.bp_systolic AS bp_systolic,
                       p.bp_diastolic AS bp_diastolic,
                       p.alt AS alt,
                       p.ldl AS ldl,
                       p.hdl AS hdl,
                       p.triglycerides AS triglycerides,
                       p.previous_prediabetes AS previous_prediabetes,
                       p.patient_id AS patient_id
                ORDER BY rand()
                LIMIT $n_samples
                """

                result = session.run(query, n_samples=n_samples)

                background_patients = []
                for record in result:
                    # Get comorbidities for this patient
                    comorbidity_query = """
                    MATCH (p:Patient {patient_id: $patient_id})-[:HAS_CONDITION]->(c:Comorbidity)
                    RETURN c.condition_name AS condition
                    """
                    comorbidities_result = session.run(comorbidity_query, patient_id=record['patient_id'])
                    comorbidities_set = {r['condition'] for r in comorbidities_result}

                    # Convert to dictionary matching user input format
                    patient = {
                        'age': int(record['age']),
                        'gender': record['gender'],
                        'ethnicity': record['ethnicity'],
                        'hba1c_baseline': float(record['hba1c_baseline']),
                        'diabetes_duration': float(record['diabetes_duration']),
                        'fasting_glucose': float(record['fasting_glucose']),
                        'c_peptide': float(record['c_peptide']),
                        'egfr': float(record['egfr']),
                        'bmi': float(record['bmi']),
                        'bp_systolic': float(record['bp_systolic']),
                        'bp_diastolic': float(record['bp_diastolic']),
                        'alt': float(record['alt']),
                        'ldl': float(record['ldl']),
                        'hdl': float(record['hdl']),
                        'triglycerides': float(record['triglycerides']),
                        'previous_prediabetes': int(record['previous_prediabetes']),
                        # Comorbidities as binary flags
                        'hypertension': 1 if 'Hypertension' in comorbidities_set else 0,
                        'ckd': 1 if 'CKD' in comorbidities_set else 0,
                        'cvd': 1 if 'CVD' in comorbidities_set else 0,
                        'nafld': 1 if 'NAFLD' in comorbidities_set else 0,
                        'retinopathy': 1 if 'Retinopathy' in comorbidities_set else 0
                    }
                    background_patients.append(patient)

                logger.info(f"Retrieved {len(background_patients)} background patients for SHAP")
                return background_patients
        except Exception as e:
            logger.error(f"Error getting background data: {e}")
            return []

    def check_contraindications(self,
                                treatment: str,
                                patient_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for contraindications for specific treatment.

        Args:
            treatment: Treatment name
            patient_profile: Patient data dictionary

        Returns:
            List of contraindication dictionaries
        """
        if not self.is_connected():
            return []

        egfr = patient_profile.get('egfr', 100)
        age = patient_profile.get('age', 50)
        c_peptide = patient_profile.get('c_peptide', 1.5)

        try:
            with self.driver.session() as session:
                query = """
                MATCH (t:Treatment {drug_name: $treatment})-[:CONTRAINDICATED_BY]->(c:Contraindication)
                WHERE (c.egfr_max IS NULL OR $egfr <= c.egfr_max)
                   OR (c.age_min IS NULL OR $age >= c.age_min)
                   OR (c.c_peptide_max IS NULL OR $c_peptide <= c.c_peptide_max)
                RETURN c.condition AS condition,
                       c.severity AS severity,
                       c.reason AS reason,
                       c.alternative AS alternative
                """

                result = session.run(
                    query,
                    treatment=treatment,
                    egfr=egfr,
                    age=age,
                    c_peptide=c_peptide
                )

                contraindications = []
                for record in result:
                    contraindications.append({
                        "condition": record["condition"],
                        "severity": record["severity"],
                        "reason": record["reason"],
                        "alternative": record["alternative"]
                    })

                return contraindications
        except Exception as e:
            logger.error(f"Error checking contraindications: {e}")
            return []

    def get_drug_interactions(self,
                              treatment: str,
                              comorbidities: List[str]) -> List[Dict[str, Any]]:
        """
        Get potential drug interactions based on comorbidities.

        Args:
            treatment: Treatment name
            comorbidities: List of comorbidity flags (e.g., ["hypertension", "cvd"])

        Returns:
            List of interaction dictionaries
        """
        if not self.is_connected() or not comorbidities:
            return []

        try:
            with self.driver.session() as session:
                query = """
                MATCH (t:Treatment {drug_name: $treatment})-[:INTERACTS_WITH]->(di:DrugInteraction)
                WHERE di.condition IN $comorbidities
                RETURN di.condition AS condition,
                       di.drug_class AS drug_class,
                       di.interaction AS interaction,
                       di.recommendation AS recommendation,
                       di.severity AS severity
                """

                result = session.run(query, treatment=treatment, comorbidities=comorbidities)

                interactions = []
                for record in result:
                    interactions.append({
                        "condition": record["condition"],
                        "drug_class": record["drug_class"],
                        "interaction": record["interaction"],
                        "recommendation": record["recommendation"],
                        "severity": record["severity"]
                    })

                return interactions
        except Exception as e:
            logger.error(f"Error getting drug interactions: {e}")
            return []

    def find_similar_cases(self,
                           patient_profile: Dict[str, Any],
                           limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar patient cases using on-demand computation.

        Args:
            patient_profile: Patient data dictionary
            limit: Maximum number of similar cases to return

        Returns:
            List of similar case dictionaries with similarity scores and outcomes
        """
        if not self.is_connected():
            return []

        age = patient_profile.get('age', 50)
        hba1c = patient_profile.get('hba1c_baseline', 8.0)
        c_peptide = patient_profile.get('c_peptide', 1.5)
        bmi = patient_profile.get('bmi', 30)

        # Categorical groupings
        age_group = self._get_age_group(age)
        hba1c_severity = self._get_hba1c_severity(hba1c)

        try:
            with self.driver.session() as session:
                query = """
                MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t:Treatment)
                WHERE p.age_group = $age_group
                  AND p.hba1c_severity = $hba1c_severity
                  AND abs(p.hba1c_baseline - $hba1c) <= 2.0
                  AND abs(p.c_peptide - $c_peptide) <= 0.5

                MATCH (o:Outcome {patient_id: p.patient_id})

                WITH p, t, o,
                     abs(p.age - $age) + 
                     abs(p.hba1c_baseline - $hba1c) * 2.0 + 
                     abs(p.c_peptide - $c_peptide) * 3.0 AS distance

                ORDER BY distance ASC
                LIMIT $limit

                RETURN p.patient_id AS case_id,
                       p.age AS age,
                       p.hba1c_baseline AS hba1c,
                       p.c_peptide AS c_peptide,
                       p.bmi AS bmi,
                       t.drug_name AS treatment,
                       o.hba1c_reduction AS reduction,
                       o.time_to_target AS time_to_target,
                       o.adverse_events AS adverse_events,
                       distance
                """

                result = session.run(
                    query,
                    age_group=age_group,
                    hba1c_severity=hba1c_severity,
                    age=age,
                    hba1c=hba1c,
                    c_peptide=c_peptide,
                    limit=limit
                )

                similar_cases = []
                for record in result:
                    distance = record['distance']
                    similarity_score = max(0.0, 1.0 - (distance / 15.0))

                    similar_cases.append({
                        "case_id": record["case_id"],
                        "similarity_score": round(similarity_score, 2),
                        "profile": {
                            "age": int(record["age"]),
                            "hba1c_baseline": round(record["hba1c"], 1),
                            "c_peptide": round(record["c_peptide"], 2),
                            "bmi": round(record["bmi"], 1)
                        },
                        "treatment_given": record["treatment"],
                        "outcome": {
                            "hba1c_reduction": round(record["reduction"], 1),
                            "time_to_target": record["time_to_target"],
                            "adverse_events": record["adverse_events"]
                        }
                    })

                return similar_cases
        except Exception as e:
            logger.error(f"Error finding similar cases: {e}")
            return []

    def find_similar_patients(self,
                              patient_profile: Dict[str, Any],
                              limit: int = 5,
                              treatment_filter: str = None,
                              min_similarity: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find similar patient cases with enhanced matching algorithm.

        USE THIS METHOD FOR REST API ENDPOINTS - Provides comprehensive similarity
        matching with normalized scoring, comorbidity overlap, and treatment filtering.

        This is the improved version of find_similar_cases() with:
        - Normalized distance calculation across all clinical features
        - Comorbidity similarity using Jaccard index (30% weight)
        - Optional treatment filtering for targeted comparisons
        - Configurable similarity threshold
        - Detailed similarity breakdown (clinical + comorbidity scores)

        The original find_similar_cases() method is kept for backwards compatibility
        with existing ML/explainability modules. Use this method for all new API
        integrations.

        Args:
            patient_profile: Patient data dictionary with all 21 base features
                Required fields: age, hba1c_baseline, c_peptide, bmi, egfr,
                               diabetes_duration, and comorbidity flags
            limit: Maximum number of similar cases to return (1-20, default: 5)
            treatment_filter: Optional treatment name to filter results
                             (e.g., "Metformin", "GLP-1", "SGLT-2")
            min_similarity: Minimum similarity score threshold (0.0-1.0, default: 0.5)
                           Lower values = more permissive matching

        Returns:
            List of dictionaries, each containing:
            - case_id: Patient identifier
            - similarity_score: Overall weighted similarity (0.0-1.0)
            - clinical_similarity: Feature-based similarity score
            - comorbidity_similarity: Comorbidity overlap score (Jaccard index)
            - profile: Patient demographics and clinical features
            - comorbidities: List of condition names
            - treatment_given: Treatment received by similar patient
            - drug_class: Drug classification
            - outcome: Complete treatment outcome data including:
                * hba1c_reduction: Actual HbA1c reduction achieved
                * hba1c_followup: Follow-up HbA1c value
                * time_to_target: Time taken to reach target
                * adverse_events: Side effects experienced
                * outcome_category: Success/Partial/Failure classification
                * success: Boolean success indicator

        Example:
            patient = {
                'age': 55,
                'hba1c_baseline': 8.5,
                'c_peptide': 1.2,
                'bmi': 32.0,
                'egfr': 75,
                'diabetes_duration': 7.0,
                'hypertension': 1,
                'cvd': 0,
                'ckd': 0,
                'nafld': 1,
                'retinopathy': 0
            }
            similar = db.find_similar_patients(
                patient_profile=patient,
                limit=5,
                treatment_filter="Metformin",
                min_similarity=0.7
            )
            for case in similar:
                print(f"Case {case['case_id']}: {case['similarity_score']:.2%} similar")
                print(f"  Treatment: {case['treatment_given']}")
                print(f"  Outcome: {case['outcome']['outcome_category']}")

        Note:
            - Requires Neo4j connection to be active
            - Returns empty list if connection fails or no matches found
            - Similarity calculation: 70% clinical features + 30% comorbidity overlap
            - Clinical features are normalized to ensure equal weighting
        """
        if not self.is_connected():
            logger.warning("Neo4j not connected for similar patients search")
            return []

        # Extract features
        age = patient_profile.get('age', 50)
        hba1c = patient_profile.get('hba1c_baseline', 8.0)
        c_peptide = patient_profile.get('c_peptide', 1.5)
        bmi = patient_profile.get('bmi', 30)
        egfr = patient_profile.get('egfr', 90)
        diabetes_duration = patient_profile.get('diabetes_duration', 5.0)

        # Extract comorbidities
        comorbidities = []
        comorbidity_map = {
            'hypertension': 'Hypertension',
            'ckd': 'CKD',
            'cvd': 'CVD',
            'nafld': 'NAFLD',
            'retinopathy': 'Retinopathy'
        }

        for key, condition_name in comorbidity_map.items():
            if patient_profile.get(key, 0) == 1:
                comorbidities.append(condition_name)

        # Categorical groupings
        age_group = self._get_age_group(age)
        hba1c_severity = self._get_hba1c_severity(hba1c)

        # Validate inputs
        limit = max(1, min(limit, 20))
        min_similarity = max(0.0, min(min_similarity, 1.0))

        try:
            with self.driver.session() as session:
                # Build treatment filter
                treatment_clause = ""
                if treatment_filter:
                    treatment_clause = "AND t.drug_name = $treatment_filter"

                query = f"""
                // Find similar patients with treatment and outcome
                MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t:Treatment)
                WHERE p.age_group = $age_group
                  AND p.hba1c_severity = $hba1c_severity
                  AND abs(p.hba1c_baseline - $hba1c) <= 2.0
                  AND abs(p.c_peptide - $c_peptide) <= 0.5
                  {treatment_clause}

                // Get outcome
                MATCH (o:Outcome {{patient_id: p.patient_id}})

                // Get comorbidities
                OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Comorbidity)

                WITH p, t, o, collect(DISTINCT c.condition_name) AS patient_comorbidities,
                     // Normalized distance calculation
                     (abs(p.age - $age) / 60.0) +                          // Age normalized by range 60
                     (abs(p.hba1c_baseline - $hba1c) / 7.0) * 2.0 +       // HbA1c normalized by range 7
                     (abs(p.c_peptide - $c_peptide) / 3.0) * 3.0 +        // C-peptide normalized by range 3
                     (abs(p.bmi - $bmi) / 20.0) +                         // BMI normalized by range 20
                     (abs(p.egfr - $egfr) / 100.0) +                      // eGFR normalized by range 100
                     (abs(p.diabetes_duration - $diabetes_duration) / 30.0) AS normalized_distance

                // Calculate comorbidity similarity (Jaccard index)
                WITH p, t, o, patient_comorbidities, normalized_distance,
                     size([x IN patient_comorbidities WHERE x IN $comorbidities]) AS overlap_count,
                     size(patient_comorbidities) + size($comorbidities) - 
                     size([x IN patient_comorbidities WHERE x IN $comorbidities]) AS union_count

                WITH p, t, o, patient_comorbidities, normalized_distance,
                     CASE 
                       WHEN union_count > 0 
                       THEN toFloat(overlap_count) / union_count 
                       ELSE CASE WHEN size(patient_comorbidities) = 0 AND size($comorbidities) = 0 THEN 1.0 ELSE 0.0 END
                     END AS comorbidity_similarity

                // Calculate clinical similarity from normalized distance
                WITH p, t, o, patient_comorbidities, normalized_distance, comorbidity_similarity,
                     1.0 - (normalized_distance / 6.0) AS clinical_similarity  // 6.0 = sum of weights

                // Final weighted similarity: 70% clinical + 30% comorbidities
                WITH p, t, o, patient_comorbidities, 
                     clinical_similarity,
                     comorbidity_similarity,
                     (clinical_similarity * 0.7) + (comorbidity_similarity * 0.3) AS similarity_score

                WHERE similarity_score >= $min_similarity

                ORDER BY similarity_score DESC
                LIMIT $limit

                RETURN p.patient_id AS patient_id,
                       p.age AS age,
                       p.gender AS gender,
                       p.ethnicity AS ethnicity,
                       p.hba1c_baseline AS hba1c_baseline,
                       p.c_peptide AS c_peptide,
                       p.bmi AS bmi,
                       p.egfr AS egfr,
                       p.diabetes_duration AS diabetes_duration,
                       p.bp_systolic AS bp_systolic,
                       p.fasting_glucose AS fasting_glucose,
                       patient_comorbidities,
                       t.drug_name AS treatment,
                       t.drug_class AS drug_class,
                       o.hba1c_reduction AS hba1c_reduction,
                       o.hba1c_followup AS hba1c_followup,
                       o.time_to_target AS time_to_target,
                       o.adverse_events AS adverse_events,
                       o.outcome_category AS outcome_category,
                       o.success AS success,
                       similarity_score,
                       clinical_similarity,
                       comorbidity_similarity
                """

                result = session.run(
                    query,
                    age_group=age_group,
                    hba1c_severity=hba1c_severity,
                    age=age,
                    hba1c=hba1c,
                    c_peptide=c_peptide,
                    bmi=bmi,
                    egfr=egfr,
                    diabetes_duration=diabetes_duration,
                    comorbidities=comorbidities,
                    limit=limit,
                    min_similarity=min_similarity,
                    treatment_filter=treatment_filter
                )

                similar_cases = []
                for record in result:
                    similar_cases.append({
                        "case_id": record["patient_id"],
                        "similarity_score": round(record["similarity_score"], 3),
                        "clinical_similarity": round(record["clinical_similarity"], 3),
                        "comorbidity_similarity": round(record["comorbidity_similarity"], 3),
                        "profile": {
                            "age": int(record["age"]),
                            "gender": record["gender"],
                            "ethnicity": record["ethnicity"],
                            "hba1c_baseline": round(record["hba1c_baseline"], 1),
                            "c_peptide": round(record["c_peptide"], 2),
                            "bmi": round(record["bmi"], 1),
                            "egfr": round(record["egfr"], 1),
                            "diabetes_duration": round(record["diabetes_duration"], 1),
                            "bp_systolic": int(record["bp_systolic"]),
                            "fasting_glucose": round(record["fasting_glucose"], 1)
                        },
                        "comorbidities": record["patient_comorbidities"] or [],
                        "treatment_given": record["treatment"],
                        "drug_class": record["drug_class"],
                        "outcome": {
                            "hba1c_reduction": round(record["hba1c_reduction"], 1),
                            "hba1c_followup": round(record["hba1c_followup"], 1),
                            "time_to_target": record["time_to_target"],
                            "adverse_events": record["adverse_events"],
                            "outcome_category": record["outcome_category"],
                            "success": bool(record["success"])
                        }
                    })

                logger.info(
                    f"Found {len(similar_cases)} similar patients "
                    f"(filter: {treatment_filter or 'none'}, min_similarity: {min_similarity})"
                )

                return similar_cases

        except Exception as e:
            logger.error(f"Error finding similar patients: {e}")
            return []

    def find_similar_cases_graph(self,
                                 patient_profile: Dict[str, Any],
                                 limit: int = 5,
                                 treatment_filter: str = None) -> Dict[str, Any]:
        """
        Find similar patient cases and return as GRAPH structure for visualization.

        Returns nodes (patients, treatments, outcomes) and edges (relationships)
        suitable for frontend graph visualization libraries like D3.js, Cytoscape, etc.

        Args:
            patient_profile: Patient data dictionary
            limit: Maximum number of similar cases to return
            treatment_filter: Optional treatment name to filter by (e.g., "Metformin")

        Returns:
            Dictionary with:
            - nodes: List of node objects (patients, treatments, outcomes)
            - edges: List of edge objects (relationships)
            - metadata: Query metadata (similarity scores, filters applied)
        """
        if not self.is_connected():
            return {"nodes": [], "edges": [], "metadata": {"error": "Neo4j not connected"}}

        age = patient_profile.get('age', 50)
        hba1c = patient_profile.get('hba1c_baseline', 8.0)
        c_peptide = patient_profile.get('c_peptide', 1.5)
        bmi = patient_profile.get('bmi', 30)

        # Extract comorbidities from patient profile
        comorbidities = []
        for condition in ['hypertension', 'ckd', 'cvd', 'nafld', 'retinopathy']:
            if patient_profile.get(condition, 0) == 1:
                comorbidities.append(condition.upper())

        # Categorical groupings
        age_group = self._get_age_group(age)
        hba1c_severity = self._get_hba1c_severity(hba1c)

        try:
            with self.driver.session() as session:
                # Build treatment filter clause
                treatment_clause = ""
                if treatment_filter:
                    treatment_clause = "AND t.drug_name = $treatment_filter"

                query = f"""
                // Find similar patients
                MATCH (p:Patient)-[:RECEIVED_TREATMENT]->(t:Treatment)
                WHERE p.age_group = $age_group
                  AND p.hba1c_severity = $hba1c_severity
                  AND abs(p.hba1c_baseline - $hba1c) <= 2.0
                  AND abs(p.c_peptide - $c_peptide) <= 0.5
                  {treatment_clause}

                // Get outcomes
                MATCH (o:Outcome {{patient_id: p.patient_id}})

                // Get comorbidities
                OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Comorbidity)

                // Calculate similarity
                WITH p, t, o, collect(DISTINCT c.condition_name) AS patient_comorbidities,
                     abs(p.age - $age) + 
                     abs(p.hba1c_baseline - $hba1c) * 2.0 + 
                     abs(p.c_peptide - $c_peptide) * 3.0 AS distance

                // Calculate comorbidity overlap score
                WITH p, t, o, patient_comorbidities, distance,
                     size([x IN patient_comorbidities WHERE x IN $comorbidities]) AS overlap_count,
                     size(patient_comorbidities) + size($comorbidities) AS total_comorbidities

                WITH p, t, o, patient_comorbidities, distance,
                     CASE 
                       WHEN total_comorbidities > 0 
                       THEN toFloat(overlap_count * 2) / total_comorbidities 
                       ELSE 0.5 
                     END AS comorbidity_similarity

                // Final similarity score (weighted)
                WITH p, t, o, patient_comorbidities,
                     (1.0 - (distance / 15.0)) * 0.7 + comorbidity_similarity * 0.3 AS similarity_score

                WHERE similarity_score >= 0.5

                ORDER BY similarity_score DESC
                LIMIT $limit

                RETURN p.patient_id AS patient_id,
                       p.age AS age,
                       p.gender AS gender,
                       p.hba1c_baseline AS hba1c,
                       p.c_peptide AS c_peptide,
                       p.bmi AS bmi,
                       p.egfr AS egfr,
                       patient_comorbidities,
                       t.drug_name AS treatment,
                       t.drug_class AS drug_class,
                       o.hba1c_reduction AS reduction,
                       o.hba1c_followup AS hba1c_followup,
                       o.time_to_target AS time_to_target,
                       o.adverse_events AS adverse_events,
                       o.outcome_category AS outcome_category,
                       similarity_score
                """

                result = session.run(
                    query,
                    age_group=age_group,
                    hba1c_severity=hba1c_severity,
                    age=age,
                    hba1c=hba1c,
                    c_peptide=c_peptide,
                    limit=limit,
                    comorbidities=comorbidities,
                    treatment_filter=treatment_filter
                )

                records = [dict(record) for record in result]

                if not records:
                    return {
                        "nodes": [],
                        "edges": [],
                        "metadata": {
                            "query_patient": patient_profile,
                            "filters_applied": {
                                "age_group": age_group,
                                "hba1c_severity": hba1c_severity,
                                "treatment": treatment_filter,
                                "comorbidities": comorbidities
                            },
                            "results_found": 0
                        }
                    }

                # Build graph structure
                nodes = []
                edges = []

                # Add query patient node (center node)
                nodes.append({
                    "id": "query_patient",
                    "type": "query_patient",
                    "label": "Query Patient",
                    "data": {
                        "age": age,
                        "hba1c_baseline": hba1c,
                        "c_peptide": c_peptide,
                        "bmi": bmi,
                        "comorbidities": comorbidities
                    },
                    "style": {
                        "color": "#FF6B6B",
                        "size": "large",
                        "shape": "star"
                    }
                })

                # Add similar patient nodes, treatment nodes, and outcome nodes
                treatment_nodes_added = set()

                for idx, record in enumerate(records):
                    patient_id = record['patient_id']
                    treatment_name = record['treatment']
                    similarity = round(record['similarity_score'], 2)

                    # 1. Add similar patient node
                    nodes.append({
                        "id": patient_id,
                        "type": "patient",
                        "label": f"Patient {idx + 1}",
                        "data": {
                            "patient_id": patient_id,
                            "age": int(record['age']),
                            "gender": record['gender'],
                            "hba1c_baseline": round(record['hba1c'], 1),
                            "c_peptide": round(record['c_peptide'], 2),
                            "bmi": round(record['bmi'], 1),
                            "egfr": round(record['egfr'], 1),
                            "comorbidities": record['patient_comorbidities'],
                            "similarity_score": similarity
                        },
                        "style": {
                            "color": self._get_similarity_color(similarity),
                            "size": "medium",
                            "shape": "circle"
                        }
                    })

                    # 2. Add treatment node (if not already added)
                    treatment_id = f"treatment_{treatment_name}"
                    if treatment_id not in treatment_nodes_added:
                        nodes.append({
                            "id": treatment_id,
                            "type": "treatment",
                            "label": treatment_name,
                            "data": {
                                "treatment": treatment_name,
                                "drug_class": record['drug_class']
                            },
                            "style": {
                                "color": "#4ECDC4",
                                "size": "medium",
                                "shape": "square"
                            }
                        })
                        treatment_nodes_added.add(treatment_id)

                    # 3. Add outcome node
                    outcome_id = f"outcome_{patient_id}"
                    nodes.append({
                        "id": outcome_id,
                        "type": "outcome",
                        "label": f"Outcome {idx + 1}",
                        "data": {
                            "hba1c_reduction": round(record['reduction'], 1),
                            "hba1c_followup": round(record['hba1c_followup'], 1),
                            "time_to_target": record['time_to_target'],
                            "adverse_events": record['adverse_events'],
                            "outcome_category": record['outcome_category']
                        },
                        "style": {
                            "color": self._get_outcome_color(record['outcome_category']),
                            "size": "small",
                            "shape": "diamond"
                        }
                    })

                    # 4. Add edges

                    # Query patient -> Similar patient (SIMILAR_TO)
                    edges.append({
                        "id": f"edge_query_{patient_id}",
                        "source": "query_patient",
                        "target": patient_id,
                        "type": "SIMILAR_TO",
                        "label": f"{similarity * 100:.0f}% similar",
                        "data": {
                            "similarity_score": similarity,
                            "matched_features": ["age", "hba1c", "c_peptide", "comorbidities"]
                        },
                        "style": {
                            "width": self._get_edge_width(similarity),
                            "color": "#95A5A6"
                        }
                    })

                    # Similar patient -> Treatment (RECEIVED_TREATMENT)
                    edges.append({
                        "id": f"edge_{patient_id}_{treatment_id}",
                        "source": patient_id,
                        "target": treatment_id,
                        "type": "RECEIVED_TREATMENT",
                        "label": "received",
                        "data": {},
                        "style": {
                            "width": 2,
                            "color": "#3498DB"
                        }
                    })

                    # Treatment -> Outcome (RESULTED_IN)
                    edges.append({
                        "id": f"edge_{treatment_id}_{outcome_id}",
                        "source": treatment_id,
                        "target": outcome_id,
                        "type": "RESULTED_IN",
                        "label": f"Δ{record['reduction']:.1f}%",
                        "data": {
                            "hba1c_reduction": round(record['reduction'], 1)
                        },
                        "style": {
                            "width": 2,
                            "color": "#2ECC71"
                        }
                    })

                # Metadata
                metadata = {
                    "query_patient": {
                        "age": age,
                        "hba1c_baseline": hba1c,
                        "c_peptide": c_peptide,
                        "bmi": bmi,
                        "comorbidities": comorbidities
                    },
                    "filters_applied": {
                        "age_group": age_group,
                        "hba1c_severity": hba1c_severity,
                        "treatment": treatment_filter,
                        "comorbidities": comorbidities
                    },
                    "results_found": len(records),
                    "similarity_range": {
                        "min": round(min(r['similarity_score'] for r in records), 2),
                        "max": round(max(r['similarity_score'] for r in records), 2),
                        "avg": round(sum(r['similarity_score'] for r in records) / len(records), 2)
                    }
                }

                return {
                    "nodes": nodes,
                    "edges": edges,
                    "metadata": metadata
                }

        except Exception as e:
            logger.error(f"Error finding similar cases (graph): {e}")
            return {
                "nodes": [],
                "edges": [],
                "metadata": {"error": str(e)}
            }

    def get_patient_by_id(self, patient_id: str) -> Dict[str, Any]:
        """
        Get complete patient details from Neo4j by patient ID.

        Args:
            patient_id: Patient ID from Neo4j (e.g., "P000123")

        Returns:
            Dictionary containing:
            - patient_id: Patient identifier
            - demographics: Age, gender, ethnicity
            - clinical_features: All 21 base features (HbA1c, BMI, eGFR, etc.)
            - comorbidities: List of conditions
            - treatment: Treatment received
            - outcome: Treatment outcome details

            Returns None if patient not found

        Example:
            patient = db.get_patient_by_id("P000123")
            if patient:
                print(f"Patient: {patient['patient_id']}")
                print(f"Age: {patient['demographics']['age']}")
                print(f"Treatment: {patient['treatment']['drug_name']}")
                print(f"Outcome: {patient['outcome']['outcome_category']}")
        """
        if not self.is_connected():
            logger.warning("Neo4j not connected for patient lookup")
            return None

        try:
            with self.driver.session() as session:
                query = """
                // Get patient node
                MATCH (p:Patient {patient_id: $patient_id})

                // Get treatment
                OPTIONAL MATCH (p)-[:RECEIVED_TREATMENT]->(t:Treatment)

                // Get outcome
                OPTIONAL MATCH (o:Outcome {patient_id: $patient_id})

                // Get comorbidities
                OPTIONAL MATCH (p)-[:HAS_CONDITION]->(c:Comorbidity)

                WITH p, t, o, collect(DISTINCT c.condition_name) AS comorbidities

                RETURN p.patient_id AS patient_id,
                       p.age AS age,
                       p.gender AS gender,
                       p.ethnicity AS ethnicity,
                       p.hba1c_baseline AS hba1c_baseline,
                       p.diabetes_duration AS diabetes_duration,
                       p.fasting_glucose AS fasting_glucose,
                       p.c_peptide AS c_peptide,
                       p.egfr AS egfr,
                       p.bmi AS bmi,
                       p.bp_systolic AS bp_systolic,
                       p.bp_diastolic AS bp_diastolic,
                       p.alt AS alt,
                       p.ldl AS ldl,
                       p.hdl AS hdl,
                       p.triglycerides AS triglycerides,
                       p.previous_prediabetes AS previous_prediabetes,
                       p.age_group AS age_group,
                       p.bmi_category AS bmi_category,
                       p.hba1c_severity AS hba1c_severity,
                       p.kidney_function AS kidney_function,
                       comorbidities,
                       t.drug_name AS treatment_name,
                       t.drug_class AS drug_class,
                       t.cost_category AS cost_category,
                       t.evidence_level AS evidence_level,
                       o.hba1c_reduction AS hba1c_reduction,
                       o.hba1c_followup AS hba1c_followup,
                       o.time_to_target AS time_to_target,
                       o.adverse_events AS adverse_events,
                       o.outcome_category AS outcome_category,
                       o.success AS success
                """

                result = session.run(query, patient_id=patient_id)
                record = result.single()

                if not record:
                    logger.info(f"Patient {patient_id} not found in Neo4j")
                    return None

                # Build response dictionary
                patient_data = {
                    "patient_id": record["patient_id"],
                    "demographics": {
                        "age": int(record["age"]),
                        "gender": record["gender"],
                        "ethnicity": record["ethnicity"],
                        "age_group": record["age_group"]
                    },
                    "clinical_features": {
                        "hba1c_baseline": round(float(record["hba1c_baseline"]), 1),
                        "diabetes_duration": round(float(record["diabetes_duration"]), 1),
                        "fasting_glucose": round(float(record["fasting_glucose"]), 1),
                        "c_peptide": round(float(record["c_peptide"]), 2),
                        "egfr": round(float(record["egfr"]), 1),
                        "bmi": round(float(record["bmi"]), 1),
                        "bp_systolic": int(record["bp_systolic"]),
                        "bp_diastolic": int(record["bp_diastolic"]),
                        "alt": round(float(record["alt"]), 1),
                        "ldl": round(float(record["ldl"]), 1),
                        "hdl": round(float(record["hdl"]), 1),
                        "triglycerides": round(float(record["triglycerides"]), 1),
                        "previous_prediabetes": bool(record["previous_prediabetes"])
                    },
                    "clinical_categories": {
                        "bmi_category": record["bmi_category"],
                        "hba1c_severity": record["hba1c_severity"],
                        "kidney_function": record["kidney_function"]
                    },
                    "comorbidities": record["comorbidities"] or [],
                    "treatment": None,
                    "outcome": None
                }

                # Add treatment if exists
                if record["treatment_name"]:
                    patient_data["treatment"] = {
                        "drug_name": record["treatment_name"],
                        "drug_class": record["drug_class"],
                        "cost_category": record["cost_category"],
                        "evidence_level": record["evidence_level"]
                    }

                # Add outcome if exists
                if record["hba1c_reduction"] is not None:
                    patient_data["outcome"] = {
                        "hba1c_reduction": round(float(record["hba1c_reduction"]), 1),
                        "hba1c_followup": round(float(record["hba1c_followup"]), 1),
                        "time_to_target": record["time_to_target"],
                        "adverse_events": record["adverse_events"],
                        "outcome_category": record["outcome_category"],
                        "success": bool(record["success"])
                    }

                logger.info(f"Retrieved patient {patient_id} from Neo4j")
                return patient_data

        except Exception as e:
            logger.error(f"Error retrieving patient {patient_id} from Neo4j: {e}")
            return None

    # =========================================================================
    # HELPER METHODS FOR GRAPH VISUALIZATION
    # =========================================================================

    def _get_similarity_color(self, similarity_score: float) -> str:
        """Return color based on similarity score (0.0-1.0)"""
        if similarity_score >= 0.9:
            return "#27AE60"  # Dark green - very similar
        elif similarity_score >= 0.8:
            return "#2ECC71"  # Green - highly similar
        elif similarity_score >= 0.7:
            return "#F39C12"  # Orange - moderately similar
        else:
            return "#E74C3C"  # Red - less similar

    def _get_outcome_color(self, outcome_category: str) -> str:
        """Return color based on outcome category"""
        color_map = {
            "Success": "#27AE60",  # Green
            "Partial": "#F39C12",  # Orange
            "Failure": "#E74C3C"  # Red
        }
        return color_map.get(outcome_category, "#95A5A6")  # Gray default

    def _get_edge_width(self, similarity_score: float) -> int:
        """Return edge width based on similarity (thicker = more similar)"""
        if similarity_score >= 0.9:
            return 5
        elif similarity_score >= 0.8:
            return 4
        elif similarity_score >= 0.7:
            return 3
        else:
            return 2

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _get_fallback_guidelines(self, treatment: str) -> Dict[str, Any]:
        """Return fallback guidelines when database unavailable"""
        return {
            "treatment": treatment,
            "indication": f"Standard indication for {treatment}",
            "guidelines": [f"{treatment} per standard clinical practice"],
            "dosing": "Individualize based on patient response",
            "monitoring": "Regular follow-up recommended"
        }

    def _get_age_group(self, age: float) -> str:
        """Convert age to categorical group"""
        if age < 40:
            return '<40'
        elif age < 50:
            return '40-50'
        elif age < 60:
            return '50-60'
        elif age < 70:
            return '60-70'
        else:
            return '>70'

    def _get_bmi_category(self, bmi: float) -> str:
        """Convert BMI to categorical group"""
        if bmi < 25:
            return 'Normal'
        elif bmi < 30:
            return 'Overweight'
        elif bmi < 35:
            return 'Obese'
        else:
            return 'Severe_Obese'

    def _get_hba1c_severity(self, hba1c: float) -> str:
        """Convert HbA1c to severity category"""
        if hba1c < 7:
            return 'Mild'
        elif hba1c < 8:
            return 'Moderate'
        elif hba1c < 9:
            return 'Severe'
        else:
            return 'Very_Severe'


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_neo4j_graph_database(uri: str, username: str, password: str) -> Neo4jGraphDatabase:
    """
    Factory function to create and connect Neo4j graph database.

    Args:
        uri: Neo4j URI (e.g., 'neo4j://localhost:7687')
        username: Database username
        password: Database password

    Returns:
        Connected Neo4jGraphDatabase instance

    Raises:
        RuntimeError: If connection fails
    """
    db = Neo4jGraphDatabase(uri, username, password)

    if not db.connect():
        raise RuntimeError(
            f"Failed to connect to Neo4j at {uri}. "
            "Ensure Neo4j is running and credentials are correct."
        )

    return db