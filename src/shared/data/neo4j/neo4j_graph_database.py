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
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
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