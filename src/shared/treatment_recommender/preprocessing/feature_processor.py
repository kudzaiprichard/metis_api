"""
Patient feature preprocessing for diabetes treatment recommendation.

This module provides stateless feature processing:
- Gender/ethnicity encoding
- Engineered feature creation (8 derived features)
- Continuous feature scaling
- Binary feature preservation
- Feature vector construction
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class PatientFeatureProcessor:
    """
    Stateless feature processor for patient data.

    Processes patient dictionaries into 21-feature vectors:
    - 2 categorical features (gender, ethnicity) → encoded
    - 13 base continuous features → scaled to mean=0, std=1
    - 6 base binary features → preserved as 0/1
    - 8 engineered features → created from base features, then scaled/preserved

    Pipeline:
    1. Create 8 engineered features from base features
    2. Encode categorical variables (gender, ethnicity)
    3. Scale continuous features (base + engineered)
    4. Return 21-feature array

    This processor is:
    - Stateless: No internal state between calls
    - Reusable: Can be shared across multiple pipelines
    - Thread-safe: Immutable after initialization

    Usage:
        processor = PatientFeatureProcessor(
            scaler_path='features/feature_scaler.pkl',
            metadata_path='features/preprocessing_metadata.json',
            verbose=False
        )

        features = processor.process_patient(patient_dict)
        # Returns: numpy array of shape (21,)
    """

    def __init__(self,
                 scaler_path: str,
                 metadata_path: str,
                 verbose: bool = False):
        """
        Initialize feature processor.

        Args:
            scaler_path: Path to fitted StandardScaler pickle file
            metadata_path: Path to preprocessing metadata JSON
            verbose: If True, print detailed processing logs

        Raises:
            FileNotFoundError: If scaler or metadata file doesn't exist
            ValueError: If metadata is invalid
        """
        self.verbose = verbose

        # Validate paths
        scaler_path = Path(scaler_path)
        metadata_path = Path(metadata_path)

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        # Load scaler
        if self.verbose:
            print(f"[FeatureProcessor] Loading scaler from: {scaler_path}")

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load metadata
        if self.verbose:
            print(f"[FeatureProcessor] Loading metadata from: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Extract feature information
        self.feature_cols = self.metadata['feature_cols']
        self.continuous_features = self.metadata['continuous_features']
        self.binary_features = self.metadata['binary_features']
        self.ethnicity_map = self.metadata['ethnicity_map']
        self.treatment_map = self.metadata.get('treatment_map', {
            'Metformin': 0, 'GLP-1': 1, 'SGLT-2': 2, 'DPP-4': 3, 'Insulin': 4
        })

        # Validate metadata
        if len(self.feature_cols) != 21:
            raise ValueError(f"Expected 21 features, got {len(self.feature_cols)}")

        if self.verbose:
            print(f"[FeatureProcessor] Initialized successfully")
            print(f"[FeatureProcessor] Feature count: {len(self.feature_cols)}")
            print(f"[FeatureProcessor] Continuous: {len(self.continuous_features)}")
            print(f"[FeatureProcessor] Binary: {len(self.binary_features)}")

    def _add_engineered_features(self, patient_dict: Dict) -> Dict:
        """
        Create 8 engineered features from base patient data.

        This matches the notebook (Cell 4) feature engineering exactly.

        Engineered features:
        1. insulin_deficiency_score - Measures insulin production decline
        2. beta_cell_reserve - Measures remaining beta cell function
        3. glucose_severity - Combined measure of glucose dysregulation
        4. disease_progression - Duration × severity interaction
        5. metabolic_syndrome_score - Composite metabolic risk
        6. cv_risk_score - Cardiovascular risk factors
        7. kidney_severity - CKD stage category
        8. comorbidity_count - Total burden of complications

        Args:
            patient_dict: Raw patient dictionary with base features

        Returns:
            Enriched patient dictionary with 8 additional features

        Example:
            raw_patient = {'age': 58, 'c_peptide': 1.5, ...}
            enriched = self._add_engineered_features(raw_patient)
            # Now has: {..., 'insulin_deficiency_score': 1.23, ...}
        """
        # Convert to DataFrame for easier computation
        df = pd.DataFrame([patient_dict])

        # 1. Insulin deficiency score (lower C-peptide + longer duration = higher score)
        df['insulin_deficiency_score'] = (2.0 - df['c_peptide']) * (1 + df['diabetes_duration'] / 15)

        # 2. Beta cell reserve (higher C-peptide + shorter duration = better reserve)
        df['beta_cell_reserve'] = df['c_peptide'] * (1 / (1 + df['diabetes_duration'] / 10))

        # 3. Glucose severity (combination of fasting glucose and HbA1c)
        df['glucose_severity'] = (df['fasting_glucose'] / 100) * df['hba1c_baseline']

        # 4. Disease progression (duration * severity)
        df['disease_progression'] = df['diabetes_duration'] * df['hba1c_baseline']

        # 5. Metabolic syndrome score (BMI + BP + lipids)
        df['metabolic_syndrome_score'] = (
                (df['bmi'] - 25) / 10 +
                (df['bp_systolic'] - 120) / 40 +
                (df['triglycerides'] - 150) / 100 +
                (60 - df['hdl']) / 20
        )

        # 6. Cardiovascular risk (CVD + hypertension + age)
        df['cv_risk_score'] = df['cvd'] + df['hypertension'] + (df['age'] > 65).astype(int)

        # 7. Kidney severity (based on eGFR stages)
        df['kidney_severity'] = pd.cut(
            df['egfr'],
            bins=[0, 30, 60, 90, 200],
            labels=[3, 2, 1, 0]
        ).astype(int)

        # 8. Total comorbidity burden
        df['comorbidity_count'] = (
                df['hypertension'] + df['ckd'] + df['cvd'] +
                df['nafld'] + df['retinopathy']
        )

        # Convert back to dictionary
        return df.iloc[0].to_dict()

    def process_patient(self, patient_dict: Dict) -> np.ndarray:
        """
        Process patient dictionary into feature vector.

        Pipeline:
        1. Add 8 engineered features (CRITICAL STEP)
        2. Extract raw feature values
        3. Encode categorical variables
        4. Scale continuous features
        5. Return 21-feature array

        Args:
            patient_dict: Dictionary with patient data
                Required keys: age, gender, ethnicity, hba1c_baseline, diabetes_duration,
                              fasting_glucose, c_peptide, egfr, bmi,
                              bp_systolic, bp_diastolic, alt, ldl, hdl, triglycerides,
                              previous_prediabetes, hypertension, ckd, cvd, nafld, retinopathy

        Returns:
            Preprocessed feature array of shape (21,)

        Example:
            patient = {
                'age': 58, 'gender': 'Female', 'ethnicity': 'Caucasian',
                'hba1c_baseline': 8.2, 'diabetes_duration': 5.0,
                'fasting_glucose': 165, 'c_peptide': 1.5, 'egfr': 75, 'bmi': 31.5,
                'bp_systolic': 135, 'bp_diastolic': 85, 'alt': 28,
                'ldl': 120, 'hdl': 45, 'triglycerides': 160,
                'previous_prediabetes': 1, 'hypertension': 1, 'ckd': 0,
                'cvd': 0, 'nafld': 1, 'retinopathy': 0
            }
            features = processor.process_patient(patient)
            # Returns: array of shape (21,) with all features
        """
        if self.verbose:
            print("[FeatureProcessor] Processing patient data")

        # STEP 1: Add engineered features (CRITICAL - must happen first)
        patient_enriched = self._add_engineered_features(patient_dict)

        if self.verbose:
            print("[FeatureProcessor] Engineered features created")

        # STEP 2: Extract raw features (now including engineered ones)
        raw_features = self._extract_raw_features(patient_enriched)

        if self.verbose:
            print(f"[FeatureProcessor] Raw features extracted: {len(raw_features)}")

        # STEP 3: Create DataFrame for scaling
        df_temp = pd.DataFrame([raw_features], columns=self.feature_cols)

        # STEP 4: Scale continuous features only
        df_temp[self.continuous_features] = self.scaler.transform(
            df_temp[self.continuous_features]
        )

        if self.verbose:
            print("[FeatureProcessor] Continuous features scaled")

        # STEP 5: Return as flat array
        result = df_temp.values.flatten()

        if self.verbose:
            print(f"[FeatureProcessor] Feature vector shape: {result.shape}")
            print(f"[FeatureProcessor] Sample values: {result[:5]}")

        return result

    def _extract_raw_features(self, patient_dict: Dict) -> np.ndarray:
        """
        Extract raw feature values and encode categorical variables.

        This method handles both base features AND engineered features.

        Args:
            patient_dict: Patient data dictionary (with engineered features already added)

        Returns:
            Raw feature array (21,) with categorical encoding applied
        """
        raw_features = np.zeros(len(self.feature_cols))

        for i, col in enumerate(self.feature_cols):
            if col == 'gender':
                # Encode: Female=1, Male=0
                raw_features[i] = 1 if patient_dict.get('gender', 'Male') == 'Female' else 0

            elif col == 'ethnicity':
                # Encode: African=0, Asian=1, Caucasian=2, Hispanic=3, Other=4
                ethnicity = patient_dict.get('ethnicity', 'Caucasian')
                raw_features[i] = self.ethnicity_map.get(ethnicity, 2)

            else:
                # All other features (base + engineered): use value or default to 0
                raw_features[i] = patient_dict.get(col, 0)

        return raw_features

    def process_batch(self, patients: List[Dict]) -> np.ndarray:
        """
        Process multiple patients at once.

        Args:
            patients: List of patient dictionaries

        Returns:
            Feature matrix of shape (n_patients, 21)

        Example:
            patients = [patient1, patient2, patient3]
            features = processor.process_batch(patients)
            # Returns: (3, 21) array
        """
        if self.verbose:
            print(f"[FeatureProcessor] Processing batch of {len(patients)} patients")

        feature_vectors = []

        for patient in patients:
            features = self.process_patient(patient)
            feature_vectors.append(features)

        result = np.vstack(feature_vectors)

        if self.verbose:
            print(f"[FeatureProcessor] Batch shape: {result.shape}")

        return result

    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names.

        Returns:
            List of 21 feature names in processing order
        """
        return self.feature_cols.copy()

    def get_feature_info(self) -> Dict:
        """
        Get detailed feature information.

        Returns:
            Dictionary with feature categorization and mappings
        """
        return {
            'all_features': self.feature_cols,
            'continuous_features': self.continuous_features,
            'binary_features': self.binary_features,
            'ethnicity_encoding': self.ethnicity_map,
            'treatment_encoding': self.treatment_map,
            'gender_encoding': {'Female': 1, 'Male': 0},
            'total_features': len(self.feature_cols),
            'engineered_features': [
                'insulin_deficiency_score', 'beta_cell_reserve', 'glucose_severity',
                'disease_progression', 'metabolic_syndrome_score', 'cv_risk_score',
                'kidney_severity', 'comorbidity_count'
            ]
        }

    def encode_treatment(self, treatment_name: str) -> int:
        """
        Encode treatment name to integer ID.

        Args:
            treatment_name: Treatment name (e.g., 'Metformin', 'GLP-1')

        Returns:
            Treatment ID (0-4)

        Raises:
            ValueError: If treatment name is invalid
        """
        if treatment_name not in self.treatment_map:
            valid = ', '.join(self.treatment_map.keys())
            raise ValueError(f"Invalid treatment '{treatment_name}'. Valid: {valid}")

        return self.treatment_map[treatment_name]

    def decode_treatment(self, treatment_id: int) -> str:
        """
        Decode treatment ID to name.

        Args:
            treatment_id: Treatment ID (0-4)

        Returns:
            Treatment name

        Raises:
            ValueError: If treatment ID is invalid
        """
        # Reverse mapping
        id_to_name = {v: k for k, v in self.treatment_map.items()}

        if treatment_id not in id_to_name:
            raise ValueError(f"Invalid treatment ID {treatment_id}. Valid: 0-4")

        return id_to_name[treatment_id]

    def get_feature_statistics(self, patient_dict: Dict) -> Dict:
        """
        Get statistics about processed features.

        Args:
            patient_dict: Patient data

        Returns:
            Dictionary with feature statistics
        """
        features = self.process_patient(patient_dict)

        return {
            'feature_count': len(features),
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'non_zero_count': int(np.count_nonzero(features))
        }

    def validate_patient_dict(self, patient_dict: Dict) -> tuple:
        """
        Validate that patient dictionary has all required base features.

        Args:
            patient_dict: Patient data dictionary

        Returns:
            Tuple of (is_valid, missing_fields)

        Example:
            is_valid, missing = processor.validate_patient_dict(patient)
            if not is_valid:
                print(f"Missing fields: {missing}")
        """
        required_base_fields = [
            'age', 'gender', 'ethnicity', 'hba1c_baseline', 'diabetes_duration',
            'fasting_glucose', 'c_peptide', 'egfr', 'bmi',
            'bp_systolic', 'bp_diastolic', 'alt', 'ldl', 'hdl', 'triglycerides',
            'previous_prediabetes', 'hypertension', 'ckd', 'cvd', 'nafld', 'retinopathy'
        ]

        missing = []
        for field in required_base_fields:
            if field not in patient_dict:
                missing.append(field)

        return len(missing) == 0, missing

    def get_engineered_feature_values(self, patient_dict: Dict) -> Dict:
        """
        Get the computed values of engineered features for inspection.

        Args:
            patient_dict: Patient data

        Returns:
            Dictionary with engineered feature values

        Example:
            engineered = processor.get_engineered_feature_values(patient)
            print(f"Insulin deficiency: {engineered['insulin_deficiency_score']:.2f}")
            print(f"Beta cell reserve: {engineered['beta_cell_reserve']:.2f}")
        """
        patient_enriched = self._add_engineered_features(patient_dict)

        engineered_features = {
            'insulin_deficiency_score': patient_enriched['insulin_deficiency_score'],
            'beta_cell_reserve': patient_enriched['beta_cell_reserve'],
            'glucose_severity': patient_enriched['glucose_severity'],
            'disease_progression': patient_enriched['disease_progression'],
            'metabolic_syndrome_score': patient_enriched['metabolic_syndrome_score'],
            'cv_risk_score': patient_enriched['cv_risk_score'],
            'kidney_severity': patient_enriched['kidney_severity'],
            'comorbidity_count': patient_enriched['comorbidity_count']
        }

        return engineered_features

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_feature_processor(scaler_path: str = 'features/feature_scaler.pkl',
                             metadata_path: str = 'features/preprocessing_metadata.json',
                             verbose: bool = False) -> PatientFeatureProcessor:
    """
    Factory function to create a feature processor instance.

    Args:
        scaler_path: Path to feature scaler
        metadata_path: Path to preprocessing metadata
        verbose: Enable detailed logging

    Returns:
        Configured PatientFeatureProcessor instance

    Example:
        processor = create_feature_processor()
        features = processor.process_patient(patient_dict)
    """
    return PatientFeatureProcessor(
        scaler_path=scaler_path,
        metadata_path=metadata_path,
        verbose=verbose
    )