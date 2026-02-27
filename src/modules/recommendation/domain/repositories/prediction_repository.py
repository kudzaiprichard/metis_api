from typing import Optional, List

from src.modules.recommendation.domain.models.prediction import Prediction
from src.modules.patients.domain.models.patient_medical_data import PatientMedicalData
from src.shared.data.base.repository import BaseRepository


class PredictionRepository(BaseRepository[Prediction]):
    """
    Repository for Prediction entity operations.
    """

    def __init__(self):
        super().__init__(Prediction)

    def find_by_medical_data_id(self, medical_data_id: str, include_deleted: bool = False) -> Optional[Prediction]:
        """
        Find prediction for a specific medical data record (1-to-1).

        Args:
            medical_data_id: The medical data ID to search for
            include_deleted: Whether to include soft-deleted predictions

        Returns:
            Prediction instance if found, None otherwise
        """
        return self.find_one_by({"medical_data_id": medical_data_id}, include_deleted=include_deleted)

    def find_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> List[Prediction]:
        """
        Find all predictions for a patient (via medical data join).

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted predictions

        Returns:
            List of Prediction instances ordered by created_at desc
        """
        query = self.model.query.join(
            PatientMedicalData,
            Prediction.medical_data_id == PatientMedicalData.id
        ).filter(PatientMedicalData.patient_id == patient_id)

        if not include_deleted:
            query = query.filter(Prediction.is_deleted == False)

        return query.order_by(Prediction.created_at.desc()).all()

    def find_latest_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> Optional[Prediction]:
        """
        Find the most recent prediction for a patient.

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted predictions

        Returns:
            Latest Prediction instance if found, None otherwise
        """
        query = self.model.query.join(
            PatientMedicalData,
            Prediction.medical_data_id == PatientMedicalData.id
        ).filter(PatientMedicalData.patient_id == patient_id)

        if not include_deleted:
            query = query.filter(Prediction.is_deleted == False)

        return query.order_by(Prediction.created_at.desc()).first()

    def find_by_model_version(self, model_version: str, include_deleted: bool = False) -> List[Prediction]:
        """
        Find all predictions made by a specific model version.

        Args:
            model_version: The model version to search for
            include_deleted: Whether to include soft-deleted predictions

        Returns:
            List of Prediction instances
        """
        return self.find_many_by({"model_version": model_version}, include_deleted=include_deleted)

    def count_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> int:
        """
        Count predictions for a patient.

        Args:
            patient_id: The patient ID to count predictions for
            include_deleted: Whether to include soft-deleted predictions

        Returns:
            Number of predictions
        """
        query = self.model.query.join(
            PatientMedicalData,
            Prediction.medical_data_id == PatientMedicalData.id
        ).filter(PatientMedicalData.patient_id == patient_id)

        if not include_deleted:
            query = query.filter(Prediction.is_deleted == False)

        return query.count()

    def exists_for_medical_data(self, medical_data_id: str, include_deleted: bool = False) -> bool:
        """
        Check if a prediction already exists for a medical data record.

        Args:
            medical_data_id: The medical data ID to check
            include_deleted: Whether to include soft-deleted predictions

        Returns:
            True if prediction exists, False otherwise
        """
        return self.exists({"medical_data_id": medical_data_id}, include_deleted=include_deleted)