from typing import Optional, List

from src.modules.recommendation.domain.models.prediction import Prediction
from src.shared.data.base.repository import BaseRepository


class PredictionRepository(BaseRepository[Prediction]):
    """
    Repository for Prediction entity operations.
    """

    def __init__(self):
        super().__init__(Prediction)

    def find_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> List[Prediction]:
        """
        Find all predictions for a patient.

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted predictions (default: False)

        Returns:
            List of Prediction instances
        """
        return self.find_many_by({"patient_id": patient_id}, include_deleted=include_deleted)

    def find_latest_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> Optional[Prediction]:
        """
        Find the most recent prediction for a patient.

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted predictions (default: False)

        Returns:
            Latest Prediction instance if found, None otherwise
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(patient_id=patient_id).order_by(Prediction.created_at.desc()).first()

    def find_by_model_version(self, model_version: str, include_deleted: bool = False) -> List[Prediction]:
        """
        Find all predictions made by a specific model version.

        Args:
            model_version: The model version to search for
            include_deleted: Whether to include soft-deleted predictions (default: False)

        Returns:
            List of Prediction instances
        """
        return self.find_many_by({"model_version": model_version}, include_deleted=include_deleted)

    def count_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> int:
        """
        Count predictions for a patient.

        Args:
            patient_id: The patient ID to count predictions for
            include_deleted: Whether to include soft-deleted predictions (default: False)

        Returns:
            Number of predictions
        """
        return self.count({"patient_id": patient_id}, include_deleted=include_deleted)