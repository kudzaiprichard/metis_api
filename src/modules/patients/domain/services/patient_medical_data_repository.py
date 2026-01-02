from typing import Optional

from src.modules.patients.domain.models.patient_medical_data import PatientMedicalData
from src.shared.data.base.repository import BaseRepository


class PatientMedicalDataRepository(BaseRepository[PatientMedicalData]):
    """
    Repository for PatientMedicalData entity operations.
    """

    def __init__(self):
        super().__init__(PatientMedicalData)

    def find_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> Optional[PatientMedicalData]:
        """
        Find medical data by patient ID.

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            PatientMedicalData instance if found, None otherwise
        """
        return self.find_one_by({"patient_id": patient_id}, include_deleted=include_deleted)

    def exists_for_patient(self, patient_id: str, include_deleted: bool = False) -> bool:
        """
        Check if medical data exists for a given patient.

        Args:
            patient_id: The patient ID to check
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            True if medical data exists, False otherwise
        """
        return self.exists({"patient_id": patient_id}, include_deleted=include_deleted)