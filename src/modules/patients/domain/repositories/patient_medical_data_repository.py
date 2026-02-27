from typing import Optional, List

from src.modules.patients.domain.models.patient_medical_data import PatientMedicalData
from src.shared.data.base.repository import BaseRepository


class PatientMedicalDataRepository(BaseRepository[PatientMedicalData]):
    """
    Repository for PatientMedicalData entity operations.
    """

    def __init__(self):
        super().__init__(PatientMedicalData)

    def find_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> List[PatientMedicalData]:
        """
        Find all medical data records for a patient (ordered by most recent first).

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted records

        Returns:
            List of PatientMedicalData instances ordered by created_at desc
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(patient_id=patient_id).order_by(PatientMedicalData.created_at.desc()).all()

    def find_latest_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> Optional[PatientMedicalData]:
        """
        Find the most recent medical data record for a patient.

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted records

        Returns:
            Latest PatientMedicalData instance if found, None otherwise
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(patient_id=patient_id).order_by(PatientMedicalData.created_at.desc()).first()

    def count_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> int:
        """
        Count medical data records for a patient.

        Args:
            patient_id: The patient ID to count records for
            include_deleted: Whether to include soft-deleted records

        Returns:
            Number of medical data records
        """
        return self.count({"patient_id": patient_id}, include_deleted=include_deleted)

    def exists_for_patient(self, patient_id: str, include_deleted: bool = False) -> bool:
        """
        Check if any medical data exists for a given patient.

        Args:
            patient_id: The patient ID to check
            include_deleted: Whether to include soft-deleted records

        Returns:
            True if medical data exists, False otherwise
        """
        return self.exists({"patient_id": patient_id}, include_deleted=include_deleted)