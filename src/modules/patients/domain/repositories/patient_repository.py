from typing import Optional, List

from src.modules.patients.domain.models.patient import Patient
from src.shared.data.base.repository import BaseRepository


class PatientRepository(BaseRepository[Patient]):
    """
    Repository for Patient entity operations.
    """

    def __init__(self):
        super().__init__(Patient)

    def find_by_email(self, email: str, include_deleted: bool = False) -> Optional[Patient]:
        """
        Find a patient by their email address.

        Args:
            email: The email address to search for
            include_deleted: Whether to include soft-deleted patients (default: False)

        Returns:
            Patient instance if found, None otherwise
        """
        return self.find_one_by({"email": email}, include_deleted=include_deleted)

    def find_by_mobile_number(self, mobile_number: str, include_deleted: bool = False) -> Optional[Patient]:
        """
        Find a patient by their mobile number.

        Args:
            mobile_number: The mobile number to search for
            include_deleted: Whether to include soft-deleted patients (default: False)

        Returns:
            Patient instance if found, None otherwise
        """
        return self.find_one_by({"mobile_number": mobile_number}, include_deleted=include_deleted)

    def search_by_name(self, search_term: str, include_deleted: bool = False) -> List[Patient]:
        """
        Search patients by first name or last name (case-insensitive).

        Args:
            search_term: The name to search for
            include_deleted: Whether to include soft-deleted patients (default: False)

        Returns:
            List of Patient instances matching the search term
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        search_pattern = f"%{search_term}%"
        return query.filter(
            (Patient.first_name.ilike(search_pattern)) |
            (Patient.last_name.ilike(search_pattern))
        ).all()