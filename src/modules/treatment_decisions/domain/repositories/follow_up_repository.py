from typing import List, Optional
from datetime import date

from src.modules.treatment_decisions.domain.models.enums import FollowUpStatus
from src.modules.treatment_decisions.domain.models.follow_up import FollowUp
from src.shared.data.base.repository import BaseRepository


class FollowUpRepository(BaseRepository[FollowUp]):
    """
    Repository for FollowUp entity operations.
    """

    def __init__(self):
        super().__init__(FollowUp)

    def find_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> List[FollowUp]:
        """
        Find all follow-ups for a patient.

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted follow-ups (default: False)

        Returns:
            List of FollowUp instances ordered by scheduled_date desc
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(patient_id=patient_id).order_by(FollowUp.scheduled_date.desc()).all()

    def find_by_decision_id(self, decision_id: str, include_deleted: bool = False) -> List[FollowUp]:
        """
        Find all follow-ups for a treatment decision.

        Args:
            decision_id: The decision ID to search for
            include_deleted: Whether to include soft-deleted follow-ups (default: False)

        Returns:
            List of FollowUp instances ordered by scheduled_date
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(decision_id=decision_id).order_by(FollowUp.scheduled_date).all()

    def find_by_status(self, status: FollowUpStatus, include_deleted: bool = False) -> List[FollowUp]:
        """
        Find all follow-ups with a specific status.

        Args:
            status: The follow-up status to search for
            include_deleted: Whether to include soft-deleted follow-ups (default: False)

        Returns:
            List of FollowUp instances
        """
        return self.find_many_by({"status": status}, include_deleted=include_deleted)

    def find_upcoming(self, include_deleted: bool = False) -> List[FollowUp]:
        """
        Find all upcoming scheduled follow-ups (status=scheduled, future date).

        Args:
            include_deleted: Whether to include soft-deleted follow-ups (default: False)

        Returns:
            List of upcoming FollowUp instances ordered by scheduled_date
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        today = date.today()
        return query.filter(
            FollowUp.status == FollowUpStatus.SCHEDULED,
            FollowUp.scheduled_date >= today
        ).order_by(FollowUp.scheduled_date).all()

    def find_overdue(self, include_deleted: bool = False) -> List[FollowUp]:
        """
        Find all overdue follow-ups (status=scheduled, past date).

        Args:
            include_deleted: Whether to include soft-deleted follow-ups (default: False)

        Returns:
            List of overdue FollowUp instances ordered by scheduled_date
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        today = date.today()
        return query.filter(
            FollowUp.status == FollowUpStatus.SCHEDULED,
            FollowUp.scheduled_date < today
        ).order_by(FollowUp.scheduled_date).all()

    def find_completed_by_patient(self, patient_id: str, include_deleted: bool = False) -> List[FollowUp]:
        """
        Find all completed follow-ups for a patient.

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted follow-ups (default: False)

        Returns:
            List of completed FollowUp instances ordered by visit_date desc
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter(
            FollowUp.patient_id == patient_id,
            FollowUp.status == FollowUpStatus.COMPLETED
        ).order_by(FollowUp.visit_date.desc()).all()

    def find_latest_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> Optional[FollowUp]:
        """
        Find the most recent follow-up for a patient.

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted follow-ups (default: False)

        Returns:
            Latest FollowUp instance if found, None otherwise
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(patient_id=patient_id).order_by(FollowUp.created_at.desc()).first()

    def count_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> int:
        """
        Count follow-ups for a patient.

        Args:
            patient_id: The patient ID to count follow-ups for
            include_deleted: Whether to include soft-deleted follow-ups (default: False)

        Returns:
            Number of follow-ups
        """
        return self.count({"patient_id": patient_id}, include_deleted=include_deleted)