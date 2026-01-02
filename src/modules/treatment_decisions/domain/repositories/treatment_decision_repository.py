from typing import Optional, List

from src.modules.treatment_decisions.domain.models.enums import DecisionType
from src.modules.treatment_decisions.domain.models.treatment_decision import TreatmentDecision
from src.shared.data.base.repository import BaseRepository


class TreatmentDecisionRepository(BaseRepository[TreatmentDecision]):
    """
    Repository for TreatmentDecision entity operations.
    """

    def __init__(self):
        super().__init__(TreatmentDecision)

    def find_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> List[TreatmentDecision]:
        """
        Find all treatment decisions for a patient.

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted decisions (default: False)

        Returns:
            List of TreatmentDecision instances ordered by decided_at desc
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(patient_id=patient_id).order_by(TreatmentDecision.decided_at.desc()).all()

    def find_by_prediction_id(self, prediction_id: str, include_deleted: bool = False) -> Optional[TreatmentDecision]:
        """
        Find treatment decision by prediction ID.

        Args:
            prediction_id: The prediction ID to search for
            include_deleted: Whether to include soft-deleted decisions (default: False)

        Returns:
            TreatmentDecision instance if found, None otherwise
        """
        return self.find_one_by({"prediction_id": prediction_id}, include_deleted=include_deleted)

    def find_latest_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> Optional[TreatmentDecision]:
        """
        Find the most recent treatment decision for a patient.

        Args:
            patient_id: The patient ID to search for
            include_deleted: Whether to include soft-deleted decisions (default: False)

        Returns:
            Latest TreatmentDecision instance if found, None otherwise
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(patient_id=patient_id).order_by(TreatmentDecision.decided_at.desc()).first()

    def find_by_decision_type(self, decision_type: DecisionType, include_deleted: bool = False) -> List[TreatmentDecision]:
        """
        Find all decisions of a specific type.

        Args:
            decision_type: The decision type to search for
            include_deleted: Whether to include soft-deleted decisions (default: False)

        Returns:
            List of TreatmentDecision instances
        """
        return self.find_many_by({"decision_type": decision_type}, include_deleted=include_deleted)

    def find_with_outcomes(self, include_deleted: bool = False) -> List[TreatmentDecision]:
        """
        Find all decisions that have recorded outcomes.

        Args:
            include_deleted: Whether to include soft-deleted decisions (default: False)

        Returns:
            List of TreatmentDecision instances with outcomes
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter(TreatmentDecision.observed_reduction.isnot(None)).all()

    def find_for_training(self, include_deleted: bool = False) -> List[TreatmentDecision]:
        """
        Find decisions available for ML training (have outcomes, not yet used).

        Args:
            include_deleted: Whether to include soft-deleted decisions (default: False)

        Returns:
            List of TreatmentDecision instances ready for training
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter(
            TreatmentDecision.observed_reduction.isnot(None),
            TreatmentDecision.used_for_training == False
        ).all()

    def count_by_patient_id(self, patient_id: str, include_deleted: bool = False) -> int:
        """
        Count treatment decisions for a patient.

        Args:
            patient_id: The patient ID to count decisions for
            include_deleted: Whether to include soft-deleted decisions (default: False)

        Returns:
            Number of treatment decisions
        """
        return self.count({"patient_id": patient_id}, include_deleted=include_deleted)