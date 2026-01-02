from typing import List

from src.modules.recommendation.models.enums import SafetySeverity
from src.modules.recommendation.models.safety_warning import SafetyWarning
from src.shared.data.base.repository import BaseRepository


class SafetyWarningRepository(BaseRepository[SafetyWarning]):
    """
    Repository for SafetyWarning entity operations.
    """

    def __init__(self):
        super().__init__(SafetyWarning)

    def find_by_prediction_id(self, prediction_id: str, include_deleted: bool = False) -> List[SafetyWarning]:
        """
        Find all safety warnings for a prediction.

        Args:
            prediction_id: The prediction ID to search for
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            List of SafetyWarning instances
        """
        return self.find_many_by({"prediction_id": prediction_id}, include_deleted=include_deleted)

    def find_by_severity(self, prediction_id: str, severity: SafetySeverity, include_deleted: bool = False) -> List[SafetyWarning]:
        """
        Find warnings by severity level for a prediction.

        Args:
            prediction_id: The prediction ID to search for
            severity: The severity level to filter by
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            List of SafetyWarning instances matching the severity
        """
        return self.find_many_by({"prediction_id": prediction_id, "severity": severity}, include_deleted=include_deleted)

    def has_critical_warnings(self, prediction_id: str, include_deleted: bool = False) -> bool:
        """
        Check if prediction has any critical or warning severity flags.

        Args:
            prediction_id: The prediction ID to check
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            True if critical or warning severity exists, False otherwise
        """
        critical = self.exists({"prediction_id": prediction_id, "severity": SafetySeverity.CRITICAL}, include_deleted)
        warning = self.exists({"prediction_id": prediction_id, "severity": SafetySeverity.WARNING}, include_deleted)
        return critical or warning

    def count_by_prediction_id(self, prediction_id: str, include_deleted: bool = False) -> int:
        """
        Count safety warnings for a prediction.

        Args:
            prediction_id: The prediction ID to count warnings for
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            Number of safety warnings
        """
        return self.count({"prediction_id": prediction_id}, include_deleted=include_deleted)