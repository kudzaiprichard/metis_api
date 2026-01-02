from typing import Optional

from src.modules.recommendation.models.prediction_explanation import PredictionExplanation
from src.shared.data.base.repository import BaseRepository


class PredictionExplanationRepository(BaseRepository[PredictionExplanation]):
    """
    Repository for PredictionExplanation entity operations.
    """

    def __init__(self):
        super().__init__(PredictionExplanation)

    def find_by_prediction_id(self, prediction_id: str, include_deleted: bool = False) -> Optional[PredictionExplanation]:
        """
        Find explanation by prediction ID (1-to-1 relationship).

        Args:
            prediction_id: The prediction ID to search for
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            PredictionExplanation instance if found, None otherwise
        """
        return self.find_one_by({"prediction_id": prediction_id}, include_deleted=include_deleted)

    def exists_for_prediction(self, prediction_id: str, include_deleted: bool = False) -> bool:
        """
        Check if explanation exists for a given prediction.

        Args:
            prediction_id: The prediction ID to check
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            True if explanation exists, False otherwise
        """
        return self.exists({"prediction_id": prediction_id}, include_deleted=include_deleted)