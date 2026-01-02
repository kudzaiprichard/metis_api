from typing import List

from src.modules.recommendation.domain.models.prediction_q_value import PredictionQValue
from src.shared.data.base.repository import BaseRepository


class PredictionQValueRepository(BaseRepository[PredictionQValue]):
    """
    Repository for PredictionQValue entity operations.
    """

    def __init__(self):
        super().__init__(PredictionQValue)

    def find_by_prediction_id(self, prediction_id: str, include_deleted: bool = False) -> List[PredictionQValue]:
        """
        Find all Q-values for a prediction (always returns 5 records).

        Args:
            prediction_id: The prediction ID to search for
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            List of PredictionQValue instances ordered by rank
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(prediction_id=prediction_id).order_by(PredictionQValue.rank).all()

    def find_top_n_by_prediction_id(self, prediction_id: str, n: int = 3, include_deleted: bool = False) -> List[
        PredictionQValue]:
        """
        Find top N Q-values for a prediction.

        Args:
            prediction_id: The prediction ID to search for
            n: Number of top Q-values to return (default: 3)
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            List of top N PredictionQValue instances ordered by rank
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(prediction_id=prediction_id).order_by(PredictionQValue.rank).limit(n).all()