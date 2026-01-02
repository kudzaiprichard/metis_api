from typing import List

from src.modules.recommendation.domain.models.explanation_feature import ExplanationFeature
from src.shared.data.base.repository import BaseRepository


class ExplanationFeatureRepository(BaseRepository[ExplanationFeature]):
    """
    Repository for ExplanationFeature entity operations.
    """

    def __init__(self):
        super().__init__(ExplanationFeature)

    def find_by_explanation_id(self, explanation_id: str, include_deleted: bool = False) -> List[ExplanationFeature]:
        """
        Find all features for an explanation (ordered by rank).

        Args:
            explanation_id: The explanation ID to search for
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            List of ExplanationFeature instances ordered by rank
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(explanation_id=explanation_id).order_by(ExplanationFeature.rank).all()

    def find_top_n_by_explanation_id(self, explanation_id: str, n: int = 5, include_deleted: bool = False) -> List[
        ExplanationFeature]:
        """
        Find top N features for an explanation.

        Args:
            explanation_id: The explanation ID to search for
            n: Number of top features to return (default: 5)
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            List of top N ExplanationFeature instances ordered by rank
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(explanation_id=explanation_id).order_by(ExplanationFeature.rank).limit(n).all()