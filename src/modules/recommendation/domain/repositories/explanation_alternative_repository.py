from typing import List

from src.modules.recommendation.domain.models.explanation_alternative import ExplanationAlternative
from src.shared.data.base.repository import BaseRepository


class ExplanationAlternativeRepository(BaseRepository[ExplanationAlternative]):
    """
    Repository for ExplanationAlternative entity operations.
    """

    def __init__(self):
        super().__init__(ExplanationAlternative)

    def find_by_explanation_id(self, explanation_id: str, include_deleted: bool = False) -> List[
        ExplanationAlternative]:
        """
        Find all alternative treatments for an explanation (ordered by rank).

        Args:
            explanation_id: The explanation ID to search for
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            List of ExplanationAlternative instances ordered by rank
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(explanation_id=explanation_id).order_by(ExplanationAlternative.rank).all()

    def find_top_alternatives(self, explanation_id: str, n: int = 3, include_deleted: bool = False) -> List[
        ExplanationAlternative]:
        """
        Find top N alternative treatments for an explanation.

        Args:
            explanation_id: The explanation ID to search for
            n: Number of top alternatives to return (default: 3)
            include_deleted: Whether to include soft-deleted records (default: False)

        Returns:
            List of top N ExplanationAlternative instances ordered by rank
        """
        query = self.model.query
        if not include_deleted:
            query = query.filter_by(is_deleted=False)

        return query.filter_by(explanation_id=explanation_id).order_by(ExplanationAlternative.rank).limit(n).all()