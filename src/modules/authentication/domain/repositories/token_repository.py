from typing import Optional

from src.modules.authentication.domain.models.token import Token
from src.shared.data.base.repository import BaseRepository


class TokenRepository(BaseRepository[Token]):
    """
    Repository for Token entity operations.
    """

    def __init__(self):
        super().__init__(Token)

    def find_by_token(self, token: str, include_deleted: bool = False) -> Optional[Token]:
        """
        Find a token by its token string.

        Args:
            token: The token string to search for
            include_deleted: Whether to include soft-deleted tokens (default: False)

        Returns:
            Token instance if found, None otherwise
        """
        return self.find_one_by({"token": token}, include_deleted=include_deleted)