from typing import Optional

from src.modules.authentication.domain.models.user import User
from src.shared.data.base.repository import BaseRepository


class UserRepository(BaseRepository[User]):
    """
    Repository for User entity operations.
    """

    def __init__(self):
        super().__init__(User)

    def find_by_email(self, email: str, include_deleted: bool = False) -> Optional[User]:
        """
        Find a user by their email address.

        Args:
            email: The email address to search for
            include_deleted: Whether to include soft-deleted users (default: False)

        Returns:
            User instance if found, None otherwise
        """
        return self.find_one_by({"email": email}, include_deleted=include_deleted)