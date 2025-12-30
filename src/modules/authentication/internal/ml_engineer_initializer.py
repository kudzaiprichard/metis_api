from typing import Optional

from src.modules.authentication.domain.services.user_service import UserService
from src.modules.authentication.presentation.dtos.user_dtos import CreateUserRequest
from src.shared.exceptions.exceptions import ConflictException, NotFoundException
import logging

logger = logging.getLogger(__name__)


class MLEngineerInitializer:
    """Service to initialize default ML Engineer user on system startup."""

    # Default ML Engineer credentials
    DEFAULT_ML_ENGINEER_EMAIL = "ml_engineer@gmail.com"
    DEFAULT_ML_ENGINEER_PASSWORD = "@Qwerty12"
    DEFAULT_ML_ENGINEER_FIRST_NAME = "ML"
    DEFAULT_ML_ENGINEER_LAST_NAME = "Engineer"

    @classmethod
    def create_default_ml_engineer(cls) -> Optional[bool]:
        """
        Create default ML Engineer user if it doesn't exist.

        Returns:
            True if created, False if already exists
        """
        user_service = UserService()

        try:
            # Check if ML Engineer already exists
            user_service.get_user_by_email(cls.DEFAULT_ML_ENGINEER_EMAIL)
            logger.info(f"ML Engineer user already exists: {cls.DEFAULT_ML_ENGINEER_EMAIL}")
            return False
        except NotFoundException:
            # User doesn't exist, proceed to create
            pass

        try:
            # Create ML Engineer user request
            create_request = CreateUserRequest(
                email=cls.DEFAULT_ML_ENGINEER_EMAIL,
                password=cls.DEFAULT_ML_ENGINEER_PASSWORD,
                first_name=cls.DEFAULT_ML_ENGINEER_FIRST_NAME,
                last_name=cls.DEFAULT_ML_ENGINEER_LAST_NAME,
                role="ML_ENGINEER"
            )

            # Create user via service
            user_service.create_user(create_request)

            logger.info(f"Default ML Engineer user created successfully")
            logger.info(f"Email: {cls.DEFAULT_ML_ENGINEER_EMAIL}")
            logger.info(f"Password: {cls.DEFAULT_ML_ENGINEER_PASSWORD}")
            logger.warning("IMPORTANT: Please change the default ML Engineer password immediately")

            return True

        except ConflictException:
            logger.info(f"ML Engineer user already exists: {cls.DEFAULT_ML_ENGINEER_EMAIL}")
            return False
        except Exception as e:
            logger.error(f"Failed to create default ML Engineer user: {str(e)}")
            raise

    @classmethod
    def initialize(cls) -> None:
        """Initialize default ML Engineer user on system startup."""
        logger.info("Starting ML Engineer initialization")
        cls.create_default_ml_engineer()
        logger.info("ML Engineer initialization complete")