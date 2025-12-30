"""
Authentication setup and initialization.
Wires together the generic decorators with project-specific implementations.
"""
from src.modules.authentication.domain.services.auth_service import AuthenticationService
from src.modules.authentication.internal.adapters.auth_adapters import (
    JwtTokenVerifier,
    DatabaseUserProvider
)
from src.modules.authentication.domain.services.jwt_service import JwtService
from src.modules.authentication.domain.services.user_service import UserService
from src.modules.authentication.domain.repositories.user_repository import UserRepository
from src.shared.security.jwt_decorators import JwtAuthDecorators

# ============================================
# Create Service Instances
# ============================================

# Repositories
user_repository = UserRepository()

# Services
jwt_service = JwtService()
auth_service = AuthenticationService()
user_service = UserService()


# ============================================
# Create Adapters (Interface Implementations)
# ============================================

# Adapt JWT service to ITokenVerifier interface
token_verifier = JwtTokenVerifier(jwt_service)

# Adapt User repository to IUserProvider interface
user_provider = DatabaseUserProvider(user_repository)


# ============================================
# Create Configured JWT Auth Decorators
# ============================================

# Wire everything together - decorators use adapters
jwt_auth = JwtAuthDecorators(
    token_verifier=token_verifier,
    user_provider=user_provider
)


# ============================================
# Export for Use in Controllers
# ============================================

__all__ = [
    'jwt_auth',
    'auth_service',
    'user_service',
    'jwt_service',
    'user_repository'
]