from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator


# ============ Registration Feature ============

class RegisterRequest(BaseModel):
    """DTO for user registration with automatic validation."""
    email: EmailStr  # Automatic email validation
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str = Field(..., min_length=2, max_length=255)
    last_name: str = Field(..., min_length=2, max_length=255)
    role: Optional[str] = None  # Optional: DOCTOR or ML_ENGINEER

    @classmethod
    @field_validator('password')
    def validate_password_strength(cls, v):
        """Validate password contains required character types."""
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one number')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(char.islower() for char in v):
            raise ValueError('Password must contain at least one lowercase letter')
        return v

    @classmethod
    @field_validator('first_name', 'last_name')
    def validate_name_characters(cls, v, info):
        """Validate name contains only valid characters."""
        import re
        if not re.match(r"^[a-zA-Z\s\-']+$", v):
            field_name = info.field_name.replace('_', ' ').title()
            raise ValueError(f'{field_name} contains invalid characters')
        return v.strip()

    @classmethod
    @field_validator('role')
    def validate_role(cls, v):
        """Validate role is valid."""
        if v is not None and v not in ['DOCTOR', 'ML_ENGINEER']:
            raise ValueError('Role must be either DOCTOR or ML_ENGINEER')
        return v

    model_config = {
        'str_strip_whitespace': True
    }


class UserResponse(BaseModel):
    """Standard user response DTO - matches User model exactly."""
    id: str
    email: str
    first_name: str
    last_name: str
    role: str
    created_at: datetime
    updated_at: datetime

    model_config = {
        'from_attributes': True  # Allows model_validate(sqlalchemy_model)
    }


# ============ Login Feature ============

class LoginRequest(BaseModel):
    """DTO for user login."""
    email: EmailStr  # Automatic email validation
    password: str = Field(..., min_length=1)  # Check not empty

    model_config = {
        'str_strip_whitespace': True
    }


# ============ Logout Feature ============

class LogoutRequest(BaseModel):
    """DTO for logout request."""
    user_id: str = Field(..., min_length=1)


# ============ Refresh Token Feature ============

class RefreshTokenRequest(BaseModel):
    """DTO for refresh token request."""
    refresh_token: str = Field(..., min_length=1)


# ============ Token Info ============

class TokenInfo(BaseModel):
    """Token information DTO."""
    token: str
    token_type: str
    expires_at: str
    created_at: str