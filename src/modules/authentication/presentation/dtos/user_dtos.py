from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator


# ============ Shared User Response ============

class UserResponse(BaseModel):
    """Standard user response DTO."""
    id: str
    email: str
    first_name: str
    last_name: str
    role: str
    created_at: datetime
    updated_at: datetime

    model_config = {
        'from_attributes': True
    }


# ============ Create User ============

class CreateUserRequest(BaseModel):
    """DTO for creating a new user (admin functionality)."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    first_name: str = Field(..., min_length=2, max_length=255)
    last_name: str = Field(..., min_length=2, max_length=255)
    role: str = Field(..., description="Must be DOCTOR or ML_ENGINEER")

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
        if v not in ['DOCTOR', 'ML_ENGINEER']:
            raise ValueError('Role must be either DOCTOR or ML_ENGINEER')
        return v

    model_config = {
        'str_strip_whitespace': True
    }


# ============ Update User ============

class UpdateUserRequest(BaseModel):
    """DTO for updating user details."""
    user_id: str
    email: Optional[EmailStr] = None
    first_name: Optional[str] = Field(None, min_length=2, max_length=255)
    last_name: Optional[str] = Field(None, min_length=2, max_length=255)
    role: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8, max_length=128)

    @classmethod
    @field_validator('password')
    def validate_password_strength(cls, v):
        """Validate password contains required character types."""
        if v is None:
            return v
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
        if v is None:
            return v
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


# ============ Get Single User ============

class GetUserRequest(BaseModel):
    """DTO for getting a single user by ID."""
    user_id: str = Field(..., min_length=1)


# ============ Delete User ============

class DeleteUserRequest(BaseModel):
    """DTO for deleting a user."""
    user_id: str = Field(..., min_length=1)


# ============ List Users (Pagination) ============

class ListUsersRequest(BaseModel):
    """DTO for listing users with pagination and filters."""
    page: int = Field(default=1, ge=1, description="Page number (starts at 1)")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page (max 100)")
    role: Optional[str] = None
    search: Optional[str] = None  # Search by email, first_name, or last_name

    @classmethod
    @field_validator('role')
    def validate_role(cls, v):
        """Validate role filter."""
        if v is not None and v not in ['DOCTOR', 'ML_ENGINEER']:
            raise ValueError('Role must be either DOCTOR or ML_ENGINEER')
        return v

    def get_offset(self) -> int:
        """Calculate database offset for pagination."""
        return (self.page - 1) * self.per_page

    def get_limit(self) -> int:
        """Get limit for database query."""
        return self.per_page