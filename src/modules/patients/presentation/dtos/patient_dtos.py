from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, field_validator
from decimal import Decimal


# ============ Shared Patient Response ============

class PatientResponse(BaseModel):
    """Standard patient response DTO."""
    id: str
    first_name: str
    last_name: str
    email: Optional[str]
    mobile_number: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = {
        'from_attributes': True
    }


# ============ Patient Medical Data Response ============

class PatientMedicalDataResponse(BaseModel):
    """Patient medical data response DTO."""
    id: str
    patient_id: str
    age: int
    gender: str
    ethnicity: str
    hba1c_baseline: Decimal
    diabetes_duration: Decimal
    fasting_glucose: Decimal
    c_peptide: Decimal
    egfr: Decimal
    bmi: Decimal
    bp_systolic: int
    bp_diastolic: int
    alt: Decimal
    ldl: Decimal
    hdl: Decimal
    triglycerides: Decimal
    previous_prediabetes: bool
    hypertension: bool
    ckd: bool
    cvd: bool
    nafld: bool
    retinopathy: bool
    updated_at: datetime

    model_config = {
        'from_attributes': True
    }


# ============ Patient Detail Response (with medical data) ============

class PatientDetailResponse(BaseModel):
    """Patient with medical data response DTO."""
    id: str
    first_name: str
    last_name: str
    email: Optional[str]
    mobile_number: Optional[str]
    created_at: datetime
    updated_at: datetime
    medical_data: Optional[PatientMedicalDataResponse]

    model_config = {
        'from_attributes': True
    }


# ============ Create Patient ============

class CreatePatientRequest(BaseModel):
    """DTO for creating a new patient."""
    first_name: str = Field(..., min_length=2, max_length=100)
    last_name: str = Field(..., min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    mobile_number: Optional[str] = Field(None, max_length=20)

    @field_validator('first_name', 'last_name')
    @classmethod
    def validate_name_characters(cls, v, info):
        """Validate name contains only valid characters."""
        import re
        if not re.match(r"^[a-zA-Z\s\-']+$", v):
            field_name = info.field_name.replace('_', ' ').title()
            raise ValueError(f'{field_name} contains invalid characters')
        return v.strip()

    @field_validator('mobile_number')
    @classmethod
    def validate_mobile_number(cls, v):
        """Validate mobile number format."""
        if v is None:
            return v
        import re
        # Remove spaces and dashes for validation
        cleaned = re.sub(r'[\s\-]', '', v)
        if not re.match(r'^\+?[0-9]{10,15}$', cleaned):
            raise ValueError('Mobile number must be 10-15 digits with optional + prefix')
        return v.strip()

    model_config = {
        'str_strip_whitespace': True
    }


# ============ Update Patient Contact Info ============

class UpdatePatientContactRequest(BaseModel):
    """DTO for updating patient contact information."""
    first_name: Optional[str] = Field(None, min_length=2, max_length=100)
    last_name: Optional[str] = Field(None, min_length=2, max_length=100)
    email: Optional[EmailStr] = None
    mobile_number: Optional[str] = Field(None, max_length=20)

    @field_validator('first_name', 'last_name')
    @classmethod
    def validate_name_characters(cls, v, info):
        """Validate name contains only valid characters."""
        if v is None:
            return v
        import re
        if not re.match(r"^[a-zA-Z\s\-']+$", v):
            field_name = info.field_name.replace('_', ' ').title()
            raise ValueError(f'{field_name} contains invalid characters')
        return v.strip()

    @field_validator('mobile_number')
    @classmethod
    def validate_mobile_number(cls, v):
        """Validate mobile number format."""
        if v is None:
            return v
        import re
        cleaned = re.sub(r'[\s\-]', '', v)
        if not re.match(r'^\+?[0-9]{10,15}$', cleaned):
            raise ValueError('Mobile number must be 10-15 digits with optional + prefix')
        return v.strip()

    model_config = {
        'str_strip_whitespace': True
    }


# ============ Create/Update Patient Medical Data ============

class CreatePatientMedicalDataRequest(BaseModel):
    """DTO for creating patient medical data."""
    patient_id: str
    age: int = Field(..., ge=18, le=120)
    gender: str = Field(..., description="Must be 'Male' or 'Female'")
    ethnicity: str = Field(..., description="Must be 'Caucasian', 'African', 'Asian', 'Hispanic', or 'Other'")
    hba1c_baseline: Decimal = Field(..., ge=4.0, le=20.0)
    diabetes_duration: Decimal = Field(..., ge=0.0, le=50.0)
    fasting_glucose: Decimal = Field(..., ge=50.0, le=500.0)
    c_peptide: Decimal = Field(..., ge=0.0, le=10.0)
    egfr: Decimal = Field(..., ge=0.0, le=150.0)
    bmi: Decimal = Field(..., ge=10.0, le=80.0)
    bp_systolic: int = Field(..., ge=70, le=250)
    bp_diastolic: int = Field(..., ge=40, le=150)
    alt: Decimal = Field(..., ge=0.0, le=500.0)
    ldl: Decimal = Field(..., ge=0.0, le=500.0)
    hdl: Decimal = Field(..., ge=0.0, le=200.0)
    triglycerides: Decimal = Field(..., ge=0.0, le=1000.0)
    previous_prediabetes: bool
    hypertension: bool
    ckd: bool
    cvd: bool
    nafld: bool
    retinopathy: bool

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        """Validate gender is valid."""
        if v not in ['Male', 'Female']:
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v

    @field_validator('ethnicity')
    @classmethod
    def validate_ethnicity(cls, v):
        """Validate ethnicity is valid."""
        valid_ethnicities = ['Caucasian', 'African', 'Asian', 'Hispanic', 'Other']
        if v not in valid_ethnicities:
            raise ValueError(f"Ethnicity must be one of: {', '.join(valid_ethnicities)}")
        return v


class UpdatePatientMedicalDataRequest(BaseModel):
    """DTO for updating patient medical data."""
    age: Optional[int] = Field(None, ge=18, le=120)
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    hba1c_baseline: Optional[Decimal] = Field(None, ge=4.0, le=20.0)
    diabetes_duration: Optional[Decimal] = Field(None, ge=0.0, le=50.0)
    fasting_glucose: Optional[Decimal] = Field(None, ge=50.0, le=500.0)
    c_peptide: Optional[Decimal] = Field(None, ge=0.0, le=10.0)
    egfr: Optional[Decimal] = Field(None, ge=0.0, le=150.0)
    bmi: Optional[Decimal] = Field(None, ge=10.0, le=80.0)
    bp_systolic: Optional[int] = Field(None, ge=70, le=250)
    bp_diastolic: Optional[int] = Field(None, ge=40, le=150)
    alt: Optional[Decimal] = Field(None, ge=0.0, le=500.0)
    ldl: Optional[Decimal] = Field(None, ge=0.0, le=500.0)
    hdl: Optional[Decimal] = Field(None, ge=0.0, le=200.0)
    triglycerides: Optional[Decimal] = Field(None, ge=0.0, le=1000.0)
    previous_prediabetes: Optional[bool] = None
    hypertension: Optional[bool] = None
    ckd: Optional[bool] = None
    cvd: Optional[bool] = None
    nafld: Optional[bool] = None
    retinopathy: Optional[bool] = None

    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        """Validate gender is valid."""
        if v is not None and v not in ['Male', 'Female']:
            raise ValueError("Gender must be 'Male' or 'Female'")
        return v

    @field_validator('ethnicity')
    @classmethod
    def validate_ethnicity(cls, v):
        """Validate ethnicity is valid."""
        if v is not None:
            valid_ethnicities = ['Caucasian', 'African', 'Asian', 'Hispanic', 'Other']
            if v not in valid_ethnicities:
                raise ValueError(f"Ethnicity must be one of: {', '.join(valid_ethnicities)}")
        return v


# ============ Get Single Patient ============

class GetPatientRequest(BaseModel):
    """DTO for getting a single patient by ID."""
    patient_id: str = Field(..., min_length=1)


# ============ Delete Patient ============

class DeletePatientRequest(BaseModel):
    """DTO for deleting a patient."""
    patient_id: str = Field(..., min_length=1)


# ============ List Patients (Pagination & Search) ============

class ListPatientsRequest(BaseModel):
    """DTO for listing patients with pagination and search."""
    page: int = Field(default=1, ge=1, description="Page number (starts at 1)")
    per_page: int = Field(default=20, ge=1, le=100, description="Items per page (max 100)")
    search: Optional[str] = None  # Search by name, email, or mobile

    def get_offset(self) -> int:
        """Calculate database offset for pagination."""
        return (self.page - 1) * self.per_page

    def get_limit(self) -> int:
        """Get limit for database query."""
        return self.per_page