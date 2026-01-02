from src.modules.patients.domain.models.enums import Gender, Ethnicity
from src.shared.data.base.model import BaseModel
from src.shared.data.database import db


class PatientMedicalData(BaseModel):
    """
    Patient medical data model storing 21 base features for AI predictions.
    1-to-1 relationship with Patient.
    """
    __tablename__ = 'patient_medical_data'

    # Foreign key to patients table (UNIQUE for 1-to-1)
    patient_id = db.Column(db.String(36), db.ForeignKey('patients.id'), unique=True, nullable=False, index=True)

    # Demographics
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Enum(Gender), nullable=False)
    ethnicity = db.Column(db.Enum(Ethnicity), nullable=False)

    # Diabetes metrics
    hba1c_baseline = db.Column(db.Numeric(4, 2), nullable=False)
    diabetes_duration = db.Column(db.Numeric(4, 1), nullable=False)
    fasting_glucose = db.Column(db.Numeric(5, 1), nullable=False)
    c_peptide = db.Column(db.Numeric(3, 2), nullable=False)

    # Kidney function
    egfr = db.Column(db.Numeric(5, 1), nullable=False)

    # Body metrics
    bmi = db.Column(db.Numeric(4, 1), nullable=False)

    # Blood pressure
    bp_systolic = db.Column(db.Integer, nullable=False)
    bp_diastolic = db.Column(db.Integer, nullable=False)

    # Liver function
    alt = db.Column(db.Numeric(5, 1), nullable=False)

    # Lipid profile
    ldl = db.Column(db.Numeric(5, 1), nullable=False)
    hdl = db.Column(db.Numeric(5, 1), nullable=False)
    triglycerides = db.Column(db.Numeric(5, 1), nullable=False)

    # Comorbidities (Boolean flags)
    previous_prediabetes = db.Column(db.Boolean, nullable=False)
    hypertension = db.Column(db.Boolean, nullable=False)
    ckd = db.Column(db.Boolean, nullable=False)
    cvd = db.Column(db.Boolean, nullable=False)
    nafld = db.Column(db.Boolean, nullable=False)
    retinopathy = db.Column(db.Boolean, nullable=False)

    def to_dict(self, exclude: list = None) -> dict:
        """Override to handle enum serialization and datetime conversion."""
        data = super().to_dict(exclude=exclude)

        # Handle enum serialization
        if 'gender' in data:
            data['gender'] = self.gender.value if hasattr(self.gender, 'value') else self.gender
        if 'ethnicity' in data:
            data['ethnicity'] = self.ethnicity.value if hasattr(self.ethnicity, 'value') else self.ethnicity

        # Handle datetime serialization
        if 'updated_at' in data and self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        if 'created_at' in data and self.created_at:
            data['created_at'] = self.created_at.isoformat()

        return data

    def __repr__(self) -> str:
        return f"<PatientMedicalData(patient_id={self.patient_id}, age={self.age})>"