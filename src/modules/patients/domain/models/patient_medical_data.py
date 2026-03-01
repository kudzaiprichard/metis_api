from src.modules.patients.domain.models.enums import Gender, Ethnicity
from src.shared.data.base.model import BaseModel
from src.shared.data.database import db


class PatientMedicalData(BaseModel):
    __tablename__ = 'patient_medical_data'

    patient_id = db.Column(db.String(36), db.ForeignKey('patients.id'), nullable=False, index=True)

    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Enum(Gender), nullable=False)
    ethnicity = db.Column(db.Enum(Ethnicity), nullable=False)

    hba1c_baseline = db.Column(db.Numeric(4, 2), nullable=False)
    diabetes_duration = db.Column(db.Numeric(4, 1), nullable=False)
    fasting_glucose = db.Column(db.Numeric(5, 1), nullable=False)
    c_peptide = db.Column(db.Numeric(3, 2), nullable=False)

    egfr = db.Column(db.Numeric(5, 1), nullable=False)
    bmi = db.Column(db.Numeric(4, 1), nullable=False)

    bp_systolic = db.Column(db.Integer, nullable=False)
    bp_diastolic = db.Column(db.Integer, nullable=False)

    alt = db.Column(db.Numeric(5, 1), nullable=False)
    ldl = db.Column(db.Numeric(5, 1), nullable=False)
    hdl = db.Column(db.Numeric(5, 1), nullable=False)
    triglycerides = db.Column(db.Numeric(5, 1), nullable=False)

    previous_prediabetes = db.Column(db.Boolean, nullable=False)
    hypertension = db.Column(db.Boolean, nullable=False)
    ckd = db.Column(db.Boolean, nullable=False)
    cvd = db.Column(db.Boolean, nullable=False)
    nafld = db.Column(db.Boolean, nullable=False)
    retinopathy = db.Column(db.Boolean, nullable=False)

    prediction = db.relationship('Prediction', backref='medical_data', uselist=False, cascade='all, delete-orphan')

    def to_dict(self, exclude: list = None) -> dict:
        data = super().to_dict(exclude=exclude)

        if 'gender' in data:
            data['gender'] = self.gender.value if hasattr(self.gender, 'value') else self.gender
        if 'ethnicity' in data:
            data['ethnicity'] = self.ethnicity.value if hasattr(self.ethnicity, 'value') else self.ethnicity

        if 'updated_at' in data and self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        if 'created_at' in data and self.created_at:
            data['created_at'] = self.created_at.isoformat()

        # Serialize prediction with patient summary injected to satisfy PredictionDetailResponse
        if self.prediction:
            prediction_dict = self.prediction.to_dict()

            # Inject patient summary from the backref to satisfy the required
            # `patient: PatientSummaryResponse` field in PredictionDetailResponse.
            # Without this the patient detail endpoint returns 400 because
            # Pydantic validation fails on the missing required field.
            patient = self.prediction.medical_data.patient if self.prediction.medical_data else None
            if patient:
                # Get age from this medical record (most relevant context)
                prediction_dict['patient'] = {
                    'id': patient.id,
                    'first_name': patient.first_name,
                    'last_name': patient.last_name,
                    'age': self.age,
                    'gender': self.gender.value if hasattr(self.gender, 'value') else self.gender,
                }
            else:
                prediction_dict['patient'] = None

            data['prediction'] = prediction_dict
        else:
            data['prediction'] = None

        return data

    def __repr__(self) -> str:
        return f"<PatientMedicalData(patient_id={self.patient_id}, age={self.age})>"