from src.modules.recommendation.models.enums import Treatment
from src.shared.data.base.model import BaseModel
from src.shared.data.database import db


class Prediction(BaseModel):
    """
    Prediction model for AI-generated treatment recommendations.
    """
    __tablename__ = 'predictions'

    # Foreign keys
    patient_id = db.Column(db.String(36), db.ForeignKey('patients.id'), nullable=False, index=True)
    created_by = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, index=True)

    # Model information
    model_version = db.Column(db.String(20), nullable=False, index=True)

    # recommendation
    recommended_treatment = db.Column(db.Enum(Treatment), nullable=False)
    treatment_index = db.Column(db.Integer, nullable=False)  # 0-4
    predicted_reduction = db.Column(db.Numeric(4, 2), nullable=False)

    # Confidence
    confidence_score = db.Column(db.Numeric(5, 2), nullable=False)  # 0-100
    confidence_margin = db.Column(db.Numeric(4, 2), nullable=False)  # Gap between top 2 Q-values

    # Relationships
    q_values = db.relationship('PredictionQValue', backref='prediction', lazy='dynamic', cascade='all, delete-orphan')
    explanation = db.relationship('PredictionExplanation', backref='prediction', uselist=False,
                                  cascade='all, delete-orphan')
    safety_warnings = db.relationship('SafetyWarning', backref='prediction', lazy='dynamic',
                                      cascade='all, delete-orphan')

    def to_dict(self, exclude: list = None) -> dict:
        """Override to handle enum serialization and datetime conversion."""
        data = super().to_dict(exclude=exclude)

        # Handle enum serialization
        if 'recommended_treatment' in data:
            data['recommended_treatment'] = self.recommended_treatment.value if hasattr(self.recommended_treatment,
                                                                                        'value') else self.recommended_treatment

        # Handle datetime serialization
        if 'created_at' in data and self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if 'updated_at' in data and self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()

        return data

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, patient_id={self.patient_id}, treatment={self.recommended_treatment.value})>"