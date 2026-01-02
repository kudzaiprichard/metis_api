from src.modules.treatment_decisions.domain.models.enums import DecisionType
from src.shared.data.base.model import BaseModel
from src.shared.data.database import db


class TreatmentDecision(BaseModel):
    """
    Treatment decision model recording doctor's final treatment choice.
    Links prediction to actual treatment given.
    """
    __tablename__ = 'treatment_decisions'

    # Foreign keys
    prediction_id = db.Column(db.String(36), db.ForeignKey('predictions.id'), nullable=False, index=True)
    patient_id = db.Column(db.String(36), db.ForeignKey('patients.id'), nullable=False, index=True)
    decided_by = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, index=True)

    # Decision details
    decision_type = db.Column(db.Enum(DecisionType), nullable=False)
    treatment_given = db.Column(db.String(50), nullable=False)
    reasoning_notes = db.Column(db.Text, nullable=True)  # Required if decision_type='custom'
    dosage = db.Column(db.String(100), nullable=True)

    # Outcome tracking (filled during follow-up)
    observed_reduction = db.Column(db.Numeric(4, 2), nullable=True)  # Actual HbA1c reduction
    outcome_recorded_at = db.Column(db.DateTime, nullable=True)
    used_for_training = db.Column(db.Boolean, nullable=False, default=False)

    # Decision timestamp
    decided_at = db.Column(db.DateTime, nullable=False, default=db.func.now())

    # Relationships
    follow_ups = db.relationship('FollowUp', backref='decision', lazy='dynamic', cascade='all, delete-orphan')

    def to_dict(self, exclude: list = None) -> dict:
        """Override to handle enum serialization and datetime conversion."""
        data = super().to_dict(exclude=exclude)

        # Handle enum serialization
        if 'decision_type' in data:
            data['decision_type'] = self.decision_type.value if hasattr(self.decision_type, 'value') else self.decision_type

        # Handle datetime serialization
        if 'decided_at' in data and self.decided_at:
            data['decided_at'] = self.decided_at.isoformat()
        if 'outcome_recorded_at' in data and self.outcome_recorded_at:
            data['outcome_recorded_at'] = self.outcome_recorded_at.isoformat()
        if 'created_at' in data and self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if 'updated_at' in data and self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()

        return data

    def __repr__(self) -> str:
        return f"<TreatmentDecision(id={self.id}, patient_id={self.patient_id}, type={self.decision_type.value})>"