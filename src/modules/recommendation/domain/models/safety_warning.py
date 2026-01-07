from src.modules.recommendation.domain.models.enums import SafetySeverity
from src.shared.data.base.model import BaseModel
from src.shared.data.database import db


class SafetyWarning(BaseModel):
    """
    Stores safety checks and contraindications.
    """
    __tablename__ = 'safety_warnings'

    # Foreign key
    prediction_id = db.Column(db.String(36), db.ForeignKey('predictions.id'), nullable=False, index=True)

    # Warning details
    severity = db.Column(db.Enum(SafetySeverity), nullable=False)
    concern = db.Column(db.Text, nullable=False)  # What the concern is
    patient_factor = db.Column(db.String(100), nullable=False)  # Relevant patient factor
    mitigation = db.Column(db.Text, nullable=False)  # How to address the concern
    reason = db.Column(db.Text, nullable=True)

    def to_dict(self, exclude: list = None) -> dict:
        """Override to handle enum serialization and datetime conversion."""
        data = super().to_dict(exclude=exclude)

        # Handle enum serialization
        if 'severity' in data:
            data['severity'] = self.severity.value if hasattr(self.severity, 'value') else self.severity

        # Handle datetime serialization
        if 'created_at' in data and self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if 'updated_at' in data and self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()

        return data

    def __repr__(self) -> str:
        return f"<SafetyWarning(severity={self.severity.value}, concern={self.concern[:50]})>"