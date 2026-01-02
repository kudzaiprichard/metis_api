from src.modules.recommendation.models.enums import Treatment
from src.shared.data.base.model import BaseModel
from src.shared.data.database import db

class ExplanationAlternative(BaseModel):
    """
    Stores alternative treatment options with pros/cons.
    Rank 2-5 (rank 1 is the primary recommendation).
    """
    __tablename__ = 'explanation_alternatives'

    # Foreign key
    explanation_id = db.Column(db.String(36), db.ForeignKey('prediction_explanations.id'), nullable=False, index=True)

    # Alternative treatment
    rank = db.Column(db.Integer, nullable=False)  # 2-5 (rank 1 is primary recommendation)
    treatment = db.Column(db.Enum(Treatment), nullable=False)
    predicted_reduction = db.Column(db.Numeric(4, 2), nullable=False)  # Expected reduction (%)

    # Pros and cons
    pros = db.Column(db.Text, nullable=False)  # Comma-separated advantages
    cons = db.Column(db.Text, nullable=False)  # Comma-separated disadvantages
    when_to_consider = db.Column(db.Text, nullable=False)  # Clinical scenario description

    def to_dict(self, exclude: list = None) -> dict:
        """Override to handle enum serialization and datetime conversion."""
        data = super().to_dict(exclude=exclude)

        # Handle enum serialization
        if 'treatment' in data:
            data['treatment'] = self.treatment.value if hasattr(self.treatment, 'value') else self.treatment

        # Handle datetime serialization
        if 'created_at' in data and self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if 'updated_at' in data and self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()

        return data

    def __repr__(self) -> str:
        return f"<ExplanationAlternative(treatment={self.treatment.value}, rank={self.rank})>"