from src.shared.data.base.model import BaseModel
from src.shared.data.database import db


class ExplanationFeature(BaseModel):
    """
    Stores top SHAP features (top 5 by default).
    """
    __tablename__ = 'explanation_features'

    # Foreign key
    explanation_id = db.Column(db.String(36), db.ForeignKey('prediction_explanations.id'), nullable=False, index=True)

    # Feature information
    feature_name = db.Column(db.String(50), nullable=False)
    scaled_value = db.Column(db.Numeric(8, 4), nullable=False)  # Z-score value (used by model)
    raw_value = db.Column(db.Numeric(8, 2), nullable=False)  # Actual patient value (for display)
    shap_value = db.Column(db.Numeric(8, 4), nullable=False)  # SHAP attribution score
    rank = db.Column(db.Integer, nullable=False)  # 1-5 importance ranking
    interpretation = db.Column(db.Text, nullable=False)  # Human-readable explanation
    reference_range = db.Column(db.String(100), nullable=True)  # Normal range (e.g., "1.1-4.4 ng/mL")

    def to_dict(self, exclude: list = None) -> dict:
        """Override to handle datetime conversion."""
        data = super().to_dict(exclude=exclude)

        # Handle datetime serialization
        if 'created_at' in data and self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if 'updated_at' in data and self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()

        return data

    def __repr__(self) -> str:
        return f"<ExplanationFeature(feature={self.feature_name}, rank={self.rank})>"