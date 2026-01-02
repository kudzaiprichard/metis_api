from src.modules.recommendation.models.enums import Treatment
from src.shared.data.base.model import BaseModel
from src.shared.data.database import db


class PredictionQValue(BaseModel):
    """
    Stores Q-values for all 5 treatments from each prediction.
    Always 5 records per prediction (one for each treatment).
    """
    __tablename__ = 'prediction_q_values'

    # Foreign key
    prediction_id = db.Column(db.String(36), db.ForeignKey('predictions.id'), nullable=False, index=True)

    # Treatment and Q-value
    treatment = db.Column(db.Enum(Treatment), nullable=False)
    q_value = db.Column(db.Numeric(5, 2), nullable=False)  # Predicted reduction (%)
    rank = db.Column(db.Integer, nullable=False)  # 1-5 ranking

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
        return f"<PredictionQValue(prediction_id={self.prediction_id}, treatment={self.treatment.value}, rank={self.rank})>"