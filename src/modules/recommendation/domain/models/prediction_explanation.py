from src.modules.recommendation.domain.models.enums import ConfidenceLevel, ClinicalPriority
from src.shared.data.base.model import BaseModel
from src.shared.data.database import db


class PredictionExplanation(BaseModel):
    """
    Stores SHAP-based explanations and LLM reasoning (optional per prediction).
    1-to-1 relationship with Prediction.
    """
    __tablename__ = 'prediction_explanations'

    # Foreign key (UNIQUE for 1-to-1)
    prediction_id = db.Column(db.String(36), db.ForeignKey('predictions.id'), unique=True, nullable=False, index=True)

    # Summary
    summary_text = db.Column(db.Text, nullable=False)
    confidence_level = db.Column(db.Enum(ConfidenceLevel), nullable=False)
    clinical_priority = db.Column(db.Enum(ClinicalPriority), nullable=False)

    # LLM Explanations
    why_this_treatment = db.Column(db.Text, nullable=False)  # 2-3 sentences
    why_not_alternatives = db.Column(db.Text, nullable=False)  # Why other treatments weren't chosen

    # SHAP values
    base_value = db.Column(db.Numeric(5, 2), nullable=False)  # SHAP base prediction
    prediction_value = db.Column(db.Numeric(5, 2), nullable=False)  # Final prediction value
    feature_interactions = db.Column(db.Text, nullable=True)  # Feature interaction description

    # Relationships
    features = db.relationship('ExplanationFeature', backref='explanation', lazy='dynamic',
                               cascade='all, delete-orphan')
    alternatives = db.relationship('ExplanationAlternative', backref='explanation', lazy='dynamic',
                                   cascade='all, delete-orphan')

    def to_dict(self, exclude: list = None) -> dict:
        """Override to handle enum serialization, datetime conversion, and relationships."""
        data = super().to_dict(exclude=exclude)

        # Handle enum serialization
        if 'confidence_level' in data:
            data['confidence_level'] = self.confidence_level.value if hasattr(self.confidence_level,
                                                                              'value') else self.confidence_level
        if 'clinical_priority' in data:
            data['clinical_priority'] = self.clinical_priority.value if hasattr(self.clinical_priority,
                                                                                'value') else self.clinical_priority

        # Handle datetime serialization
        if 'created_at' in data and self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if 'updated_at' in data and self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()

        # Load relationships (lazy='dynamic' returns query, need .all())
        data['features'] = [f.to_dict() for f in self.features.all()]
        data['alternatives'] = [a.to_dict() for a in self.alternatives.all()]

        return data

    def __repr__(self) -> str:
        return f"<PredictionExplanation(prediction_id={self.prediction_id}, confidence={self.confidence_level.value})>"