from src.modules.authentication.domain.models.enums import TokenType
from src.shared.data.base.model import BaseModel
from src.shared.data.database import db
from datetime import datetime


class Token(BaseModel):
    __tablename__ = 'tokens'

    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False, index=True)
    token = db.Column(db.Text, nullable=False, unique=True, index=True)  # Longer + unique + indexed
    token_type = db.Column(db.Enum(TokenType), nullable=False)
    is_expired = db.Column(db.Boolean, nullable=False, default=False)
    is_revoked = db.Column(db.Boolean, nullable=False, default=False)
    expires_at = db.Column(db.DateTime, nullable=False)  # Store actual expiry datetime, not duration

    def is_valid(self) -> bool:
        """Check if token is still valid."""
        return not self.is_expired and not self.is_revoked and self.expires_at > datetime.now()

    def __repr__(self):
        return f"<Token(id={self.id}, type={self.token_type.value}, user_id={self.user_id})>"