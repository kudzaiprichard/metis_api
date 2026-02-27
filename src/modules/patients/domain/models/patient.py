from src.shared.data.base.model import BaseModel
from src.shared.data.database import db


class Patient(BaseModel):
    """
    Patient model for storing contact and demographic information.
    """
    __tablename__ = 'patients'

    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), nullable=True)
    mobile_number = db.Column(db.String(20), nullable=True)

    # Relationships (one-to-many: one patient, multiple medical data records)
    medical_records = db.relationship(
        'PatientMedicalData',
        backref='patient',
        lazy='dynamic',
        cascade='all, delete-orphan',
        order_by='PatientMedicalData.created_at.desc()'
    )

    def to_dict(self, exclude: list = None) -> dict:
        """Override to handle datetime conversion."""
        data = super().to_dict(exclude=exclude)

        if 'created_at' in data and self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if 'updated_at' in data and self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()

        return data

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, name={self.first_name} {self.last_name})>"