from src.modules.treatment_decisions.domain.models.enums import FollowUpStatus, PatientStatus, TreatmentAction, \
    Adherence
from src.shared.data.base.model import BaseModel
from src.shared.data.database import db


class FollowUp(BaseModel):
    """
    Follow-up model tracking scheduled and completed patient visits.
    """
    __tablename__ = 'follow_ups'

    # Foreign keys
    patient_id = db.Column(db.String(36), db.ForeignKey('patients.id'), nullable=False, index=True)
    decision_id = db.Column(db.String(36), db.ForeignKey('treatment_decisions.id'), nullable=False, index=True)
    recorded_by = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=True)  # NULL until completed

    # Scheduling
    scheduled_date = db.Column(db.Date, nullable=False, index=True)
    status = db.Column(db.Enum(FollowUpStatus), nullable=False, default=FollowUpStatus.SCHEDULED, index=True)

    # Visit details (NULL until completed)
    visit_date = db.Column(db.Date, nullable=True)

    # New measurements
    hba1c_new = db.Column(db.Numeric(4, 2), nullable=True)
    weight_new = db.Column(db.Numeric(5, 2), nullable=True)  # kg
    egfr_new = db.Column(db.Numeric(5, 1), nullable=True)
    bp_systolic_new = db.Column(db.Integer, nullable=True)
    bp_diastolic_new = db.Column(db.Integer, nullable=True)

    # Assessment
    patient_status = db.Column(db.Enum(PatientStatus), nullable=True)
    adherence = db.Column(db.Enum(Adherence), nullable=True)
    adverse_events = db.Column(db.Text, nullable=True)
    patient_feedback = db.Column(db.Text, nullable=True)

    # Treatment action
    treatment_action = db.Column(db.Enum(TreatmentAction), nullable=True)
    action_notes = db.Column(db.Text, nullable=True)  # Required if action='adjust' or 'change'

    def to_dict(self, exclude: list = None) -> dict:
        """Override to handle enum serialization and datetime conversion."""
        data = super().to_dict(exclude=exclude)

        # Handle enum serialization
        if 'status' in data:
            data['status'] = self.status.value if hasattr(self.status, 'value') else self.status
        if 'patient_status' in data and self.patient_status:
            data['patient_status'] = self.patient_status.value if hasattr(self.patient_status, 'value') else self.patient_status
        if 'adherence' in data and self.adherence:
            data['adherence'] = self.adherence.value if hasattr(self.adherence, 'value') else self.adherence
        if 'treatment_action' in data and self.treatment_action:
            data['treatment_action'] = self.treatment_action.value if hasattr(self.treatment_action, 'value') else self.treatment_action

        # Handle date serialization
        if 'scheduled_date' in data and self.scheduled_date:
            data['scheduled_date'] = self.scheduled_date.isoformat()
        if 'visit_date' in data and self.visit_date:
            data['visit_date'] = self.visit_date.isoformat()

        # Handle datetime serialization
        if 'created_at' in data and self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if 'updated_at' in data and self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()

        return data

    def __repr__(self) -> str:
        return f"<FollowUp(id={self.id}, patient_id={self.patient_id}, status={self.status.value})>"