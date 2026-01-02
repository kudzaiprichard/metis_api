from pydantic import BaseModel


# ============ Patient Summary for Follow-ups ============

class PatientSummaryResponse(BaseModel):
    """Minimal patient info for follow-ups."""
    id: str
    first_name: str
    last_name: str
    age: int
    gender: str

    model_config = {
        'from_attributes': True
    }