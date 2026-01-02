# src/modules/monitoring/domain/models/enums.py
from enum import Enum


class DecisionType(Enum):
    ACCEPTED = 'accepted'
    CUSTOM = 'custom'


class PatientStatus(Enum):
    IMPROVING = 'improving'
    STABLE = 'stable'
    WORSENING = 'worsening'


class Adherence(Enum):
    GOOD = 'good'
    FAIR = 'fair'
    POOR = 'poor'


class TreatmentAction(Enum):
    CONTINUE = 'continue'
    ADJUST = 'adjust'
    CHANGE = 'change'


class FollowUpStatus(Enum):
    SCHEDULED = 'scheduled'
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'