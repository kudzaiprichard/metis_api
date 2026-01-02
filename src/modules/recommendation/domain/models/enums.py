from enum import Enum


class Treatment(Enum):
    METFORMIN = 'Metformin'
    GLP1 = 'GLP-1'
    SGLT2 = 'SGLT-2'
    DPP4 = 'DPP-4'
    INSULIN = 'Insulin'


class ConfidenceLevel(Enum):
    CRITICAL = 'critical'
    LOW = 'low'
    MODERATE = 'moderate'
    HIGH = 'high'
    VERY_HIGH = 'very_high'


class ClinicalPriority(Enum):
    ROUTINE = 'routine'
    STANDARD = 'standard'
    URGENT = 'urgent'
    CRITICAL = 'critical'


class SafetySeverity(Enum):
    INFO = 'info'
    CAUTION = 'caution'
    WARNING = 'warning'
    CRITICAL = 'critical'