from enum import Enum


class Role(Enum):
    DOCTOR = 'doctor'
    ML_ENGINEER = 'ml_engineer'

class TokenType(Enum):
    ACCESS = 'ACCESS'
    REFRESH = 'REFRESH'