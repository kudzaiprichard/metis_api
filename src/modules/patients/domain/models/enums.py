from enum import Enum


class Gender(Enum):
    MALE = 'Male'
    FEMALE = 'Female'


class Ethnicity(Enum):
    CAUCASIAN = 'Caucasian'
    AFRICAN = 'African'
    ASIAN = 'Asian'
    HISPANIC = 'Hispanic'
    OTHER = 'Other'