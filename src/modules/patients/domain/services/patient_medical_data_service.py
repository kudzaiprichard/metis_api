"""
Patient medical data service for CRUD operations.
Handles creating, reading, updating, and deleting patient medical data.
"""

from typing import List

from src.modules.patients.domain.models.patient_medical_data import PatientMedicalData
from src.modules.patients.domain.models.enums import Gender, Ethnicity
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.presentation.dtos.patient_dtos import (
    CreatePatientMedicalDataRequest,
    UpdatePatientMedicalDataRequest,
    PatientMedicalDataResponse
)
from src.modules.patients.domain.repositories.patient_medical_data_repository import PatientMedicalDataRepository
from src.shared.exceptions.exceptions import NotFoundException
from src.shared.response.error_detail import ErrorDetail


class PatientMedicalDataService:
    """
    Service for patient medical data CRUD operations.
    Supports multiple medical data records per patient (one per visit).
    """

    def __init__(self):
        self.medical_data_repository = PatientMedicalDataRepository()
        self.patient_repository = PatientRepository()

    def create_medical_data(self, request: CreatePatientMedicalDataRequest) -> PatientMedicalDataResponse:
        """
        Create medical data for a patient.
        Multiple records allowed per patient (one per visit).

        Args:
            request: CreatePatientMedicalDataRequest DTO

        Returns:
            PatientMedicalDataResponse DTO

        Raises:
            NotFoundException: If patient not found
        """
        # Check if patient exists
        patient = self.patient_repository.find_by_id(request.patient_id)
        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {request.patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're trying to add medical data for doesn't exist",
                error_detail=error
            )

        # Create medical data
        medical_data = PatientMedicalData(
            patient_id=request.patient_id,
            age=request.age,
            gender=Gender[request.gender.upper()],
            ethnicity=Ethnicity[request.ethnicity.upper()],
            hba1c_baseline=request.hba1c_baseline,
            diabetes_duration=request.diabetes_duration,
            fasting_glucose=request.fasting_glucose,
            c_peptide=request.c_peptide,
            egfr=request.egfr,
            bmi=request.bmi,
            bp_systolic=request.bp_systolic,
            bp_diastolic=request.bp_diastolic,
            alt=request.alt,
            ldl=request.ldl,
            hdl=request.hdl,
            triglycerides=request.triglycerides,
            previous_prediabetes=request.previous_prediabetes,
            hypertension=request.hypertension,
            ckd=request.ckd,
            cvd=request.cvd,
            nafld=request.nafld,
            retinopathy=request.retinopathy
        )

        saved_data = self.medical_data_repository.create(medical_data)

        return PatientMedicalDataResponse.model_validate(saved_data)

    def get_medical_data(self, medical_data_id: str) -> PatientMedicalDataResponse:
        """
        Get a specific medical data record by ID.

        Args:
            medical_data_id: Medical data record ID

        Returns:
            PatientMedicalDataResponse DTO

        Raises:
            NotFoundException: If medical data not found
        """
        medical_data = self.medical_data_repository.find_by_id(medical_data_id)

        if not medical_data:
            error = ErrorDetail(
                title="Medical Data Not Found",
                code="MEDICAL_DATA_NOT_FOUND",
                status=404,
                details=[f"Medical data with ID {medical_data_id} does not exist"]
            )
            raise NotFoundException(
                message="Medical data record not found",
                error_detail=error
            )

        return PatientMedicalDataResponse.model_validate(medical_data)

    def get_patient_medical_records(self, patient_id: str) -> List[PatientMedicalDataResponse]:
        """
        Get all medical data records for a patient (most recent first).

        Args:
            patient_id: Patient ID

        Returns:
            List of PatientMedicalDataResponse DTOs

        Raises:
            NotFoundException: If patient not found
        """
        patient = self.patient_repository.find_by_id(patient_id)
        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're looking for doesn't exist",
                error_detail=error
            )

        records = self.medical_data_repository.find_by_patient_id(patient_id)

        return [PatientMedicalDataResponse.model_validate(r) for r in records]

    def get_latest_medical_data(self, patient_id: str) -> PatientMedicalDataResponse:
        """
        Get the most recent medical data record for a patient.

        Args:
            patient_id: Patient ID

        Returns:
            PatientMedicalDataResponse DTO

        Raises:
            NotFoundException: If patient or medical data not found
        """
        patient = self.patient_repository.find_by_id(patient_id)
        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're looking for doesn't exist",
                error_detail=error
            )

        medical_data = self.medical_data_repository.find_latest_by_patient_id(patient_id)

        if not medical_data:
            error = ErrorDetail(
                title="Medical Data Not Found",
                code="MEDICAL_DATA_NOT_FOUND",
                status=404,
                details=[f"No medical data exists for patient ID {patient_id}"]
            )
            raise NotFoundException(
                message="No medical data found for this patient",
                error_detail=error
            )

        return PatientMedicalDataResponse.model_validate(medical_data)

    def update_medical_data(self, medical_data_id: str, request: UpdatePatientMedicalDataRequest) -> PatientMedicalDataResponse:
        """
        Update a specific medical data record.

        Args:
            medical_data_id: Medical data record ID
            request: UpdatePatientMedicalDataRequest DTO

        Returns:
            PatientMedicalDataResponse DTO

        Raises:
            NotFoundException: If medical data not found
        """
        medical_data = self.medical_data_repository.find_by_id(medical_data_id)
        if not medical_data:
            error = ErrorDetail(
                title="Medical Data Not Found",
                code="MEDICAL_DATA_NOT_FOUND",
                status=404,
                details=[f"Medical data with ID {medical_data_id} does not exist"]
            )
            raise NotFoundException(
                message="Medical data record not found",
                error_detail=error
            )

        # Update fields
        if request.age is not None:
            medical_data.age = request.age
        if request.gender:
            medical_data.gender = Gender[request.gender.upper()]
        if request.ethnicity:
            medical_data.ethnicity = Ethnicity[request.ethnicity.upper()]
        if request.hba1c_baseline is not None:
            medical_data.hba1c_baseline = request.hba1c_baseline
        if request.diabetes_duration is not None:
            medical_data.diabetes_duration = request.diabetes_duration
        if request.fasting_glucose is not None:
            medical_data.fasting_glucose = request.fasting_glucose
        if request.c_peptide is not None:
            medical_data.c_peptide = request.c_peptide
        if request.egfr is not None:
            medical_data.egfr = request.egfr
        if request.bmi is not None:
            medical_data.bmi = request.bmi
        if request.bp_systolic is not None:
            medical_data.bp_systolic = request.bp_systolic
        if request.bp_diastolic is not None:
            medical_data.bp_diastolic = request.bp_diastolic
        if request.alt is not None:
            medical_data.alt = request.alt
        if request.ldl is not None:
            medical_data.ldl = request.ldl
        if request.hdl is not None:
            medical_data.hdl = request.hdl
        if request.triglycerides is not None:
            medical_data.triglycerides = request.triglycerides
        if request.previous_prediabetes is not None:
            medical_data.previous_prediabetes = request.previous_prediabetes
        if request.hypertension is not None:
            medical_data.hypertension = request.hypertension
        if request.ckd is not None:
            medical_data.ckd = request.ckd
        if request.cvd is not None:
            medical_data.cvd = request.cvd
        if request.nafld is not None:
            medical_data.nafld = request.nafld
        if request.retinopathy is not None:
            medical_data.retinopathy = request.retinopathy

        updated_data = self.medical_data_repository.update(medical_data)

        return PatientMedicalDataResponse.model_validate(updated_data)

    def delete_medical_data(self, medical_data_id: str) -> None:
        """
        Soft delete a specific medical data record.

        Args:
            medical_data_id: Medical data record ID

        Raises:
            NotFoundException: If medical data not found
        """
        medical_data = self.medical_data_repository.find_by_id(medical_data_id)

        if not medical_data:
            error = ErrorDetail(
                title="Medical Data Not Found",
                code="MEDICAL_DATA_NOT_FOUND",
                status=404,
                details=[f"Medical data with ID {medical_data_id} does not exist"]
            )
            raise NotFoundException(
                message="Medical data record not found",
                error_detail=error
            )

        self.medical_data_repository.delete(medical_data)