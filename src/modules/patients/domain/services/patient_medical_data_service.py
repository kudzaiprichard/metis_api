"""
Patient medical data service for CRUD operations.
Handles creating, reading, updating, and deleting patient medical data.
"""

from src.modules.patients.domain.models.patient_medical_data import PatientMedicalData
from src.modules.patients.domain.models.enums import Gender, Ethnicity
from src.modules.patients.presentation.dtos.patient_dtos import (
    CreatePatientMedicalDataRequest,
    UpdatePatientMedicalDataRequest,
    PatientMedicalDataResponse
)
from src.modules.patients.domain.repositories.patient_medical_data_repository import PatientMedicalDataRepository
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.shared.exceptions.exceptions import ConflictException, NotFoundException, ValidationException
from src.shared.response.error_detail import ErrorDetail


class PatientMedicalDataService:
    """
    Service for patient medical data CRUD operations.
    """

    def __init__(self):
        self.medical_data_repository = PatientMedicalDataRepository()
        self.patient_repository = PatientRepository()

    def create_medical_data(self, request: CreatePatientMedicalDataRequest) -> PatientMedicalDataResponse:
        """
        Create medical data for a patient.

        Args:
            request: CreatePatientMedicalDataRequest DTO

        Returns:
            PatientMedicalDataResponse DTO

        Raises:
            NotFoundException: If patient not found
            ConflictException: If medical data already exists for patient
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

        # Check if medical data already exists
        existing_data = self.medical_data_repository.find_by_patient_id(request.patient_id)
        if existing_data:
            error = ErrorDetail(
                title="Medical Data Creation Failed",
                code="MEDICAL_DATA_EXISTS",
                status=409,
                details=["Medical data already exists for this patient"]
            )
            raise ConflictException(
                message="Medical data already exists for this patient",
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

        # Save to database
        saved_data = self.medical_data_repository.create(medical_data)

        # Convert to response DTO
        return PatientMedicalDataResponse.model_validate(saved_data)

    def get_medical_data(self, patient_id: str) -> PatientMedicalDataResponse:
        """
        Get medical data by patient ID.

        Args:
            patient_id: Patient ID

        Returns:
            PatientMedicalDataResponse DTO

        Raises:
            NotFoundException: If medical data not found
        """
        medical_data = self.medical_data_repository.find_by_patient_id(patient_id)

        if not medical_data:
            error = ErrorDetail(
                title="Medical Data Not Found",
                code="MEDICAL_DATA_NOT_FOUND",
                status=404,
                details=[f"Medical data for patient ID {patient_id} does not exist"]
            )
            raise NotFoundException(
                message="Medical data not found for this patient",
                error_detail=error
            )

        return PatientMedicalDataResponse.model_validate(medical_data)

    def update_medical_data(self, patient_id: str, request: UpdatePatientMedicalDataRequest) -> PatientMedicalDataResponse:
        """
        Update patient medical data.

        Args:
            patient_id: Patient ID
            request: UpdatePatientMedicalDataRequest DTO

        Returns:
            PatientMedicalDataResponse DTO

        Raises:
            NotFoundException: If medical data not found
        """
        # Find medical data
        medical_data = self.medical_data_repository.find_by_patient_id(patient_id)
        if not medical_data:
            error = ErrorDetail(
                title="Medical Data Not Found",
                code="MEDICAL_DATA_NOT_FOUND",
                status=404,
                details=[f"Medical data for patient ID {patient_id} does not exist"]
            )
            raise NotFoundException(
                message="Medical data not found for this patient",
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

        # Save changes
        updated_data = self.medical_data_repository.update(medical_data)

        return PatientMedicalDataResponse.model_validate(updated_data)

    def delete_medical_data(self, patient_id: str) -> None:
        """
        Soft delete patient medical data.

        Args:
            patient_id: Patient ID

        Raises:
            NotFoundException: If medical data not found
        """
        medical_data = self.medical_data_repository.find_by_patient_id(patient_id)

        if not medical_data:
            error = ErrorDetail(
                title="Medical Data Not Found",
                code="MEDICAL_DATA_NOT_FOUND",
                status=404,
                details=[f"Medical data for patient ID {patient_id} does not exist"]
            )
            raise NotFoundException(
                message="Medical data not found for this patient",
                error_detail=error
            )

        # Soft delete
        self.medical_data_repository.delete(medical_data)

    def exists_for_patient(self, patient_id: str) -> bool:
        """
        Check if medical data exists for a given patient.

        Args:
            patient_id: Patient ID

        Returns:
            True if medical data exists, False otherwise
        """
        return self.medical_data_repository.exists_for_patient(patient_id)