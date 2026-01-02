"""
Patient management service for CRUD operations.
Handles creating, reading, updating, and deleting patients.
"""

from typing import List, Tuple

from src.modules.patients.domain.models.patient import Patient
from src.modules.patients.presentation.dtos.patient_dtos import (
    CreatePatientRequest,
    UpdatePatientContactRequest,
    PatientResponse,
    PatientDetailResponse,
    GetPatientRequest,
    DeletePatientRequest,
    ListPatientsRequest
)
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.patient_medical_data_repository import PatientMedicalDataRepository
from src.shared.exceptions.exceptions import ConflictException, NotFoundException, ValidationException
from src.shared.response.error_detail import ErrorDetail


class PatientService:
    """
    Service for patient CRUD operations.
    """

    def __init__(self):
        self.patient_repository = PatientRepository()
        self.medical_data_repository = PatientMedicalDataRepository()

    def create_patient(self, request: CreatePatientRequest) -> PatientResponse:
        """
        Create a new patient.

        Args:
            request: CreatePatientRequest DTO

        Returns:
            PatientResponse DTO

        Raises:
            ConflictException: If patient with same first name and last name already exists
        """
        # Check if patient with same first name and last name already exists
        existing_patients = self.patient_repository.find_many_by({
            "first_name": request.first_name,
            "last_name": request.last_name
        })

        if existing_patients:
            error = ErrorDetail(
                title="Patient Creation Failed",
                code="PATIENT_NAME_EXISTS",
                status=409,
                details=[f"Patient with name {request.first_name} {request.last_name} already exists"]
            )
            error.add_field_error("name", "Patient with this name already exists")
            raise ConflictException(
                message="A patient with this name already exists in the system",
                error_detail=error
            )

        # Create patient
        patient = Patient(
            first_name=request.first_name,
            last_name=request.last_name,
            email=request.email,
            mobile_number=request.mobile_number
        )

        # Save to database
        saved_patient = self.patient_repository.create(patient)

        # Convert to response DTO
        return PatientResponse.model_validate(saved_patient)

    def get_patient(self, request: GetPatientRequest) -> PatientResponse:
        """
        Get a single patient by ID (contact info only).

        Args:
            request: GetPatientRequest DTO

        Returns:
            PatientResponse DTO

        Raises:
            NotFoundException: If patient not found
        """
        patient = self.patient_repository.find_by_id(request.patient_id)

        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {request.patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're looking for doesn't exist",
                error_detail=error
            )

        return PatientResponse.model_validate(patient)

    def get_patient_detail(self, request: GetPatientRequest) -> PatientDetailResponse:
        """
        Get patient with medical data.

        Args:
            request: GetPatientRequest DTO

        Returns:
            PatientDetailResponse DTO

        Raises:
            NotFoundException: If patient not found
        """
        patient = self.patient_repository.find_by_id(request.patient_id)

        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {request.patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're looking for doesn't exist",
                error_detail=error
            )

        return PatientDetailResponse.model_validate(patient)

    def update_patient_contact(self, patient_id: str, request: UpdatePatientContactRequest) -> PatientResponse:
        """
        Update patient contact information.

        Args:
            patient_id: Patient ID
            request: UpdatePatientContactRequest DTO

        Returns:
            PatientResponse DTO

        Raises:
            NotFoundException: If patient not found
            ConflictException: If name combination already exists
        """
        # Find patient
        patient = self.patient_repository.find_by_id(patient_id)
        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're trying to update doesn't exist",
                error_detail=error
            )

        # Check if name is being changed
        first_name = request.first_name if request.first_name else patient.first_name
        last_name = request.last_name if request.last_name else patient.last_name

        # Check if name combination already exists (excluding current patient)
        if request.first_name or request.last_name:
            existing_patients = self.patient_repository.find_many_by({
                "first_name": first_name,
                "last_name": last_name
            })

            # Filter out the current patient from results
            existing_patients = [p for p in existing_patients if p.id != patient_id]

            if existing_patients:
                error = ErrorDetail(
                    title="Update Failed",
                    code="PATIENT_NAME_EXISTS",
                    status=409,
                    details=[f"Patient with name {first_name} {last_name} already exists"]
                )
                error.add_field_error("first_name", "Patient with this name already exists")
                error.add_field_error("last_name", "Patient with this name already exists")
                raise ConflictException(
                    message="A patient with this name already exists in the system",
                    error_detail=error
                )

        # Update fields
        if request.first_name:
            patient.first_name = request.first_name

        if request.last_name:
            patient.last_name = request.last_name

        if request.email is not None:
            patient.email = request.email

        if request.mobile_number is not None:
            patient.mobile_number = request.mobile_number

        # Save changes
        updated_patient = self.patient_repository.update(patient)

        return PatientResponse.model_validate(updated_patient)

    def delete_patient(self, request: DeletePatientRequest) -> None:
        """
        Soft delete a patient.

        Args:
            request: DeletePatientRequest DTO

        Raises:
            NotFoundException: If patient not found
        """
        patient = self.patient_repository.find_by_id(request.patient_id)

        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {request.patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're trying to delete doesn't exist",
                error_detail=error
            )

        # Soft delete
        self.patient_repository.delete(patient)

    def list_patients(self, request: ListPatientsRequest) -> Tuple[List[PatientResponse], int]:
        """
        List patients with pagination and optional search.

        Args:
            request: ListPatientsRequest DTO

        Returns:
            Tuple of (list of PatientResponse DTOs, total count)
        """
        # Get total count
        total = self.patient_repository.count()

        # Get paginated patients
        pagination = self.patient_repository.paginate(
            page=request.page,
            per_page=request.per_page,
            include_deleted=False
        )

        patients = pagination.items

        # Apply search filter if specified
        if request.search:
            patients = self.patient_repository.search_by_name(request.search)

            # Also search by email and mobile
            search_lower = request.search.lower()
            additional_patients = [
                p for p in pagination.items
                if (p.email and search_lower in p.email.lower()) or
                   (p.mobile_number and search_lower in p.mobile_number.lower())
            ]

            # Combine and remove duplicates
            patient_ids = {p.id for p in patients}
            for p in additional_patients:
                if p.id not in patient_ids:
                    patients.append(p)
                    patient_ids.add(p.id)

        # Convert to response DTOs
        patient_responses = [PatientResponse.model_validate(patient) for patient in patients]

        return patient_responses, total

    def restore_patient(self, patient_id: str) -> PatientResponse:
        """
        Restore a soft-deleted patient.

        Args:
            patient_id: Patient ID to restore

        Returns:
            PatientResponse DTO

        Raises:
            NotFoundException: If patient not found
            ValidationException: If patient is not deleted
        """
        # Find patient including deleted ones
        patient = self.patient_repository.find_by_id(patient_id, include_deleted=True)

        if not patient:
            error = ErrorDetail(
                title="Patient Not Found",
                code="PATIENT_NOT_FOUND",
                status=404,
                details=[f"Patient with ID {patient_id} does not exist"]
            )
            raise NotFoundException(
                message="The patient you're trying to restore doesn't exist",
                error_detail=error
            )

        # Check if already active
        if not patient.is_deleted:
            error = ErrorDetail(
                title="Patient Already Active",
                code="PATIENT_ACTIVE",
                status=400,
                details=["Patient is not deleted"]
            )
            raise ValidationException(
                message="This patient is already active",
                error_detail=error
            )

        # Restore patient
        restored_patient = self.patient_repository.restore(patient)

        return PatientResponse.model_validate(restored_patient)