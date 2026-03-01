"""
Similar patient search service.

Handles finding similar patient cases using Neo4j graph database with
normalized clinical feature matching and comorbidity overlap scoring.
"""

from typing import Dict, Any

from src.modules.patients.presentation.dtos.similar_patient_dtos import (
    FindSimilarPatientsRequest,
    FindSimilarPatientsGraphRequest,
    GetSimilarPatientDetailRequest,
    SimilarPatientsResponse,
    SimilarPatientsGraphResponse,
    SimilarPatientDetailResponse,
    SimilarPatientCaseResponse,
    PatientProfileResponse,
    OutcomeResponse,
    DemographicsResponse,
    ClinicalFeaturesResponse,
    ClinicalCategoriesResponse,
    TreatmentInfoResponse,
    GraphNodeResponse,
    GraphEdgeResponse,
    GraphMetadataResponse,
    GraphNodeStyleResponse,
    GraphEdgeStyleResponse
)
from src.modules.patients.domain.repositories.patient_repository import PatientRepository
from src.modules.patients.domain.repositories.patient_medical_data_repository import PatientMedicalDataRepository
from src.shared.data.neo4j.neo4j_manager import get_neo4j_manager
from src.shared.exceptions.exceptions import (
    NotFoundException,
    BadRequestException,
    ServiceUnavailableException
)
from src.shared.response.error_detail import ErrorDetail
import logging

logger = logging.getLogger(__name__)


class SimilarPatientService:
    """
    Service for finding similar patient cases.
    Uses the latest medical data record for similarity matching.
    """

    def __init__(self):
        self.patient_repository = PatientRepository()
        self.medical_data_repository = PatientMedicalDataRepository()
        self.neo4j_manager = get_neo4j_manager()

    def _validate_patient_and_get_medical_data(self, patient_id: str):
        """
        Validate patient exists and has medical data.
        Returns the latest medical data record.

        Args:
            patient_id: Patient ID

        Returns:
            Latest PatientMedicalData instance

        Raises:
            NotFoundException: If patient not found
            BadRequestException: If no medical data exists
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
                message="The patient you're searching for doesn't exist",
                error_detail=error
            )

        medical_data = self.medical_data_repository.find_latest_by_patient_id(patient_id)
        if not medical_data:
            error = ErrorDetail(
                title="Medical Data Missing",
                code="NO_MEDICAL_DATA",
                status=400,
                details=["Patient must have medical data to find similar cases"]
            )
            raise BadRequestException(
                message="Cannot find similar cases without patient medical data",
                error_detail=error
            )

        return medical_data

    def _check_neo4j_availability(self):
        """
        Check Neo4j is available.

        Raises:
            ServiceUnavailableException: If Neo4j unavailable
        """
        if not self.neo4j_manager.is_available():
            error = ErrorDetail(
                title="Service Unavailable",
                code="NEO4J_UNAVAILABLE",
                status=503,
                details=["Graph database is currently unavailable"]
            )
            raise ServiceUnavailableException(
                message="Similar patient search is temporarily unavailable",
                error_detail=error
            )

    def find_similar_patients(self, request: FindSimilarPatientsRequest) -> SimilarPatientsResponse:
        """
        Find similar patient cases in tabular format.

        Args:
            request: FindSimilarPatientsRequest DTO

        Returns:
            SimilarPatientsResponse DTO with list of similar cases

        Raises:
            NotFoundException: If patient not found
            BadRequestException: If patient has no medical data
            ServiceUnavailableException: If Neo4j unavailable
        """
        medical_data = self._validate_patient_and_get_medical_data(request.patient_id)
        self._check_neo4j_availability()

        patient_profile = self._build_patient_profile(medical_data)
        neo4j_db = self.neo4j_manager.get_database()

        try:
            similar_cases_raw = neo4j_db.find_similar_patients(
                patient_profile=patient_profile,
                limit=request.limit,
                treatment_filter=request.treatment_filter,
                min_similarity=request.min_similarity
            )

            similar_cases = []
            for case in similar_cases_raw:
                similar_case = SimilarPatientCaseResponse(
                    case_id=case['case_id'],
                    similarity_score=case['similarity_score'],
                    clinical_similarity=case['clinical_similarity'],
                    comorbidity_similarity=case['comorbidity_similarity'],
                    profile=PatientProfileResponse(**case['profile']),
                    comorbidities=case['comorbidities'],
                    treatment_given=case['treatment_given'],
                    drug_class=case['drug_class'],
                    outcome=OutcomeResponse(**case['outcome'])
                )
                similar_cases.append(similar_case)

            filters_applied = {
                "treatment": request.treatment_filter,
                "min_similarity": request.min_similarity,
                "limit": request.limit
            }

            return SimilarPatientsResponse(
                patient_id=request.patient_id,
                similar_cases=similar_cases,
                total_found=len(similar_cases),
                filters_applied=filters_applied
            )

        except Exception as e:
            logger.error(f"Error finding similar patients: {e}")
            error = ErrorDetail(
                title="Search Failed",
                code="SIMILAR_PATIENTS_ERROR",
                status=500,
                details=["Failed to search for similar patients"]
            )
            raise ServiceUnavailableException(
                message="Similar patient search failed. Please try again later",
                error_detail=error
            )

    def find_similar_patients_graph(self, request: FindSimilarPatientsGraphRequest) -> SimilarPatientsGraphResponse:
        medical_data = self._validate_patient_and_get_medical_data(request.patient_id)
        self._check_neo4j_availability()

        patient_profile = self._build_patient_profile(medical_data)
        print(f"[DEBUG] Patient profile: {patient_profile}")

        neo4j_db = self.neo4j_manager.get_database()

        try:
            graph_data = neo4j_db.find_similar_cases_graph(
                patient_profile=patient_profile,
                limit=request.limit,
                treatment_filter=request.treatment_filter
            )

            print(f"[DEBUG] Nodes found: {len(graph_data['nodes'])}")
            print(f"[DEBUG] Edges found: {len(graph_data['edges'])}")
            print(f"[DEBUG] Metadata: {graph_data['metadata']}")

            nodes = []
            for node in graph_data['nodes']:
                graph_node = GraphNodeResponse(
                    id=node['id'],
                    type=node['type'],
                    label=node['label'],
                    data=node['data'],
                    style=GraphNodeStyleResponse(**node['style'])
                )
                nodes.append(graph_node)

            edges = []
            for edge in graph_data['edges']:
                graph_edge = GraphEdgeResponse(
                    id=edge['id'],
                    source=edge['source'],
                    target=edge['target'],
                    type=edge['type'],
                    label=edge['label'],
                    data=edge['data'],
                    style=GraphEdgeStyleResponse(**edge['style'])
                )
                edges.append(graph_edge)

            metadata = GraphMetadataResponse(**graph_data['metadata'])

            return SimilarPatientsGraphResponse(
                patient_id=request.patient_id,
                nodes=nodes,
                edges=edges,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Error finding similar patients (graph): {e}")
            error = ErrorDetail(
                title="Search Failed",
                code="SIMILAR_PATIENTS_GRAPH_ERROR",
                status=500,
                details=["Failed to generate graph for similar patients"]
            )
            raise ServiceUnavailableException(
                message="Similar patient graph generation failed. Please try again later",
                error_detail=error
            )

    def get_similar_patient_detail(self, request: GetSimilarPatientDetailRequest) -> SimilarPatientDetailResponse:
        """
        Get complete details of a similar patient case from Neo4j.

        Args:
            request: GetSimilarPatientDetailRequest DTO with case_id

        Returns:
            SimilarPatientDetailResponse DTO

        Raises:
            NotFoundException: If patient case not found in Neo4j
            ServiceUnavailableException: If Neo4j unavailable
        """
        self._check_neo4j_availability()

        neo4j_db = self.neo4j_manager.get_database()

        try:
            patient_data = neo4j_db.get_patient_by_id(request.case_id)

            if not patient_data:
                error = ErrorDetail(
                    title="Patient Case Not Found",
                    code="CASE_NOT_FOUND",
                    status=404,
                    details=[f"Patient case with ID {request.case_id} does not exist in historical dataset"]
                )
                raise NotFoundException(
                    message="The patient case you're looking for doesn't exist",
                    error_detail=error
                )

            demographics = DemographicsResponse(**patient_data['demographics'])
            clinical_features = ClinicalFeaturesResponse(**patient_data['clinical_features'])
            clinical_categories = ClinicalCategoriesResponse(**patient_data['clinical_categories'])

            treatment = None
            if patient_data['treatment']:
                treatment = TreatmentInfoResponse(**patient_data['treatment'])

            outcome = None
            if patient_data['outcome']:
                outcome = OutcomeResponse(**patient_data['outcome'])

            return SimilarPatientDetailResponse(
                patient_id=patient_data['patient_id'],
                demographics=demographics,
                clinical_features=clinical_features,
                clinical_categories=clinical_categories,
                comorbidities=patient_data['comorbidities'],
                treatment=treatment,
                outcome=outcome
            )

        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving patient case {request.case_id}: {e}")
            error = ErrorDetail(
                title="Lookup Failed",
                code="PATIENT_CASE_LOOKUP_ERROR",
                status=500,
                details=["Failed to retrieve patient case details"]
            )
            raise ServiceUnavailableException(
                message="Patient case lookup failed. Please try again later",
                error_detail=error
            )

    def _build_patient_profile(self, medical_data) -> Dict[str, Any]:
        """
        Build patient profile dictionary from medical data entity.

        Args:
            medical_data: PatientMedicalData entity

        Returns:
            Dictionary with all 21 base features
        """
        return {
            'age': int(medical_data.age),
            'gender': medical_data.gender.value,
            'ethnicity': medical_data.ethnicity.value,
            'hba1c_baseline': float(medical_data.hba1c_baseline),
            'diabetes_duration': float(medical_data.diabetes_duration),
            'fasting_glucose': float(medical_data.fasting_glucose),
            'c_peptide': float(medical_data.c_peptide),
            'egfr': float(medical_data.egfr),
            'bmi': float(medical_data.bmi),
            'bp_systolic': int(medical_data.bp_systolic),
            'bp_diastolic': int(medical_data.bp_diastolic),
            'alt': float(medical_data.alt),
            'ldl': float(medical_data.ldl),
            'hdl': float(medical_data.hdl),
            'triglycerides': float(medical_data.triglycerides),
            'previous_prediabetes': int(medical_data.previous_prediabetes),
            'hypertension': int(medical_data.hypertension),
            'ckd': int(medical_data.ckd),
            'cvd': int(medical_data.cvd),
            'nafld': int(medical_data.nafld),
            'retinopathy': int(medical_data.retinopathy)
        }