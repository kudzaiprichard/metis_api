"""
Model Manager service for ML model CRUD operations.
Handles listing, activating, deleting, and managing model versions.
"""

from typing import List, Optional, Dict
from src.modules.ml_models.presentation.dtos.model_dtos import (
    ModelVersionResponse,
    ModelListResponse,
    ActiveModelResponse,
    ModelStatusResponse,
    ModelComparisonResponse,
    ModelLineageResponse,
    ActivateModelRequest,
    DeleteModelRequest,
    ListModelsRequest,
    CompareModelsRequest
)
from src.shared.exceptions.exceptions import (
    NotFoundException,
    ConflictException,
    InternalServerException,
    BadRequestException
)
from src.shared.response.error_detail import ErrorDetail

# Import ML Service Manager
from src.shared.ml.service_initializer import get_ml_service


class ModelManagerService:
    """
    Service for ML model management operations.
    Uses the shared MLServiceManager singleton initialized in app factory.
    """

    def __init__(self):
        """
        Initialize ModelManagerService.
        Uses the shared ML service manager which is already configured.
        """
        try:
            # Get the shared ML service manager (already initialized)
            ml_service = get_ml_service()
            self.model_manager = ml_service.get_model_manager()

        except RuntimeError as e:
            error = ErrorDetail(
                title="Model Manager Not Available",
                code="MANAGER_NOT_INITIALIZED",
                status=500,
                details=[str(e), "ML services may not be initialized yet"]
            )
            raise InternalServerException(
                message="Model manager is not available",
                error_detail=error
            )
        except Exception as e:
            error = ErrorDetail(
                title="Model Manager Error",
                code="MANAGER_ERROR",
                status=500,
                details=[f"Unexpected error: {str(e)}"]
            )
            raise InternalServerException(
                message="An error occurred while accessing model manager",
                error_detail=error
            )

    def list_models(self, request: ListModelsRequest) -> ModelListResponse:
        """
        List all available model versions.

        Args:
            request: ListModelsRequest with sorting options

        Returns:
            ModelListResponse with all versions

        Raises:
            BadRequestException: If invalid sort parameter
            InternalServerException: If listing fails
        """
        try:
            # Validate sort_by
            valid_sorts = ['version', 'date', 'avg_reward', 'accuracy']
            if request.sort_by not in valid_sorts:
                error = ErrorDetail(
                    title="Invalid Sort Parameter",
                    code="INVALID_SORT",
                    status=400,
                    details=[f"sort_by must be one of: {', '.join(valid_sorts)}"]
                )
                raise BadRequestException(
                    message="Invalid sorting parameter",
                    error_detail=error
                )

            # Get versions from ModelManager
            versions = self.model_manager.list_versions(
                sort_by=request.sort_by,
                reverse=request.reverse
            )

            # Get active and latest versions
            active_version = self.model_manager.get_active_version()
            latest_version = self.model_manager.get_latest_version()
            total_versions = self.model_manager.get_version_count()

            # Convert to response DTOs
            version_responses = [
                ModelVersionResponse(**v) for v in versions
            ]

            return ModelListResponse(
                total_versions=total_versions,
                active_version=active_version,
                latest_version=latest_version,
                versions=version_responses
            )

        except BadRequestException:
            raise
        except Exception as e:
            error = ErrorDetail(
                title="Failed to List Models",
                code="LIST_MODELS_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to retrieve model list",
                error_detail=error
            )

    def get_model_info(self, version: str) -> ModelVersionResponse:
        """
        Get detailed information for a specific model version.

        Args:
            version: Model version number (e.g., 'v1_2')

        Returns:
            ModelVersionResponse with version details

        Raises:
            NotFoundException: If version not found
            InternalServerException: If retrieval fails
        """
        try:
            model_info = self.model_manager.get_model_info(version)

            if model_info is None:
                available = [v['version_number'] for v in self.model_manager.list_versions()]
                error = ErrorDetail(
                    title="Model Version Not Found",
                    code="MODEL_NOT_FOUND",
                    status=404,
                    details=[
                        f"Model version '{version}' does not exist",
                        f"Available versions: {', '.join(available)}"
                    ]
                )
                raise NotFoundException(
                    message=f"Model version '{version}' not found",
                    error_detail=error
                )

            return ModelVersionResponse(**model_info)

        except NotFoundException:
            raise
        except Exception as e:
            error = ErrorDetail(
                title="Failed to Get Model Info",
                code="GET_MODEL_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to retrieve model information",
                error_detail=error
            )

    def get_active_model(self) -> ActiveModelResponse:
        """
        Get currently active model version info.

        Returns:
            ActiveModelResponse with active model details

        Raises:
            NotFoundException: If no active model set
            InternalServerException: If retrieval fails
        """
        try:
            active_version = self.model_manager.get_active_version()

            if active_version is None:
                # Fallback to latest
                latest_version = self.model_manager.get_latest_version()
                if latest_version is None:
                    error = ErrorDetail(
                        title="No Models Available",
                        code="NO_MODELS",
                        status=404,
                        details=["No models have been registered yet"]
                    )
                    raise NotFoundException(
                        message="No active model found",
                        error_detail=error
                    )
                active_version = latest_version

            # Get model info
            model_info = self.model_manager.get_model_info(active_version)

            return ActiveModelResponse(
                version_number=model_info['version_number'],
                trained_timestamp=model_info['trained_timestamp'],
                performance_metrics=model_info['performance_metrics'],
                is_active=model_info['is_active']
            )

        except NotFoundException:
            raise
        except Exception as e:
            error = ErrorDetail(
                title="Failed to Get Active Model",
                code="GET_ACTIVE_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to retrieve active model",
                error_detail=error
            )

    def activate_model(self, request: ActivateModelRequest) -> ModelVersionResponse:
        """
        Activate a specific model version.

        This updates both:
        1. ModelManager metadata (model_metadata.json)
        2. MLServiceManager cache (reloads active pipeline)

        Args:
            request: ActivateModelRequest with version to activate

        Returns:
            ModelVersionResponse for newly activated version

        Raises:
            NotFoundException: If version not found
            InternalServerException: If activation fails
        """
        try:
            # Get ML service to update cache
            ml_service = get_ml_service()

            # Switch active version (updates both registry and cache)
            switch_result = ml_service.switch_active_version(request.version)

            # Get updated model info
            model_info = self.model_manager.get_model_info(request.version)

            return ModelVersionResponse(**model_info)

        except ValueError as e:
            # Version not found
            available = [v['version_number'] for v in self.model_manager.list_versions()]
            error = ErrorDetail(
                title="Model Version Not Found",
                code="MODEL_NOT_FOUND",
                status=404,
                details=[
                    str(e),
                    f"Available versions: {', '.join(available)}"
                ]
            )
            raise NotFoundException(
                message=f"Cannot activate version '{request.version}' - not found",
                error_detail=error
            )
        except Exception as e:
            error = ErrorDetail(
                title="Model Activation Failed",
                code="ACTIVATION_ERROR",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to activate model version",
                error_detail=error
            )

    def delete_model(self, request: DeleteModelRequest) -> Dict:
        """
        Delete a model version.

        Args:
            request: DeleteModelRequest with version and delete_files flag

        Returns:
            Dict with success message

        Raises:
            NotFoundException: If version not found
            ConflictException: If trying to delete active version
            InternalServerException: If deletion fails
        """
        try:
            # Attempt deletion
            success = self.model_manager.delete_version(
                version=request.version,
                delete_files=request.delete_files
            )

            if not success:
                error = ErrorDetail(
                    title="Model Version Not Found",
                    code="MODEL_NOT_FOUND",
                    status=404,
                    details=[f"Model version '{request.version}' does not exist"]
                )
                raise NotFoundException(
                    message=f"Model version '{request.version}' not found",
                    error_detail=error
                )

            return {
                "message": f"Model version '{request.version}' deleted successfully",
                "version": request.version,
                "files_deleted": request.delete_files
            }

        except Exception as e:
            # Check if it's an active version conflict
            if "Cannot delete active version" in str(e):
                error = ErrorDetail(
                    title="Cannot Delete Active Model",
                    code="DELETE_ACTIVE_MODEL",
                    status=409,
                    details=[
                        f"Version '{request.version}' is currently active",
                        "Activate a different version before deleting"
                    ]
                )
                raise ConflictException(
                    message="Cannot delete the currently active model",
                    error_detail=error
                )

            # Check if it's a not found error
            if not success:
                raise

            # Other errors
            error = ErrorDetail(
                title="Model Deletion Failed",
                code="DELETE_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to delete model version",
                error_detail=error
            )

    def get_status(self) -> ModelStatusResponse:
        """
        Get model manager status.

        Returns:
            ModelStatusResponse with system status

        Raises:
            InternalServerException: If status retrieval fails
        """
        try:
            status = self.model_manager.get_status()

            return ModelStatusResponse(**status)

        except Exception as e:
            error = ErrorDetail(
                title="Failed to Get Status",
                code="STATUS_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to retrieve model manager status",
                error_detail=error
            )

    def compare_models(self, request: CompareModelsRequest) -> ModelComparisonResponse:
        """
        Compare performance between two model versions.

        Args:
            request: CompareModelsRequest with two versions

        Returns:
            ModelComparisonResponse with comparison details

        Raises:
            NotFoundException: If either version not found
            InternalServerException: If comparison fails
        """
        try:
            comparison = self.model_manager.compare_versions(
                v1=request.version_1,
                v2=request.version_2
            )

            # Get full model info for both versions
            model_info_1 = self.model_manager.get_model_info(request.version_1)
            model_info_2 = self.model_manager.get_model_info(request.version_2)

            if model_info_1 is None or model_info_2 is None:
                missing = []
                if model_info_1 is None:
                    missing.append(request.version_1)
                if model_info_2 is None:
                    missing.append(request.version_2)

                error = ErrorDetail(
                    title="Model Version Not Found",
                    code="MODEL_NOT_FOUND",
                    status=404,
                    details=[f"Version(s) not found: {', '.join(missing)}"]
                )
                raise NotFoundException(
                    message="One or more model versions not found",
                    error_detail=error
                )

            return ModelComparisonResponse(
                version_1=request.version_1,
                version_2=request.version_2,
                metrics_v1=model_info_1['performance_metrics'],
                metrics_v2=model_info_2['performance_metrics'],
                differences=comparison
            )

        except NotFoundException:
            raise
        except Exception as e:
            error = ErrorDetail(
                title="Model Comparison Failed",
                code="COMPARISON_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to compare model versions",
                error_detail=error
            )

    def get_lineage(self, version: str) -> ModelLineageResponse:
        """
        Get version lineage (ancestry) for a model.

        Args:
            version: Model version number

        Returns:
            ModelLineageResponse with lineage chain

        Raises:
            NotFoundException: If version not found
            InternalServerException: If lineage retrieval fails
        """
        try:
            lineage = self.model_manager.get_version_lineage(version)

            if not lineage:
                error = ErrorDetail(
                    title="Model Version Not Found",
                    code="MODEL_NOT_FOUND",
                    status=404,
                    details=[f"Model version '{version}' does not exist"]
                )
                raise NotFoundException(
                    message=f"Model version '{version}' not found",
                    error_detail=error
                )

            return ModelLineageResponse(
                version=version,
                lineage=lineage,
                depth=len(lineage)
            )

        except NotFoundException:
            raise
        except Exception as e:
            error = ErrorDetail(
                title="Failed to Get Lineage",
                code="LINEAGE_FAILED",
                status=500,
                details=[str(e)]
            )
            raise InternalServerException(
                message="Failed to retrieve model lineage",
                error_detail=error
            )