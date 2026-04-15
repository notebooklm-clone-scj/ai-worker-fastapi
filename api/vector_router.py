import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from services.vector_service import delete_document_vectors

router = APIRouter(prefix="/api/v1/vector", tags=["Vector Store"])
logger = logging.getLogger(__name__)


class DocumentVectorDeleteRequest(BaseModel):
    filename: str


@router.delete("/documents/{document_id}")
async def delete_document_vector_endpoint(
    document_id: int,
    request: DocumentVectorDeleteRequest,
    http_request: Request
):
    request_id = getattr(http_request.state, "request_id", "unknown")
    start_time = time.perf_counter()

    try:
        deleted_count = delete_document_vectors(
            document_id=document_id,
            filename=request.filename,
            request_id=request_id
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "event=vector_delete_request_success requestId=%s documentId=%s latencyMs=%s deletedCount=%s",
            request_id,
            document_id,
            latency_ms,
            deleted_count
        )
        return {"deleted_count": deleted_count}
    except ValueError as ve:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.warning(
            "event=vector_delete_request_failure requestId=%s documentId=%s latencyMs=%s error=%s",
            request_id,
            document_id,
            latency_ms,
            str(ve)
        )
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.exception(
            "event=vector_delete_request_unhandled_failure requestId=%s documentId=%s latencyMs=%s",
            request_id,
            document_id,
            latency_ms
        )
        raise HTTPException(status_code=500, detail="서버 내부 에러가 발생했습니다.")
