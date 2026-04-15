import logging
import time

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from services.pdf_service import extract_text_from_pdf
from services.llm_service import summarize_text
from services.vector_service import process_and_store_document

router = APIRouter(prefix="/api/v1/pdf", tags=["PDF Extraction"])
logger = logging.getLogger(__name__)

# PDF 업로드 전체 흐름을 단계별로 나눠 시간 로그와 함께 처리한다.
@router.post("/extract")
async def extract_pdf_endpoint(
    http_request: Request,
    file: UploadFile = File(...),
    notebook_id: int = Form(...),
    document_id: int = Form(...)
):
    request_id = getattr(http_request.state, "request_id", "unknown")
    total_start = time.perf_counter()

    # 요청 검증 (확장자 확인)
    if not file.filename.lower().endswith(".pdf"):
        logger.warning("event=pdf_request_invalid_extension requestId=%s filename=%s", request_id, file.filename)
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능")
    
    try:
        # 파일 데이터 읽기
        file_bytes = await file.read()
        logger.info(
            "event=pdf_request_start requestId=%s filename=%s fileSizeBytes=%s",
            request_id,
            file.filename,
            len(file_bytes)
        )
        
        # pdf_service에 전달 (PDF에서 전체 글자(full_text) 확인)
        extract_start = time.perf_counter()
        result = extract_text_from_pdf(file_bytes)
        extract_latency_ms = int((time.perf_counter() - extract_start) * 1000)
        full_text = result["full_text"]
        pages_data = result["pages_data"] # 페이지별 데이터
        logger.info(
            "event=pdf_extract_stage_success requestId=%s filename=%s latencyMs=%s totalPages=%s fullTextLength=%s",
            request_id,
            file.filename,
            extract_latency_ms,
            result["total_pages"],
            result["full_text_length"]
        )

        # llm_service에 전달 (full_text를 AI에게 전달 후 요약)
        summarize_start = time.perf_counter()
        summary = summarize_text(full_text)
        summarize_latency_ms = int((time.perf_counter() - summarize_start) * 1000)
        logger.info(
            "event=pdf_summary_stage_success requestId=%s filename=%s latencyMs=%s summaryLength=%s",
            request_id,
            file.filename,
            summarize_latency_ms,
            len(summary)
        )

        # vector_service에 전달 (페이지별로 나눈 전체 글자를 chunking해서 벡터db에 저장)
        vector_start = time.perf_counter()
        chunk_count = process_and_store_document(
            pages_data=pages_data,
            filename=file.filename,
            document_id=document_id,
            notebook_id=notebook_id,
            request_id=request_id
        )
        vector_latency_ms = int((time.perf_counter() - vector_start) * 1000)
        logger.info(
            "event=pdf_vector_stage_success requestId=%s filename=%s latencyMs=%s chunkCount=%s",
            request_id,
            file.filename,
            vector_latency_ms,
            chunk_count
        )
        
        result["filename"] = file.filename # 결과에 파일이름 추가
        result["summary"] = summary # 결과에 요약 추가
        result["chunks_saved"] = chunk_count # 몇 조각으로 나눴는지 추가

        result.pop("full_text", None) # 전체 텍스트 숨김, 응답 화면이 터지지 않게 전체 텍스트는 응답에서 제외
        result.pop("pages_data", None) # 이것도 줄 필요없지

        total_latency_ms = int((time.perf_counter() - total_start) * 1000)
        logger.info(
            "event=pdf_request_success requestId=%s filename=%s latencyMs=%s totalPages=%s chunkCount=%s",
            request_id,
            file.filename,
            total_latency_ms,
            result["total_pages"],
            chunk_count
        )
        return result

    except ValueError as ve:
        total_latency_ms = int((time.perf_counter() - total_start) * 1000)
        logger.warning(
            "event=pdf_request_failure requestId=%s filename=%s latencyMs=%s error=%s",
            request_id,
            file.filename,
            total_latency_ms,
            str(ve)
        )
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        total_latency_ms = int((time.perf_counter() - total_start) * 1000)
        logger.exception(
            "event=pdf_request_unhandled_failure requestId=%s filename=%s latencyMs=%s",
            request_id,
            file.filename,
            total_latency_ms
        )
        raise HTTPException(status_code=500, detail="서버 내부 에러 발생")
