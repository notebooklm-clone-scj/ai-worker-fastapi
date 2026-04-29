import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from services.chat_service import ask_question_to_pdf, summarize_conversation
from typing import List, Dict, Optional

router = APIRouter(prefix="/api/v1/chat", tags=["Chat Q&A"])
logger = logging.getLogger(__name__)

# 유저가 보낼 JSON 데이터 모양
class ChatRequest(BaseModel):
    notebook_id: int = Field(..., gt=0)
    document_id: Optional[int] = Field(default=None, gt=0)
    question: str
    conversation_summary: Optional[str] = None

    # [{"role": "USER", "message": "..."}, ...] 형태로 들어온다
    history: Optional[List[Dict[str, str]]] = None

class ChatSummaryRequest(BaseModel):
    existing_summary: Optional[str] = None
    history: List[Dict[str, str]]

# 채팅 요청 단위의 시작/성공/실패 로그를 남기고 실제 생성은 service에 위임한다.
@router.post("/")
async def chat_with_document(request: ChatRequest, http_request: Request):
    request_id = getattr(http_request.state, "request_id", "unknown")
    start_time = time.perf_counter()

    try:
        logger.info(
            "event=chat_request_start requestId=%s notebookId=%s documentId=%s questionLength=%s historyCount=%s hasConversationSummary=%s",
            request_id,
            request.notebook_id,
            request.document_id,
            len(request.question),
            len(request.history or []),
            bool(request.conversation_summary)
        )

        result = ask_question_to_pdf(
            notebook_id=request.notebook_id,
            document_id=request.document_id,
            question=request.question,
            history=request.history or [],
            conversation_summary=request.conversation_summary,
            request_id=request_id
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "event=chat_request_success requestId=%s latencyMs=%s referenceCount=%s answerLength=%s",
            request_id,
            latency_ms,
            len(result.get("reference_chunks", [])),
            len(result.get("answer", ""))
        )
        return result
    except ValueError as ve:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.warning(
            "event=chat_request_failure requestId=%s latencyMs=%s error=%s",
            request_id,
            latency_ms,
            str(ve)
        )
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.exception(
            "event=chat_request_unhandled_failure requestId=%s latencyMs=%s",
            request_id,
            latency_ms
        )
        raise HTTPException(status_code=500, detail="서버 내부 에러가 발생했습니다.")
    
# 오래된 대화를 summary memory로 압축하는 전용 엔드포인트다.
@router.post("/summary")
async def summarize_chat_memory(request: ChatSummaryRequest, http_request: Request):
    request_id = getattr(http_request.state, "request_id", "unknown")
    start_time = time.perf_counter()

    try:
        logger.info(
            "event=chat_summary_request_start requestId=%s historyCount=%s hasExistingSummary=%s",
            request_id,
            len(request.history or []),
            bool(request.existing_summary)
        )

        result = summarize_conversation(
            existing_summary=request.existing_summary,
            history=request.history,
            request_id=request_id
        )

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.info(
            "event=chat_summary_request_success requestId=%s latencyMs=%s summaryLength=%s",
            request_id,
            latency_ms,
            len(result.get("summary", ""))
        )
        return result
    except ValueError as ve:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.warning(
            "event=chat_summary_request_failure requestId=%s latencyMs=%s error=%s",
            request_id,
            latency_ms,
            str(ve)
        )
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        logger.exception(
            "event=chat_summary_request_unhandled_failure requestId=%s latencyMs=%s",
            request_id,
            latency_ms
        )
        raise HTTPException(status_code=500, detail="서버 내부 에러가 발생했습니다.")
