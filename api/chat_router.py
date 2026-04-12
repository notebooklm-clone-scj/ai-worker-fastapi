from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.chat_service import ask_question_to_pdf, summarize_conversation
from typing import List, Dict, Optional

router = APIRouter(prefix="/api/v1/chat", tags=["Chat Q&A"])

# 유저가 보낼 JSON 데이터 모양
class ChatRequest(BaseModel):
    question: str
    conversation_summary: Optional[str] = None

    # [{"role": "USER", "message": "..."}, ...] 형태로 들어온다
    history: Optional[List[Dict[str, str]]] = []

class ChatSummaryRequest(BaseModel):
    existing_summary: Optional[str] = None
    history: List[Dict[str, str]]

@router.post("/")
async def chat_with_document(request: ChatRequest):
    try:
        # service에 유저 질문 전달
        return ask_question_to_pdf(
            question=request.question,
            history=request.history,
            conversation_summary=request.conversation_summary
        )
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="서버 내부 에러가 발생했습니다.")
    
@router.post("/summary")
async def summarize_chat_memory(request: ChatSummaryRequest):
    try:
        return summarize_conversation(
            existing_summary=request.existing_summary,
            history=request.history
        )
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="서버 내부 에러가 발생했습니다.")