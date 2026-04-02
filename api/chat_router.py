from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.chat_service import ask_question_to_pdf
from typing import List, Dict, Optional

router = APIRouter(prefix="/api/v1/chat", tags=["Chat Q&A"])

# 유저가 보낼 JSON 데이터 모양
class ChatRequest(BaseModel):
    question: str

    # [{"role": "USER", "message": "..."}, ...] 형태로 들어온다
    history: Optional[List[Dict[str, str]]] = []

@router.post("/")
async def chat_with_document(request: ChatRequest):
    try:
        # service에 유저 질문 전달
        result = ask_question_to_pdf(request.question, request.history)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="서버 내부 에러가 발생했습니다.")