from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.chat_service import ask_question_to_pdf

router = APIRouter(prefix="/api/v1/chat", tags=["Chat Q&A"])

# 유저가 보낼 JSON 데이터 모양
class ChatRequest(BaseModel):
    question: str

@router.post("/")
async def chat_with_document(request: ChatRequest):
    try:
        # service에 유저 질문 전달
        result = ask_question_to_pdf(request.question)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="서버 내부 에러가 발생했습니다.")