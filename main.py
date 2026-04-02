from fastapi import FastAPI
from api.pdf_router import router as pdf_router
from api.chat_router import router as chat_router


# FastAPI 객체 생성 (이 한 줄이 스프링의 @SpringBootApplication 역할)
app = FastAPI(
    title="NotebookLM AI Worker",
    description="PDF 텍스트 추출 및 AI 처리용 파이썬 서버",
    version="1.0.0"
)

# 1. 서버 잘 켜졌는지 확인
@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Worker 서버가 정상 작동"}

# 2. PDF 파일 업로드 테스트 API
app.include_router(pdf_router)
app.include_router(chat_router)