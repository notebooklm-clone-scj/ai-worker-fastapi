from fastapi import FastAPI, UploadFile, File

# FastAPI 객체 생성 (이 한 줄이 스프링의 @SpringBootApplication 역할을 합니다!)
app = FastAPI(
    title="NotebookLM AI Worker",
    description="PDF 텍스트 추출 및 AI 처리용 파이썬 서버",
    version="1.0.0"
)

# 1. 서버 헬스체크 (잘 켜졌는지 확인용)
@app.get("/")
def health_check():
    return {"status": "ok", "message": "AI Worker 서버가 정상 작동 중입니다! 🚀"}

# 2. PDF 파일 업로드 테스트 API
@app.post("/api/v1/pdf/extract")
async def extract_pdf_text(file: UploadFile = File(...)):
    # 지금은 파일 이름만 돌려주고, 다음 스텝에서 진짜 텍스트를 추출할 겁니다!
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "파일을 성공적으로 받았습니다!"
    }