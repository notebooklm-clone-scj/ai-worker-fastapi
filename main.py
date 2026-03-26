import fitz
from fastapi import FastAPI, UploadFile, File, HTTPException


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
    # 1. 파일 확장자 검사 (PDF가 아니면 입구컷!)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    try:
        # 2. 유저가 올린 파일을 메모리로 스윽 읽어옵니다.
        file_bytes = await file.read()

        # 3. PyMuPDF(fitz)에게 "이거 PDF니까 열어봐!" 하고 던져줍니다.
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")

        # 4. 1페이지부터 끝페이지까지 돌면서 글자만 쏙쏙 뽑아냅니다.
        extracted_text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            extracted_text += page.get_text()

        total_pages = len(pdf_document)
        pdf_document.close() # 다 읽었으면 책 덮기!

        # 5. 추출된 결과를 반환합니다. (너무 길면 화면이 터지니까 앞부분 500자만 미리보기로 줍니다!)
        return {
            "filename": file.filename,
            "total_pages": total_pages,
            "text_preview": extracted_text[:500] + "\n\n... (이하 생략) ...",
            "full_text_length": len(extracted_text)
        }

    except Exception as e:
        # 만약 알 수 없는 에러가 터지면 500 에러를 뱉습니다.
        raise HTTPException(status_code=500, detail=f"PDF 추출 중 에러 발생: {str(e)}")