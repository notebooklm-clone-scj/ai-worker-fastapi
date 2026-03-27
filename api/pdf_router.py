from fastapi import APIRouter, UploadFile, File, HTTPException
from services.pdf_service import extract_text_from_pdf
from services.llm_service import summarize_text

router = APIRouter(prefix="/api/v1/pdf", tags=["PDF Extraction"])

@router.post("/extract")
async def extract_pdf_endpoint(file: UploadFile = File(...)):

    # 요청 검증 (확장자 확인)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능")
    
    try:
        # 파일 데이터 읽기
        file_bytes = await file.read()
        
        # Service에 전달
        result = extract_text_from_pdf(file_bytes)

        # full_text를 AI에게 전달 후 요약
        summary = summarize_text(result["full_text"])
        
        result["filename"] = file.filename # 결과에 파일이름 추가
        result["summary"] = summary # 결과에 요약 추가
        result.pop("full_text", None) # 전체 텍스트 숨김, 응답 화면이 터지지 않게 전체 텍스트는 응답에서 제외 (추후 AI한테 전달)
        
        return result

    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="서버 내부 에러 발생")