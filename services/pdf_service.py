import fitz

def extract_text_from_pdf(file_bytes: bytes) -> dict:
    try:
        # pdf 열기
        pdf_document = fitz.open(stream=file_bytes, filetype = "pdf")

        # 텍스트 추출
        extracted_text = ""
        total_pages = len(pdf_document)

        for page_num in range(total_pages):
            page = pdf_document.load_page(page_num)
            extracted_text += page.get_text()
        
        pdf_document.close()

        # 결과 반환 (추후 LLM에 던질 때는 full_text 사용)
        return {
            "total_pages": total_pages,
            "text_preview": extracted_text[:500] + "\n\n... (이하 생략) ...",
            "full_text_length": len(extracted_text),
            "full_text": extracted_text
        }
    
    except Exception as e:
        raise ValueError(f"PDF 텍스트 추출 실패: {str(e)}")