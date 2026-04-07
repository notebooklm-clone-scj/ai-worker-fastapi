import fitz

def extract_text_from_pdf(file_bytes: bytes) -> dict:
    try:
        # pdf 열기
        pdf_document = fitz.open(stream=file_bytes, filetype = "pdf")

        # 텍스트 추출
        full_text = ""
        total_pages = len(pdf_document)
        pages_data = [] # 페이지별로 번호와 텍스트를 따로 담음

        for page_num in range(total_pages):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()

            full_text += page_text # 전체 텍스트용

            # 추후 임베딩할 때 쓰기 위해 페이지 번호와 글을 함께 저장
            pages_data.append({
                "page_number": page_num + 1,
                "text": page_text
            })
        
        pdf_document.close()

        # 결과 반환 (추후 LLM에 던질 때는 full_text 사용)
        return {
            "total_pages": total_pages,
            "text_preview": full_text[:500] + "\n\n... (이하 생략) ...",
            "full_text_length": len(full_text),
            "full_text": full_text,
            "pages_data": pages_data # 딕셔너리에 추가
        }
    
    except Exception as e:
        raise ValueError(f"PDF 텍스트 추출 실패: {str(e)}")