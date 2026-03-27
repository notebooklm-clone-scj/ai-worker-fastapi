import os
import google.generativeai as genai
from dotenv import load_dotenv

# 환경변수(.env) 파일에서 비밀키 확인
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 제미나이에게 키를 건네주며 인증을 완료
genai.configure(api_key=GEMINI_API_KEY)

def summarize_text(text: str) -> str:
    # pdf의 텍스트를 AI에 전달하여 요약본을 받음
    
    # 사용할 모델 선택 (빠르고 무료인 1.5 Flash 모델 사용)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # AI에게 내릴 프롬프트(명령어) 작성
    prompt = f"너는 문서 요약 전문가야. 다음 내용을 읽고 핵심만 3줄로 요약해줘:\n\n{text}"
    
    try:
        # AI에게 질문을 던지고 대답 대기
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise ValueError(f"AI 요약 중 에러 발생: {str(e)}")