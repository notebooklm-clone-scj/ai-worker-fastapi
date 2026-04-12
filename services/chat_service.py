import os
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

def build_history_text(history: list) -> str:
    if not history:
        return "최근 대화 기록이 없습니다."
    
    lines = []
    for h in history:
        role_name = "유저" if h["role"] == "USER" else "AI"
        lines.append(f"{role_name}: {h['message']}")
    
    return "\n".join(lines)


def ask_question_to_pdf(question: str, history: list, conversation_summary: Optional[str] = None):
    try:
        # 검색 준비: 유저 질문을 임베딩하기 위해 준비
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=GEMINI_API_KEY
        )

        # DB 연결
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name="notebook_documents",
            connection=DATABASE_URL,
            use_jsonb=True
        )

        # 유사도 검색: 유저 질문과 가장 비슷한 글자 조각(k개)를 가져온다.
        docs = vectorstore.similarity_search(query=question, k = 3)

        # 찾아온 조각을 하나의 텍스트로 
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        history_text = build_history_text(history)
        summary_text = conversation_summary if conversation_summary else "이전 대화 요약이 없습니다."

        # 답변 생성 준비: 글을 읽고 대답할 챗봇 AI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=GEMINI_API_KEY,
            temperature=0.1 # 0에 가까울수록 창의성 없이 문서에 있는 내용만 말함 (환각 방지)
        )

        # 프롬프트
        prompt = f"""
        당신은 제공된 [참고 문서]만을 바탕으로 질문에 대답하는 친절한 AI 어시스턴트입니다.
        [이전 대화 요약]과 [최근 대화 기록]을 함께 참고해, 유저의 질문에 자연스럽게 이어지도록 답변하세요.
        만약 [참고 문서]에 질문에 대한 정답이 없다면, 절대 지어내지 말고 "제공된 문서에서는 해당 내용을 찾을 수 없습니다." 라고 대답하세요.

        [이전 대화 요약]
        {summary_text}
        
        [대화 기록]
        {history_text}

        [참고 문서]
        {context}

        [유저 질문]
        {question}
        """

        # AI에게 질문을 던지고 답변 수령
        response = llm.invoke(prompt)

        # 참고한 문서 조각과 페이지 번호
        references = []
        for doc in docs:
            references.append({
                "page_number": doc.metadata.get("page_number", 0), # db에서 페이지 꺼내는데 없을 경우 0
                "content": doc.page_content
            })

        # 유저에게 AI의 대답과 함께 어떤 자료를 참고했는지 함께 제출
        return {
            "answer": response.content,
            "reference_chunks": references
        }
    
    except Exception as e:
        raise ValueError(f"채팅 생성 중 에러 발생: {str(e)}")
    

def summarize_conversation(existing_summary: Optional[str], history: list):
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=GEMINI_API_KEY,
            temperature=0.1
        )

        previous_summary = existing_summary if existing_summary else "이전 요약 없음"
        history_text = build_history_text(history)

        prompt = f"""
        당신은 대화 메모리 압축기입니다.
        기존 요약과 새 대화를 합쳐서, 이후 AI가 참고할 핵심 정보만 남겨주세요.

        규칙:
        1. 사용자 목표, 문서 주제, 핵심 사실, 미해결 질문만 남깁니다.
        2. 인사말, 반복, 잡담은 제거합니다.
        3. 한국어 평문으로 작성합니다.
        4. 최대 8문장 또는 600자 이내로 유지합니다.
        5. 다음 요약만 출력하고 다른 설명은 붙이지 않습니다.

        [기존 요약]
        {previous_summary}

        [새로 압축할 대화]
        {history_text}
        """

        response = llm.invoke(prompt)

        return {
            "summary": response.content.strip()
        }

    except Exception as e:
        raise ValueError(f"대화 요약 중 에러 발생: {str(e)}")