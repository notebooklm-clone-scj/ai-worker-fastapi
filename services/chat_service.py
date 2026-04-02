import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

def ask_question_to_pdf(question: str):
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

        # 답변 생성 준비: 글을 읽고 대답할 챗봇 AI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=GEMINI_API_KEY,
            temperature=0.1 # 0에 가까울수록 창의성 없이 문서에 있는 내용만 말함 (환각 방지)
        )

        # 프롬프트
        prompt = f"""
        당신은 제공된 [참고 문서]만을 바탕으로 질문에 대답하는 친절한 AI 어시스턴트입니다.
        만약 [참고 문서]에 질문에 대한 정답이 없다면, 절대 지어내지 말고 "제공된 문서에서는 해당 내용을 찾을 수 없습니다." 라고 대답하세요.

        [참고 문서]
        {context}

        [유저 질문]
        {question}
        """

        # AI에게 질문을 던지고 답변 수령
        response = llm.invoke(prompt)

        # 유저에게 AI의 대답과 함께 어떤 자료를 참고했는지 함께 제출
        return {
            "answer": response.content,
            "reference_chunks": [doc.page_content for doc in docs]
        }
    
    except Exception as e:
        raise ValueError(f"채팅 생성 중 에러 발생: {str(e)}")