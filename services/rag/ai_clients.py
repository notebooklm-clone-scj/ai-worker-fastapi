import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = "notebook_documents"


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """질문/문서 embedding 클라이언트를 프로세스 내에서 재사용한다."""
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=GEMINI_API_KEY
    )


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatGoogleGenerativeAI:
    """채팅 답변 생성용 Gemini LLM 클라이언트를 재사용한다."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.1
    )


@lru_cache(maxsize=1)
def get_summary_llm() -> ChatGoogleGenerativeAI:
    """대화 summary memory 생성용 Gemini LLM 클라이언트를 재사용한다."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=GEMINI_API_KEY,
        temperature=0.1
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> PGVector:
    """공통 PGVector 클라이언트를 재사용한다."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True
    )
