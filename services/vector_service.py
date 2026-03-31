import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

print("="*50)
print(f"✅ [Vector 요리사] API 키: {'있음!' if GEMINI_API_KEY else '없음 (None) 🚨'}")
print(f"✅ [Vector 요리사] DB 주소: {'있음!' if DATABASE_URL else '없음 (None) 🚨'}")
print("="*50)

def process_and_store_document(text: str, filename: str):
    
    try:
        print("[1단계] 시작...")
        # Chunking : 500자씩 자르되, 50자씩 겹치게
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        print("[1단계] 종료...")

        print("[2단계] 시작...")
        # 구글 번역기 (Embedding 모델)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=GEMINI_API_KEY
        )
        print("[2단계] 종료...")

        print("[3단계] 시작...")
        # PostgreSQL(pgvector)에 저장
        metadatas = [{"filename": filename} for _ in chunks]
        
        
        # PGVector 객체 생성 
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name="notebook_documents", # DB 안에 생길 논리적인 컬렉션 이름
            connection=DATABASE_URL,
            use_jsonb=True
        )
        print("[3단계] 종료...")
        
        print("[4단계] 시작...")
        # chunking 글자와 메타데이터를 DB에 저장
        vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        print("[4단계] 종료...")

        return len(chunks)

    except Exception as e:
        raise ValueError(f"PostgreSQL 벡터 DB 저장 중 에러 발생: {str(e)}")