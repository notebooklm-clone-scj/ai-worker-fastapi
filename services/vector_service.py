import os
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv
import psycopg2

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
logger = logging.getLogger(__name__)
COLLECTION_NAME = "notebook_documents"

# TODO: 추후 상용 배포 시 print 대신 python logging 모듈로 교체 예정
# print("="*50)
# print(f" [Vector 요리사] API 키: {'있음!' if GEMINI_API_KEY else '없음 (None) '}")
# print(f" [Vector 요리사] DB 주소: {'있음!' if DATABASE_URL else '없음 (None) '}")
# print("="*50)

# PDF 페이지를 chunk로 나누고 pgvector에 저장하는 단계다.
def process_and_store_document(
    pages_data: list,
    filename: str,
    document_id: int,
    notebook_id: int,
    request_id: str | None = None
):
    
    try:
        # print("[1단계] 시작...")
        # Chunking : 500자씩 자르되, 50자씩 겹치게
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        all_chunks = []
        all_metadatas = []

        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["text"]

            if not page_text.strip():
                continue

            # 해당 페이지만 분할
            chunks = text_splitter.split_text(page_text)

            # 나눠진 조각들을 모으고 각 조각마다 파일명, 페이지 번호 추가
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({
                    "filename": filename,
                    "page_number": page_num,
                    "document_id": document_id,
                    "notebook_id": notebook_id
                })
        # print("[1단계] 종료...")

        # print("[2단계] 시작...")
        # 구글 번역기 (Embedding 모델)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=GEMINI_API_KEY
        )
        # print("[2단계] 종료...")

        # print("[3단계] 시작...")
        # PostgreSQL(pgvector)에 저장
        
        # PGVector 객체 생성 
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME, # DB 안에 생길 논리적인 컬렉션 이름
            connection=DATABASE_URL,
            use_jsonb=True
        )
        # print("[3단계] 종료...")
        
        # print("[4단계] 시작...")
        # chunking 글자와 메타데이터를 DB에 저장
        vectorstore.add_texts(texts=all_chunks, metadatas=all_metadatas)
        # print("[4단계] 종료...")

        logger.info(
            "event=vector_store_success requestId=%s filename=%s chunkCount=%s",
            request_id,
            filename,
            len(all_chunks)
        )

        return len(all_chunks)

    except Exception as e:
        logger.exception(
            "event=vector_store_failure requestId=%s filename=%s error=%s",
            request_id,
            filename,
            str(e)
        )
        raise ValueError(f"PostgreSQL 벡터 DB 저장 중 에러 발생: {str(e)}")


def normalize_connection_string(raw_url: str) -> str:
    return raw_url.replace("postgresql+psycopg://", "postgresql://")


def delete_document_vectors(document_id: int, filename: str, request_id: str | None = None):
    try:
        parsed = urlparse(normalize_connection_string(DATABASE_URL))

        connection = psycopg2.connect(
            dbname=parsed.path.lstrip("/"),
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port or 5432
        )

        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        DELETE FROM langchain_pg_embedding
                        WHERE collection_id = (
                            SELECT uuid
                            FROM langchain_pg_collection
                            WHERE name = %s
                        )
                        AND (
                            cmetadata ->> 'document_id' = %s
                            OR cmetadata ->> 'filename' = %s
                        )
                        """,
                        (COLLECTION_NAME, str(document_id), filename)
                    )

                    deleted_count = cursor.rowcount

            logger.info(
                "event=vector_delete_success requestId=%s documentId=%s filename=%s deletedCount=%s",
                request_id,
                document_id,
                filename,
                deleted_count
            )

            return deleted_count
        finally:
            connection.close()
    except Exception as e:
        logger.exception(
            "event=vector_delete_failure requestId=%s documentId=%s filename=%s error=%s",
            request_id,
            document_id,
            filename,
            str(e)
        )
        raise ValueError(f"문서 벡터 삭제 중 에러 발생: {str(e)}")
