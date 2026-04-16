import os
import logging
import time
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
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
EMBEDDING_BATCH_DELAY_SECONDS = float(os.getenv("EMBEDDING_BATCH_DELAY_SECONDS", "1.5"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "2"))
EMBEDDING_RETRY_BASE_DELAY_SECONDS = float(
    os.getenv("EMBEDDING_RETRY_BASE_DELAY_SECONDS", "2.0")
)


def is_quota_error(error: Exception) -> bool:
    message = str(error)
    return "RESOURCE_EXHAUSTED" in message or "429" in message


def add_texts_with_retry(
    vectorstore: PGVector,
    texts: list[str],
    metadatas: list[dict],
    request_id: str | None,
    filename: str,
    batch_index: int,
    total_batches: int,
):
    attempt = 0

    while True:
        try:
            vectorstore.add_texts(texts=texts, metadatas=metadatas)
            return
        except Exception as error:
            if not is_quota_error(error) or attempt >= EMBEDDING_MAX_RETRIES:
                raise

            attempt += 1
            retry_delay = EMBEDDING_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))

            logger.warning(
                "event=vector_store_batch_retry requestId=%s filename=%s batchIndex=%s totalBatches=%s attempt=%s retryDelaySeconds=%s error=%s",
                request_id,
                filename,
                batch_index,
                total_batches,
                attempt,
                retry_delay,
                str(error)
            )
            time.sleep(retry_delay)

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
    stored_chunk_count = 0

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

        if not all_chunks:
            logger.info(
                "event=vector_store_skip_empty_document requestId=%s filename=%s",
                request_id,
                filename
            )
            return 0
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

        total_batches = (len(all_chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        # 외부 임베딩 API 처리량 제한을 피하려고 chunk를 배치 단위로 나눠 저장한다.
        for batch_index, batch_start in enumerate(range(0, len(all_chunks), EMBEDDING_BATCH_SIZE), start=1):
            batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, len(all_chunks))
            batch_chunks = all_chunks[batch_start:batch_end]
            batch_metadatas = all_metadatas[batch_start:batch_end]

            add_texts_with_retry(
                vectorstore=vectorstore,
                texts=batch_chunks,
                metadatas=batch_metadatas,
                request_id=request_id,
                filename=filename,
                batch_index=batch_index,
                total_batches=total_batches,
            )
            stored_chunk_count = batch_end

            logger.info(
                "event=vector_store_batch_success requestId=%s filename=%s batchIndex=%s totalBatches=%s batchSize=%s processedCount=%s",
                request_id,
                filename,
                batch_index,
                total_batches,
                len(batch_chunks),
                batch_end
            )

            if batch_end < len(all_chunks) and EMBEDDING_BATCH_DELAY_SECONDS > 0:
                time.sleep(EMBEDDING_BATCH_DELAY_SECONDS)

        logger.info(
            "event=vector_store_success requestId=%s filename=%s chunkCount=%s",
            request_id,
            filename,
            len(all_chunks)
        )

        return len(all_chunks)

    except Exception as e:
        error_message = str(e)

        if stored_chunk_count > 0:
            try:
                delete_document_vectors(
                    document_id=document_id,
                    filename=filename,
                    request_id=request_id
                )
                logger.info(
                    "event=vector_store_partial_cleanup_success requestId=%s filename=%s storedChunkCount=%s",
                    request_id,
                    filename,
                    stored_chunk_count
                )
            except Exception as cleanup_error:
                logger.warning(
                    "event=vector_store_partial_cleanup_failure requestId=%s filename=%s storedChunkCount=%s error=%s",
                    request_id,
                    filename,
                    stored_chunk_count,
                    str(cleanup_error)
                )

        if is_quota_error(e):
            logger.warning(
                "event=vector_store_quota_exceeded requestId=%s filename=%s error=%s",
                request_id,
                filename,
                error_message
            )
            raise ValueError(
                "Gemini Embedding API 처리량 또는 무료 티어 한도를 초과했습니다. 잠시 후 다시 시도하거나 더 작은 문서를 업로드해 주세요."
            )

        logger.exception(
            "event=vector_store_failure requestId=%s filename=%s error=%s",
            request_id,
            filename,
            error_message
        )
        raise ValueError(f"PostgreSQL 벡터 DB 저장 중 에러 발생: {error_message}")


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
