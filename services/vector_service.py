import os
import logging
import re
import time
from urllib.parse import urlparse
from dotenv import load_dotenv
import psycopg2

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from services.rag.ai_clients import COLLECTION_NAME, DATABASE_URL, get_vectorstore

load_dotenv()
logger = logging.getLogger(__name__)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
EMBEDDING_BATCH_DELAY_SECONDS = float(os.getenv("EMBEDDING_BATCH_DELAY_SECONDS", "1.5"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "2"))
EMBEDDING_RETRY_BASE_DELAY_SECONDS = float(
    os.getenv("EMBEDDING_RETRY_BASE_DELAY_SECONDS", "2.0")
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "20"))


def get_document_title(filename: str) -> str:
    title = os.path.splitext(filename)[0].strip()
    return title or filename


def looks_like_section_title(line: str) -> bool:
    text = line.strip()

    if len(text) < 2 or len(text) > 80:
        return False

    if text.startswith(("-", "*", "•")):
        return False

    if text.endswith((".", ",", ";", "!", "?")):
        return False

    if len(text) > 12 and text.endswith(("다", "요", "니다")):
        return False

    # 번호형 제목(1. 개요), 마크다운 제목, 짧은 독립 라인을 섹션 후보로 본다.
    return bool(
        text.startswith("#")
        or re.match(r"^\d+(\.\d+)*[.)]?\s+\S+", text)
        or (len(text) <= 30 and len(text.split()) <= 6)
    )


def infer_section_title(page_text: str, chunk: str) -> str | None:
    for source_text in (chunk, page_text):
        for line in source_text.splitlines():
            if looks_like_section_title(line):
                return line.strip().lstrip("#").strip()

    return None


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
        # Chunking : 문단/줄바꿈 경계를 우선 살리되, 설정된 길이를 넘으면 재귀적으로 분할한다.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        all_chunks = []
        all_metadatas = []
        document_title = get_document_title(filename)
        chunk_index = 0

        for page in pages_data:
            page_num = page["page_number"]
            page_text = page["text"]

            if not page_text.strip():
                continue

            # 해당 페이지만 분할
            chunks = [
                chunk.strip()
                for chunk in text_splitter.split_text(page_text)
                if len(chunk.strip()) >= MIN_CHUNK_CHARS
            ]

            # 나눠진 조각들을 모으고 각 조각마다 파일명, 페이지 번호 추가
            for page_chunk_index, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "filename": filename,
                    "document_title": document_title,
                    "page_number": page_num,
                    "document_id": document_id,
                    "notebook_id": notebook_id,
                    "chunk_index": chunk_index,
                    "page_chunk_index": page_chunk_index,
                    "section_title": infer_section_title(page_text, chunk)
                })
                chunk_index += 1

        if not all_chunks:
            logger.info(
                "event=vector_store_skip_empty_document requestId=%s filename=%s",
                request_id,
                filename
            )
            return 0
        # print("[1단계] 종료...")

        # 공통 PGVector 클라이언트를 재사용해 객체 생성 비용을 줄인다.
        vectorstore = get_vectorstore()

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
            "event=vector_store_success requestId=%s filename=%s chunkCount=%s chunkSize=%s chunkOverlap=%s minChunkChars=%s",
            request_id,
            filename,
            len(all_chunks),
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            MIN_CHUNK_CHARS
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
