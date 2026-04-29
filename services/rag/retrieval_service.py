import logging
import os
import re
import time
from contextlib import closing
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv
from services.rag.ai_clients import COLLECTION_NAME, DATABASE_URL, get_vectorstore

load_dotenv()
logger = logging.getLogger(__name__)
RETRIEVAL_CONTEXT_K = int(os.getenv("RETRIEVAL_CONTEXT_K", "5"))
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "10"))
RETRIEVAL_MMR_LAMBDA = float(os.getenv("RETRIEVAL_MMR_LAMBDA", "0.5"))
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
KEYWORD_FETCH_K = int(os.getenv("KEYWORD_FETCH_K", "5"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_POOL_K = int(os.getenv("RERANK_POOL_K", "8"))
RERANK_MIN_TOKEN_LENGTH = int(os.getenv("RERANK_MIN_TOKEN_LENGTH", "2"))
RERANK_STOPWORDS = {
    "그리고", "그런데", "그러면", "이것", "저것", "해줘", "알려줘", "설명해줘",
    "무엇", "뭐야", "어떤", "어느", "문서", "내용", "핵심", "정리", "요약"
}


@dataclass
class RetrievedDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    docs: list
    latency_ms: int
    dense_count: int
    keyword_count: int
    effective_fetch_k: int
    rerank_pool_k: int
    context_k: int
    keyword_fetch_k: int
    mmr_lambda: float
    hybrid_enabled: bool
    rerank_enabled: bool


def normalize_connection_string(raw_url: str) -> str:
    """psycopg2가 이해할 수 있도록 SQLAlchemy 스타일 드라이버명을 제거한다."""
    return raw_url.replace("postgresql+psycopg2://", "postgresql://").replace(
        "postgresql+psycopg://",
        "postgresql://"
    )


def create_keyword_search_connection():
    """LangChain PGVector가 저장한 embedding 테이블을 keyword 검색하기 위한 DB 연결을 만든다."""
    parsed = urlparse(normalize_connection_string(DATABASE_URL))

    return psycopg2.connect(
        dbname=parsed.path.lstrip("/"),
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port or 5432
    )


def tokenize_for_rerank(text: str) -> list[str]:
    """질문과 문서 내용을 비교하기 쉽도록 의미 있는 검색 토큰만 추출한다."""
    tokens = re.findall(r"[A-Za-z0-9가-힣]+", text.lower())
    return [
        token
        for token in tokens
        if len(token) >= RERANK_MIN_TOKEN_LENGTH and token not in RERANK_STOPWORDS
    ]


def build_keyword_terms(question: str) -> list[str]:
    """고유명사, 숫자, 날짜처럼 keyword search에 유용한 질문 토큰을 준비한다."""
    seen_terms = set()
    terms = []

    for token in tokenize_for_rerank(question):
        if token in seen_terms:
            continue

        seen_terms.add(token)
        terms.append(token)

    return terms


def search_keyword_documents(
    question: str,
    notebook_id: int,
    document_id: Optional[int],
    limit: int
) -> list[RetrievedDocument]:
    """질문 키워드가 직접 포함된 chunk를 pgvector 저장 테이블에서 조회한다."""
    if not HYBRID_SEARCH_ENABLED or limit <= 0:
        return []

    terms = build_keyword_terms(question)
    if not terms:
        return []

    metadata_filters = ["cmetadata ->> 'notebook_id' = %s"]
    params = [str(notebook_id)]

    if document_id is not None:
        metadata_filters.append("cmetadata ->> 'document_id' = %s")
        params.append(str(document_id))

    keyword_conditions = []
    for term in terms:
        keyword_conditions.append("""
            (
                document ILIKE %s
                OR cmetadata ->> 'document_title' ILIKE %s
                OR cmetadata ->> 'filename' ILIKE %s
                OR cmetadata ->> 'section_title' ILIKE %s
            )
        """)
        like_term = f"%{term}%"
        params.extend([like_term, like_term, like_term, like_term])

    params.extend([COLLECTION_NAME, limit])

    query = f"""
        SELECT document, cmetadata
        FROM langchain_pg_embedding
        WHERE {" AND ".join(metadata_filters)}
        AND ({" OR ".join(keyword_conditions)})
        AND collection_id = (
            SELECT uuid
            FROM langchain_pg_collection
            WHERE name = %s
        )
        LIMIT %s
    """

    try:
        with closing(create_keyword_search_connection()) as connection:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
    except Exception as error:
        logger.warning("event=keyword_search_failure error=%s", str(error))
        return []

    return [
        RetrievedDocument(page_content=row[0], metadata=row[1] or {})
        for row in rows
    ]


def get_document_identity(doc) -> tuple:
    """dense 검색과 keyword 검색에 동시에 걸린 같은 chunk를 제거하기 위한 식별자를 만든다."""
    metadata = doc.metadata or {}

    return (
        metadata.get("document_id"),
        metadata.get("chunk_index"),
        metadata.get("page_number"),
        doc.page_content[:120]
    )


def merge_retrieved_documents(*doc_groups: list) -> list:
    """여러 retrieval 전략에서 나온 후보를 순서 보존 방식으로 합치고 중복을 제거한다."""
    merged_docs = []
    seen_identities = set()

    for docs in doc_groups:
        for doc in docs:
            identity = get_document_identity(doc)
            if identity in seen_identities:
                continue

            seen_identities.add(identity)
            merged_docs.append(doc)

    return merged_docs


def build_rerank_text(doc) -> str:
    """chunk 본문뿐 아니라 문서명/섹션/페이지 metadata도 rerank 비교 대상에 포함한다."""
    metadata = doc.metadata or {}
    metadata_text = " ".join(
        str(value)
        for value in [
            metadata.get("document_title"),
            metadata.get("filename"),
            metadata.get("section_title"),
            metadata.get("page_number"),
        ]
        if value is not None
    )

    return f"{metadata_text}\n{doc.page_content}"


def calculate_rerank_score(question_terms: set[str], doc, original_rank: int) -> float:
    """질문 토큰과 chunk의 겹침 정도, 직접 매칭, 기존 MMR 순위를 섞어 relevance 점수를 계산한다."""
    if not question_terms:
        return 1 / (original_rank + 1)

    rerank_text = build_rerank_text(doc)
    doc_terms = set(tokenize_for_rerank(rerank_text))
    overlap_count = len(question_terms & doc_terms)
    overlap_score = overlap_count / len(question_terms)

    exact_match_count = sum(1 for term in question_terms if term in rerank_text.lower())
    exact_match_score = exact_match_count / len(question_terms)

    # MMR이 이미 relevance/diversity를 반영했으므로 기존 순위도 약하게 보존한다.
    rank_score = 1 / (original_rank + 1)

    return (overlap_score * 0.6) + (exact_match_score * 0.3) + (rank_score * 0.1)


def rerank_documents(question: str, docs: list, limit: int) -> list:
    """MMR 후보 chunk를 로컬 relevance 점수로 재정렬하고 최종 context 개수만 남긴다."""
    if not RERANK_ENABLED or len(docs) <= limit:
        return docs[:limit]

    question_terms = set(tokenize_for_rerank(question))
    scored_docs = [
        (calculate_rerank_score(question_terms, doc, index), index, doc)
        for index, doc in enumerate(docs)
    ]

    scored_docs.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored_docs[:limit]]


def retrieve_documents(
    question: str,
    notebook_id: int,
    document_id: Optional[int] = None
) -> RetrievalResult:
    """dense MMR 검색과 keyword 검색 후보를 합친 뒤 rerank해서 최종 context 문서를 반환한다."""
    retrieval_start = time.perf_counter()
    vectorstore = get_vectorstore()

    metadata_filter = {"notebook_id": notebook_id}
    if document_id is not None:
        metadata_filter["document_id"] = document_id

    rerank_pool_k = max(RETRIEVAL_CONTEXT_K, RERANK_POOL_K)
    effective_fetch_k = max(RETRIEVAL_FETCH_K, rerank_pool_k)
    dense_docs = vectorstore.max_marginal_relevance_search(
        query=question,
        k=rerank_pool_k,
        fetch_k=effective_fetch_k,
        lambda_mult=RETRIEVAL_MMR_LAMBDA,
        filter=metadata_filter
    )
    keyword_docs = search_keyword_documents(
        question=question,
        notebook_id=notebook_id,
        document_id=document_id,
        limit=KEYWORD_FETCH_K
    )

    docs = merge_retrieved_documents(dense_docs, keyword_docs)
    docs = rerank_documents(question, docs, RETRIEVAL_CONTEXT_K)
    retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)

    return RetrievalResult(
        docs=docs,
        latency_ms=retrieval_latency_ms,
        dense_count=len(dense_docs),
        keyword_count=len(keyword_docs),
        effective_fetch_k=effective_fetch_k,
        rerank_pool_k=rerank_pool_k,
        context_k=RETRIEVAL_CONTEXT_K,
        keyword_fetch_k=KEYWORD_FETCH_K,
        mmr_lambda=RETRIEVAL_MMR_LAMBDA,
        hybrid_enabled=HYBRID_SEARCH_ENABLED,
        rerank_enabled=RERANK_ENABLED
    )
