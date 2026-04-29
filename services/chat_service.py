import os
import logging
import re
import time
from textwrap import dedent
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
logger = logging.getLogger(__name__)
RETRIEVAL_CONTEXT_K = int(os.getenv("RETRIEVAL_CONTEXT_K", "5"))
RETRIEVAL_FETCH_K = int(os.getenv("RETRIEVAL_FETCH_K", "10"))
RETRIEVAL_MMR_LAMBDA = float(os.getenv("RETRIEVAL_MMR_LAMBDA", "0.5"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_POOL_K = int(os.getenv("RERANK_POOL_K", "8"))
RERANK_MIN_TOKEN_LENGTH = int(os.getenv("RERANK_MIN_TOKEN_LENGTH", "2"))
RERANK_STOPWORDS = {
    "그리고", "그런데", "그러면", "이것", "저것", "해줘", "알려줘", "설명해줘",
    "무엇", "뭐야", "어떤", "어느", "문서", "내용", "핵심", "정리", "요약"
}

def build_history_text(history: list) -> str:
    if not history:
        return "최근 대화 기록이 없습니다."
    
    lines = []
    for h in history:
        role_name = "유저" if h["role"] == "USER" else "AI"
        lines.append(f"{role_name}: {h['message']}")
    
    return "\n".join(lines)


def tokenize_for_rerank(text: str) -> list[str]:
    """질문과 문서 내용을 비교하기 쉽도록 의미 있는 검색 토큰만 추출한다."""
    tokens = re.findall(r"[A-Za-z0-9가-힣]+", text.lower())
    return [
        token
        for token in tokens
        if len(token) >= RERANK_MIN_TOKEN_LENGTH and token not in RERANK_STOPWORDS
    ]


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


def build_reference_label(doc, index: int) -> str:
    metadata = doc.metadata or {}
    document_title = metadata.get("document_title") or metadata.get("filename") or "문서명 없음"
    page_number = metadata.get("page_number", "페이지 정보 없음")
    section_title = metadata.get("section_title")

    label = f"[참고 {index}] 문서: {document_title} / 페이지: {page_number}"
    if section_title:
        label += f" / 섹션: {section_title}"

    return label


def build_context_text(docs: list) -> str:
    if not docs:
        return "검색된 참고 문서가 없습니다."

    chunks = []
    for index, doc in enumerate(docs, start=1):
        chunks.append(f"{build_reference_label(doc, index)}\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)


def build_chat_prompt(summary_text: str, history_text: str, context: str, question: str) -> str:
    return dedent(f"""
    당신은 제공된 [참고 문서]만을 근거로 답변하는 RAG 어시스턴트입니다.
    [이전 대화 요약]과 [최근 대화 기록]은 후속 질문을 이해하는 데만 사용하고, 사실 판단은 반드시 [참고 문서]에 근거하세요.

    규칙:
    1. [참고 문서]에 없는 내용은 추측하지 마세요.
    2. 답을 찾을 수 없으면 답변 형식을 사용하지 말고 정확히 "제공된 문서에서는 해당 내용을 찾을 수 없습니다." 라고만 답하세요.
    3. 답변에는 참고한 문서명과 페이지를 함께 언급하세요.
    4. 여러 참고 문서가 함께 필요하면 핵심 근거를 묶어서 설명하세요.
    5. 답변은 아래 형식을 유지하세요.

    답변 형식:
    요약:
    - 질문에 대한 핵심 답변을 1~3문장으로 작성

    핵심 근거:
    - 근거 1
    - 근거 2

    참고 위치:
    - [참고 번호] 문서명, 페이지

    [이전 대화 요약]
    {summary_text}

    [대화 기록]
    {history_text}

    [참고 문서]
    {context}

    [유저 질문]
    {question}
    """).strip()


def build_reference_chunks(docs: list) -> list[dict]:
    references = []
    for doc in docs:
        metadata = doc.metadata or {}
        references.append({
            "page_number": metadata.get("page_number", 0),
            "content": doc.page_content
        })

    return references


# 문서 검색과 LLM 답변 생성을 묶는 RAG 채팅 서비스다.
def ask_question_to_pdf(
    notebook_id: int,
    question: str,
    history: list,
    document_id: Optional[int] = None,
    conversation_summary: Optional[str] = None,
    request_id: Optional[str] = None
):
    try:
        # 검색 준비: 유저 질문을 임베딩하기 위해 준비
        retrieval_start = time.perf_counter()
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

        # 유사도 검색 범위를 현재 노트북으로 제한해 다른 노트북 문서가 섞이지 않게 한다.
        metadata_filter = {"notebook_id": notebook_id}
        if document_id is not None:
            metadata_filter["document_id"] = document_id

        rerank_pool_k = max(RETRIEVAL_CONTEXT_K, RERANK_POOL_K)
        effective_fetch_k = max(RETRIEVAL_FETCH_K, rerank_pool_k)
        docs = vectorstore.max_marginal_relevance_search(
            query=question,
            k=rerank_pool_k,
            fetch_k=effective_fetch_k,
            lambda_mult=RETRIEVAL_MMR_LAMBDA,
            filter=metadata_filter
        )
        docs = rerank_documents(question, docs, RETRIEVAL_CONTEXT_K)
        retrieval_latency_ms = int((time.perf_counter() - retrieval_start) * 1000)

        # 찾아온 조각에 문서명/페이지/섹션 정보를 붙여 LLM이 출처를 구분해서 쓰도록 한다.
        context = build_context_text(docs)
        logger.info(
            "event=chat_retrieval_success requestId=%s notebookId=%s documentId=%s searchType=mmr rerankEnabled=%s fetchK=%s rerankPoolK=%s contextK=%s lambdaMult=%s latencyMs=%s retrievedCount=%s",
            request_id,
            notebook_id,
            document_id,
            RERANK_ENABLED,
            effective_fetch_k,
            rerank_pool_k,
            RETRIEVAL_CONTEXT_K,
            RETRIEVAL_MMR_LAMBDA,
            retrieval_latency_ms,
            len(docs)
        )

        history_text = build_history_text(history)
        summary_text = conversation_summary if conversation_summary else "이전 대화 요약이 없습니다."

        # 답변 생성 준비: 글을 읽고 대답할 챗봇 AI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            api_key=GEMINI_API_KEY,
            temperature=0.1 # 0에 가까울수록 창의성 없이 문서에 있는 내용만 말함 (환각 방지)
        )

        prompt = build_chat_prompt(
            summary_text=summary_text,
            history_text=history_text,
            context=context,
            question=question
        )

        # AI에게 질문을 던지고 답변 수령
        llm_start = time.perf_counter()
        response = llm.invoke(prompt)
        llm_latency_ms = int((time.perf_counter() - llm_start) * 1000)
        logger.info(
            "event=chat_llm_success requestId=%s latencyMs=%s answerLength=%s",
            request_id,
            llm_latency_ms,
            len(response.content)
        )

        # 유저에게 AI의 대답과 함께 어떤 자료를 참고했는지 함께 제출
        return {
            "answer": response.content,
            "reference_chunks": build_reference_chunks(docs)
        }
    
    except Exception as e:
        logger.exception("event=chat_service_failure requestId=%s error=%s", request_id, str(e))
        raise ValueError(f"채팅 생성 중 에러 발생: {str(e)}")
    

# 기존 요약과 새 대화를 합쳐 장기 대화용 summary memory를 만든다.
def summarize_conversation(existing_summary: Optional[str], history: list, request_id: Optional[str] = None):
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

        llm_start = time.perf_counter()
        response = llm.invoke(prompt)
        llm_latency_ms = int((time.perf_counter() - llm_start) * 1000)
        logger.info(
            "event=chat_summary_llm_success requestId=%s latencyMs=%s summaryLength=%s",
            request_id,
            llm_latency_ms,
            len(response.content.strip())
        )

        return {
            "summary": response.content.strip()
        }

    except Exception as e:
        logger.exception("event=chat_summary_service_failure requestId=%s error=%s", request_id, str(e))
        raise ValueError(f"대화 요약 중 에러 발생: {str(e)}")
