import os
import logging
import time
from textwrap import dedent
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from services.retrieval_service import retrieve_documents

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
logger = logging.getLogger(__name__)


def build_history_text(history: list) -> str:
    if not history:
        return "최근 대화 기록이 없습니다."
    
    lines = []
    for h in history:
        role_name = "유저" if h["role"] == "USER" else "AI"
        lines.append(f"{role_name}: {h['message']}")
    
    return "\n".join(lines)


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
        retrieval_result = retrieve_documents(
            question=question,
            notebook_id=notebook_id,
            document_id=document_id
        )
        docs = retrieval_result.docs

        # 찾아온 조각에 문서명/페이지/섹션 정보를 붙여 LLM이 출처를 구분해서 쓰도록 한다.
        context = build_context_text(docs)
        logger.info(
            "event=chat_retrieval_success requestId=%s notebookId=%s documentId=%s searchType=hybrid hybridEnabled=%s rerankEnabled=%s fetchK=%s rerankPoolK=%s keywordFetchK=%s contextK=%s lambdaMult=%s denseCount=%s keywordCount=%s latencyMs=%s retrievedCount=%s",
            request_id,
            notebook_id,
            document_id,
            retrieval_result.hybrid_enabled,
            retrieval_result.rerank_enabled,
            retrieval_result.effective_fetch_k,
            retrieval_result.rerank_pool_k,
            retrieval_result.keyword_fetch_k,
            retrieval_result.context_k,
            retrieval_result.mmr_lambda,
            retrieval_result.dense_count,
            retrieval_result.keyword_count,
            retrieval_result.latency_ms,
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
