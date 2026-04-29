import logging
import time
from textwrap import dedent
from typing import Optional
from services.rag.ai_clients import get_chat_llm, get_summary_llm
from services.rag.retrieval_service import retrieve_documents

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
    [구조화된 대화 메모리]와 [최근 대화 기록]은 후속 질문의 지시어와 사용자 목표를 이해하는 데만 사용하세요.
    사실 판단과 답변 근거는 반드시 [참고 문서]에 한정하세요.

    규칙:
    1. [참고 문서]에 없는 내용은 추측하지 마세요.
    2. 답을 찾을 수 없으면 답변 형식을 사용하지 말고 정확히 "제공된 문서에서는 해당 내용을 찾을 수 없습니다." 라고만 답하세요.
    3. 답변에는 참고한 문서명과 페이지를 함께 언급하세요.
    4. 여러 참고 문서가 함께 필요하면 핵심 근거를 묶어서 설명하세요.
    5. [구조화된 대화 메모리]의 핵심 사실과 [참고 문서]가 충돌하면 [참고 문서]를 우선하세요.
    6. 답변은 아래 형식을 유지하세요.

    답변 형식:
    요약:
    - 질문에 대한 핵심 답변을 1~3문장으로 작성

    핵심 근거:
    - 근거 1
    - 근거 2

    참고 위치:
    - [참고 번호] 문서명, 페이지

    [구조화된 대화 메모리]
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
        document_title = metadata.get("document_title") or metadata.get("filename")
        references.append({
            "document_id": metadata.get("document_id"),
            "document_title": document_title,
            "section_title": metadata.get("section_title"),
            "page_number": metadata.get("page_number", 0),
            "chunk_index": metadata.get("chunk_index"),
            "page_chunk_index": metadata.get("page_chunk_index"),
            "content": doc.page_content
        })

    return references


def build_summary_prompt(previous_summary: str, history_text: str) -> str:
    return dedent(f"""
    당신은 NotebookLM 스타일의 대화 메모리 압축기입니다.
    기존 요약과 새 대화를 합쳐, 이후 후속 질문을 이해하는 데 필요한 정보만 구조화해 남기세요.

    핵심 원칙:
    1. 문서나 대화에서 확인된 사실만 남기고 추측은 제거합니다.
    2. 인사말, 반복, 감탄, 잡담은 제거합니다.
    3. 사용자의 목표와 후속 질문에서 지시어로 다시 참조될 수 있는 대상을 보존합니다.
    4. 새 대화가 기존 요약을 갱신하거나 정정하면 최신 내용을 우선합니다.
    5. 빈 항목은 "없음"으로 적습니다.
    6. 전체는 700자 이내로 유지합니다.
    7. 아래 형식만 출력하고 다른 설명은 붙이지 않습니다.

    출력 형식:
    사용자 목표:
    - ...

    문서 주제:
    - ...

    핵심 사실:
    - ...

    미해결 질문:
    - ...

    후속 질문 맥락:
    - ...

    [기존 요약]
    {previous_summary}

    [새로 압축할 대화]
    {history_text}
    """).strip()


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
        llm = get_chat_llm()

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
        llm = get_summary_llm()

        previous_summary = existing_summary if existing_summary else "이전 요약 없음"
        history_text = build_history_text(history)

        prompt = build_summary_prompt(
            previous_summary=previous_summary,
            history_text=history_text
        )

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
