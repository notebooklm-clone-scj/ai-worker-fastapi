# ai-worker-fastapi

PDF 파싱, 임베딩, 유사도 검색, LLM 응답 생성을 담당하는 FastAPI 서비스입니다.

## 역할

- PDF 텍스트 추출
- 문서 요약 생성
- 문서 청크 분할 및 pgvector 저장
- 노트북 단위 metadata filter 검색
- MMR, reranker, hybrid search 기반 reference chunk 검색
- 답변과 reference chunks 생성
- reference metadata 생성
- 요청별 requestId, 단계별 처리 로그 기록

## 처리 흐름

```txt
PDF 업로드
  -> 텍스트 추출
  -> 요약 생성
  -> 청크 분할
  -> 벡터 저장

채팅 질문
  -> 하이브리드 검색
  -> 로컬 reranker 재정렬
  -> 참고 청크 구성
  -> LLM 답변 생성
  -> 답변 + reference chunks 반환
```

## RAG 개선 포인트

| 개선 | 내용 |
| --- | --- |
| 검색 범위 제한 | `notebook_id` metadata filter를 적용해 현재 노트북 문서 안에서만 검색합니다. |
| 선택적 문서 필터 | 필요하면 `document_id`로 특정 문서 검색 범위를 더 좁힐 수 있습니다. |
| MMR 검색 | 후보를 넓게 가져온 뒤 관련성과 다양성을 함께 고려해 중복 chunk를 줄입니다. |
| Hybrid search | pgvector dense 검색에 keyword 검색을 더해 인물명, 숫자, 날짜, 고유명사 질문을 보강합니다. |
| Local reranker | 추가 AI 호출 없이 질문과 chunk의 토큰 겹침을 기준으로 후보를 재정렬합니다. |
| Prompt grounding | 참고 chunk마다 문서명, 페이지, 섹션 라벨을 붙여 답변 근거를 명확히 합니다. |
| Reference metadata | `document_id`, `document_title`, `section_title`, `page_number`, `chunk_index`, `page_chunk_index`를 응답에 포함합니다. |
| Runtime reuse | LLM, embedding, PGVector 객체를 프로세스 내에서 재사용합니다. |

## Reference response 예시

```json
{
  "answer": "문서 기반 답변",
  "reference_chunks": [
    {
      "document_id": 1,
      "document_title": "운수 좋은 날",
      "section_title": "운수좋은날",
      "page_number": 1,
      "chunk_index": 0,
      "page_chunk_index": 0,
      "content": "검색된 근거 chunk 본문"
    }
  ]
}
```

## 실행

```bash
cd /Users/seochanjin/workspace/notebooklm/ai-worker-fastapi
docker compose up --build
```

## 주요 설정

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `CHUNK_SIZE` | `500` | 벡터 저장 전 문서를 나눌 최대 chunk 크기 |
| `CHUNK_OVERLAP` | `50` | 인접 chunk 사이에 겹쳐둘 문자 수 |
| `MIN_CHUNK_CHARS` | `20` | 너무 짧아 검색 품질을 낮출 수 있는 chunk 제외 기준 |
| `RETRIEVAL_CONTEXT_K` | `5` | 최종 프롬프트에 넣을 reference chunk 수 |
| `RETRIEVAL_FETCH_K` | `10` | MMR 검색 시 pgvector에서 가져올 후보 수 |
| `RETRIEVAL_MMR_LAMBDA` | `0.5` | MMR의 relevance/diversity 균형값 |
| `HYBRID_SEARCH_ENABLED` | `true` | dense 검색 후보와 keyword 검색 후보를 함께 사용할지 여부 |
| `KEYWORD_FETCH_K` | `5` | keyword search로 추가로 가져올 후보 chunk 수 |
| `RERANK_ENABLED` | `true` | MMR 후보를 로컬 relevance 점수로 재정렬할지 여부 |
| `RERANK_POOL_K` | `8` | reranker가 재정렬할 MMR 후보 chunk 수 |
| `RERANK_MIN_TOKEN_LENGTH` | `2` | reranker 키워드로 사용할 최소 토큰 길이 |

## RAG 평가

- RAG 품질 비교용 수동 평가셋은 `evaluation/README.md`에서 확인할 수 있습니다.
- 평가셋은 retrieval recall, faithfulness, relevance, citation quality, latency를 기준으로 기록합니다.
- 포트폴리오 설명 시에는 같은 질문 세트를 기준으로 retrieval 결과와 답변 근거가 어떻게 달라졌는지 비교할 수 있습니다.

## 런타임 최적화

- Gemini Embedding, Gemini Chat LLM, Summary LLM, PGVector 클라이언트는 `services/rag/ai_clients.py`에서 생성하고 프로세스 내에서 재사용합니다.
- 채팅/검색/벡터 저장 서비스는 공통 클라이언트 factory를 통해 객체 생성 중복을 줄입니다.

## Conversation Memory

- 오래된 대화는 summary memory로 압축해 후속 질문에 함께 전달합니다.
- summary memory는 `사용자 목표`, `문서 주제`, `핵심 사실`, `미해결 질문`, `후속 질문 맥락` 슬롯으로 구조화합니다.
- 잡담과 반복은 제거하고, 후속 질문에서 다시 참조될 수 있는 대상과 문서 기반 사실만 보존합니다.

## 관련 문서

- 전체 문서는 `../infra-config/README.md` 와 `../infra-config/docs/architecture.md` 에서 확인할 수 있습니다.
