# ai-worker-fastapi

PDF 파싱, 임베딩, 유사도 검색, LLM 응답 생성을 담당하는 FastAPI 서비스입니다.

## 역할

- PDF 텍스트 추출
- 문서 요약 생성
- 문서 청크 분할 및 pgvector 저장
- 사용자 질문 기반 유사도 검색
- 답변과 reference chunks 생성
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
  -> 참고 청크 구성
  -> LLM 답변 생성
  -> 답변 + reference chunks 반환
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

## 런타임 최적화

- Gemini Embedding, Gemini Chat LLM, Summary LLM, PGVector 클라이언트는 `services/rag/ai_clients.py`에서 생성하고 프로세스 내에서 재사용합니다.
- 채팅/검색/벡터 저장 서비스는 공통 클라이언트 factory를 통해 객체 생성 중복을 줄입니다.

## Conversation Memory

- 오래된 대화는 summary memory로 압축해 후속 질문에 함께 전달합니다.
- summary memory는 `사용자 목표`, `문서 주제`, `핵심 사실`, `미해결 질문`, `후속 질문 맥락` 슬롯으로 구조화합니다.
- 잡담과 반복은 제거하고, 후속 질문에서 다시 참조될 수 있는 대상과 문서 기반 사실만 보존합니다.

## 관련 문서

- 전체 문서는 `../infra-config/README.md` 와 `../infra-config/docs/architecture.md` 에서 확인할 수 있습니다.
