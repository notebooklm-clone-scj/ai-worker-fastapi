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
  -> 유사도 검색
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

## RAG 평가

- RAG 품질 비교용 수동 평가셋은 `evaluation/README.md`에서 확인할 수 있습니다.
- 평가셋은 retrieval recall, faithfulness, relevance, citation quality, latency를 기준으로 기록합니다.

## 관련 문서

- 전체 문서는 `../infra-config/README.md` 와 `../infra-config/docs/architecture.md` 에서 확인할 수 있습니다.
