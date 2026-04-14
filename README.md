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

## 관련 문서

- 전체 문서는 `../infra-config/README.md` 와 `../infra-config/docs/architecture.md` 에서 확인할 수 있습니다.
