# Manual RAG Scorecard

| Date | Branch | Case ID | Retrieval Recall | Faithfulness | Relevance | Citation | Latency(ms) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD | main | rag-001 |  |  |  |  |  |  |
| YYYY-MM-DD | feature branch | rag-001 |  |  |  |  |  |  |

## Score Guide

- `0`: 실패 또는 근거 없음
- `1`: 부분 성공
- `2`: 기대 수준 충족

## How To Fill

- `Retrieval Recall`: 답변 아래 reference 또는 응답 근거에 기대한 문서/페이지가 있으면 2점입니다.
- `Faithfulness`: 답변 내용이 reference chunk 안에서 확인되면 2점입니다.
- `Relevance`: 질문에 직접 답하면 2점입니다.
- `Citation`: 문서명과 페이지를 사용자가 확인할 수 있으면 2점입니다.
- `Latency(ms)`: 브라우저 체감 시간 또는 서버 로그의 latency를 적습니다. 처음에는 비워도 됩니다.

## Comparison Summary

| Branch | Avg Retrieval | Avg Faithfulness | Avg Relevance | Avg Citation | Avg Latency(ms) |
| --- | --- | --- | --- | --- | --- |
| main |  |  |  |  |  |
| feature branch |  |  |  |  |  |
