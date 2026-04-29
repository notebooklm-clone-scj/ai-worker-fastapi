# RAG Evaluation

RAG 개선 전후를 감이 아니라 같은 기준으로 비교하기 위한 수동 평가셋입니다.
처음에는 10개 내외의 질문으로 작게 시작하고, 자주 묻는 질문이나 실패 사례가 생길 때마다 케이스를 추가합니다.

## 평가 대상

- Retrieval Recall: 정답 근거 chunk가 검색 결과에 포함되는가
- Answer Faithfulness: 답변이 검색된 참고 문서에 근거하는가
- Answer Relevance: 질문에 직접 답하는가
- Citation Quality: 문서명, 페이지, 참고 위치가 확인 가능한가
- Latency: 응답 시간이 사용자가 기다릴 만한 수준인가

## 평가 흐름

1. 평가용 문서를 새 노트북에 업로드합니다.
2. `rag_eval_cases.sample.json`의 `document_title`, `question`, `expected_sources`를 실제 문서에 맞게 채웁니다.
3. 각 질문을 같은 노트북에서 실행합니다.
4. 응답의 `reference_chunks`와 최종 답변을 기준으로 점수를 기록합니다.
5. PR 전후 또는 브랜치 전후 평균 점수를 비교합니다.

## 15분 수동 테스트 방법

처음에는 자동화하지 말고 아래 순서대로 3개 질문만 실행합니다.

1. 평가할 PDF 하나를 고릅니다.
   - 예: 포트폴리오 문서, 사업계획서, 강의자료처럼 정답 위치를 사람이 확인할 수 있는 문서
2. 새 노트북을 만들고 해당 PDF를 업로드합니다.
   - 최근 chunk metadata 개선 효과를 보려면 기존 문서가 아니라 새로 업로드한 문서를 사용합니다.
3. `rag_eval_cases.sample.json`에서 질문 3개만 고릅니다.
   - 추천 시작 조합: `rag-001` 요약, `rag-005` 페이지 특정, `rag-008` 답 없음
4. 선택한 질문의 `document_title`, `question`, `expected_sources`를 실제 PDF에 맞게 바꿔 생각합니다.
   - 파일을 꼭 수정하지 않아도 됩니다. 처음에는 머릿속 기준이나 메모로 시작해도 됩니다.
5. 웹 화면에서 같은 노트북에 질문을 하나씩 입력합니다.
6. 답변과 참고 위치를 보고 `manual_scorecard.md`에 점수를 적습니다.
7. 평균을 계산해 현재 브랜치의 기준 점수로 둡니다.

## 실제 기록 예시

예를 들어 3페이지에 "프로젝트 리스크" 섹션이 있는 PDF를 평가한다면:

```txt
질문: 3페이지에서 설명하는 핵심 내용은 뭐야?
기대 근거: 문서 제목 / 3페이지 / 프로젝트 리스크
```

답변이 3페이지 리스크 내용을 말하고 참고 위치도 3페이지를 보여주면:

```txt
Retrieval Recall: 2
Faithfulness: 2
Relevance: 2
Citation: 2
Notes: 기대한 3페이지 근거가 reference에 포함됨
```

답변 내용은 맞지만 참고 위치가 없거나 다른 페이지를 보여주면:

```txt
Retrieval Recall: 1
Faithfulness: 2
Relevance: 2
Citation: 0
Notes: 답변은 맞지만 근거 페이지 표시가 불안정함
```

문서에 없는 내용을 그럴듯하게 만들어내면:

```txt
Retrieval Recall: 0
Faithfulness: 0
Relevance: 0
Citation: 0
Notes: 문서 밖 내용을 추측함
```

## 최소 통과 기준

처음에는 아래 기준을 넘기는 것을 목표로 합니다.

| 항목 | 목표 |
| --- | --- |
| Retrieval Recall | 평균 1.5 이상 |
| Answer Faithfulness | 평균 1.5 이상 |
| Answer Relevance | 평균 1.5 이상 |
| Citation Quality | 평균 1.0 이상 |

이 기준을 넘지 못한 케이스는 다음 RAG 개선 후보로 기록합니다.

## 점수 기준

| 항목 | 0점 | 1점 | 2점 |
| --- | --- | --- | --- |
| Retrieval Recall | 근거 없음 | 일부 근거만 있음 | 핵심 근거가 있음 |
| Answer Faithfulness | 문서 밖 추측 | 일부 추측 포함 | 참고 문서에 충실 |
| Answer Relevance | 질문과 어긋남 | 일부만 답함 | 질문에 직접 답함 |
| Citation Quality | 출처 없음 | 페이지 또는 문서명 일부만 있음 | 문서명과 페이지 확인 가능 |

## 기록 예시

```txt
case_id: rag-001
branch: feature/rag-prompt-grounding
retrieval_recall: 2
answer_faithfulness: 2
answer_relevance: 2
citation_quality: 1
latency_ms: 3210
notes: 답변은 정확하지만 참고 위치에 섹션명이 빠짐
```

## 운영 팁

- 평가셋은 쉬운 질문, 키워드 질문, 페이지 특정 질문, 요약 질문, 답이 없는 질문을 섞습니다.
- 한 문서만 쓰는 질문과 여러 문서를 함께 봐야 하는 질문을 분리합니다.
- 답이 없는 질문은 환각 여부를 확인하기 좋아서 반드시 포함합니다.
- 기존 vector metadata 변경은 과거 문서에 소급되지 않으므로, metadata 관련 평가 전에는 문서를 다시 업로드합니다.
