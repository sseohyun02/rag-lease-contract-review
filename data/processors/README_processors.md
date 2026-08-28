# data/processors/ — QA 데이터 처리 파이프라인 정리

이 폴더 안 QA 관련 파일이 여러 개라 헷갈리기 쉬워서, 역할과 실행 순서를 정리한다.

## 파일별 역할

| 파일 | 역할 | 언제 실행하나 |
|---|---|---|
| `qa_1_process_raw.py` | 원본 크롤링(`posts_index_*.json`) → 규칙 기반 필터링(`qa_both.json` 등) → LLM 정제(`qa_clauses_processed.json`) | **원본 데이터부터 처음부터 다시 만들 때만.** 지금 QA 데이터가 추가되면 이 파이프라인을 다시 돌려야 함 |
| `qa_2_fix_ids.py` | 이미 만들어진 `qa_processed.json`(2026-XX 이전, 버그 있던 시절 산출물)에 `url` 부여 + `precedent_numbers` 재계산 | **일회성 마이그레이션, 이미 실행 완료.** `qa_1_process_raw.py`가 고쳐졌으므로 이제 새로 만드는 데이터에는 이 스크립트가 필요 없음. 과거 기록용으로만 보관 |
| `qa_3_build_gt.py` | 답변 원문에서 번호 없이 설명된 법조문을 찾아 실제 DB와 대조 후 gt_laws 보강 | eval_set을 새로 만들 때마다, `qa_4_make_eval_set.py`보다 먼저 |
| `qa_4_make_eval_set.py` | `qa_processed.json`(+ `qa_3_build_gt.py` 결과) → `eval_set_tune.json` / `eval_set_test.json` | eval_set을 새로 만들 때마다, 마지막 단계 |

## 실행 순서 (eval_set을 처음부터 다시 만들 때)

```
1. (원본 QA 데이터가 바뀌었을 때만) qa_1_process_raw.py의 run() + process_rule_based_filtered_clauses()
2. qa_3_build_gt.py       — gt_laws 보강 파일 생성
3. qa_4_make_eval_set.py — eval_set_tune.json / eval_set_test.json 생성
```