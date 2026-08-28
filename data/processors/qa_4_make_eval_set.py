"""검수 완료된 gt_from_reasoning.json을 평가용 eval_set으로 변환한다.

하는 일 (DB/API 불필요 - 형식 변환만)
----
1. gt_from_reasoning.json을 읽는다.
2. gt_laws_filtered(검수 후 정답)가 1개 이상인 케이스만 남긴다.
3. 평가 스크립트(retrieval_eval.py)가 읽는 형식으로 변환한다:
   - clause          -> clauses[0].normalized
   - gt_laws_filtered -> gt_laws (그리고 gt_laws_explicit도 동일값으로)
   - concepts/detail 등 검수용 필드는 제거
4. tune : test = 7 : 3 으로 분할한다 (seed 42, 기존 eval_set과 동일 규칙).

이 정답은 '변호사 답변의 법리 설명'을 근거로 만들고 사람이 검수한 것이라,
gt_laws_explicit(직접인용)과 구분할 필요가 없다. 전부 신뢰 gt로 취급한다.

실행: python data/processors/convert_reasoning_to_eval_set.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

INPUT_PATH = Path("evaluation/gt_from_reasoning.json")
OUTPUT_TUNE_PATH = Path("evaluation/eval_set_tune.json")
OUTPUT_TEST_PATH = Path("evaluation/eval_set_test.json")

TEST_RATIO = 0.3
RANDOM_SEED = 42


def main() -> None:
    records = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    eval_set = []
    for r in records:
        gt = r.get("gt_laws_filtered", [])
        if not gt:
            continue  # 검수 후 정답이 없는 케이스(무관/포괄조항 등)는 평가에서 제외
        eval_set.append({
            "id": f"qa_{r['index']}_{r['url'].split('/')[-1].split('?')[0]}",
            "source_type": "qa",
            "source_id": r["url"],
            "clauses": [{"normalized": r["clause"]}],
            "gt_laws": gt,
            "gt_laws_explicit": gt,   # 전부 검수된 신뢰 gt이므로 동일값
            "gt_laws_inferred": [],
            "gt_cases": [],
            "meta": {"url": r["url"], "index": r["index"]},
        })

    print(f"변환된 케이스: {len(eval_set)}건")

    # tune / test 분할 (기존 eval_set과 동일 규칙: seed 42, test 30%)
    rng = random.Random(RANDOM_SEED)
    shuffled = eval_set[:]
    rng.shuffle(shuffled)
    test_size = round(len(shuffled) * TEST_RATIO)
    test_set = shuffled[:test_size]
    tune_set = shuffled[test_size:]

    OUTPUT_TUNE_PATH.write_text(
        json.dumps(tune_set, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUTPUT_TEST_PATH.write_text(
        json.dumps(test_set, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_gt = sum(len(r["gt_laws"]) for r in eval_set)
    print(f"  총 gt_laws: {total_gt}개 (특약당 평균 {total_gt/len(eval_set):.2f})")
    print(f"tune 저장: {OUTPUT_TUNE_PATH} ({len(tune_set)}건)")
    print(f"test 저장: {OUTPUT_TEST_PATH} ({len(test_set)}건)")


if __name__ == "__main__":
    main()
