"""검수 완료된 gt_from_reasoning.json을 평가용 eval_set으로 변환한다.

하는 일 (DB/API 불필요 - 형식 변환만)
----
1. gt_from_reasoning.json을 읽는다.
2. 전체 레코드를 먼저 tune : test = 7 : 3 으로 분할한다 (seed 42).
   - GT 수정으로 정답 0개 케이스가 생겨도 다른 케이스의 소속이 바뀌지 않도록,
     '분할 → 빈 케이스 제외' 순서로 처리한다.
3. 분할된 각 셋에서 gt_laws_filtered(검수 후 정답)가 1개 이상인 케이스만 남긴다.
4. 평가 스크립트(retrieval_eval.py)가 읽는 형식으로 변환한다:
   - clause           -> clauses[0].normalized
   - gt_laws_filtered -> gt_laws (그리고 gt_laws_explicit도 동일값으로)
   - concepts/detail 등 검수용 필드는 제외

이 정답은 '변호사 답변의 법리 설명'을 근거로 만들고 사람이 검수한 것이라,
gt_laws_explicit(직접인용)과 구분할 필요가 없다. 전부 신뢰 gt로 취급한다.

실행 (프로젝트 루트에서): python data/processors/qa_4_make_eval_set.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

# 프로젝트 루트 기준 경로 (어느 위치에서 실행해도 동일하게 동작)
ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "evaluation" / "results"

INPUT_PATH = RESULTS_DIR / "gt_from_reasoning.json"
OUTPUT_TUNE_PATH = RESULTS_DIR / "eval_set_tune.json"
OUTPUT_TEST_PATH = RESULTS_DIR / "eval_set_test.json"

TEST_RATIO = 0.3
RANDOM_SEED = 42

# 조 단위 집계용 (항/호/목 제거) — retrieval_eval.py의 매칭 기준과 동일
_ARTICLE_STRIP = re.compile(r"_(제\d+항|제\d+호|제\d+목).*$")


def _article_of(key: str) -> str:
    return _ARTICLE_STRIP.sub("", str(key))


def _to_eval_record(r: dict) -> dict:
    """gt_from_reasoning 레코드 1건을 eval_set 형식으로 변환."""
    url = str(r.get("url", ""))
    url_id = url.split("/")[-1].split("?")[0] if url else "nourl"
    gt = list(r.get("gt_laws_filtered", []) or [])
    return {
        "id": f"qa_{r.get('index')}_{url_id}",
        "source_type": "qa",
        "source_id": url,
        "clauses": [{"normalized": r.get("clause", "")}],
        "gt_laws": gt,
        "gt_laws_explicit": gt,   # 전부 검수된 신뢰 gt이므로 동일값
        "gt_laws_inferred": [],
        "gt_cases": [],
        "meta": {"url": url, "index": r.get("index")},
    }


def _build_split(records: list[dict]) -> list[dict]:
    """정답 1개 이상인 레코드만 eval 형식으로 변환."""
    out = []
    for r in records:
        if not (r.get("gt_laws_filtered") or []):
            continue  # 검수 후 정답이 없는 케이스는 평가에서 제외
        if not str(r.get("clause", "")).strip():
            continue  # 특약 문장이 비어있으면 검색 입력이 없으므로 제외
        out.append(_to_eval_record(r))
    return out


def _summary(name: str, split: list[dict]) -> str:
    n = len(split)
    n_hang = sum(len(r["gt_laws"]) for r in split)
    n_art = sum(len({_article_of(g) for g in r["gt_laws"]}) for r in split)
    avg = (n_art / n) if n else 0.0
    return f"{name}: {n}건 | 정답(항 단위) {n_hang}개 | 정답(조 단위) {n_art}개 | 케이스당 조 평균 {avg:.2f}"


def main() -> None:
    if not INPUT_PATH.exists():
        sys.exit(f"[오류] 입력 파일이 없습니다: {INPUT_PATH}")

    try:
        records = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"[오류] JSON 파싱 실패 ({INPUT_PATH}): {e}")

    if not isinstance(records, list) or not records:
        sys.exit("[오류] 입력 JSON은 비어있지 않은 리스트여야 합니다.")

    # 1) 전체 레코드를 먼저 셔플·분할 → 케이스 소속이 GT 수정과 무관하게 고정
    rng = random.Random(RANDOM_SEED)
    shuffled = records[:]
    rng.shuffle(shuffled)
    test_size = round(len(shuffled) * TEST_RATIO)
    test_records = shuffled[:test_size]
    tune_records = shuffled[test_size:]

    # 2) 분할 후 정답 0개 케이스 제외 + 형식 변환
    test_set = _build_split(test_records)
    tune_set = _build_split(tune_records)

    # 3) 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TUNE_PATH.write_text(
        json.dumps(tune_set, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    OUTPUT_TEST_PATH.write_text(
        json.dumps(test_set, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4) 요약 출력
    print(f"입력 레코드: {len(records)}건 (분할: test {len(test_records)} / tune {len(tune_records)})")
    print(f"정답 0개로 제외: test {len(test_records) - len(test_set)}건 / tune {len(tune_records) - len(tune_set)}건")
    print(_summary("test", test_set))
    print(_summary("tune", tune_set))
    print(f"저장: {OUTPUT_TEST_PATH}")
    print(f"저장: {OUTPUT_TUNE_PATH}")


if __name__ == "__main__":
    main()