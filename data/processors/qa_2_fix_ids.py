"""qa_processed.json의 각 레코드를 원본 12개 파일과 내용(special_clauses) 기준으로
정확히 매칭하여:
  1. 전역 고유 식별자인 url을 부여하고 (index 충돌 문제 해결)
  2. 매칭된 원본의 답변 텍스트로 precedent_numbers를 개선된 정규식으로 재계산한다.

배경
----
qa_processed.json의 index는 원본 12개 파일 각각에서 1~250로 리셋되는 로컬
번호라서, 파일 간에 값이 겹친다 (예: 1_250.json의 23번과 251_500.json의
23번이 둘 다 index=23). 이 때문에 index만으로는 원본을 특정할 수 없다.
대신 special_clauses는 question_body에서 그대로 뽑힌 문장이므로, 이 문장이
어느 원본 글의 question_body에 들어있는지 검색해서 원본을 역추적한다.

실행 위치: 리포지토리 루트
사전 준비: data/raw/ 폴더에 posts_index_*.json 12개 전부 저장
실행 방법: python rebuild_qa_data.py
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

RAW_DIR = Path("data/qa_raw")
QA_PROCESSED_PATH = Path("data/lawtalk_qa_filtered/qa_processed.json")

# 실제 사건부호만 허용 (민사/형사/행정/가사 1~3심 기준, 대법원 예규 기준 주요 부호).
CASE_TYPE_CODES = (
    "가단", "가합", "가소", "나", "다",
    "카단", "카합", "카확", "카기", "카명", "카불",
    "고단", "고합", "고약", "노", "도",
    "구단", "구합", "누", "두",
    "드단", "드합", "르",
)
_CODES_PATTERN = "|".join(sorted(CASE_TYPE_CODES, key=len, reverse=True))
PRECEDENT_PATTERN = re.compile(
    r"(?<!\d)(\d{2}|\d{4})\s*"
    rf"(?!년|월|일|조|항|호)({_CODES_PATTERN})"
    r"\s*(\d{3,7})(?!\d)"
)


def extract_precedent_numbers(text: str) -> list[str]:
    numbers: list[str] = []
    for match in PRECEDENT_PATTERN.finditer(text):
        number = match.group(1) + match.group(2) + match.group(3)
        if number not in numbers:
            numbers.append(number)
    return numbers


def load_raw_records() -> list[dict]:
    raw_files = sorted(RAW_DIR.glob("posts_index_*.json"))
    if len(raw_files) < 12:
        raise FileNotFoundError(
            f"{RAW_DIR}에 posts_index_*.json 파일이 12개 미만입니다 "
            f"(현재 {len(raw_files)}개). 전부 받아서 저장해주세요."
        )

    records = []
    for path in raw_files:
        with open(path, encoding="utf-8") as f:
            records.extend(json.load(f))
    return records


def find_matching_raw_record(
    special_clauses: list[str], raw_records: list[dict]
) -> tuple[dict | None, int]:
    """special_clauses 문장들이 모두 question_body에 포함되는 원본을 찾는다.

    반환: (매칭된 레코드 또는 None, 후보 개수)
    후보가 정확히 1개일 때만 매칭 성공으로 취급한다.
    """
    if not special_clauses:
        return None, 0

    candidates = raw_records
    for clause in special_clauses:
        candidates = [
            r for r in candidates if clause in r.get("question_body", "")
        ]
        if len(candidates) <= 1:
            break

    if len(candidates) == 1:
        return candidates[0], 1
    return None, len(candidates)


def main() -> None:
    with open(QA_PROCESSED_PATH, encoding="utf-8") as f:
        qa_records = json.load(f)

    raw_records = load_raw_records()
    print(f"원본 레코드 수: {len(raw_records)}")
    print(f"qa_processed.json 레코드 수: {len(qa_records)}")

    stats = {
        "matched": 0,
        "ambiguous": 0,
        "not_found": 0,
        "old_precedent_count_total": 0,
        "new_precedent_count_total": 0,
    }

    updated_records = []
    review_needed = []

    for record in qa_records:
        clauses = record.get("special_clauses", [])
        matched_raw, candidate_count = find_matching_raw_record(clauses, raw_records)

        if matched_raw is None:
            if candidate_count == 0:
                stats["not_found"] += 1
            else:
                stats["ambiguous"] += 1
            review_needed.append(
                {
                    "index": record.get("index"),
                    "candidate_count": candidate_count,
                    "special_clauses": clauses,
                }
            )
            # 매칭 실패한 레코드는 url 없이, precedent_numbers도 원래 값 그대로 둔다
            # (뒤에서 gt 매칭 단계에서 자연히 걸러지도록)
            updated_records.append(record)
            continue

        stats["matched"] += 1

        new_precedents: list[str] = []
        for answer_item in matched_raw.get("all_answers", []):
            answer_text = answer_item.get("answer", "")
            for number in extract_precedent_numbers(answer_text):
                if number not in new_precedents:
                    new_precedents.append(number)

        stats["old_precedent_count_total"] += len(record.get("precedent_numbers", []))
        stats["new_precedent_count_total"] += len(new_precedents)

        record["url"] = matched_raw.get("url")
        record["precedent_numbers"] = new_precedents
        updated_records.append(record)

    # 원본 백업 후 덮어쓰기
    backup_path = QA_PROCESSED_PATH.with_suffix(".json.bak")
    shutil.copy(QA_PROCESSED_PATH, backup_path)

    with open(QA_PROCESSED_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_records, f, ensure_ascii=False, indent=2)

    review_path = QA_PROCESSED_PATH.parent / "qa_processed_review_needed.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_needed, f, ensure_ascii=False, indent=2)

    # url 기준 진짜 중복(같은 원본 글이 여러 레코드로 존재하는 경우) 확인
    urls = [r.get("url") for r in updated_records if r.get("url")]
    unique_urls = set(urls)
    stats["records_with_url"] = len(urls)
    stats["unique_urls"] = len(unique_urls)
    stats["true_duplicate_records"] = len(urls) - len(unique_urls)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n백업: {backup_path}")
    print(f"업데이트 완료: {QA_PROCESSED_PATH}")
    print(f"검토 필요 목록: {review_path} ({len(review_needed)}건)")


if __name__ == "__main__":
    main()
