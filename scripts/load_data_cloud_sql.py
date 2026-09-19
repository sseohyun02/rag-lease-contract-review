"""Cloud SQL 데이터 통합 적재 스크립트.

법령 parent/child, 판례, 판례-법령 참조, 판례-판례 참조를 로컬 CSV에서 읽어
Cloud SQL PostgreSQL에 적재한다.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SCHEMA_SQL = PROJECT_ROOT / "infra" / "cloud_sql" / "init_schema.sql"
DEFAULT_PARENT_CSV = PROJECT_ROOT / "data" / "law_chunks" / "law_parent.csv"
DEFAULT_CHILD_CSV = PROJECT_ROOT / "data" / "law_chunks" / "law_child_vertex.csv"
DEFAULT_CASE_LAW_CSV = PROJECT_ROOT / "data" / "case_law_with_embeddings_vertex.csv"
DEFAULT_QA_JSON = (
    PROJECT_ROOT
    / "data"
    / "lawtalk_qa_preprocessed"
    / "lawtalk_qa_db_ready_from_predictions.json"
)
COPY_BATCH_SIZE = 1000

EXPECTED_PARENT_COLUMNS = [
    "article_key",
    "law_name",
    "law_abbr",
    "ministry",
    "enforcement_date",
    "article_no",
    "article_title",
    "article_date",
    "is_amended",
    "is_deleted",
    "parent_text",
    "is_article_only",
]

EXPECTED_CHILD_COLUMNS = [
    "clause_key",
    "article_key",
    "law_name",
    "article_no",
    "paragraph_no",
    "child_text",
    "embed_vertex",
]

EXPECTED_CASE_LAW_COLUMNS = [
    "case_id",
    "case_name",
    "case_number",
    "judgment_date",
    "judgment_result",
    "court_name",
    "court_type_code",
    "judgment_type",
    "issue",
    "judgment_summary",
    "referenced_law",
    "referenced_case",
    "case_detail",
    "embed_vertex",
]

PARENT_DB_COLUMNS = [
    "article_key",
    "law_name",
    "law_abbr",
    "ministry",
    "enforcement_date",
    "article_no",
    "article_title",
    "article_date",
    "is_amended",
    "is_deleted",
    "parent_text",
    "is_article_only",
]

CHILD_DB_COLUMNS = [
    "clause_key",
    "article_key",
    "parent_id",
    "law_name",
    "article_no",
    "paragraph_no",
    "child_text",
    "embed_vertex",
    "embed_kure",
    "embed_e5",
]

CASE_LAW_DB_COLUMNS = [
    "case_id",
    "case_name",
    "case_number",
    "judgment_date",
    "judgment_result",
    "court_name",
    "court_type_code",
    "judgment_type",
    "issue",
    "judgment_summary",
    "referenced_law",
    "referenced_case",
    "case_detail",
    "embed_vertex",
    "embed_kure",
    "embed_e5",
]

REFERENCED_LAW_DB_COLUMNS = [
    "case_id",
    "clause_key",
    "law_name",
    "article_no",
    "paragraph_no",
    "parent_id",
    "child_id",
]

REFERENCED_CASE_DB_COLUMNS = [
    "case_id",
    "referenced_case_number",
]

QUESTION_DB_COLUMNS = [
    "id",
    "title",
    "body",
    "tags",
    "written_at",
    "embedding",
]

ANSWER_DB_COLUMNS = [
    "id",
    "question_id",
    "lawyer_name",
    "answer_body",
    "written_at",
    "dispute_background",
    "lawyer_conclusion",
    "lawyer_reasoning",
    "action_checklist",
]

ANSWER_REFERENCED_LAW_DB_COLUMNS = [
    "answer_id",
    "clause_key",
    "law_name",
    "article_no",
    "paragraph_no",
    "parent_id",
    "child_id",
]

ANSWER_REFERENCED_CASE_DB_COLUMNS = [
    "answer_id",
    "referenced_case_number",
]

ARTICLE_PATTERN = re.compile(
    r"(?P<law_name>[가-힣A-Za-z0-9ㆍ·\s]+?)\s*"
    r"제(?P<article_no>\d+(?:의\d+)?)조(?:의(?P<article_suffix>\d+))?"
    r"(?:\s*제(?P<paragraph_no>\d+)항)?"
)

CASE_NUMBER_PATTERN = re.compile(r"\b\d{2,4}[가-힣]{1,4}\d+\b")


@dataclass(frozen=True)
class ReferenceMaps:
    parent_by_article_key: dict[str, int]
    child_by_clause_key: dict[str, int]


def parse_date_yyyymmdd(value: str) -> date | None:
    """CSV 날짜 문자열을 date로 변환한다. 빈 값은 NULL로 둔다."""
    value = value.strip()
    if value == "":
        return None

    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"date must be YYYYMMDD: {value!r}")

    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def parse_bool_flag(value: str) -> bool:
    """CSV의 0/1 값을 boolean으로 변환한다."""
    if value == "1":
        return True
    if value == "0":
        return False

    raise ValueError(f"boolean flag must be 0 or 1: {value!r}")


def to_nullable_text(value: str) -> str | None:
    value = value.strip()
    return value or None


def to_nullable_int(value: str) -> int | None:
    value = value.strip()
    if value == "":
        return None
    return int(float(value))


def normalize_date_value(value: str | None) -> str | None:
    """QA 날짜 값을 PostgreSQL date에 넣을 수 있는 YYYY-MM-DD로 정리한다."""
    if value is None:
        return None

    value = value.strip()
    if value == "":
        return None

    if re.match(r"^\d{4}-\d{2}-\d{2}", value):
        return value[:10]

    return None


def parse_pgvector(value: str, *, expected_dimensions: int = 3072) -> str | None:
    """CSV 임베딩 문자열을 pgvector COPY 입력 형식으로 변환한다."""
    value = value.strip()
    if value == "":
        return None

    if value.startswith("["):
        inner = value[1:-1].strip()
        dimensions = 0 if inner == "" else inner.count(",") + 1
        if dimensions != expected_dimensions:
            raise ValueError(
                f"embedding dimension mismatch: expected {expected_dimensions}, "
                f"got {dimensions}"
            )
        return "[" + ",".join(item.strip() for item in inner.split(",")) + "]"

    parsed = [float(item) for item in value.split(",") if item.strip()]
    if len(parsed) != expected_dimensions:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected_dimensions}, "
            f"got {len(parsed)}"
        )

    return "[" + ",".join(str(float(item)) for item in parsed) + "]"


def normalize_pgvector(
    value: str | list[int | float] | None,
    *,
    expected_dimensions: int,
) -> str | None:
    """JSON/CSV에서 온 임베딩 값을 pgvector 입력 형식으로 맞춘다."""
    if value is None:
        return None

    if isinstance(value, str):
        return parse_pgvector(value, expected_dimensions=expected_dimensions)

    if len(value) != expected_dimensions:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected_dimensions}, "
            f"got {len(value)}"
        )

    return "[" + ",".join(str(float(item)) for item in value) + "]"


def read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    """CSV를 읽고 필수 컬럼이 모두 있는지 확인한다. 추가 컬럼은 허용한다."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        actual = set(reader.fieldnames or [])
        missing = [c for c in expected_columns if c not in actual]
        if missing:
            raise ValueError(
                f"{path} missing required columns: {missing} "
                f"(actual: {list(reader.fieldnames)})"
            )

        return list(reader)


def iter_csv_rows(path: Path, expected_columns: list[str]) -> Iterable[dict[str, str]]:
    """CSV를 메모리에 전부 올리지 않고 한 행씩 읽는다. 추가 컬럼은 허용한다."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        actual = set(reader.fieldnames or [])
        missing = [c for c in expected_columns if c not in actual]
        if missing:
            raise ValueError(
                f"{path} missing required columns: {missing} "
                f"(actual: {list(reader.fieldnames)})"
            )

        yield from reader


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def make_postgres_copy_stream(
    rows: list[dict[str, Any]],
    columns: list[str],
) -> io.StringIO:
    """Postgres COPY에 넘길 CSV 문자열을 만든다."""
    stream = io.StringIO()
    writer = csv.writer(stream)

    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if value is None:
                values.append("")
            elif isinstance(value, bool):
                values.append("true" if value else "false")
            elif isinstance(value, date):
                values.append(value.isoformat())
            elif isinstance(value, (dict, list)):
                values.append(json.dumps(value, ensure_ascii=False))
            else:
                values.append(value)

        writer.writerow(values)

    stream.seek(0)
    return stream


def copy_rows(
    raw_connection: Any,
    table_name: str,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """pg8000 COPY stream으로 여러 row를 빠르게 적재한다."""
    if not rows:
        return

    column_text = ", ".join(columns)
    stream = make_postgres_copy_stream(rows, columns)
    cursor = raw_connection.cursor()
    cursor.execute(
        f"COPY {table_name} ({column_text}) FROM STDIN WITH (FORMAT csv, NULL '')",
        stream=stream,
    )


def copy_rows_in_batches(
    raw_connection: Any,
    table_name: str,
    columns: list[str],
    rows: Iterable[dict[str, Any]],
    *,
    batch_size: int = COPY_BATCH_SIZE,
    total_count: int | None = None,
    label: str | None = None,
) -> int:
    """큰 row iterator를 batch 단위로 COPY한다."""
    copied_count = 0
    batch = []
    log_label = label or table_name

    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            copy_rows(raw_connection, table_name, columns, batch)
            copied_count += len(batch)
            log_copy_progress(log_label, copied_count, total_count)
            batch = []

    if batch:
        copy_rows(raw_connection, table_name, columns, batch)
        copied_count += len(batch)
        log_copy_progress(log_label, copied_count, total_count)

    return copied_count


def log_copy_progress(label: str, copied_count: int, total_count: int | None) -> None:
    if total_count is None:
        print(f"copied {label} rows: {copied_count:,}", flush=True)
        return

    print(f"copied {label} rows: {copied_count:,}/{total_count:,}", flush=True)


def _find_duplicates(values: Iterable[str]) -> list[str]:
    seen = set()
    duplicates = []

    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)

    return duplicates


def validate_law_rows(
    parent_rows: list[dict[str, str]],
    child_rows: list[dict[str, str]],
) -> None:
    article_keys = [row["article_key"] for row in parent_rows]
    clause_keys = [row["clause_key"] for row in child_rows]

    duplicate_article_keys = _find_duplicates(article_keys)
    if duplicate_article_keys:
        raise ValueError(f"duplicate article_key: {duplicate_article_keys[:5]}")

    duplicate_clause_keys = _find_duplicates(clause_keys)
    if duplicate_clause_keys:
        raise ValueError(f"duplicate clause_key: {duplicate_clause_keys[:5]}")

    parent_key_set = set(article_keys)
    missing_parent_keys = []
    for row in child_rows:
        article_key = row["article_key"]
        if article_key not in parent_key_set and article_key not in missing_parent_keys:
            missing_parent_keys.append(article_key)

    if missing_parent_keys:
        raise ValueError(f"missing parent article_key: {missing_parent_keys[:5]}")

    for row in parent_rows:
        parse_date_yyyymmdd(row["enforcement_date"])
        parse_date_yyyymmdd(row["article_date"])
        parse_bool_flag(row["is_amended"])
        parse_bool_flag(row["is_deleted"])
        parse_bool_flag(row["is_article_only"])

    for row in child_rows:
        to_nullable_int(row["paragraph_no"])
        parse_pgvector(row["embed_vertex"])
        if "embed_kure" in row:
            parse_pgvector(row["embed_kure"], expected_dimensions=1024)
        if "embed_e5" in row:
            parse_pgvector(row["embed_e5"], expected_dimensions=1024)


def validate_child_rows_stream(
    child_rows: Iterable[dict[str, str]],
    parent_key_set: set[str],
) -> int:
    """큰 child CSV를 list로 만들지 않고 키/형식 검증을 수행한다."""
    seen_clause_keys = set()
    duplicate_clause_keys = []
    missing_parent_keys = []
    row_count = 0

    for row in child_rows:
        row_count += 1

        clause_key = row["clause_key"]
        if clause_key in seen_clause_keys and clause_key not in duplicate_clause_keys:
            duplicate_clause_keys.append(clause_key)
        seen_clause_keys.add(clause_key)

        article_key = row["article_key"]
        if article_key not in parent_key_set and article_key not in missing_parent_keys:
            missing_parent_keys.append(article_key)

        to_nullable_int(row["paragraph_no"])
        parse_pgvector(row["embed_vertex"])
        if "embed_kure" in row:
            parse_pgvector(row["embed_kure"], expected_dimensions=1024)
        if "embed_e5" in row:
            parse_pgvector(row["embed_e5"], expected_dimensions=1024)

    if duplicate_clause_keys:
        raise ValueError(f"duplicate clause_key: {duplicate_clause_keys[:5]}")

    if missing_parent_keys:
        raise ValueError(f"missing parent article_key: {missing_parent_keys[:5]}")

    return row_count


def validate_case_law_rows(case_rows: list[dict[str, str]]) -> None:
    case_ids = [row["case_id"] for row in case_rows]
    duplicate_case_ids = _find_duplicates(case_ids)
    if duplicate_case_ids:
        raise ValueError(f"duplicate case_id: {duplicate_case_ids[:5]}")

    for row in case_rows:
        parse_date_yyyymmdd(row["judgment_date"])
        to_nullable_int(row["court_type_code"])
        parse_pgvector(row["embed_vertex"])
        if "embed_kure" in row:
            parse_pgvector(row["embed_kure"], expected_dimensions=1024)
        if "embed_e5" in row:
            parse_pgvector(row["embed_e5"], expected_dimensions=1024)


def map_parent_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "article_key": row["article_key"],
        "law_name": row["law_name"],
        "law_abbr": to_nullable_text(row["law_abbr"]),
        "ministry": row["ministry"],
        "enforcement_date": parse_date_yyyymmdd(row["enforcement_date"]),
        "article_no": row["article_no"],
        "article_title": to_nullable_text(row["article_title"]),
        "article_date": parse_date_yyyymmdd(row["article_date"]),
        "is_amended": parse_bool_flag(row["is_amended"]),
        "is_deleted": parse_bool_flag(row["is_deleted"]),
        "parent_text": row["parent_text"],
        "is_article_only": parse_bool_flag(row["is_article_only"]),
    }


def map_child_row(row: dict[str, str], *, parent_id: int) -> dict[str, Any]:
    return {
        "clause_key": row["clause_key"],
        "article_key": row["article_key"],
        "parent_id": parent_id,
        "law_name": row["law_name"],
        "article_no": row["article_no"],
        "paragraph_no": to_nullable_int(row["paragraph_no"]),
        "child_text": row["child_text"],
        "embed_vertex": parse_pgvector(row["embed_vertex"]),
        "embed_kure": parse_pgvector(row.get("embed_kure", ""), expected_dimensions=1024),
        "embed_e5": parse_pgvector(row.get("embed_e5", ""), expected_dimensions=1024),
    }


def map_case_law_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "case_name": to_nullable_text(row["case_name"]),
        "case_number": to_nullable_text(row["case_number"]),
        "judgment_date": parse_date_yyyymmdd(row["judgment_date"]),
        "judgment_result": to_nullable_text(row["judgment_result"]),
        "court_name": to_nullable_text(row["court_name"]),
        "court_type_code": to_nullable_int(row["court_type_code"]),
        "judgment_type": to_nullable_text(row["judgment_type"]),
        "issue": to_nullable_text(row["issue"]),
        "judgment_summary": to_nullable_text(row["judgment_summary"]),
        "referenced_law": to_nullable_text(row["referenced_law"]),
        "referenced_case": to_nullable_text(row["referenced_case"]),
        "case_detail": to_nullable_text(row["case_detail"]),
        "embed_vertex": parse_pgvector(row["embed_vertex"]),
        "embed_kure": parse_pgvector(row.get("embed_kure", ""), expected_dimensions=1024),
        "embed_e5": parse_pgvector(row.get("embed_e5", ""), expected_dimensions=1024),
    }


def map_question_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": to_nullable_text(row.get("title") or ""),
        "body": to_nullable_text(row.get("body") or ""),
        "tags": row.get("tags") or [],
        "written_at": normalize_date_value(row.get("written_at")),
        "embedding": normalize_pgvector(
            row.get("embedding"),
            expected_dimensions=3072,
        ),
    }


def map_answer_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "question_id": row["question_id"],
        "lawyer_name": to_nullable_text(row.get("lawyer_name") or ""),
        "answer_body": row.get("answer_body") or {},
        "written_at": normalize_date_value(row.get("written_at")),
        "dispute_background": to_nullable_text(row.get("dispute_background") or ""),
        "lawyer_conclusion": to_nullable_text(row.get("lawyer_conclusion") or ""),
        "lawyer_reasoning": to_nullable_text(row.get("lawyer_reasoning") or ""),
        "action_checklist": to_nullable_text(row.get("action_checklist") or ""),
    }


def build_qa_rows(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    question_rows = [map_question_row(row) for row in data.get("questions", [])]
    answer_rows = [map_answer_row(row) for row in data.get("answers", [])]

    question_ids = [row["id"] for row in question_rows]
    answer_ids = [row["id"] for row in answer_rows]

    duplicate_question_ids = _find_duplicates(question_ids)
    if duplicate_question_ids:
        raise ValueError(f"duplicate question id: {duplicate_question_ids[:5]}")

    duplicate_answer_ids = _find_duplicates(answer_ids)
    if duplicate_answer_ids:
        raise ValueError(f"duplicate answer id: {duplicate_answer_ids[:5]}")

    question_id_set = set(question_ids)
    missing_question_ids = []
    for row in answer_rows:
        question_id = row["question_id"]
        if question_id not in question_id_set and question_id not in missing_question_ids:
            missing_question_ids.append(question_id)

    if missing_question_ids:
        raise ValueError(f"answer references missing question id: {missing_question_ids[:5]}")

    return question_rows, answer_rows


def article_key_for(law_name: str, article_no: str) -> str:
    if "의" in article_no:
        base, suffix = article_no.split("의", 1)
        return f"{law_name}_제{base}조의{suffix}"
    return f"{law_name}_제{article_no}조"


def clause_key_for(
    law_name: str,
    article_no: str,
    paragraph_no: str | None,
) -> str:
    article_key = article_key_for(law_name, article_no)
    if paragraph_no:
        return f"{article_key}_제{paragraph_no}항"
    return article_key


def normalize_reference_clause_key(clause_key: str) -> str:
    """DB reference keys must match law_child/law_parent keys exactly."""
    return clause_key.removesuffix("_조문")


def article_no_from_match(match: re.Match[str]) -> str:
    article_no = match.group("article_no")
    article_suffix = match.group("article_suffix")
    if article_suffix and "의" not in article_no:
        return f"{article_no}의{article_suffix}"
    return article_no


def _law_name_from_article_key(article_key: str) -> str:
    return article_key.rsplit("_제", 1)[0]


def resolve_law_name(
    raw_law_name: str,
    reference_maps: ReferenceMaps,
) -> str:
    law_names = {
        _law_name_from_article_key(article_key)
        for article_key in reference_maps.parent_by_article_key
    }
    matches = [
        law_name
        for law_name in law_names
        if raw_law_name == law_name or raw_law_name.endswith(law_name)
    ]

    if not matches:
        return raw_law_name

    return max(matches, key=len)


def parse_referenced_law_text(
    case_id: str,
    referenced_law: str | None,
    reference_maps: ReferenceMaps,
) -> tuple[list[dict[str, Any]], int]:
    """참조조문 문자열을 referenced_law row로 변환한다."""
    if not referenced_law:
        return [], 0

    rows = []
    unmatched_count = 0
    seen = set()

    for match in ARTICLE_PATTERN.finditer(referenced_law):
        raw_law_name = " ".join(match.group("law_name").split())
        law_name = resolve_law_name(raw_law_name, reference_maps)
        if law_name in {"", "부칙"} or law_name.endswith("부칙"):
            continue

        article_no = article_no_from_match(match)
        paragraph_no = match.group("paragraph_no")
        clause_key = normalize_reference_clause_key(
            clause_key_for(law_name, article_no, paragraph_no)
        )
        unique_key = (case_id, clause_key)
        if unique_key in seen:
            continue
        seen.add(unique_key)

        parent_id = reference_maps.parent_by_article_key.get(
            article_key_for(law_name, article_no)
        )
        child_id = None
        if paragraph_no:
            child_id = reference_maps.child_by_clause_key.get(clause_key)

        if parent_id is None and child_id is None:
            unmatched_count += 1

        rows.append(
            {
                "case_id": case_id,
                "clause_key": clause_key,
                "law_name": law_name,
                "article_no": article_no,
                "paragraph_no": paragraph_no,
                "parent_id": parent_id,
                "child_id": child_id,
            }
        )

    return rows, unmatched_count


def parse_referenced_case_text(
    case_id: str,
    referenced_case: str | None,
) -> list[dict[str, str]]:
    """참조판례 문자열에서 판례번호 후보를 뽑는다."""
    if not referenced_case:
        return []

    rows = []
    seen = set()
    for match in CASE_NUMBER_PATTERN.finditer(referenced_case):
        case_number = match.group(0)
        unique_key = (case_id, case_number)
        if unique_key in seen:
            continue
        seen.add(unique_key)
        rows.append(
            {
                "case_id": case_id,
                "referenced_case_number": case_number,
            }
        )

    return rows


def answer_text_from_row(row: dict[str, Any]) -> str:
    answer_body = row.get("answer_body")
    if isinstance(answer_body, dict):
        answer = answer_body.get("answer")
        if isinstance(answer, str):
            return answer

    if isinstance(answer_body, str):
        return answer_body

    return ""


def build_answer_reference_rows(
    answer_rows: list[dict[str, Any]],
    reference_maps: ReferenceMaps,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    answer_referenced_law_rows = []
    answer_referenced_case_rows = []
    unmatched_law_count = 0

    for row in answer_rows:
        answer_id = row["id"]
        answer_text = answer_text_from_row(row)

        law_rows, unmatched_count = parse_referenced_law_text(
            str(answer_id),
            answer_text,
            reference_maps,
        )
        unmatched_law_count += unmatched_count
        for law_row in law_rows:
            if law_row["parent_id"] is None and law_row["child_id"] is None:
                continue

            answer_referenced_law_rows.append(
                {
                    "answer_id": answer_id,
                    "clause_key": law_row["clause_key"],
                    "law_name": law_row["law_name"],
                    "article_no": law_row["article_no"],
                    "paragraph_no": law_row["paragraph_no"],
                    "parent_id": law_row["parent_id"],
                    "child_id": law_row["child_id"],
                }
            )

        for case_row in parse_referenced_case_text(str(answer_id), answer_text):
            answer_referenced_case_rows.append(
                {
                    "answer_id": answer_id,
                    "referenced_case_number": case_row["referenced_case_number"],
                }
            )

    return answer_referenced_law_rows, answer_referenced_case_rows, unmatched_law_count


def build_reference_rows(
    case_rows: list[dict[str, str]],
    reference_maps: ReferenceMaps,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    referenced_law_rows = []
    referenced_case_rows = []
    unmatched_law_count = 0

    for row in case_rows:
        law_rows, unmatched_count = parse_referenced_law_text(
            row["case_id"],
            to_nullable_text(row["referenced_law"]),
            reference_maps,
        )
        referenced_law_rows.extend(law_rows)
        unmatched_law_count += unmatched_count

        referenced_case_rows.extend(
            parse_referenced_case_text(
                row["case_id"],
                to_nullable_text(row["referenced_case"]),
            )
        )

    return referenced_law_rows, referenced_case_rows, unmatched_law_count


def load_schema(raw_connection: Any, schema_sql: Path) -> None:
    """스키마 SQL을 실행한다."""
    sql = schema_sql.read_text(encoding="utf-8")
    cursor = raw_connection.cursor()
    for statement in sql.split(";"):
        statement = statement.strip()
        if statement:
            cursor.execute(statement)


def fetch_reference_maps(raw_connection: Any) -> ReferenceMaps:
    cursor = raw_connection.cursor()

    cursor.execute("SELECT id, article_key FROM law_parent")
    parent_by_article_key = {row[1]: row[0] for row in cursor.fetchall()}

    cursor.execute("SELECT id, clause_key FROM law_child")
    child_by_clause_key = {row[1]: row[0] for row in cursor.fetchall()}

    return ReferenceMaps(
        parent_by_article_key=parent_by_article_key,
        child_by_clause_key=child_by_clause_key,
    )


def load_legal_data(
    *,
    parent_csv: Path = DEFAULT_PARENT_CSV,
    child_csv: Path = DEFAULT_CHILD_CSV,
    case_law_csv: Path = DEFAULT_CASE_LAW_CSV,
    schema_sql: Path = DEFAULT_SCHEMA_SQL,
    init_schema: bool = False,
    truncate: bool = False,
) -> dict[str, int]:
    """로컬 CSV 파일을 Cloud SQL에 적재한다."""
    from shared.db.connection import get_db_client

    parent_rows = read_csv(parent_csv, EXPECTED_PARENT_COLUMNS)
    case_rows = read_csv(case_law_csv, EXPECTED_CASE_LAW_COLUMNS)

    validate_law_rows(parent_rows, [])
    parent_key_set = {row["article_key"] for row in parent_rows}
    child_count = validate_child_rows_stream(
        iter_csv_rows(child_csv, EXPECTED_CHILD_COLUMNS),
        parent_key_set,
    )
    validate_case_law_rows(case_rows)

    db = get_db_client()
    raw_connection = db.engine.raw_connection()

    try:
        cursor = raw_connection.cursor()

        if init_schema:
            print(f"running schema: {schema_sql}", flush=True)
            load_schema(raw_connection, schema_sql)

        if truncate:
            cursor.execute(
                "TRUNCATE TABLE referenced_case, referenced_law, case_law, "
                "law_child, law_parent RESTART IDENTITY CASCADE"
            )

        mapped_parents = [map_parent_row(row) for row in parent_rows]
        print(f"copying law_parent rows: {len(mapped_parents)}", flush=True)
        copy_rows_in_batches(
            raw_connection,
            "law_parent",
            PARENT_DB_COLUMNS,
            mapped_parents,
            total_count=len(mapped_parents),
            label="law_parent",
        )

        reference_maps = fetch_reference_maps(raw_connection)

        mapped_children = (
            map_child_row(
                row,
                parent_id=reference_maps.parent_by_article_key[row["article_key"]],
            )
            for row in iter_csv_rows(child_csv, EXPECTED_CHILD_COLUMNS)
        )
        print(f"copying law_child rows: {child_count}", flush=True)
        copied_child_count = copy_rows_in_batches(
            raw_connection,
            "law_child",
            CHILD_DB_COLUMNS,
            mapped_children,
            total_count=child_count,
            label="law_child",
        )

        reference_maps = fetch_reference_maps(raw_connection)

        mapped_cases = [map_case_law_row(row) for row in case_rows]
        print(f"copying case_law rows: {len(mapped_cases)}", flush=True)
        copy_rows_in_batches(
            raw_connection,
            "case_law",
            CASE_LAW_DB_COLUMNS,
            mapped_cases,
            total_count=len(mapped_cases),
            label="case_law",
        )

        referenced_law_rows, referenced_case_rows, unmatched_law_count = (
            build_reference_rows(case_rows, reference_maps)
        )

        print(f"copying referenced_law rows: {len(referenced_law_rows)}", flush=True)
        copy_rows_in_batches(
            raw_connection,
            "referenced_law",
            REFERENCED_LAW_DB_COLUMNS,
            referenced_law_rows,
            total_count=len(referenced_law_rows),
            label="referenced_law",
        )

        print(f"copying referenced_case rows: {len(referenced_case_rows)}", flush=True)
        copy_rows_in_batches(
            raw_connection,
            "referenced_case",
            REFERENCED_CASE_DB_COLUMNS,
            referenced_case_rows,
            total_count=len(referenced_case_rows),
            label="referenced_case",
        )

        raw_connection.commit()
    except Exception:
        raw_connection.rollback()
        raise
    finally:
        raw_connection.close()
        db.close()

    return {
        "law_parent_count": len(parent_rows),
        "law_child_count": copied_child_count,
        "case_law_count": len(case_rows),
        "referenced_law_count": len(referenced_law_rows),
        "referenced_case_count": len(referenced_case_rows),
        "unmatched_referenced_law_count": unmatched_law_count,
    }


def load_qa_data(
    *,
    qa_json: Path = DEFAULT_QA_JSON,
    schema_sql: Path = DEFAULT_SCHEMA_SQL,
    init_schema: bool = False,
    truncate_qa: bool = False,
    with_answer_references: bool = False,
) -> dict[str, int]:
    """DB-ready QA JSON을 questions/answers 테이블에 적재한다."""
    from shared.db.connection import get_db_client

    question_rows, answer_rows = build_qa_rows(read_json(qa_json))

    db = get_db_client()
    raw_connection = db.engine.raw_connection()

    try:
        cursor = raw_connection.cursor()

        if init_schema:
            print(f"running schema: {schema_sql}", flush=True)
            load_schema(raw_connection, schema_sql)

        if truncate_qa:
            cursor.execute(
                "TRUNCATE TABLE answer_referenced_case, answer_referenced_law, "
                "answers, questions RESTART IDENTITY CASCADE"
            )

        print(f"copying questions rows: {len(question_rows)}", flush=True)
        copy_rows_in_batches(
            raw_connection,
            "questions",
            QUESTION_DB_COLUMNS,
            question_rows,
            total_count=len(question_rows),
            label="questions",
        )

        print(f"copying answers rows: {len(answer_rows)}", flush=True)
        copy_rows_in_batches(
            raw_connection,
            "answers",
            ANSWER_DB_COLUMNS,
            answer_rows,
            total_count=len(answer_rows),
            label="answers",
        )

        answer_referenced_law_rows = []
        answer_referenced_case_rows = []
        unmatched_answer_law_count = 0
        if with_answer_references:
            reference_maps = fetch_reference_maps(raw_connection)
            (
                answer_referenced_law_rows,
                answer_referenced_case_rows,
                unmatched_answer_law_count,
            ) = build_answer_reference_rows(answer_rows, reference_maps)

            print(
                f"copying answer_referenced_law rows: "
                f"{len(answer_referenced_law_rows)}",
                flush=True,
            )
            copy_rows_in_batches(
                raw_connection,
                "answer_referenced_law",
                ANSWER_REFERENCED_LAW_DB_COLUMNS,
                answer_referenced_law_rows,
                total_count=len(answer_referenced_law_rows),
                label="answer_referenced_law",
            )

            print(
                f"copying answer_referenced_case rows: "
                f"{len(answer_referenced_case_rows)}",
                flush=True,
            )
            copy_rows_in_batches(
                raw_connection,
                "answer_referenced_case",
                ANSWER_REFERENCED_CASE_DB_COLUMNS,
                answer_referenced_case_rows,
                total_count=len(answer_referenced_case_rows),
                label="answer_referenced_case",
            )

        raw_connection.commit()
    except Exception:
        raw_connection.rollback()
        raise
    finally:
        raw_connection.close()
        db.close()

    return {
        "questions_count": len(question_rows),
        "answers_count": len(answer_rows),
        "answer_referenced_law_count": len(answer_referenced_law_rows),
        "answer_referenced_case_count": len(answer_referenced_case_rows),
        "unmatched_answer_referenced_law_count": unmatched_answer_law_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load local CSV files into Cloud SQL")
    parser.add_argument("--parent-csv", type=Path, default=DEFAULT_PARENT_CSV)
    parser.add_argument("--child-csv", type=Path, default=DEFAULT_CHILD_CSV)
    parser.add_argument("--case-law-csv", type=Path, default=DEFAULT_CASE_LAW_CSV)
    parser.add_argument("--qa-json", type=Path)
    parser.add_argument("--schema-sql", type=Path, default=DEFAULT_SCHEMA_SQL)
    parser.add_argument(
        "--init-schema",
        action="store_true",
        help="Run infra/cloud_sql/init_schema.sql before loading data",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Clear loaded legal-data tables before loading",
    )
    parser.add_argument(
        "--only-qa",
        action="store_true",
        help="Load only questions and answers from --qa-json",
    )
    parser.add_argument(
        "--truncate-qa",
        action="store_true",
        help="Clear questions and answers before loading QA data",
    )
    parser.add_argument(
        "--with-answer-references",
        action="store_true",
        help="Also extract and load law/case references from answer bodies",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.only_qa:
        stats = load_qa_data(
            qa_json=args.qa_json or DEFAULT_QA_JSON,
            schema_sql=args.schema_sql,
            init_schema=args.init_schema,
            truncate_qa=args.truncate_qa,
            with_answer_references=args.with_answer_references,
        )
        for key, value in stats.items():
            print(f"{key}: {value}")
        return

    stats = load_legal_data(
        parent_csv=args.parent_csv,
        child_csv=args.child_csv,
        case_law_csv=args.case_law_csv,
        schema_sql=args.schema_sql,
        init_schema=args.init_schema,
        truncate=args.truncate,
    )

    if args.qa_json:
        stats.update(
            load_qa_data(
                qa_json=args.qa_json,
                schema_sql=args.schema_sql,
                init_schema=False,
                truncate_qa=args.truncate_qa,
                with_answer_references=args.with_answer_references,
            )
        )

    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
