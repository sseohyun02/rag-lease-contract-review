"""로톡 QA 데이터에서 특약 질문과 법령/판례 참조를 추출한다."""

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field


TARGET_LAWS = [
    "민법",
    "부동산등기법",
    "공인중개사법",
    "주택임대차보호법",
    "화물자동차운수사업법",
    "민사소송법",
    "민사조정법",
    "민사집행법",
    "소액사건심판법",
    "공동주택관리법",
    "주민등록법",
]

SPECIAL_KEYWORDS = ["특약", "특약사항", "특약조항", "특약 조항"]
RULE_BASED_FILTERED_FILES = [
    "qa_both.json",
    "qa_law_only.json",
    "qa_precedent_only.json",
]


class ExtractedClausesResult(BaseModel):
    """rule-based 특약 후보에서 실제 계약서 특약 문구를 고른 결과."""

    extracted_clauses: list[str] = Field(
        description=(
            "special_clauses 안에서 실제 계약서에 존재할 만한 특약 문구만 넣습니다. "
            "질문, 상담 요청, 배경 설명, 법률 판단 질문은 제외합니다."
        )
    )


class LawtalkAnswerAnalysis(BaseModel):
    """DB 적재용 answers 확장 필드 추출 결과."""

    dispute_background: str = Field(
        description=(
            "질문 본문에서 추출한 분쟁 배경입니다. 당사자 관계, 계약 내용, "
            "문제 발생 경위를 중심으로 3~5문장으로 작성합니다."
        )
    )
    lawyer_conclusion: str = Field(
        description=(
            "변호사 답변의 최종 결론입니다. 유효, 무효, 일부무효 등 "
            "법적 판단을 중심으로 1~2문장으로 작성합니다."
        )
    )
    lawyer_reasoning: str = Field(
        description=(
            "변호사 답변의 법리 근거입니다. 답변에 명시된 내용만 "
            "개조식 3~5줄로 작성하고, 추론은 '(추론)'으로 표시합니다."
        )
    )
    action_checklist: str = Field(
        description=(
            "변호사 답변에서 질문자가 취해야 할 행동 또는 주의사항입니다. "
            "답변 본문에 명시된 내용만 항목별로 작성합니다."
        )
    )


LAWTALK_DB_ANALYSIS_SYSTEM_PROMPT = """당신은 한국 임대차 법률 상담 분석 전문가입니다.
주어진 법률 상담 QA를 읽고 DB 스키마에 필요한 필드만 정확히 추출하세요.

---

[필드별 추출 지침]

dispute_background
- 질문 본문에서 분쟁 배경을 3~5문장으로 압축
- 당사자 관계, 계약 내용, 문제 발생 경위 중심으로 서술

lawyer_conclusion
- 변호사 답변의 최종 결론을 1~2문장으로 서술
- 유효/무효/일부무효 등 법적 판단을 중심으로 명확하게 서술

lawyer_reasoning
- 변호사 답변의 법리 근거를 개조식 3~5줄로 정리
- 답변에 명시된 내용은 팩트로, 답변에서 추론된 내용은 "(추론)"으로 표시
- 새로운 해석이나 추가 설명 금지

action_checklist
- 변호사 답변에서 질문자가 취해야 할 행동이나 주의해야 할 사항을 항목별로 추출
- 답변 본문에 명시된 내용만 작성
"""


def add_unique(items, value):
    """리스트에 같은 값이 없을 때만 값을 추가한다."""
    if value not in items:
        items.append(value)


def debug_log(enabled, message):
    """디버그 모드일 때만 추적 로그를 출력한다."""
    if enabled:
        print(f"[lawtalk_qa][debug] {message}", flush=True)


def make_law_reference(law_name, article_number, sub_article_number):
    """법령명과 조문 번호를 출력용 문자열로 만든다."""
    if article_number is None:
        return law_name

    reference = law_name + " 제" + article_number + "조"

    if sub_article_number is not None:
        reference = reference + "의" + sub_article_number

    return reference


def find_direct_law_matches(text):
    """답변에서 직접 등장한 법령명을 찾는다."""
    matches = []

    for law_name in TARGET_LAWS:
        escaped_law_name = re.escape(law_name)
        pattern = (
            escaped_law_name
            + r"\s*(?:제\s*)?(\d+)?\s*"
            + r"(?:조\s*(?:의\s*(\d+))?)?"
        )

        for match in re.finditer(pattern, text):
            article_number = match.group(1)
            sub_article_number = match.group(2)

            matches.append(
                {
                    "start": match.start(),
                    "law_name": law_name,
                    "article_number": article_number,
                    "sub_article_number": sub_article_number,
                    "is_same_law": False,
                }
            )

    return matches


def find_same_law_matches(text):
    """답변에서 '동법'으로 표현된 조문을 찾는다."""
    matches = []
    pattern = r"동법\s*(?:제\s*)?(\d+)?\s*(?:조\s*(?:의\s*(\d+))?)?"

    for match in re.finditer(pattern, text):
        matches.append(
            {
                "start": match.start(),
                "law_name": None,
                "article_number": match.group(1),
                "sub_article_number": match.group(2),
                "is_same_law": True,
            }
        )

    return matches


def extract_law_references(text):
    """답변에서 대상 법령 참조를 추출한다.

    '동법'은 같은 답변 안에서 앞서 나온 대상 법령명으로 바꾼다.
    """
    all_matches = find_direct_law_matches(text) + find_same_law_matches(text)
    all_matches.sort(key=lambda item: item["start"])

    references = []
    last_law_name = None

    for match in all_matches:
        if match["is_same_law"]:
            if last_law_name is None:
                continue

            reference = make_law_reference(
                last_law_name,
                match["article_number"],
                match["sub_article_number"],
            )
            add_unique(references, reference)
            continue

        last_law_name = match["law_name"]
        reference = make_law_reference(
            match["law_name"],
            match["article_number"],
            match["sub_article_number"],
        )
        add_unique(references, reference)

    return references


def extract_precedent_numbers(text):
    """답변에서 한국 사건번호 형식의 판례번호를 추출한다."""
    pattern = r"(?<!\d)(\d{2}|\d{4})\s*(?!년|월|일|조|항|호)([가-힣]{1,4})\s*(\d+)(?!\d)"
    numbers = []

    for match in re.finditer(pattern, text):
        precedent_number = match.group(1) + match.group(2) + match.group(3)
        add_unique(numbers, precedent_number)

    return numbers


def split_sentences(text):
    """줄바꿈과 문장 종료 기호를 기준으로 문장을 나눈다."""
    sentences = []
    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if line == "":
            continue

        line_sentences = re.findall(r"[^.!?。]+[.!?。]?", line)

        for sentence in line_sentences:
            sentence = sentence.strip()
            if sentence != "":
                sentences.append(sentence)

    return sentences


def extract_special_clauses(question_body):
    """질문 본문에서 특약이 언급된 문장만 추출한다."""
    special_clauses = []
    sentences = split_sentences(question_body)

    for sentence in sentences:
        for keyword in SPECIAL_KEYWORDS:
            if keyword in sentence:
                special_clauses.append(sentence)
                break

    return special_clauses


def build_clause_postprocess_prompt(special_clauses):
    """1차 rule-based 결과의 special_clauses만 사용해 특약 후처리 프롬프트를 만든다."""
    lines = [
        "다음 special_clauses 목록에서 실제 계약서에 존재할 만한 특약 문구만 골라주세요.",
        "규칙:",
        "1. 질문자의 질문, 상담 요청, 배경 설명은 제외합니다.",
        "2. 계약서에 실제로 적혀 있을 만한 약정 문구만 extracted_clauses에 넣습니다.",
        "3. 원문에 없는 새로운 내용을 만들지 않습니다.",
        "4. 문장을 다듬더라도 의미를 바꾸지 않습니다.",
        "5. 실제 특약 문구가 없으면 extracted_clauses에 빈 리스트를 넣습니다.",
        "",
        "special_clauses:",
    ]

    for index, clause in enumerate(special_clauses, start=1):
        lines.append(f"{index}. {clause}")

    return "\n".join(lines)


def extract_clauses_from_special_clauses_with_llm(special_clauses):
    """1차 rule-based special_clauses에서 실제 계약서 특약 문구만 LLM으로 추출한다."""
    from shared.llm.gemini_client import gemini_client

    prompt = build_clause_postprocess_prompt(special_clauses)
    result = gemini_client.generate(
        prompt,
        response_schema=ExtractedClausesResult,
    )
    return result.extracted_clauses


def build_answer_analysis_prompt(question_record, answer_item):
    """DB 적재용 답변 분석 프롬프트를 만든다."""
    return "\n".join(
        [
            "다음 법률 상담 QA를 DB 스키마 필드에 맞게 분석하세요.",
            "출력 필드는 dispute_background, lawyer_conclusion, "
            "lawyer_reasoning, action_checklist 네 개만 사용합니다.",
            "",
            "[질문 제목]",
            str(question_record.get("question_title", "")),
            "",
            "[질문 본문]",
            str(question_record.get("question_body", "")),
            "",
            "[변호사 이름]",
            str(answer_item.get("lawyer", "")),
            "",
            "[변호사 답변]",
            str(answer_item.get("answer", "")),
        ]
    )


def analyze_answer_for_db_with_llm(question_record, answer_item):
    """질문과 답변 하나를 LLM으로 분석해 DB 적재용 확장 필드를 만든다."""
    from shared.llm.gemini_client import gemini_client

    prompt = build_answer_analysis_prompt(question_record, answer_item)
    return gemini_client.generate(
        prompt,
        system_instruction=LAWTALK_DB_ANALYSIS_SYSTEM_PROMPT,
        response_schema=LawtalkAnswerAnalysis,
    )


def parse_prediction_key(key):
    """Batch prediction key에서 question_id와 답변 순번을 읽는다."""
    match = re.fullmatch(r"q(\d+)_a(\d+)", str(key))
    if match is None:
        raise ValueError(f"invalid prediction key: {key!r}")

    return int(match.group(1)), int(match.group(2))


def extract_analysis_from_prediction(prediction):
    """Batch prediction 한 줄에서 DB 적재용 답변 분석 결과를 읽는다."""
    try:
        text = prediction["response"]["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        key = prediction.get("key")
        raise ValueError(f"prediction has no response text: {key!r}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        key = prediction.get("key")
        raise ValueError(f"prediction response is not valid JSON: {key!r}") from exc

    return LawtalkAnswerAnalysis(**payload)


def build_question_row(record):
    """원본 질문 레코드를 questions 테이블 형태로 바꾼다."""
    return {
        "id": record.get("index"),
        "title": record.get("question_title"),
        "body": record.get("question_body"),
        "tags": record.get("tags", []),
        "written_at": record.get("question_written_at_raw"),
        "embedding": None,
    }


def build_answer_row(answer_id, question_id, answer_item, analysis):
    """원본 답변과 LLM 분석 결과를 answers 테이블 형태로 바꾼다."""
    return {
        "id": answer_id,
        "question_id": question_id,
        "lawyer_name": answer_item.get("lawyer"),
        "answer_body": dict(answer_item),
        "written_at": answer_item.get("written_at_raw"),
        "dispute_background": analysis.dispute_background,
        "lawyer_conclusion": analysis.lawyer_conclusion,
        "lawyer_reasoning": analysis.lawyer_reasoning,
        "action_checklist": analysis.action_checklist,
    }


def iter_answer_items(records):
    """원본 질문 목록에서 답변을 안정적인 순서와 ID로 순회한다."""
    answer_id = 1

    for record in records:
        question_id = record.get("index")

        for answer_item in record.get("all_answers", []):
            yield answer_id, question_id, record, answer_item
            answer_id = answer_id + 1


def iter_answer_items_with_position(records):
    """원본 질문 목록에서 답변 ID와 질문 안 답변 순번을 함께 순회한다."""
    answer_id = 1

    for record in records:
        question_id = record.get("index")

        for answer_position, answer_item in enumerate(
            record.get("all_answers", []),
            start=1,
        ):
            yield answer_id, question_id, answer_position, record, answer_item
            answer_id = answer_id + 1


def process_record(record):
    """로톡 QA 레코드 하나를 필터링하고 출력 형식으로 바꾼다."""
    question_body = record.get("question_body", "")
    special_clauses = extract_special_clauses(question_body)

    if len(special_clauses) == 0:
        return None

    law_references = []
    precedent_numbers = []

    for answer_item in record.get("all_answers", []):
        answer_text = answer_item.get("answer", "")

        for law_reference in extract_law_references(answer_text):
            add_unique(law_references, law_reference)

        for precedent_number in extract_precedent_numbers(answer_text):
            add_unique(precedent_numbers, precedent_number)

    if len(law_references) == 0 and len(precedent_numbers) == 0:
        return None

    return {
        "index": record.get("index"),
        "special_clauses": special_clauses,
        "law_references": law_references,
        "precedent_numbers": precedent_numbers,
    }


def read_records_from_file(file_path):
    """JSON 파일 하나에서 레코드 목록을 읽는다."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(file_path, records):
    """레코드 목록을 JSON 파일로 저장한다."""
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def read_jsonl(file_path):
    """JSONL 파일을 한 줄씩 읽어 dict 리스트로 반환한다."""
    records = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if line == "":
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc

    return records


def prepare_db_ready_records_from_predictions(
    input_file="data/raw/lawtalk_QA_Context.json",
    prediction_jsonl=(
        "data/raw/"
        "qa-data-processed_prediction-model-2026-05-05T23_38_04.717801Z_predictions.jsonl"
    ),
    output_file=None,
):
    """원본 QA와 Batch prediction 결과를 합쳐 전체 DB 적재용 JSON을 만든다."""
    input_path = Path(input_file)
    prediction_path = Path(prediction_jsonl)
    output_path = Path(
        output_file
        or "data/lawtalk_qa_preprocessed/lawtalk_qa_db_ready_from_predictions.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_records = read_records_from_file(input_path)
    question_rows = [build_question_row(record) for record in source_records]
    prediction_rows = read_jsonl(prediction_path)

    predictions_by_key = {}
    failed_prediction_count = 0
    for prediction in prediction_rows:
        try:
            question_id, answer_position = parse_prediction_key(prediction.get("key"))
            predictions_by_key[(question_id, answer_position)] = prediction
        except ValueError:
            failed_prediction_count = failed_prediction_count + 1

    answer_rows = []
    matched_count = 0
    missing_prediction_count = 0

    for (
        answer_id,
        question_id,
        answer_position,
        question_record,
        answer_item,
    ) in iter_answer_items_with_position(source_records):
        prediction = predictions_by_key.get((question_id, answer_position))
        if prediction is None:
            missing_prediction_count = missing_prediction_count + 1
            continue

        try:
            analysis = extract_analysis_from_prediction(prediction)
        except ValueError:
            failed_prediction_count = failed_prediction_count + 1
            continue

        answer_rows.append(
            build_answer_row(
                answer_id,
                question_id,
                answer_item,
                analysis,
            )
        )
        matched_count = matched_count + 1

    write_json(
        output_path,
        {
            "questions": question_rows,
            "answers": answer_rows,
        },
    )

    return {
        "total_questions": len(question_rows),
        "total_answers": sum(
            len(record.get("all_answers", [])) for record in source_records
        ),
        "prediction_count": len(prediction_rows),
        "matched_count": matched_count,
        "missing_prediction_count": missing_prediction_count,
        "failed_prediction_count": failed_prediction_count,
    }


def load_rule_based_filtered_records(input_dir):
    """1차 rule-based 결과 세 파일을 읽고 source 메타데이터를 붙인다."""
    input_path = Path(input_dir)
    records = []

    for source_file in RULE_BASED_FILTERED_FILES:
        source_path = input_path / source_file
        source_records = read_records_from_file(source_path)

        for source_position, record in enumerate(source_records, start=1):
            merged_record = dict(record)
            merged_record["source_file"] = source_file
            merged_record["source_position"] = source_position
            records.append(merged_record)

    return records


def process_rule_based_filtered_clauses(
    input_dir="data/lawtalk_qa_filtered",
    output_file=None,
    clause_extractor=None,
    debug=False,
    limit=None,
):
    """1차 rule-based 결과를 LLM으로 후처리하고 매 레코드마다 결과 파일을 갱신한다."""
    input_path = Path(input_dir)
    output_path = (
        Path(output_file)
        if output_file is not None
        else input_path / "qa_clauses_processed.json"
    )
    extractor = clause_extractor or extract_clauses_from_special_clauses_with_llm
    source_records = load_rule_based_filtered_records(input_path)

    if output_path.exists():
        processed_records = read_records_from_file(output_path)
    else:
        processed_records = []

    processed_keys = {
        (record.get("source_file"), record.get("source_position"))
        for record in processed_records
    }
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    debug_log(
        debug,
        (
            f"clause_postprocess_start input_dir={input_path} "
            f"output_file={output_path} limit={limit}"
        ),
    )

    for record in source_records:
        key = (record["source_file"], record["source_position"])

        if key in processed_keys:
            skipped_count = skipped_count + 1
            debug_log(
                debug,
                (
                    f"clause_postprocess_skip source_file={record['source_file']} "
                    f"source_position={record['source_position']} reason=already_processed"
                ),
            )
            continue

        if limit is not None and processed_count >= limit:
            debug_log(debug, f"clause_postprocess_limit_reached limit={limit}")
            break

        special_clauses = record.get("special_clauses", [])
        debug_log(
            debug,
            (
                f"clause_postprocess_start_record source_file={record['source_file']} "
                f"source_position={record['source_position']} "
                f"special_clause_count={len(special_clauses)}"
            ),
        )

        output_record = dict(record)

        try:
            output_record["extracted_clauses"] = extractor(special_clauses)
            output_record["clause_processing_status"] = "success"
        except Exception as exc:
            output_record["extracted_clauses"] = []
            output_record["clause_processing_status"] = "failed"
            output_record["clause_processing_error"] = str(exc)
            failed_count = failed_count + 1

        processed_records.append(output_record)
        processed_keys.add(key)
        processed_count = processed_count + 1
        write_json(output_path, processed_records)

        debug_log(
            debug,
            (
                f"clause_postprocess_done_record source_file={record['source_file']} "
                f"source_position={record['source_position']} "
                f"status={output_record['clause_processing_status']} "
                f"extracted_count={len(output_record['extracted_clauses'])}"
            ),
        )

    debug_log(debug, "clause_postprocess_done")

    return {
        "total_records": len(source_records),
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
    }


def prepare_db_ready_records(
    input_file="data/raw/lawtalk_QA_Context.json",
    output_file=None,
    answer_analyzer=None,
    debug=False,
    limit=None,
):
    """로톡 QA 원본을 questions/answers DB 적재용 JSON으로 전처리한다."""
    input_path = Path(input_file)
    output_path = Path(
        output_file or "data/lawtalk_qa_preprocessed/lawtalk_qa_db_ready.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer = answer_analyzer or analyze_answer_for_db_with_llm
    source_records = read_records_from_file(input_path)
    question_rows = [build_question_row(record) for record in source_records]
    answer_items = list(iter_answer_items(source_records))

    if output_path.exists():
        output_records = read_records_from_file(output_path)
        processed_answers = output_records.get("answers", [])
    else:
        processed_answers = []

    processed_answer_ids = {answer.get("id") for answer in processed_answers}
    output_records = {
        "questions": question_rows,
        "answers": processed_answers,
    }

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    write_json(output_path, output_records)
    debug_log(
        debug,
        (
            f"db_prepare_start input_file={input_path} "
            f"output_file={output_path} limit={limit}"
        ),
    )

    for answer_id, question_id, question_record, answer_item in answer_items:
        if answer_id in processed_answer_ids:
            skipped_count = skipped_count + 1
            debug_log(
                debug,
                f"db_prepare_skip answer_id={answer_id} reason=already_processed",
            )
            continue

        if limit is not None and processed_count >= limit:
            debug_log(debug, f"db_prepare_limit_reached limit={limit}")
            break

        debug_log(
            debug,
            (
                f"db_prepare_start_answer answer_id={answer_id} "
                f"question_id={question_id}"
            ),
        )

        try:
            analysis = analyzer(question_record, answer_item)
            answer_row = build_answer_row(
                answer_id,
                question_id,
                answer_item,
                analysis,
            )
            output_records["answers"].append(answer_row)
            processed_answer_ids.add(answer_id)
        except Exception as exc:
            failed_count = failed_count + 1
            debug_log(
                debug,
                (
                    f"db_prepare_failed answer_id={answer_id} "
                    f"question_id={question_id} error={exc}"
                ),
            )

        processed_count = processed_count + 1
        write_json(output_path, output_records)
        debug_log(
            debug,
            (
                f"db_prepare_done_answer answer_id={answer_id} "
                f"question_id={question_id}"
            ),
        )

    debug_log(debug, "db_prepare_done")

    return {
        "total_questions": len(question_rows),
        "total_answers": len(answer_items),
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
    }


def run(input_dir="data/lawtalk_qa", output_dir="data/lawtalk_qa_filtered"):
    """입력 디렉터리의 로톡 QA JSON 파일들을 처리한다."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    both = []
    law_only = []
    precedent_only = []
    total_records = 0

    for file_path in sorted(input_path.glob("*.json")):
        records = read_records_from_file(file_path)

        for record in records:
            total_records = total_records + 1
            processed_record = process_record(record)

            if processed_record is None:
                continue

            has_law = len(processed_record["law_references"]) > 0
            has_precedent = len(processed_record["precedent_numbers"]) > 0

            if has_law and has_precedent:
                both.append(processed_record)
            elif has_law:
                law_only.append(processed_record)
            else:
                precedent_only.append(processed_record)

    write_json(output_path / "qa_both.json", both)
    write_json(output_path / "qa_law_only.json", law_only)
    write_json(output_path / "qa_precedent_only.json", precedent_only)

    filtered_count = len(both) + len(law_only) + len(precedent_only)

    print(f"총 {total_records}건 중 {filtered_count}건 필터링 완료")
    print(f"  법령 + 판례 모두: {len(both)}건 → qa_both.json")
    print(f"  법령만: {len(law_only)}건 → qa_law_only.json")
    print(f"  판례만: {len(precedent_only)}건 → qa_precedent_only.json")

    return {
        "total_records": total_records,
        "both_count": len(both),
        "law_only_count": len(law_only),
        "precedent_only_count": len(precedent_only),
    }
