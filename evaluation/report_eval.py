"""
LLM-as-a-Judge: report_generator.py 출력 품질 평가
- 특약 관계 분석 평가: Coherence (1-5, G-Eval) — 쌍별 평가 후 평균 집계
- 체크리스트 평가:     Consistency (1-5, G-Eval) — 전체 체크리스트 단위

사용법:
    # 단일 파일
    python evaluation/judge_eval.py \
        --report output/test_rag_104_report.json \
        --output output/test_rag_104_judge.json
    # 폴더 전체
    python evaluation/jreport_eval.py --report output/
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import vertexai
from dotenv import load_dotenv
from vertexai.generative_models import GenerationConfig, GenerativeModel

# ── 프로젝트 루트 ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
vertexai.init(project=PROJECT_ID, location=LOCATION)

MODEL_NAME = "gemini-2.5-pro"

# ── 프롬프트 ──────────────────────────────────────────────────────
RELATION_JUDGE_PROMPT = """\
당신은 주택임대차 계약 분석 시스템이 생성한 특약 간 관계 설명의 품질을 평가합니다.

[Evaluation Criteria]
Coherence (1-5): 생성된 관계 설명이 두 특약 원문에 있는 구체적 표현을 근거로
논리적으로 도출되었는가.
관계 설명은 두 특약이 동시에 적용될 때 발생하는 사항을 문장에서 문장으로
자연스럽게 이어지는 일관된 서술로 구성해야 합니다.
원문과 무관한 내용이 삽입되거나 근거 없이 결론만 제시될수록 낮은 점수.

[Evaluation Steps]
1. 분석 대상 특약과 연관 특약 원문을 주의 깊게 읽고 각각의 핵심 내용을 파악한다.
2. 생성된 관계 설명을 읽고, 두 특약 원문에 실제로 존재하는 표현에 근거하여
   논리가 전개되는지 확인한다.
3. 문장 간 흐름이 일관성 있는지, 원문에 없는 내용이 삽입되지 않았는지 확인한다.
4. 1-5 점수를 부여한다 (1=매우 낮음, 5=매우 높음).

[분석 대상 특약]
{clause_text}

[연관 특약]
{related_clause_text}

[생성된 관계 설명]
{relation}

JSON 형식으로만 응답:
{{"coherence": {{"score": int, "reason": str}}}}"""

CHECKLIST_JUDGE_PROMPT = """\
당신은 주택임대차 계약 분석 시스템이 생성한 계약 전 체크리스트의 품질을 평가합니다.

[Evaluation Criteria]
Consistency (1-5): 체크리스트의 각 항목(description)이 입력된 특약 원문 및
basis에 명시된 법령에서 도출 가능한 내용만을 포함하는가.
원문에 없는 사실을 생성하거나(hallucination), basis에 명시되지 않은 법령·판례를
근거로 주장하는 항목이 없을수록 높은 점수.

[Evaluation Steps]
1. 입력 특약 목록과 특약별 분석 결과(basis)를 주의 깊게 읽고 원문의 주요 사실을 파악한다.
2. 체크리스트의 각 항목을 읽고, description 내용이 특약 원문이나 basis 법령에서
   실제로 뒷받침되는지 확인한다.
3. 원문에 없는 사실이 포함되거나 basis와 다른 내용이 서술된 항목에 감점한다.
4. 1-5 점수를 부여한다 (1=매우 낮음, 5=매우 높음).

[입력 특약 목록]
{target_terms}

[특약별 분석 결과 (basis 근거)]
{clause_results}

[평가 대상 체크리스트]
{contract_checklist}

JSON 형식으로만 응답:
{{"consistency": {{"score": int, "reason": str}}}}"""


# ── LLM 호출 ─────────────────────────────────────────────────────
def call_judge(prompt: str) -> dict | None:
    model = GenerativeModel(MODEL_NAME)
    config = GenerationConfig(temperature=0.0, response_mime_type="application/json")
    try:
        response = model.generate_content(prompt, generation_config=config)
        text = response.text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception as e:
        print(f"  [Judge LLM 오류] {e}")
        return None


# ── 특약 관계 분석 평가 (Coherence) ─────────────────────────────
def evaluate_relations(clause_results: list[dict]) -> tuple[list[dict], float | None]:
    """
    related_clauses가 있는 모든 쌍에 대해 Coherence 평가.
    Returns: (쌍별 상세 결과 리스트, 집계 평균 점수)
    """
    pair_results = []
    pair_scores  = []

    for clause in clause_results:
        clause_id   = clause.get("clause_id", "")
        clause_text = clause.get("clause_text", "")

        for rel in (clause.get("related_clauses") or []):
            related_text = rel.get("clause_text", "")
            relation     = rel.get("relation", "")
            related_id   = rel.get("clause_id", "")

            if not relation:
                continue

            print(f"  [Coherence] {clause_id} ↔ {related_id}")
            prompt = RELATION_JUDGE_PROMPT.format(
                clause_text=clause_text,
                related_clause_text=related_text,
                relation=relation,
            )
            result = call_judge(prompt)
            score  = (result or {}).get("coherence", {}).get("score")
            reason = (result or {}).get("coherence", {}).get("reason")

            if isinstance(score, (int, float)):
                pair_scores.append(score)

            pair_results.append({
                "clause_id":           clause_id,
                "clause_text":         clause_text,
                "related_clause_id":   related_id,
                "related_clause_text": related_text,
                "relation":            relation,
                "score":               score,
                "reason":              reason,
            })

    coherence_score = (
        round(sum(pair_scores) / len(pair_scores), 2) if pair_scores else None
    )
    return pair_results, coherence_score


# ── 체크리스트 평가 (Consistency) ────────────────────────────────
def evaluate_checklist(
    target_terms: list[str],
    clause_results: list[dict],
    contract_checklist: list[dict],
) -> dict | None:
    if not contract_checklist:
        return None

    prompt = CHECKLIST_JUDGE_PROMPT.format(
        target_terms="\n".join(f"특약{i+1}: {t}" for i, t in enumerate(target_terms)),
        clause_results=json.dumps(clause_results, ensure_ascii=False, indent=2),
        contract_checklist=json.dumps(contract_checklist, ensure_ascii=False, indent=2),
    )
    result = call_judge(prompt)
    return (result or {}).get("consistency")


# ── 단일 파일 평가 ────────────────────────────────────────────────
def run_single(report_path: Path, output_path: str) -> None:
    print(f"\n[judge] 입력: {report_path.name}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    clause_results     = report.get("clause_results")     or []
    contract_checklist = report.get("contract_checklist") or []
    target_terms       = [cr.get("clause_text", "") for cr in clause_results]

    print(f"  대상 특약 수: {len(target_terms)}")
    print(f"  체크리스트 항목 수: {len(contract_checklist)}")

    print("\n  [1] Coherence 평가 중...")
    pair_results, coherence_score = evaluate_relations(clause_results)
    if pair_results:
        print(f"    쌍 수: {len(pair_results)}, 집계 점수: {coherence_score}")
    else:
        print("    평가 가능한 특약 쌍 없음")

    print("\n  [2] Consistency 평가 중...")
    consistency_result = evaluate_checklist(target_terms, clause_results, contract_checklist)
    if consistency_result:
        print(f"    consistency={consistency_result.get('score', '?')}")
    else:
        print("    체크리스트 없음 — 건너뜀")

    result = {
        "coherence_results":  pair_results,
        "coherence_score":    coherence_score,
        "consistency_result": consistency_result,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n  완료. 결과 저장: {out}")
    print(f"  Coherence: {coherence_score} | Consistency: {(consistency_result or {}).get('score')}")


# ── 메인 ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge for report_generator output")
    parser.add_argument(
        "--report",
        required=True,
        help="*_report.json 파일 경로 또는 해당 파일들이 있는 폴더",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="출력 경로 또는 폴더. 미지정 시 report와 같은 폴더에 *_judge.json으로 저장",
    )
    args = parser.parse_args()

    report_path = Path(args.report)

    if report_path.is_dir():
        files = sorted(report_path.glob("*_report.json"))
        if not files:
            print(f"[경고] {report_path} 에서 *_report.json 파일을 찾을 수 없습니다.")
            return
        print(f"[폴더 모드] {len(files)}개 파일 처리 시작")
        for f in files:
            judge_stem = f.stem.replace("_report", "_judge") if "_report" in f.stem else f"{f.stem}_judge"
            out = str(Path(args.output) / f"{judge_stem}.json") if args.output else None
            run_single(f, out or str(f.parent / f"{judge_stem}.json"))
    else:
        judge_stem = report_path.stem.replace("_report", "_judge") if "_report" in report_path.stem else f"{report_path.stem}_judge"
        out = args.output or str(report_path.parent / f"{judge_stem}.json")
        run_single(report_path, out)


if __name__ == "__main__":
    main()
