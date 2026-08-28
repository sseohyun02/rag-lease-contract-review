"""변호사 답변에서 '실제로 근거로 든 법령'을 gt로 만든다 (재설계본 v2).

기존 qa_3의 문제 (진단으로 확인됨)
----
- 기존: '특약을 직접 규율하는 조문'을 기준으로 뽑고, 손해배상/취소/형사/절차를
  '곁가지'라며 제외했다. 그 결과 변호사가 실제 든 법령과 크게 어긋났다
  (변호사 명시 인용 커버율 ~21%).
- 원인: 정답 근거를 '변호사 답변'이 아니라 'LLM이 특약 보고 판단한 관련성'에 뒀다.

재설계 원칙 (단 하나)
----
    정답 = 변호사가 답변에서 '실제로 근거로 든 법령' 전부.
    - LLM은 '곁가지인지' 판단하지 않는다. 변호사가 들었으면 넣는다.
    - LLM은 변호사가 '안 든' 법을 특약 보고 상상해서 넣지 않는다 (환각 방지).
    - 즉 LLM의 역할은 '관련성 판단'이 아니라 '변호사가 이 법을 근거로 들었는가' 판정뿐.

파이프라인
----
1. 명시 인용(자동 정답): qa_1.extract_law_references 로 답변에서 조문번호가 명시된
   법령을 뽑아 무조건 정답에 넣는다 (LLM 안 거침). 단 시행령 등 TARGET_LAWS 밖은 못 잡음 -> 2단계가 보완.
2. 서술 -> 조문 변환: 답변에서 '법령 내용을 풀어 설명했지만 조문번호를 안 쓴 부분'의
   법리를 개념으로 추출한다 (특약 관련성이 아니라 '변호사가 설명한 법리'가 기준).
3. 각 개념을 law_child BM25 로 검색해 실제 조문 후보를 얻는다.
4. 확인: 후보 조문이 '변호사 답변이 이 법리를 실제로 설명/근거로 들었는지' 판정한다
   (특약을 규율하는지 X, 곁가지인지 X). 부합하면 채택.
5. 정답 = (1의 명시 인용) ∪ (4에서 채택된 조문).

주의
- 조문 번호를 지어내지 않는다. 후보는 항상 BM25로 실재하는 law_child 에서만 온다.
- reason 을 남겨 사람이 검수한다.
- 명시 인용은 '법령명_제N조' 문자열이라, gt(clause_key: 법령명_제N조_제M항)와 형태가 다르다.
  -> 명시 인용은 'article 단위'로 gt에 넣고, 검수 시 항 단위가 필요하면 사람이 좁힌다.
  (원 파이프라인이 조 단위로도 채점되므로 article 단위 저장이 안전하다.)

실행: python data/processors/qa_3_build_gt_v2.py --limit 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.retrieval.bm25_retrieval import load_law_child_from_db, tokenize
from shared.llm.gemini_client import gemini_client, LLMError
from data.processors.qa_1_process_raw import extract_law_references  # 명시 인용 추출 재사용

QA_PROCESSED_PATH = Path("data/lawtalk_qa_filtered/qa_processed.json")
RAW_DIR = Path("data/qa_raw")
OUTPUT_PATH = Path("evaluation/gt_from_reasoning.json")

BM25_TOP_N = 5
MAX_CANDIDATES = 20
MAX_RETRIES = 5
RETRY_DELAY = 20


class ConceptExtraction(BaseModel):
    concepts: list[str]


class ConfirmJudgment(BaseModel):
    concept: str
    clause_key: str
    cited_by_lawyer: bool   # 변호사 답변이 이 법리를 실제로 설명/근거로 들었는가
    reason: str


class ConfirmResult(BaseModel):
    judgments: list[ConfirmJudgment]


# 1) 추출: '변호사가 답변에서 근거로 든 법리' 전부. 곁가지 제외하지 않는다.
EXTRACT_PROMPT = """너는 한국 법률 문서 분석 전문가다.
아래에 '특약'과 그에 대한 '변호사 답변 전문'이 주어진다.
변호사 답변은 조문 번호를 거의 쓰지 않고, 법리(법적 근거)를 자연어로 풀어서 설명한다.

과제: 변호사가 이 답변에서 '법적 근거로 실제로 설명하거나 언급한 법리'를 모두 개념으로 뽑아라.

핵심 원칙:
- 기준은 '특약과 관련 있는가'가 아니라 '변호사가 이 법리를 답변에서 근거로 들었는가'다.
- 변호사가 근거로 든 것이면 손해배상, 계약 취소, 절차(소송/집행), 설명의무 등 무엇이든 포함하라.
  (어떤 것도 '곁가지'라고 임의로 빼지 마라. 변호사가 들었으면 넣는다.)
- 반대로, 변호사가 답변에서 언급하지 않은 법리를 특약만 보고 상상해서 추가하지 마라.
- 조문 번호는 추정하지 말고, 각 법리를 간결한 법률 용어 문구로 서술하라
  (예: '차임 증액 5% 상한', '묵시적 갱신', '계약갱신요구권', '손해배상액의 예정', '기망에 의한 취소').
- 변호사가 조문 번호를 직접 쓴 경우, 그 쟁점도 개념으로 함께 넣어라(뒤 단계에서 대조된다).
- concepts 배열로 반환하라. 개수 제한은 없으나 변호사가 실제 든 것에 한정하라."""

# 법명만 명시된 경우, 그 법 안에서 조·항을 찾도록 주는 힌트 지시 (동적으로 붙임)
LAW_HINT_TEMPLATE = """

[추가 힌트]
변호사가 아래 법령을 근거로 언급했으나 조문 번호는 쓰지 않았다: {laws}
이 법령들 안에서 변호사가 '실제로 설명한 법리'에 해당하는 조·항이 무엇인지
답변 내용을 근거로 개념에 반드시 반영하라 (법령명만 나왔다고 아무 조문이나 넣지 말고,
답변이 설명한 내용과 맞는 조문의 법리를 개념으로 서술하라)."""

# 4) 확인: 후보 조문이 '변호사 답변이 실제로 든 법리'인지. 특약 규율 여부/곁가지 판단 안 함.
CONFIRM_PROMPT = """너는 변호사 답변과 법조문을 대조하는 검증자다.
아래에 '변호사 답변 전문', 답변에서 뽑은 '법리 쟁점', 그리고 '후보 조문 원문'들이 주어진다.
각 후보 조문에 대해, '변호사가 답변에서 이 조문의 법리를 실제로 설명하거나 근거로 들었는가'를 판정하라
(cited_by_lawyer: true/false).

기준:
- 변호사 답변의 서술이 이 조문의 내용과 실제로 부합하면 true
  (변호사가 조문 번호를 안 썼어도, 그 내용을 풀어 설명했으면 true).
- 변호사 답변에 근거가 없는데 특약과 관련 있어 보인다는 이유만으로는 false.
- 즉 '특약을 규율하는지'가 아니라 '변호사가 이 법리를 답변에서 들었는지'가 유일한 기준이다.
- 곁가지인지 아닌지는 판단하지 마라. 변호사가 들었으면 true.
- 애매하면(변호사 답변에 근거가 불명확하면) false.
- reason 에 답변의 어느 서술과 부합/불일치하는지 한 줄로 남겨라 (사람이 검수한다)."""


def _retry(**kwargs):
    delay = RETRY_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return gemini_client.generate(**kwargs)
        except LLMError as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    [재시도 {attempt}] {str(exc)[:80]} - {delay}초 대기")
            time.sleep(delay)
            delay *= 2


def _parse(model_cls, result):
    if isinstance(result, model_cls):
        return result
    if isinstance(result, str):
        return model_cls.model_validate_json(result)
    return model_cls.model_validate(result)


def explicit_to_article_key(ref: str) -> str:
    """'민법 제398조' / '민법 제398조의2' -> '민법_제398조' (article 단위, 항 없음)."""
    return re.sub(r"\s*제", "_제", ref.strip(), count=1)


def split_explicit_refs(refs: list[str]) -> tuple[list[str], list[str]]:
    """명시 인용을 (조문번호 있음, 법명만)으로 분리.

    - 조문번호 있음(예: '민법 제565조'): article_key 로 변환해 자동 정답.
    - 법명만(예: '주택임대차보호법'): gt 에 넣지 않고, 추출 힌트로만 사용
      (변호사가 이 법을 근거로 들었으니 어느 조·항인지 답변에서 찾게 한다).
    """
    with_article, law_only = [], []
    for r in refs:
        if "조" in r and "제" in r:            # 조문번호가 있는 경우
            with_article.append(explicit_to_article_key(r))
        else:                                   # 법령명만 있는 경우
            law_only.append(r.strip())
    return sorted(set(with_article)), sorted(set(law_only))


def load_raw_by_url():
    raw = {}
    for fp in sorted(RAW_DIR.glob("posts_index_*.json")):
        for rec in json.loads(fp.read_text(encoding="utf-8")):
            if rec.get("url"):
                raw[rec["url"]] = rec
    return raw


def build_bm25():
    df = load_law_child_from_db()
    docs = [
        {"clause_key": str(r["clause_key"]),
         "text": str(r.get("bm25_target") or r.get("child_text") or "")}
        for _, r in df.iterrows()
    ]
    return docs, BM25Okapi([tokenize(d["text"]) for d in docs])


def search(concept, docs, bm25, top_n):
    toks = tokenize(concept)
    if not toks:
        return []
    scores = bm25.get_scores(toks)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [{"clause_key": docs[i]["clause_key"], "text": docs[i]["text"]}
            for i in idx if scores[i] > 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--eval-only", action="store_true",
                        help="eval_set_tune/test.json 에 실제 포함된 url 만 처리(정확히 그 케이스만 재생성)")
    parser.add_argument("--eval-paths", nargs="+",
                        default=["evaluation/eval_set_tune.json", "evaluation/eval_set_test.json"])
    args = parser.parse_args()

    qa = json.loads(QA_PROCESSED_PATH.read_text(encoding="utf-8"))
    qa = [r for r in qa
          if r.get("url") and r.get("clause_processing_status") == "success"
          and len(r.get("extracted_clauses") or []) == 1]

    # --eval-only: eval_set 에 실제 들어간 url 로 한정
    if args.eval_only:
        eval_urls = set()
        for p in args.eval_paths:
            fp = Path(p)
            if not fp.exists():
                print(f"  (경고) eval 파일 없음: {p}")
                continue
            for c in json.loads(fp.read_text(encoding="utf-8")):
                u = (c.get("meta", {}) or {}).get("url") or c.get("source_id")
                if u:
                    eval_urls.add(u)
        before = len(qa)
        qa = [r for r in qa if r["url"] in eval_urls]
        print(f"--eval-only: eval url {len(eval_urls)}개 기준 필터 {before} -> {len(qa)}건")

    qa = qa[: args.limit]
    raw_by_url = load_raw_by_url()

    if OUTPUT_PATH.exists():
        output = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        done_urls = {r["url"] for r in output if r.get("url")}
        print(f"기존 결과 {len(output)}건 로드 - 이어서 진행\n")
    else:
        output = []
        done_urls = set()

    print("BM25 인덱스 구축 중...")
    docs, bm25 = build_bm25()
    print(f"완료 ({len(docs)}건)\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for rec in qa:
        if rec["url"] in done_urls:
            skipped += 1
            continue
        raw = raw_by_url.get(rec["url"])
        if not raw:
            continue
        answer = "\n\n".join(a.get("answer", "") for a in raw.get("all_answers", []))
        if not answer.strip():
            continue
        clause = rec["extracted_clauses"][0]

        # 1) 명시 인용 (자동 정답) - qa_1 재사용, 조문있음/법명만 분리
        explicit_refs = extract_law_references(answer)
        explicit_keys, law_only_hints = split_explicit_refs(explicit_refs)

        # 2) 서술 -> 법리 개념 추출 (변호사가 든 것 전부, 곁가지 제외 안 함)
        #    법명만 명시된 경우는 힌트로 넘겨 조·항을 답변에서 찾게 한다.
        extract_instruction = EXTRACT_PROMPT
        if law_only_hints:
            extract_instruction += LAW_HINT_TEMPLATE.format(laws=", ".join(law_only_hints))
        ext = _retry(
            contents=f"[특약]\n{clause}\n\n[변호사 답변 전문]\n{answer}",
            system_instruction=extract_instruction,
            response_schema=ConceptExtraction,
        )
        ext = _parse(ConceptExtraction, ext)
        print(f"[{rec.get('index')}] 특약: {clause[:50]}")
        print(f"   명시 인용(조문 있음, 자동정답): {explicit_keys}")
        print(f"   법명만 명시(힌트로 사용): {law_only_hints}")
        print(f"   추출 법리 쟁점: {ext.concepts}")

        record_out = {
            "index": rec.get("index"), "url": rec["url"], "clause": clause,
            "explicit_refs": explicit_keys,           # 조문번호 있는 명시 인용(자동 정답)
            "law_only_hints": law_only_hints,         # 법명만 명시(힌트, gt 아님)
            "concepts": ext.concepts,
            "confirmed_laws": [],                      # 서술->조문 확인 통과분
            "gt_laws": [],                             # 최종 = explicit ∪ confirmed
            "detail": [],
        }

        confirmed = []
        if ext.concepts:
            # 3) BM25 후보
            cands, seen = [], set()
            for c in ext.concepts:
                for cd in search(c, docs, bm25, BM25_TOP_N):
                    if cd["clause_key"] not in seen:
                        seen.add(cd["clause_key"])
                        cands.append({"concept": c, **cd})
            cands = cands[:MAX_CANDIDATES]

            # 4) 확인: 변호사가 이 법리를 실제로 들었는가
            lines = [f"[변호사 답변 전문]\n{answer}\n",
                     f"[답변에서 뽑은 법리 쟁점]\n{ext.concepts}\n",
                     "[후보 조문]"]
            for cd in cands:
                lines.append(f"- clause_key: {cd['clause_key']}\n  원문: {cd['text']}\n")
            conf = _retry(
                contents="\n".join(lines),
                system_instruction=CONFIRM_PROMPT,
                response_schema=ConfirmResult,
            )
            conf = _parse(ConfirmResult, conf)
            confirmed = [j.clause_key for j in conf.judgments if j.cited_by_lawyer]
            record_out["detail"] = [j.model_dump() for j in conf.judgments]

        record_out["confirmed_laws"] = sorted(set(confirmed))
        # 5) 최종 gt = 명시 인용(article) ∪ 확인된 조문(clause_key)
        #    안전장치: 조문번호(_제N조)가 없는 법령명 단독 항목은 gt에서 제외
        merged = set(explicit_keys) | set(confirmed)
        gt_final = sorted(k for k in merged if re.search(r"_제\d+조", k))
        record_out["gt_laws"] = gt_final
        print(f"   -> 확인된 조문: {record_out['confirmed_laws']}")
        print(f"   -> 최종 gt: {record_out['gt_laws']}\n")

        output.append(record_out)
        done_urls.add(rec["url"])
        OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if skipped:
        print(f"이미 처리되어 건너뜀: {skipped}건")
    print(f"저장: {OUTPUT_PATH} (총 {len(output)}건)")
    print("\n검수 포인트:")
    print("  - explicit_refs: 변호사가 조문번호를 직접 쓴 것(article 단위). 신뢰 높음.")
    print("  - confirmed_laws: 서술->조문 변환분. detail의 reason으로 변호사 답변 부합 검수.")
    print("  - 명시 인용은 조 단위라 항 단위 정밀화가 필요하면 검수 때 좁힌다.")


if __name__ == "__main__":
    main()