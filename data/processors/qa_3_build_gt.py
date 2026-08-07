"""변호사 답변의 '법리 설명'을 근거로 특약의 정답 조문을 만든다 (방식 1).

이전 augment_gt_laws.py의 실패 원인과 그 해결
----
이전 실패: (1) '특약 문장'만 보고 조문을 찾아 근거가 빈약했다.
           (2) 전체(수백 건)를 한 번에 돌려 토큰 비용이 컸다.
이번 해결: (1) 근거를 '변호사 답변 전문'(법리 설명이 풍부)으로 바꾼다.
               실제 답변을 보면 조문 번호는 거의 없지만, "5% 증액 상한,
               합의 전제, 갱신요구권" 처럼 조문으로 직결되는 법리를
               자연어로 충실히 설명한다. 이 설명을 근거로 삼는다.
           (2) --limit로 소수만 먼저 돌려 품질을 확인한 뒤 확대한다.

파이프라인 (조문 번호를 지어내지 않음)
----
1. 답변 전문을 LLM에 주고, 이 QA의 '특약'에 직접 관련된 법리 쟁점만
   개념으로 요약하게 한다 (특약과 무관한 곁가지 쟁점은 제외하도록 지시).
2. 각 개념을 law_child BM25로 검색해 실제 존재하는 조문 후보를 얻는다.
3. 후보 조문 원문 + 특약 + 답변 근거를 함께 주고, 그 조문이 이 특약을
   실제로 규율하는지 LLM이 확인한다 (matches true/false + reason).
4. matches=true인 것만 정답으로 채택. reason을 남겨 사람이 검수한다.

정답의 최종 형태는 '조문'이므로, 채점(recall/precision)은 기존과 동일한
문자 비교로 이뤄진다. 유사도 판정 같은 모호한 채점을 도입하지 않는다.

실행: python data/processors/build_gt_from_reasoning.py --limit 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel
from rank_bm25 import BM25Okapi

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.retrieval.bm25_retrieval import load_law_child_from_db, tokenize
from shared.llm.gemini_client import gemini_client, LLMError

QA_PROCESSED_PATH = Path("data/lawtalk_qa_filtered/qa_processed.json")
RAW_DIR = Path("data/qa_raw")
OUTPUT_PATH = Path("evaluation/gt_from_reasoning.json")

BM25_TOP_N = 5
MAX_CANDIDATES = 15
MAX_RETRIES = 5
RETRY_DELAY = 20


class ConceptExtraction(BaseModel):
    concepts: list[str]


class ConfirmJudgment(BaseModel):
    concept: str
    clause_key: str
    matches: bool
    reason: str


class ConfirmResult(BaseModel):
    judgments: list[ConfirmJudgment]


EXTRACT_PROMPT = """너는 한국 주택임대차 법률 전문가다.
아래에 '특약'과 그 특약에 대한 '변호사 답변 전문'이 주어진다.
변호사 답변은 조문 번호를 거의 쓰지 않고 법리를 자연어로 설명하며,
특약과 직접 관련 없는 곁가지 대응책(소송 절차, 형사 고소, 가압류, 조정 신청 등)도
함께 나열하는 경우가 많다.

과제: 이 '특약 문구 자체가 직접 규율하는' 핵심 법적 쟁점만 뽑아라.

엄격한 규칙:
- 최대 3개까지만 뽑아라. 애매하면 개수를 줄여라.
- 특약 문구에 직접 대응하는 실체법적 쟁점만 뽑아라
  (예: 특약이 '묵시적 갱신'이면 '묵시적 갱신', '갱신 후 해지권' 정도).
- 다음은 반드시 제외하라 (특약의 직접 쟁점이 아니라 답변의 곁가지다):
  · 소송/집행/보전 절차 (가압류, 강제집행, 처분금지가처분, 내용증명 등)
  · 형사 쟁점 (사기죄, 형사처벌, 고소 등)
  · 분쟁조정 신청, 손해배상/위자료/부당이득 같은 일반적 구제수단
    (단, 특약이 바로 그것을 정하는 경우는 예외)
- 조문 번호는 추정하지 말라. 쟁점을 간결한 법률 용어 문구로만 서술하라.
- concepts 배열로 반환하라."""

CONFIRM_PROMPT = """너는 법조문과 특약의 관련성을 엄격히 판정하는 검증자다.
아래에 특약, 그 특약에 대한 변호사 답변의 법리 설명, 그리고 후보 조문 원문들이 주어진다.
각 후보 조문이 '이 특약을 직접 규율하는지' 판정하라 (matches: true/false).

기준:
- 조문 원문이 이 특약의 법적 쟁점을 직접 다루면 true.
- 답변의 법리 설명과 조문 내용이 실제로 부합해야 한다.
- 막연히 같은 분야라서 관련 있어 보이는 정도, 키워드만 겹치는 정도는 false.
- 애매하면 false (무관한 것을 정답에 넣는 게 더 나쁘다).
- reason에 왜 그렇게 판정했는지 한 줄로 남겨라 (사람이 검수한다)."""


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
    args = parser.parse_args()

    qa = json.loads(QA_PROCESSED_PATH.read_text(encoding="utf-8"))
    qa = [r for r in qa
          if r.get("url") and r.get("clause_processing_status") == "success"
          and len(r.get("extracted_clauses") or []) == 1][: args.limit]
    raw_by_url = load_raw_by_url()

    # 이어서 실행: 이미 처리한 url은 건너뛴다
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

        # 1) 답변 법리에서 특약 관련 쟁점 추출
        ext = _retry(
            contents=f"[특약]\n{clause}\n\n[변호사 답변 전문]\n{answer}",
            system_instruction=EXTRACT_PROMPT,
            response_schema=ConceptExtraction,
        )
        ext = _parse(ConceptExtraction, ext)
        print(f"[{rec.get('index')}] 특약: {clause[:50]}")
        print(f"   추출된 쟁점: {ext.concepts}")

        record_out = {"index": rec.get("index"), "url": rec["url"], "clause": clause,
                      "concepts": ext.concepts, "gt_laws": [], "detail": []}

        if ext.concepts:
            # 2) 각 쟁점 BM25 검색 -> 후보 조문
            cands, seen = [], set()
            for c in ext.concepts:
                for cd in search(c, docs, bm25, BM25_TOP_N):
                    if cd["clause_key"] not in seen:
                        seen.add(cd["clause_key"])
                        cands.append({"concept": c, **cd})
            cands = cands[:MAX_CANDIDATES]

            # 3) 확인
            lines = [f"[특약]\n{clause}\n", f"[답변 법리 요약 쟁점]\n{ext.concepts}\n", "[후보 조문]"]
            for cd in cands:
                lines.append(f"- clause_key: {cd['clause_key']}\n  원문: {cd['text']}\n")
            conf = _retry(
                contents="\n".join(lines),
                system_instruction=CONFIRM_PROMPT,
                response_schema=ConfirmResult,
            )
            conf = _parse(ConfirmResult, conf)
            record_out["gt_laws"] = [j.clause_key for j in conf.judgments if j.matches]
            record_out["detail"] = [j.model_dump() for j in conf.judgments]
            print(f"   -> 정답 조문: {record_out['gt_laws']}\n")
        else:
            print("   -> 쟁점 없음, 건너뜀\n")

        output.append(record_out)
        done_urls.add(rec["url"])
        # 레코드마다 즉시 저장 (중간에 죽어도 여기까지 보존)
        OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if skipped:
        print(f"이미 처리되어 건너뜀: {skipped}건")
    print(f"저장: {OUTPUT_PATH} (총 {len(output)}건)")


if __name__ == "__main__":
    main()