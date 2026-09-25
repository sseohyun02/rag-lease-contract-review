"""recall 낮은 쿼리(=케이스) miss 진단 — 법령(조문) 기준.

목적
----
retrieval_eval.py가 남긴 eval_results.json을 입력받아,
recall@k(법령, 조 단위)가 낮은 케이스 N개를 뽑고,
각 케이스에서 '놓친 정답 조문(missed GT article)'을 4유형으로 분류한다.

    1. 라벨 오류      : 검색된 게 사실 맞는데 정답셋에 없음 → 라벨이 너무 좁음
    2. 추론 필요      : 정답 조문이 쿼리 어휘로는 도달 불가 → 평가 설계 한계
    3. 진짜 모델 오류 : 도달 가능한데 순위가 낮음 → 검색 개선 여지
    4. 코퍼스·청킹    : 조문이 코퍼스에 없거나 잘림

자동 판정의 한계(정직하게 명시)
------------------------------
- '진짜 모델 오류'와 '코퍼스·청킹(미존재)'은 결과 JSON만으로 자동 판정 가능.
- '추론 필요'는 코퍼스가 있어야 판정(후보 풀에 없음 + 코퍼스에는 존재).
  어휘 겹침 수치를 함께 제시하되, 최종 확정은 사람이 한다.
- '라벨 오류'는 원리상 자동 판정 불가(법리 판단 필요). 그래서 '자동 분류'하지 않고,
  케이스별로 '검색됐지만 정답셋에 없는 조문(text 포함)'을 나란히 보여줘서
  사람이 눈으로 확인하도록 review 후보로만 제시한다.
→ 따라서 이 스크립트의 suggested_type은 '제안'이며, worksheet CSV의 final_type를
  사람이 채우는 것을 전제로 한다.

매칭 규칙은 retrieval_eval.py와 동일하게 '조 단위'(_article_of)를 사용한다.

사용 예
-------
    # 1) (선택) 코퍼스 텍스트 신호까지 쓰려면 DB에서 코퍼스 덤프를 먼저 뽑는다
    python evaluation/recall_error_analysis.py \
        --dump-corpus evaluation/results/law_corpus_dump.json

    # 2) 진단 실행
    python evaluation/recall_error_analysis.py \
        --results evaluation/eval_results_test.json \
        --corpus  evaluation/results/law_corpus_dump.json \
        --bottom 20 --k 10 \
        --out-md  evaluation/results/recall_error_report.md \
        --out-csv evaluation/results/recall_error_worksheet.csv

    # 코퍼스 없이 결과만으로 (진짜 모델 오류 / 후보에 없음 까지만 분리)
    python evaluation/recall_error_analysis.py \
        --results evaluation/eval_results_test.json --bottom 20
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# retrieval_eval.py와 동일한 조 단위 매칭 로직
# ─────────────────────────────────────────────────────────────────────
_ARTICLE_STRIP = re.compile(r'_(제\d+항|제\d+호|제\d+목).*$')


def article_of(key: str) -> str:
    """clause_key에서 항/호/목을 떼고 '조' 단위까지만 남긴다.

    예) '주택임대차보호법_제7조_제1항' -> '주택임대차보호법_제7조'
        '민법_제623조'                 -> '민법_제623조' (변화 없음)
    """
    return _ARTICLE_STRIP.sub('', str(key))


# ─────────────────────────────────────────────────────────────────────
# 어휘 토크나이저 (추론 필요 판정용 어휘 겹침 계산)
#   - 가능하면 코퍼스와 동일한 Kiwi 형태소 토크나이저를 재사용한다.
#   - Kiwi/DB 의존성이 없는 환경(결과만 분석)에서도 죽지 않도록 폴백을 둔다.
# ─────────────────────────────────────────────────────────────────────
_TOKENIZE_FN = None
_TOKENIZE_MODE = None


def _get_tokenizer():
    """(tokenize_fn, mode) 반환. Kiwi가 있으면 프로젝트 토크나이저, 없으면 폴백."""
    global _TOKENIZE_FN, _TOKENIZE_MODE
    if _TOKENIZE_FN is not None:
        return _TOKENIZE_FN, _TOKENIZE_MODE
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pipeline.retrieval.bm25_retrieval import tokenize as _kiwi_tok  # noqa
        _TOKENIZE_FN, _TOKENIZE_MODE = _kiwi_tok, "kiwi"
    except Exception:
        # 폴백: 한글/영숫자 2글자 이상 토막. 형태소 분석보다 거칠지만 겹침 유무는 잡는다.
        _fallback = re.compile(r'[가-힣A-Za-z0-9]{2,}')
        _TOKENIZE_FN = lambda t: _fallback.findall(t or "")  # noqa: E731
        _TOKENIZE_MODE = "fallback"
    return _TOKENIZE_FN, _TOKENIZE_MODE


def lexical_overlap(query_text: str, doc_text: str) -> dict:
    """쿼리(특약)와 정답 조문 텍스트 사이 어휘 겹침을 계산한다.

    반환: shared(공유 토큰 수), n_query, n_doc, jaccard, shared_tokens(상위 일부)
    겹침이 0에 가까우면 '쿼리 어휘로는 도달 불가(추론 필요)' 신호.
    """
    tok, _ = _get_tokenizer()
    q = set(tok(query_text))
    d = set(tok(doc_text))
    if not q or not d:
        return {"shared": 0, "n_query": len(q), "n_doc": len(d),
                "jaccard": 0.0, "shared_tokens": []}
    inter = q & d
    union = q | d
    return {
        "shared": len(inter),
        "n_query": len(q),
        "n_doc": len(d),
        "jaccard": round(len(inter) / len(union), 4) if union else 0.0,
        "shared_tokens": sorted(inter)[:15],
    }


# ─────────────────────────────────────────────────────────────────────
# 코퍼스 로딩 / 덤프
# ─────────────────────────────────────────────────────────────────────
def dump_corpus(out_path: Path) -> None:
    """프로젝트 로더(load_law_child_from_db)로 법령 코퍼스를 JSON으로 덤프한다.

    DB 접속이 되는 환경(= retrieval_eval.py를 돌릴 수 있는 환경)에서만 동작한다.
    덤프 스키마: [{clause_key, law_name, article_no, paragraph_no, child_text, parent_text}, ...]
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.retrieval.bm25_retrieval import load_law_child_from_db

    df = load_law_child_from_db()
    keep = ["clause_key", "law_name", "article_no", "paragraph_no", "child_text", "parent_text"]
    records = []
    for _, r in df.iterrows():
        records.append({c: ("" if r.get(c) is None else str(r.get(c))) for c in keep})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    print(f"코퍼스 덤프 저장: {out_path} ({len(records)}행)")


def load_corpus(path: Path) -> dict:
    """코퍼스 덤프를 조 단위로 집계한다.

    반환:
      article_keys : set[str]  존재하는 모든 조 단위 키
      article_text : dict[str, str]  조 단위 키 -> 그 조에 속한 항 텍스트들을 이어붙인 것
      article_n    : dict[str, int]  조 단위 키 -> 코퍼스 내 항(child) 개수
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    article_keys: set[str] = set()
    article_text: dict[str, list] = {}
    article_n: Counter = Counter()
    for r in records:
        art = article_of(r["clause_key"])
        article_keys.add(art)
        article_n[art] += 1
        parts = article_text.setdefault(art, [])
        # 조 텍스트 = parent_text(조 본문) + child_text(항 본문) 결합
        chunk = " ".join(x for x in [r.get("parent_text", ""), r.get("child_text", "")] if x)
        if chunk.strip():
            parts.append(chunk.strip())
    return {
        "article_keys": article_keys,
        "article_text": {k: " ".join(v) for k, v in article_text.items()},
        "article_n": dict(article_n),
    }


# ─────────────────────────────────────────────────────────────────────
# 후보 풀 분석: 놓친 조문이 어디까지 도달했는지
# ─────────────────────────────────────────────────────────────────────
def law_ranked_ids(clause: dict, method: str) -> list:
    """clause[method]에서 법령만 순위순 doc_id 리스트로 뽑는다(rrf/bm25/dense)."""
    return [r["doc_id"] for r in clause.get(method, []) if r.get("source_type") == "law"]


def article_rank_in_list(ranked_ids: list, target_art: str) -> int | None:
    """법령 순위 리스트에서 target_art(조 단위)가 처음 등장하는 1-based 순위. 없으면 None."""
    for idx, did in enumerate(ranked_ids, 1):
        if article_of(did) == target_art:
            return idx
    return None


def reachability(clauses: list, missed_art: str) -> dict:
    """놓친 조문이 후보 풀에서 도달한 최선의 순위/채널을 케이스 전체에서 집계한다.

    - law_only_rrf_rank : rrf 결과를 '법령만' 추린 순위에서의 최선(min) 순위.
      recall@k가 이 리스트를 [:k]로 자르므로, 이 값이 k보다 크면
      '도달은 했는데 순위가 낮아 밀린 것'(진짜 모델 오류).
      None이면 두 채널 어디에서도 못 잡음.
    - bm25_rank / dense_rank : 채널별 최선 순위(있으면 그 채널이 잡은 것).
      both/dense-only/bm25-only/neither 판별에 사용.
    """
    best_rrf = None
    best_bm25 = None
    best_dense = None
    for c in clauses:
        r = article_rank_in_list(law_ranked_ids(c, "rrf"), missed_art)
        if r is not None:
            best_rrf = r if best_rrf is None else min(best_rrf, r)
        # 채널별 순위는 rrf 항목의 bm25_rank/dense_rank 필드에서 직접 읽는다
        for item in c.get("rrf", []):
            if item.get("source_type") != "law":
                continue
            if article_of(item["doc_id"]) != missed_art:
                continue
            if item.get("bm25_rank") is not None:
                best_bm25 = item["bm25_rank"] if best_bm25 is None else min(best_bm25, item["bm25_rank"])
            if item.get("dense_rank") is not None:
                best_dense = item["dense_rank"] if best_dense is None else min(best_dense, item["dense_rank"])
        # rrf에 안 실린 경우 대비: bm25/dense 리스트도 직접 스캔
        rb = article_rank_in_list(law_ranked_ids(c, "bm25"), missed_art)
        if rb is not None:
            best_bm25 = rb if best_bm25 is None else min(best_bm25, rb)
        rd = article_rank_in_list(law_ranked_ids(c, "dense"), missed_art)
        if rd is not None:
            best_dense = rd if best_dense is None else min(best_dense, rd)

    if best_bm25 is not None and best_dense is not None:
        channel = "both"
    elif best_dense is not None:
        channel = "dense-only"
    elif best_bm25 is not None:
        channel = "bm25-only"
    else:
        channel = "neither"
    return {
        "law_only_rrf_rank": best_rrf,
        "best_bm25_rank": best_bm25,
        "best_dense_rank": best_dense,
        "channel": channel,
    }


def classify_miss(reach: dict, k: int, corpus: dict | None,
                  missed_art: str, query_text: str) -> dict:
    """놓친 조문 1개에 대한 suggested_type과 근거 신호를 만든다.

    판정 트리
    ---------
    - 후보 풀(법령 rrf)에 존재 & 순위 > k        → 진짜 모델 오류
    - 후보 풀에 존재 & 순위 <= k                 → recheck(조단위 dedup 등으로 miss 처리된 경계)
    - 후보 풀에 없음:
        - 코퍼스 있음:
            - 코퍼스에 조문 없음                → 코퍼스·청킹(미존재)
            - 코퍼스에 있음                     → 추론 필요(+어휘 겹침 수치)
        - 코퍼스 없음                           → not_retrieved(추론필요/코퍼스누락 미확인)
    ※ 라벨 오류는 여기서 판정하지 않는다(케이스 단위 review 후보로 따로 제시).
    """
    rrf_rank = reach["law_only_rrf_rank"]
    sig = {
        "law_only_rrf_rank": rrf_rank,
        "best_bm25_rank": reach["best_bm25_rank"],
        "best_dense_rank": reach["best_dense_rank"],
        "channel": reach["channel"],
        "in_corpus": None,
        "corpus_n_paragraphs": None,
        "lex_shared": None,
        "lex_jaccard": None,
        "lex_shared_tokens": None,
    }

    if rrf_rank is not None and rrf_rank > k:
        sig["suggested_type"] = "3_진짜_모델_오류"
        sig["reason"] = f"법령 rrf 순위 {rrf_rank}위(>{k})로 도달했으나 밀림 · 채널={reach['channel']}"
        return sig
    if rrf_rank is not None and rrf_rank <= k:
        sig["suggested_type"] = "0_recheck"
        sig["reason"] = f"법령 rrf 순위 {rrf_rank}위(<= {k})인데 miss로 잡힘 — 조단위 매칭/dedup 경계 확인"
        return sig

    # 여기부터는 후보 풀에 아예 없음(neither)
    if corpus is not None:
        in_corpus = missed_art in corpus["article_keys"]
        sig["in_corpus"] = in_corpus
        sig["corpus_n_paragraphs"] = corpus["article_n"].get(missed_art)
        if not in_corpus:
            sig["suggested_type"] = "4_코퍼스_청킹"
            sig["reason"] = "정답 조문이 코퍼스에 존재하지 않음(미수집/키 불일치)"
            return sig
        # 코퍼스에 있는데 두 채널 모두 못 잡음 → 어휘 겹침 확인
        ov = lexical_overlap(query_text, corpus["article_text"].get(missed_art, ""))
        sig["lex_shared"] = ov["shared"]
        sig["lex_jaccard"] = ov["jaccard"]
        sig["lex_shared_tokens"] = ov["shared_tokens"]
        sig["suggested_type"] = "2_추론_필요"
        sig["reason"] = (f"코퍼스에 존재하나 두 채널 모두 미검색 · "
                         f"어휘 겹침 shared={ov['shared']} jaccard={ov['jaccard']}")
        return sig

    # 코퍼스 없이 결과만 있는 경우
    sig["suggested_type"] = "2or4_not_retrieved"
    sig["reason"] = "두 채널 모두 미검색 — 코퍼스 덤프가 있어야 추론필요/코퍼스누락 구분 가능"
    return sig


# ─────────────────────────────────────────────────────────────────────
# 케이스 단위 분석
# ─────────────────────────────────────────────────────────────────────
def analyze_case(case: dict, k: int, corpus: dict | None) -> dict:
    """케이스 1개: 법령 recall@k(조 단위) + 놓친 조문 분류 + 라벨오류 review 후보."""
    clauses = case.get("clauses", [])
    gt_arts = {article_of(g) for g in case.get("gt_laws", [])}

    # 검색 상위 k 풀(조 단위) = 각 특약 법령 rrf 상위 k의 합집합
    retrieved_ids = set()
    for c in clauses:
        retrieved_ids.update(law_ranked_ids(c, "rrf")[:k])
    retrieved_arts = {article_of(d) for d in retrieved_ids}

    hit_arts = gt_arts & retrieved_arts
    missed_arts = gt_arts - retrieved_arts
    recall = round(len(hit_arts) / len(gt_arts), 4) if gt_arts else None

    # 놓친 조문 분류
    query_text = " ".join(c.get("clause", "") for c in clauses)
    missed_records = []
    for art in sorted(missed_arts):
        reach = reachability(clauses, art)
        sig = classify_miss(reach, k, corpus, art, query_text)
        rec = {"missed_article": art}
        rec.update(sig)
        missed_records.append(rec)

    # 라벨 오류 review 후보 = 상위 k에 검색됐지만 정답셋에 없는 조문
    label_candidates = []
    # rrf 순위순으로 유니크 조 추출
    ordered_arts = []
    seen = set()
    for c in clauses:
        for did in law_ranked_ids(c, "rrf")[:k]:
            a = article_of(did)
            if a not in seen:
                seen.add(a)
                ordered_arts.append(a)
    for a in ordered_arts:
        if a in gt_arts:
            continue
        entry = {"article": a}
        if corpus is not None:
            entry["in_corpus"] = a in corpus["article_keys"]
            txt = corpus["article_text"].get(a, "")
            entry["text_preview"] = txt[:160]
        label_candidates.append(entry)

    return {
        "id": case.get("id"),
        "recall_at_k": recall,
        "n_gt_arts": len(gt_arts),
        "n_hit": len(hit_arts),
        "n_missed": len(missed_arts),
        "query_text": query_text,
        "gt_articles": sorted(gt_arts),
        "hit_articles": sorted(hit_arts),
        "missed": missed_records,
        "label_error_candidates": label_candidates[:10],
    }


# ─────────────────────────────────────────────────────────────────────
# 리포트 출력
# ─────────────────────────────────────────────────────────────────────
def _art_text_preview(corpus: dict | None, art: str, n: int = 160) -> str:
    if corpus is None:
        return ""
    return corpus["article_text"].get(art, "")[:n]


def write_markdown(analyses: list, k: int, corpus: dict | None, tally: Counter, out_path: Path) -> None:
    lines = []
    lines.append(f"# recall miss 진단 리포트 (법령 · recall@{k} · 조 단위)\n")
    lines.append(f"- 분석 케이스 수: **{len(analyses)}개** (recall@{k} 오름차순 하위)")
    lines.append(f"- 코퍼스 텍스트 신호: **{'사용' if corpus else '미사용(결과 JSON만)'}**")
    lines.append("")
    lines.append("## suggested_type 분포 (놓친 조문 기준)")
    lines.append("")
    lines.append("| suggested_type | 개수 |")
    lines.append("| --- | --- |")
    for t, n in tally.most_common():
        lines.append(f"| {t} | {n} |")
    lines.append("")
    lines.append("> suggested_type은 '제안'이다. 특히 라벨 오류는 자동 판정하지 않으므로, "
                 "각 케이스의 '라벨 오류 review 후보'를 눈으로 확인하고 worksheet의 final_type을 확정할 것.\n")
    lines.append("---\n")

    for a in analyses:
        lines.append(f"## [{a['id']}]  recall@{k} = {a['recall_at_k']}  "
                     f"(정답 {a['n_gt_arts']}조 중 {a['n_hit']} hit / {a['n_missed']} miss)\n")
        lines.append(f"**특약(쿼리)**: {a['query_text']}\n")

        lines.append("**정답 조문(GT, 조 단위)**")
        for art in a["gt_articles"]:
            mark = "✅" if art in a["hit_articles"] else "❌"
            prev = _art_text_preview(corpus, art)
            lines.append(f"- {mark} `{art}`" + (f" — {prev}" if prev else ""))
        lines.append("")

        lines.append(f"**놓친 조문 분류 (miss {a['n_missed']}개)**")
        if not a["missed"]:
            lines.append("- (없음)")
        for m in a["missed"]:
            lines.append(
                f"- `{m['missed_article']}` → **{m['suggested_type']}**  \n"
                f"  - 근거: {m['reason']}  \n"
                f"  - rrf(법령)순위={m['law_only_rrf_rank']} / bm25={m['best_bm25_rank']} / "
                f"dense={m['best_dense_rank']} / 채널={m['channel']}"
                + (f" / 코퍼스존재={m['in_corpus']} 항수={m['corpus_n_paragraphs']}" if corpus else "")
                + (f"  \n  - 공유토큰: {m['lex_shared_tokens']}" if m.get("lex_shared_tokens") else "")
            )
        lines.append("")

        lines.append("**라벨 오류 review 후보 (상위 k에 검색됐으나 정답셋에 없음 → 사실 정답인지 확인)**")
        if not a["label_error_candidates"]:
            lines.append("- (없음)")
        for lc in a["label_error_candidates"]:
            prev = lc.get("text_preview", "")
            lines.append(f"- `{lc['article']}`" + (f" — {prev}" if prev else ""))
        lines.append("\n---\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"마크다운 리포트 저장: {out_path}")


def write_csv(analyses: list, corpus: dict | None, out_path: Path) -> None:
    """놓친 조문 1개 = 1행. final_type는 사람이 채우는 빈 칸."""
    cols = [
        "case_id", "recall_at_k", "query_text", "missed_article",
        "suggested_type", "final_type", "reason",
        "law_only_rrf_rank", "best_bm25_rank", "best_dense_rank", "channel",
        "in_corpus", "corpus_n_paragraphs", "lex_shared", "lex_jaccard",
        "missed_article_text",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for a in analyses:
            for m in a["missed"]:
                w.writerow({
                    "case_id": a["id"],
                    "recall_at_k": a["recall_at_k"],
                    "query_text": a["query_text"],
                    "missed_article": m["missed_article"],
                    "suggested_type": m["suggested_type"],
                    "final_type": "",  # 사람이 확정
                    "reason": m["reason"],
                    "law_only_rrf_rank": m["law_only_rrf_rank"],
                    "best_bm25_rank": m["best_bm25_rank"],
                    "best_dense_rank": m["best_dense_rank"],
                    "channel": m["channel"],
                    "in_corpus": m["in_corpus"],
                    "corpus_n_paragraphs": m["corpus_n_paragraphs"],
                    "lex_shared": m["lex_shared"],
                    "lex_jaccard": m["lex_jaccard"],
                    "missed_article_text": _art_text_preview(corpus, m["missed_article"], 300),
                })
    print(f"worksheet CSV 저장: {out_path}")


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="recall 낮은 케이스 miss 4유형 진단")
    p.add_argument("--dump-corpus", type=Path, help="DB에서 법령 코퍼스를 이 경로로 덤프하고 종료")
    p.add_argument("--results", type=Path, help="retrieval_eval.py 결과 JSON 경로")
    p.add_argument("--corpus", type=Path, help="(선택) 법령 코퍼스 덤프 JSON 경로")
    p.add_argument("--bottom", type=int, default=20, help="분석할 하위 케이스 수(default 20)")
    p.add_argument("--k", type=int, default=10, help="recall@k의 k(default 10)")
    p.add_argument("--out-md", type=Path, default=Path("evaluation/results/recall_error_report.md"))
    p.add_argument("--out-csv", type=Path, default=Path("evaluation/results/recall_error_worksheet.csv"))
    args = p.parse_args()

    # 코퍼스 덤프 모드
    if args.dump_corpus:
        dump_corpus(args.dump_corpus)
        return

    if not args.results:
        p.error("--results 가 필요합니다 (또는 --dump-corpus 사용).")

    cases = json.loads(args.results.read_text(encoding="utf-8"))
    corpus = load_corpus(args.corpus) if args.corpus else None
    if corpus:
        _, mode = _get_tokenizer()
        print(f"코퍼스 로드 완료: 조 {len(corpus['article_keys'])}개 · 토크나이저={mode}")

    # 1) 케이스별 recall@k 계산 후 하위 N개 선정 (gt_laws 있는 케이스만)
    scored = []
    for case in cases:
        if not case.get("gt_laws"):
            continue
        a = analyze_case(case, args.k, corpus)
        if a["recall_at_k"] is None:
            continue
        scored.append(a)

    # recall 오름차순 → miss 많은 순 → id 순 (결정적)
    scored.sort(key=lambda x: (x["recall_at_k"], -x["n_missed"], str(x["id"])))
    picked = scored[: args.bottom]

    print(f"\n총 {len(scored)}개 케이스(gt_laws 보유) 중 recall@{args.k} 하위 {len(picked)}개 선정")
    print(f"선정 케이스 recall 범위: "
          f"{picked[0]['recall_at_k'] if picked else '-'} ~ "
          f"{picked[-1]['recall_at_k'] if picked else '-'}")

    # 2) 분류 집계
    tally = Counter()
    for a in picked:
        for m in a["missed"]:
            tally[m["suggested_type"]] += 1

    print("\n[suggested_type 분포 — 놓친 조문 기준]")
    for t, n in tally.most_common():
        print(f"  {t:24s} {n}")

    # 3) 리포트 출력
    write_markdown(picked, args.k, corpus, tally, args.out_md)
    write_csv(picked, corpus, args.out_csv)
    print("\n다음 단계: worksheet CSV의 final_type 칸을 사람이 확정 → 유형별 개선 액션 도출")


if __name__ == "__main__":
    main()
