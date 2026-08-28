"""가중 RRF 스윕 — 순위 기반 융합으로 리랭킹 손해 문제 개선 시도.

배경
----
현행 alpha-가중합은 BM25/Dense '점수'를 min-max 정규화 후 섞는다. 이때 BM25가
확신 없이 뿌린 낮은 점수도 정규화로 부풀려져 Dense의 상위 결과를 밀어내,
융합이 Dense 단독보다 나빠지는 구간이 생긴다.

RRF는 점수가 아니라 '순위'만 쓴다:
  score = w_bm25 * 1/(K + rank_bm25) + w_dense * 1/(K + rank_dense)
순위 기반이라 점수 스케일 왜곡이 없고, 가중치(w_dense↑)로 Dense를 우대할 수 있다.

이 스크립트는:
- 검색을 1회만 수행해 순위를 캐시하고 (추가 API 비용 없음)
- (w_bm25:w_dense) 조합과 RRF_K를 스윕해서 recall/P@1/MRR을 비교한다.
- 참고로 alpha-가중합 현행(alpha=0.2)·Dense 단독 수치도 함께 출력한다.

tune set에서만 스윕한다. test는 최종 1회용으로 건드리지 않는다.
실행: python data/processors/rrf_sweep_eval.py --eval-set evaluation/eval_set_tune.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from pipeline.retrieval.bm25_retrieval import tokenize, build_query_tokens, load_law_child_from_db
from pipeline.retrieval import dense_retrieval
from pipeline.retrieval.query_expansion.query_expansion import expand_clause
from pipeline.retrieval.query_expansion.retrieval_adapter import build_retrieval_payload
from shared.db.connection import get_db_client
from sqlalchemy import text

K_VALUES = [1, 3, 5, 10, 20, 50]
TOP_K = 50

# (w_bm25, w_dense) 조합 — 1:2 근처를 소수 단위로 촘촘하게
# recall 우선이므로 Dense 우대 구간(1:1.5 ~ 1:3)을 0.25 간격으로 세밀 탐색
WEIGHTS = [
    (1, 1.25), (1, 1.5), (1, 1.75), (1, 2), (1, 2.25), (1, 2.5), (1, 2.75), (1, 3),
    (1.25, 2), (1.5, 2), (1.75, 2),   # bm25쪽도 소수로 살짝
    (1, 1), (0, 1),                    # 참고용 기준
]
RRF_KS = [10, 20, 30, 40, 50, 60]      # K도 더 촘촘하게


import re


def _article_of(key):
    """clause_key에서 항/호/목을 떼고 '조' 단위까지만 남긴다."""
    return re.sub(r'_(제\d+항|제\d+호|제\d+목).*$', '', key)


def is_hit(gt_key, doc_id):
    # 조 단위 매칭 (항이 달라도 같은 조이면 정답)
    return _article_of(gt_key) == _article_of(doc_id)


def load_law_corpus():
    df = load_law_child_from_db()
    return [{"doc_id": str(r["clause_key"]),
             "text": str(r.get("bm25_target") or r.get("child_text") or "")}
            for _, r in df.iterrows()]


def load_law_embeddings():
    db = get_db_client()
    rows = db.fetch_all(text(
        "SELECT clause_key, embed_vertex FROM law_child WHERE embed_vertex IS NOT NULL"))
    ids, vecs = [], []
    for r in rows:
        ids.append(str(r["clause_key"]))
        v = r["embed_vertex"]
        vecs.append(np.array(json.loads(v) if isinstance(v, str) else v, dtype=np.float32))
    return ids, np.vstack(vecs)


def minmax(scores):
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def eval_ranking_fn(cache, rank_fn):
    """rank_fn(bm25_ranks, dense_ranks, bm25_raw, dense_raw) -> ranked doc_id list"""
    agg = {k: {"hit": 0, "total": 0} for k in K_VALUES}
    p1_hit = 0
    mrr_sum = 0.0
    n = 0
    for gt, bm25_ranks, dense_ranks, bm25_raw, dense_raw in cache:
        ranked = rank_fn(bm25_ranks, dense_ranks, bm25_raw, dense_raw)
        n += 1
        if ranked and any(is_hit(g, ranked[0]) for g in gt):
            p1_hit += 1
        for rank, did in enumerate(ranked, 1):
            if any(is_hit(g, did) for g in gt):
                mrr_sum += 1.0 / rank
                break
        for k in K_VALUES:
            topk = ranked[:k]
            hits = sum(1 for g in gt if any(is_hit(g, d) for d in topk))
            agg[k]["hit"] += hits
            agg[k]["total"] += len(gt)
    recalls = {k: agg[k]["hit"] / agg[k]["total"] if agg[k]["total"] else 0 for k in K_VALUES}
    return recalls, p1_hit / n if n else 0, mrr_sum / n if n else 0


def make_wrrf(w_bm25, w_dense, rrf_k):
    def fn(bm25_ranks, dense_ranks, bm25_raw, dense_raw):
        all_ids = set(bm25_ranks) | set(dense_ranks)
        scored = []
        for did in all_ids:
            s = 0.0
            if did in bm25_ranks:
                s += w_bm25 * 1.0 / (rrf_k + bm25_ranks[did])
            if did in dense_ranks:
                s += w_dense * 1.0 / (rrf_k + dense_ranks[did])
            scored.append((did, s))
        scored.sort(key=lambda x: -x[1])
        return [d for d, _ in scored]
    return fn


def make_alpha(alpha):
    def fn(bm25_ranks, dense_ranks, bm25_raw, dense_raw):
        nb = minmax(bm25_raw)
        nd = minmax(dense_raw)
        all_ids = set(nb) | set(nd)
        scored = [(did, alpha * nb.get(did, 0) + (1 - alpha) * nd.get(did, 0)) for did in all_ids]
        scored.sort(key=lambda x: -x[1])
        return [d for d, _ in scored]
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", type=Path, default=Path("evaluation/eval_set_tune.json"))
    ap.add_argument("--qe-cache", type=Path, default=Path("evaluation/_qe_cache/qe_cache.json"),
                    help="QE 결과 파일 캐시 경로 (미리 qe_cache.py로 채워두면 토큰 0)")
    args = ap.parse_args()

    # QE 캐시 래퍼 (실험 전용). 캐시에 있으면 LLM 호출 없이 재사용.
    from data.processors.qe_cache import CachedExpander
    expander = CachedExpander(args.qe_cache, auto_flush=True)

    cases = json.loads(args.eval_set.read_text(encoding="utf-8"))
    docs = load_law_corpus()
    doc_ids = [d["doc_id"] for d in docs]
    bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])
    emb_ids, emb_matrix = load_law_embeddings()

    print("검색 순위 캐시 생성 (특약당 QE+BM25+Dense 1회)...")
    cache = []
    for i, case in enumerate(cases):
        # v2 eval_set은 gt_laws 필드. 조 단위로 중복 제거해 분모 왜곡 방지.
        raw_gt = case.get("gt_laws", []) or case.get("gt_laws_filtered", [])
        gt = {_article_of(g) for g in raw_gt}
        if not gt:
            continue
        clause = case["clauses"][0]["normalized"]
        # QE 캐시 래퍼 사용 (실험 전용, 토큰 절약) — 서비스 코드 불변
        payload = build_retrieval_payload(expander.expand(clause), clause_text=clause)

        bm25_scores = bm25.get_scores(build_query_tokens(payload["bm25_keywords"]))
        bm25_top = np.argsort(bm25_scores)[::-1][:TOP_K]
        bm25_ranks = {doc_ids[j]: r + 1 for r, j in enumerate(bm25_top) if bm25_scores[j] > 0}
        bm25_raw = {doc_ids[j]: float(bm25_scores[j]) for j in bm25_top if bm25_scores[j] > 0}

        qvec = dense_retrieval.embed_query(payload["dense_query"]).reshape(1, -1)
        sims = cosine_similarity(qvec, emb_matrix)[0]
        dense_top = np.argsort(sims)[::-1][:TOP_K]
        dense_ranks = {emb_ids[j]: r + 1 for r, j in enumerate(dense_top)}
        dense_raw = {emb_ids[j]: float(sims[j]) for j in dense_top}

        cache.append((gt, bm25_ranks, dense_ranks, bm25_raw, dense_raw))
        print(f"  [{i+1}/{len(cases)}]", end="\r")
    print(f"\n캐시 완료: {len(cache)}건\n")

    def show(label, recalls, p1, mrr):
        print(f"{label:<24} " + "  ".join(f"{recalls[k]:.3f}" for k in K_VALUES) + f"   {p1:.3f}  {mrr:.3f}")

    print("=" * 90)
    print(f"{'방식':<26} " + "  ".join(f"R@{k:<2}" for k in K_VALUES) + "   P@1    MRR")
    print("=" * 90)

    # 기준선: Dense 단독, alpha 현행
    r, p, m = eval_ranking_fn(cache, make_alpha(0.0)); show("[기준] Dense 단독", r, p, m)
    r, p, m = eval_ranking_fn(cache, make_alpha(0.2)); show("[기준] alpha=0.2 (현행)", r, p, m)
    r, p, m = eval_ranking_fn(cache, make_alpha(0.3)); show("[기준] alpha=0.3", r, p, m)
    print("-" * 90)

    # 가중 RRF 스윕
    best = None
    all_results = []   # (label, recalls, p1, mrr) 전체 저장 → 상위 정렬용
    for rrf_k in RRF_KS:
        for wb, wd in WEIGHTS:
            r, p, m = eval_ranking_fn(cache, make_wrrf(wb, wd, rrf_k))
            label = f"RRF K={rrf_k} w={wb}:{wd}"
            show(label, r, p, m)
            all_results.append((label, r, p, m))
            score = r[10]  # recall@10 기준
            if best is None or score > best[0]:
                best = (score, label, r, p, m)
        print("-" * 90)

    score, label, r, p, m = best
    print(f"\n[recall@10 최적] {label}")
    print(f"  R@10={r[10]:.3f}  R@50={r[50]:.3f}  P@1={p:.3f}  MRR={m:.3f}")

    # recall@10 상위 10개 — 한 점의 최고값이 아니라 '안정 구간'을 보기 위함
    print("\n=== recall@10 상위 10 (안정 구간 확인용) ===")
    print("  주의: 70건 tune에서 R@10 0.01 차이는 케이스 1개(1/70)라 노이즈일 수 있음.")
    top = sorted(all_results, key=lambda x: -x[1][10])[:10]
    for lbl, rr, pp, mm in top:
        print(f"  R@10={rr[10]:.3f}  R@50={rr[50]:.3f}  P@1={pp:.3f}  MRR={mm:.3f}   {lbl}")
    print(f"\n{expander.stats}")


if __name__ == "__main__":
    main()