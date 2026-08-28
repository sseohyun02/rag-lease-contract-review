"""alpha-가중합 융합 비율(alpha) 스윕 — tune set 최적값 탐색.

배경
----
실제 서비스(retrieval_service.py)는 RRF가 아니라 alpha-가중합을 쓴다:
  hybrid = alpha * norm_bm25 + (1-alpha) * norm_dense   (min-max 정규화)
법령은 현재 alpha=0.20 (BM25 20% + Dense 80%)로 튜닝돼 있다.
그런데 그동안 평가 스크립트는 반반 RRF를 봤기 때문에, 새 골드셋에서
실제 서비스 방식(alpha-가중합)의 최적 alpha를 다시 확인할 필요가 있다.

방식
----
- 각 특약을 한 번만 검색(QE + BM25 + Dense)해서 원점수를 캐시한다.
- alpha만 0.0~1.0으로 바꿔가며 융합을 재계산한다 (추가 API 호출 없음).
- alpha=0.0 -> Dense 100%, alpha=1.0 -> BM25 100%.
- 정규화(min-max)와 융합식은 retrieval_service._alpha_hybrid와 동일하게 맞춘다.
- tune set에서만 스윕한다. test set은 최종 1회 측정용으로 건드리지 않는다.

gt/결과 매칭은 조 단위가 아니라 기존 평가와 동일하게 clause_key prefix로 한다
(gt가 조 단위면 검색된 항이 그 조에 속하면 hit).

실행: python evaluation/hybrid_sweep_eval.py --eval-set evaluation/eval_set_tune.json
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

from pipeline.retrieval.bm25_retrieval import (
    tokenize, build_query_tokens, load_law_child_from_db,
)
from pipeline.retrieval import dense_retrieval
from pipeline.retrieval.query_expansion.query_expansion import expand_clause
from pipeline.retrieval.query_expansion.retrieval_adapter import build_retrieval_payload
from shared.db.connection import get_db_client
from sqlalchemy import text

K_VALUES = [1, 3, 5, 10, 20, 50]
TOP_K = 50
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def is_hit(gt_set, doc_id):
    return any(doc_id == g or doc_id.startswith(g + "_") for g in gt_set)


def load_law_corpus():
    df = load_law_child_from_db()
    docs = [
        {"doc_id": str(r["clause_key"]),
         "text": str(r.get("bm25_target") or r.get("child_text") or "")}
        for _, r in df.iterrows()
    ]
    return docs


def load_law_embeddings():
    """law_child 임베딩을 DB에서 로드 (dense 검색용)."""
    db = get_db_client()
    rows = db.fetch_all(text(
        "SELECT clause_key, embed_vertex FROM law_child WHERE embed_vertex IS NOT NULL"
    ))
    ids, vecs = [], []
    for r in rows:
        ids.append(str(r["clause_key"]))
        v = r["embed_vertex"]
        vecs.append(np.array(json.loads(v) if isinstance(v, str) else v, dtype=np.float32))
    return ids, np.vstack(vecs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=Path("evaluation/eval_set_tune.json"))
    args = parser.parse_args()

    cases = json.loads(args.eval_set.read_text(encoding="utf-8"))
    print(f"평가 케이스: {len(cases)}건")

    print("법령 코퍼스 + BM25 인덱스 로드...")
    docs = load_law_corpus()
    bm25 = BM25Okapi([tokenize(d["text"]) for d in docs])
    doc_ids = [d["doc_id"] for d in docs]

    print("법령 임베딩 로드...")
    emb_ids, emb_matrix = load_law_embeddings()
    emb_id_to_row = {k: i for i, k in enumerate(emb_ids)}

    # 1) 각 특약 검색 결과(원점수) 캐시 — API/검색 1회만
    print("검색 원점수 캐시 생성 (특약당 QE+BM25+Dense 1회)...")
    cache = []  # [(gt_set, {doc_id: bm25_raw}, {doc_id: dense_raw})]
    for i, case in enumerate(cases):
        gt = set(case.get("gt_laws_explicit", case["gt_laws"]))
        if not gt:
            continue
        clause = case["clauses"][0]["normalized"]
        expansion = expand_clause(clause)
        payload = build_retrieval_payload(expansion, clause_text=clause)

        # BM25 원점수 (상위 TOP_K*4 정도만 담아 정규화 안정화)
        bm25_scores_all = bm25.get_scores(build_query_tokens(payload["bm25_keywords"]))
        top_bm25_idx = np.argsort(bm25_scores_all)[::-1][:200]
        bm25_raw = {doc_ids[j]: float(bm25_scores_all[j]) for j in top_bm25_idx if bm25_scores_all[j] > 0}

        # Dense 원점수 (코사인 유사도 상위 200)
        qvec = dense_retrieval.embed_query(payload["dense_query"]).reshape(1, -1)
        sims = cosine_similarity(qvec, emb_matrix)[0]
        top_dense_idx = np.argsort(sims)[::-1][:200]
        dense_raw = {emb_ids[j]: float(sims[j]) for j in top_dense_idx}

        cache.append((gt, bm25_raw, dense_raw))
        print(f"  [{i+1}/{len(cases)}]", end="\r")
    print(f"\n캐시 완료: {len(cache)}건\n")

    # 2) alpha 스윕 — 재계산만 (API 호출 없음)
    print("=" * 78)
    print("alpha 스윕 (alpha=0 → Dense 100%, alpha=1 → BM25 100%)")
    print("법령 recall@k, tune set 기준")
    print("=" * 78)
    header = "alpha  " + "  ".join(f"R@{k:<2}" for k in K_VALUES) + "   P@1    MRR"
    print(header)

    best = None
    for alpha in ALPHAS:
        agg = {k: {"hit": 0, "total": 0} for k in K_VALUES}
        p1_hit = 0
        mrr_sum = 0.0
        n = 0
        for gt, bm25_raw, dense_raw in cache:
            nb = minmax(bm25_raw)
            nd = minmax(dense_raw)
            all_ids = set(nb) | set(nd)
            fused = []
            for did in all_ids:
                score = alpha * nb.get(did, 0.0) + (1 - alpha) * nd.get(did, 0.0)
                fused.append((did, score))
            fused.sort(key=lambda x: -x[1])
            ranked = [d for d, _ in fused]

            n += 1
            # P@1, MRR
            if ranked and is_hit(gt, ranked[0]):
                p1_hit += 1
            for rank, did in enumerate(ranked, 1):
                if is_hit(gt, did):
                    mrr_sum += 1.0 / rank
                    break
            # recall@k
            for k in K_VALUES:
                topk = ranked[:k]
                hits = sum(1 for g in gt if any(is_hit({g}, d) for d in topk))
                agg[k]["hit"] += hits
                agg[k]["total"] += len(gt)

        recalls = {k: agg[k]["hit"] / agg[k]["total"] if agg[k]["total"] else 0 for k in K_VALUES}
        p1 = p1_hit / n if n else 0
        mrr = mrr_sum / n if n else 0
        row = f"{alpha:>4.1f}   " + "  ".join(f"{recalls[k]:.3f}" for k in K_VALUES) + f"   {p1:.3f}  {mrr:.3f}"
        print(row)

        # 최적 기준: recall@10 (실용적 상위 구간)
        if best is None or recalls[10] > best[1]:
            best = (alpha, recalls[10], recalls, p1, mrr)

    print("\n" + "=" * 78)
    a, r10, recalls, p1, mrr = best
    print(f"recall@10 최적 alpha = {a}  (BM25 {a*100:.0f}% + Dense {(1-a)*100:.0f}%)")
    print(f"  recall@10={r10:.3f}  recall@50={recalls[50]:.3f}  P@1={p1:.3f}  MRR={mrr:.3f}")
    print(f"현재 서비스 설정(법령 alpha=0.20)과 비교해보세요.")


if __name__ == "__main__":
    main()
