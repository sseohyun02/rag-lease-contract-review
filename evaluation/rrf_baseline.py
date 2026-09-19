"""표준 RRF(K=60) 베이스라인 — 평가 전용.

이 모듈은 프로덕션 리랭커가 아니다.
프로덕션 융합은 pipeline/retrieval/retrieval_service.py에서 인라인으로 수행한다
(법령: 가중 RRF K=30, BM25:Dense=1:2 / 판례: α-하이브리드 α=0.70).

여기서는 sweep으로 튠한 프로덕션 방식과 비교하기 위한 '가중 없는 표준 RRF(K=60)'
베이스라인만 제공한다. 상수 K=60은 Cormack et al.(SIGIR 2009)의 기본값이다.
현재 evaluation/legal_retrieval_eval_multi.py가 run_rrf를 호출한다.

(구 pipeline/reranking/reranker.py에서 실제로 쓰이던 run_rrf/rrf_score/K만
 그대로 옮겨온 것이며, 파일 기반 배치 실행부는 사용처가 없어 제외했다.)
"""
from __future__ import annotations

# RRF 상수 (Cormack et al. 2009 기본값)
K = 60


def rrf_score(bm25_rank: int, dense_rank: int, k: int) -> float:
    return 1 / (k + bm25_rank) + 1 / (k + dense_rank)


def run_rrf(bm25_map: dict, dense_map: dict, k: int, top_n: int) -> list:
    results = []

    for idx, bm25_item in bm25_map.items():
        special_terms = bm25_item["special_terms"]
        bm25_ranks    = bm25_item["rank_map"]

        dense_records = dense_map.get(special_terms, [])
        dense_ranks   = {r["doc_id"]: r for r in dense_records}

        # 실제 pool 크기 기반 penalty: 해당 리스트의 마지막 순위 + 1
        bm25_penalty  = (max(bm25_ranks.values()) + 1) if bm25_ranks else 1
        dense_penalty = (len(dense_records) + 1) if dense_records else 1

        all_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())

        scored = []
        for doc_id in all_ids:
            b_rank = bm25_ranks.get(doc_id, bm25_penalty)
            d_rank = dense_ranks[doc_id]["rank"] if doc_id in dense_ranks else dense_penalty

            score = rrf_score(b_rank, d_rank, k)

            doc_text = dense_ranks[doc_id]["doc_text"] if doc_id in dense_ranks else ""
            summary  = dense_ranks[doc_id]["summary"]  if doc_id in dense_ranks else ""

            scored.append({
                "doc_id"     : doc_id,
                "rrf_score"  : round(score, 6),
                "bm25_rank"  : b_rank if doc_id in bm25_ranks else None,
                "dense_rank" : d_rank if doc_id in dense_ranks else None,
                "doc_text"   : doc_text,
                "summary"    : summary,
            })

        scored.sort(key=lambda x: x["rrf_score"], reverse=True)
        top = scored[:top_n]

        for rank, item in enumerate(top, 1):
            item["rank"] = rank

        results.append({
            "index"         : idx,
            "special_terms" : special_terms,
            "top_matches"   : top,
        })

    return results
