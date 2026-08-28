"""FastAPI lifespan에서 1회 로드되는 검색 서비스 싱글턴.

Dense 유사도 검색은 in-memory numpy 대신 PostgreSQL pgvector (<=> 코사인 거리)에 위임한다.
→ load() 시 임베딩 벡터를 메모리에 올리지 않아 RAM 사용량이 대폭 줄어든다.

리랭킹 방식: alpha_hybrid
  - 법령:  가중 RRF (K=30, w_bm25:w_dense=1:2) — 순위 기반 융합, Dense 우대
  - 판례:  α=0.70 → BM25 70% + Dense 30%  (BM25 위주 — 키워드 기반, 추후 RRF 검토)
  - 법령/판례 각각 독립 랭킹 후 합산 → 두 도메인 결과 보장

alpha 값은 alpha_sweep 평가 최적값 (alpha_sweep_2d_20260530_235115) 사용.

의존 관계:
  BM25  : pipeline.retrieval.bm25_retrieval (tokenize, build_query_tokens, load_*_from_db)
  Embed : pipeline.retrieval.dense_retrieval (embed_query — Vertex AI 호출)
  DB    : shared.db.connection (pgvector 쿼리)
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

TOP_K      = 20   # 법령/판례 각각 상위 K개 → 총 최대 2*TOP_K 반환
ALPHA_LAW  = 0.20  # 법령: BM25 20% + Dense 80%  (구 방식, 미사용 — 참고용 보존)
ALPHA_PREC = 0.70  # 판례: BM25 70% + Dense 30%

# 법령 리랭킹: 가중 RRF (score = w_bm25/(K+rank_bm25) + w_dense/(K+rank_dense))
# v2 골드셋 기준 그리드서치(tune 70건)에서 recall@10 최적값. alpha=0.2 대비 R@10 0.418→0.463.
RRF_K_LAW       = 30
RRF_W_BM25_LAW  = 1.0
RRF_W_DENSE_LAW = 2.0

# pgvector <=> 는 (1 - cosine_similarity) 를 반환한다.
_MIN_SIM  = 0.2
_MAX_DIST = round(1.0 - _MIN_SIM, 6)  # 0.8


class RetrievalService:
    def __init__(self) -> None:
        self._corpus: dict | None = None

    # ------------------------------------------------------------------ #
    # 공개 상태 API                                                        #
    # ------------------------------------------------------------------ #

    @property
    def is_ready(self) -> bool:
        return self._corpus is not None

    @property
    def corpus(self) -> dict:
        if self._corpus is None:
            raise RuntimeError("RetrievalService.load()가 아직 실행되지 않았습니다.")
        return self._corpus

    # ------------------------------------------------------------------ #
    # 초기화 (서버 시작 시 1회)                                            #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """DB에서 BM25 코퍼스를 로드하고 인덱스를 구축한다.

        Dense 임베딩은 쿼리 시 pgvector에 위임하므로 여기서 로드하지 않는다.
        """
        from rank_bm25 import BM25Okapi

        from pipeline.retrieval.bm25_retrieval import (
            load_case_law_from_db,
            load_law_child_from_db,
            tokenize,
        )

        log.info("▶ BM25 코퍼스 로드 시작...")

        law_df  = load_law_child_from_db()
        prec_df = load_case_law_from_db()

        law_docs = [
            {"clause_key": str(r["clause_key"]), "text": str(r["child_text"] or "")}
            for _, r in law_df.iterrows()
        ]
        prec_docs = [
            {"case_number": str(r["case_number"]), "text": str(r.get("bm25_target") or "")}
            for _, r in prec_df.iterrows()
        ]

        log.info(
            "  BM25 인덱스 구축 중 (법령 %d건, 판례 %d건)...",
            len(law_docs),
            len(prec_docs),
        )
        law_bm25  = BM25Okapi([tokenize(d["text"]) for d in law_docs])
        prec_bm25 = BM25Okapi([tokenize(d["text"]) for d in prec_docs])

        self._corpus = {
            "law_docs":  law_docs,
            "law_bm25":  law_bm25,
            "prec_docs": prec_docs,
            "prec_bm25": prec_bm25,
        }
        log.info("▶ BM25 코퍼스 로드 완료 (Dense 검색은 쿼리 시 pgvector에 위임)")

    # ------------------------------------------------------------------ #
    # 하이브리드 검색 (BM25 + Dense → Alpha Hybrid)                       #
    # ------------------------------------------------------------------ #

    def retrieve(self, payload: dict) -> dict:
        """BM25 → Dense(pgvector) → Alpha Hybrid 실행.

        Parameters
        ----------
        payload:
            build_retrieval_payload() 결과.
            필수 키: "bm25_keywords" (list[str]), "dense_query" (str)

        Returns
        -------
        dict with keys:
            "bm25" : BM25 순위 목록
            "dense": Dense 순위 목록
            "law"  : Alpha Hybrid 법령 순위 목록 (rank 1~TOP_K, 독립)
            "prec" : Alpha Hybrid 판례 순위 목록 (rank 1~TOP_K, 독립)

        법령/판례는 평가(legal_retrieval_eval_multi)와 동일하게 각각 독립 랭킹한다.
        슬롯 경쟁 없이 두 도메인 모두 top-K를 보장한다.
        """
        from pipeline.retrieval.bm25_retrieval import build_query_tokens
        from pipeline.retrieval.dense_retrieval import embed_query

        corpus = self.corpus

        # ── BM25 (법령/판례 각각, raw score 포함) ─────────────────────
        query_tokens = build_query_tokens(payload["bm25_keywords"])

        bm25_law_scores  = self._bm25_scores(query_tokens, corpus["law_bm25"],  corpus["law_docs"],  "law",       "clause_key")
        bm25_prec_scores = self._bm25_scores(query_tokens, corpus["prec_bm25"], corpus["prec_docs"], "precedent", "case_number")

        # ── Dense (pgvector, similarity score 포함) ───────────────────
        query_vec = embed_query(payload["dense_query"], "embed_vertex")
        dense_law_scores, dense_prec_scores = self._pgvector_scores(query_vec, TOP_K * 2)

        # ── Min-Max 정규화 (평가와 동일: 법령+판례 합친 풀에서 정규화) ──
        # 법령 id(clause_key)와 판례 id(case_number)는 ID 공간이 달라 충돌 없음.
        bm25_raw_all  = {d: s for d, (_, __, s) in {**bm25_law_scores,  **bm25_prec_scores}.items()}
        dense_raw_all = {d: s for d, (_, __, s) in {**dense_law_scores, **dense_prec_scores}.items()}
        norm_bm25  = self._minmax(bm25_raw_all)
        norm_dense = self._minmax(dense_raw_all)

        # ── 법령: 가중 RRF (rank 기반) / 판례: 기존 Alpha Hybrid ──────
        law_results  = self._weighted_rrf(
            bm25_law_scores, dense_law_scores,
            RRF_W_BM25_LAW, RRF_W_DENSE_LAW, RRF_K_LAW, "law", TOP_K)
        prec_results = self._alpha_hybrid(bm25_prec_scores, dense_prec_scores, norm_bm25, norm_dense, ALPHA_PREC, "precedent", TOP_K)

        # BM25 / Dense 결과 목록 (디버깅/검증용)
        bm25_list = sorted(
            [{"doc_id": d, "source_type": st, "rank": r}
             for d, (r, st, _) in {**bm25_law_scores, **bm25_prec_scores}.items()],
            key=lambda x: x["rank"],
        )
        dense_list = sorted(
            [{"doc_id": d, "source_type": st, "rank": r}
             for d, (r, st, _) in {**dense_law_scores, **dense_prec_scores}.items()],
            key=lambda x: x["rank"],
        )

        return {"bm25": bm25_list, "dense": dense_list, "law": law_results, "prec": prec_results}

    # ------------------------------------------------------------------ #
    # BM25 점수 수집                                                       #
    # ------------------------------------------------------------------ #

    def _bm25_scores(
        self,
        query_tokens: list[str],
        bm25,
        docs: list[dict],
        source_type: str,
        id_field: str,
    ) -> dict[str, tuple[int, str, float]]:
        """doc_id → (rank, source_type, raw_score) 반환."""
        scores  = bm25.get_scores(query_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K * 2]
        return {
            docs[idx][id_field]: (rank, source_type, float(scores[idx]))
            for rank, idx in enumerate(top_idx, 1)
        }

    # ------------------------------------------------------------------ #
    # Dense 점수 수집 (pgvector)                                          #
    # ------------------------------------------------------------------ #

    def _pgvector_scores(
        self,
        query_vec,
        top_k: int,
    ) -> tuple[dict[str, tuple[int, str, float]], dict[str, tuple[int, str, float]]]:
        """법령/판례 각각 doc_id → (rank, source_type, similarity) 반환."""
        from sqlalchemy import text
        from shared.db.connection import get_db_client

        vec_literal = "[" + ",".join(str(float(x)) for x in query_vec) + "]"
        max_dist    = _MAX_DIST
        db          = get_db_client()

        law_rows = db.fetch_all(
            text(f"""
                SELECT clause_key::text AS doc_id,
                       1.0 - (embed_vertex <=> '{vec_literal}'::vector) AS similarity
                FROM   law_child
                WHERE  embed_vertex IS NOT NULL
                  AND  embed_vertex <=> '{vec_literal}'::vector <= {max_dist}
                ORDER BY embed_vertex <=> '{vec_literal}'::vector
                LIMIT  :top_k
            """),
            {"top_k": top_k},
        )

        prec_rows = db.fetch_all(
            text(f"""
                SELECT case_number::text AS doc_id,
                       1.0 - (embed_vertex <=> '{vec_literal}'::vector) AS similarity
                FROM   case_law
                WHERE  embed_vertex IS NOT NULL
                  AND  embed_vertex <=> '{vec_literal}'::vector <= {max_dist}
                ORDER BY embed_vertex <=> '{vec_literal}'::vector
                LIMIT  :top_k
            """),
            {"top_k": top_k},
        )

        law_scores  = {str(r["doc_id"]): (rank, "law",       float(r["similarity"])) for rank, r in enumerate(law_rows,  1)}
        prec_scores = {str(r["doc_id"]): (rank, "precedent", float(r["similarity"])) for rank, r in enumerate(prec_rows, 1)}

        return law_scores, prec_scores

    # ------------------------------------------------------------------ #
    # Alpha Hybrid 융합                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _minmax(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {k: 0.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def _weighted_rrf(
        self,
        bm25_scores:  dict[str, tuple[int, str, float]],
        dense_scores: dict[str, tuple[int, str, float]],
        w_bm25: float,
        w_dense: float,
        rrf_k: int,
        source_type: str,
        top_k: int,
    ) -> list[dict]:
        """가중 RRF로 융합 후 상위 top_k 반환.

        score = w_bm25 * 1/(rrf_k + rank_bm25) + w_dense * 1/(rrf_k + rank_dense)
        점수(raw)가 아니라 순위(rank)만 사용하므로 점수 정규화 왜곡이 없다.
        rank 는 bm25_scores/dense_scores 튜플의 [0] (1부터 시작하는 도메인 내 순위)이다.
        """
        all_ids = set(bm25_scores) | set(dense_scores)
        scored = []
        for doc_id in all_ids:
            s = 0.0
            b_rank = bm25_scores[doc_id][0]  if doc_id in bm25_scores  else None
            d_rank = dense_scores[doc_id][0] if doc_id in dense_scores else None
            if b_rank is not None:
                s += w_bm25 * 1.0 / (rrf_k + b_rank)
            if d_rank is not None:
                s += w_dense * 1.0 / (rrf_k + d_rank)
            scored.append({
                "doc_id":       doc_id,
                "source_type":  source_type,
                "rank":         0,
                "hybrid_score": round(s, 8),
                "bm25_rank":    b_rank,
                "dense_rank":   d_rank,
            })
        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top = scored[:top_k]
        for rank, item in enumerate(top, 1):
            item["rank"] = rank
        return top

    def _alpha_hybrid(
        self,
        bm25_scores:  dict[str, tuple[int, str, float]],
        dense_scores: dict[str, tuple[int, str, float]],
        norm_bm25:  dict[str, float],
        norm_dense: dict[str, float],
        alpha: float,
        source_type: str,
        top_k: int,
    ) -> list[dict]:
        """alpha * norm_bm25 + (1-alpha) * norm_dense 로 융합 후 상위 top_k 반환.

        norm_bm25/norm_dense 는 법령+판례 합친 풀에서 이미 정규화된 값(평가와 동일).
        이 도메인(source_type)에 속한 doc_id 들만 골라 랭킹한다.
        """
        all_ids = set(bm25_scores) | set(dense_scores)
        scored  = []
        for doc_id in all_ids:
            nb     = norm_bm25.get(doc_id,  0.0)
            nd     = norm_dense.get(doc_id, 0.0)
            hybrid = alpha * nb + (1.0 - alpha) * nd
            scored.append({
                "doc_id":       doc_id,
                "source_type":  source_type,
                "rank":         0,
                "hybrid_score": round(hybrid, 6),
                "bm25_rank":    bm25_scores[doc_id][0]  if doc_id in bm25_scores  else None,
                "dense_rank":   dense_scores[doc_id][0] if doc_id in dense_scores else None,
            })

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top = scored[:top_k]
        for rank, item in enumerate(top, 1):  # 도메인 내 독립 rank 1~top_k
            item["rank"] = rank
        return top


retrieval_service = RetrievalService()