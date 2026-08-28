"""법령/판례 검색 평가 — 실제 DB 임베딩 pipeline 사용.

원본(evaluation/retrieval_judge_eval.py) 대비 차이:
  1. recall@k를 법령/판례 타입별로 독립적으로 계산
     (원본은 법령+판례가 뒤섞인 리스트를 통째로 [:k]로 잘라서, 판례 후보가
     상위권을 차지하면 법령 recall이 부당하게 낮아지는 버그가 있었음)
  2. eval_set 입력 경로와 결과 출력 경로를 CLI로 지정 가능
     (원본은 evaluation/eval_set.json 고정이라 tune/test를 분리해서
     따로 돌릴 수 없었음)
  3. precision@1, MRR(Mean Reciprocal Rank) 추가
     (recall@k만으로는 "가장 자신 있게 1등으로 뽑은 게 얼마나 정확한지"를
     알 수 없어서 별도로 계산한다)

참고: query expansion의 eval_set 힌트 주입("overfit_mode") 기능은
pipeline/retrieval/query_expansion/query_expansion.py에서 완전히
삭제했으므로, 이 스크립트는 그 옵션 없이 항상 정직하게 평가한다.

사용 예
-------
    python evaluation/legal_retrieval_eval.py \
        --eval-set evaluation/eval_set_test.json \
        --results evaluation/eval_results_test.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

from rank_bm25 import BM25Okapi

CLAUSE_WORKERS = 4   # 특약 동시 처리 수 (API rate limit에 맞게 조절)
EMBED_WORKERS = 1    # dense_retrieval: embed_vertex only

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.retrieval.bm25_retrieval import (
    build_query_tokens,
    load_case_law_from_db,
    load_law_child_from_db,
    tokenize,
)
from pipeline.retrieval import dense_retrieval
from pipeline.retrieval.query_expansion.query_expansion import expand_clause
from pipeline.retrieval.query_expansion.retrieval_adapter import build_retrieval_payload

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

TOP_K = 50   # 후보군 깊이 (recall@20 vs recall@50 비교 진단용으로 확대)
RRF_K = 60
RECALL_K_VALUES = [1, 3, 5, 10, 20, 50]
EMBED_COLS = ["embed_vertex"]
LAW_KEEP_COLS = ["clause_key", "law_name", "article_no", "paragraph_no", "child_text"]
PREC_KEEP_COLS = ["case_id", "case_number", "judgment_summary"]


def load_corpus() -> dict:
    """BM25 + Dense 검색에 필요한 코퍼스를 DB에서 로드한다."""
    log.info("▶ BM25 코퍼스 로드 중...")
    law_df = load_law_child_from_db()
    prec_df = load_case_law_from_db()

    law_docs = [
        {
            "clause_key": str(r["clause_key"]),
            "text": str(r.get("bm25_target") or r.get("child_text") or ""),
        }
        for _, r in law_df.iterrows()
    ]
    prec_docs = [
        {"case_number": str(r["case_number"]), "text": str(r.get("bm25_target") or "")}
        for _, r in prec_df.iterrows()
    ]

    log.info("  법령 BM25 인덱스 구축 중 (%d건)...", len(law_docs))
    law_bm25 = BM25Okapi([tokenize(d["text"]) for d in law_docs])
    log.info("  판례 BM25 인덱스 구축 중 (%d건)...", len(prec_docs))
    prec_bm25 = BM25Okapi([tokenize(d["text"]) for d in prec_docs])

    log.info("▶ Dense 임베딩 청크 로드 중 (%s)...", ", ".join(EMBED_COLS))
    law_chunks = {
        col: dense_retrieval.load_chunks(dense_retrieval.LAW_TABLE, col, LAW_KEEP_COLS)
        for col in EMBED_COLS
    }
    prec_chunks = {
        col: dense_retrieval.load_chunks(dense_retrieval.PREC_TABLE, col, PREC_KEEP_COLS)
        for col in EMBED_COLS
    }

    log.info("▶ 코퍼스 로드 완료 (법령 %d건, 판례 %d건)", len(law_docs), len(prec_docs))
    return {
        "law_docs": law_docs, "law_bm25": law_bm25,
        "prec_docs": prec_docs, "prec_bm25": prec_bm25,
        "law_chunks": law_chunks, "prec_chunks": prec_chunks,
    }


def retrieve_clause(clause: str, corpus: dict) -> dict:
    """단일 특약에 대해 Query Expansion → BM25 → Dense → RRF를 실행한다."""
    expansion = expand_clause(clause)
    payload = build_retrieval_payload(expansion, clause_text=clause)
    log.info("    QE keywords: %s", payload["bm25_keywords"])

    query_tokens = build_query_tokens(payload["bm25_keywords"])
    bm25_hits: dict[str, tuple[int, str]] = {}
    for docs, bm25, source_type, id_field in [
        (corpus["law_docs"], corpus["law_bm25"], "law", "clause_key"),
        (corpus["prec_docs"], corpus["prec_bm25"], "precedent", "case_number"),
    ]:
        scores = bm25.get_scores(query_tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        for rank, idx in enumerate(top_idx, 1):
            bm25_hits[docs[idx][id_field]] = (rank, source_type)
    log.info(
        "    BM25: 법령 %d건, 판례 %d건",
        sum(1 for _, st in bm25_hits.values() if st == "law"),
        sum(1 for _, st in bm25_hits.values() if st == "precedent"),
    )

    def _embed_and_search(col: str) -> list[tuple[str, int, str]]:
        query_vec = dense_retrieval.embed_query(payload["dense_query"], col)
        hits = []
        for chunks, source_type, id_field in [
            (corpus["law_chunks"][col], "law", "clause_key"),
            (corpus["prec_chunks"][col], "precedent", "case_number"),
        ]:
            rows = dense_retrieval.search_similar(query_vec, chunks, col, TOP_K)
            for rank, (_, row) in enumerate(rows.iterrows(), 1):
                hits.append((str(row[id_field]), rank, source_type))
        return hits

    dense_hits: dict[str, tuple[int, str]] = {}
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        for hit_list in pool.map(_embed_and_search, EMBED_COLS):
            for doc_id, rank, source_type in hit_list:
                if doc_id not in dense_hits or rank < dense_hits[doc_id][0]:
                    dense_hits[doc_id] = (rank, source_type)
    log.info(
        "    Dense: 법령 %d건, 판례 %d건",
        sum(1 for _, st in dense_hits.values() if st == "law"),
        sum(1 for _, st in dense_hits.values() if st == "precedent"),
    )

    all_ids = set(bm25_hits) | set(dense_hits)
    scored = []
    for doc_id in all_ids:
        b_rank = bm25_hits[doc_id][0] if doc_id in bm25_hits else 1000
        d_rank = dense_hits[doc_id][0] if doc_id in dense_hits else 1000
        source_type = (bm25_hits.get(doc_id) or dense_hits.get(doc_id))[1]
        scored.append({
            "doc_id": doc_id,
            "source_type": source_type,
            "rrf_score": round(1 / (RRF_K + b_rank) + 1 / (RRF_K + d_rank), 6),
            "bm25_rank": b_rank if b_rank != 1000 else None,
            "dense_rank": d_rank if d_rank != 1000 else None,
        })

    scored.sort(key=lambda x: (
        -x["rrf_score"],
        x["bm25_rank"] if x["bm25_rank"] is not None else 1000,
        x["dense_rank"] if x["dense_rank"] is not None else 1000,
        x["doc_id"],
    ))

    # 주의: 여기서 상위 TOP_K개로 미리 자르지 않는다. law/precedent가 섞인
    # 상태로 자르면 타입별 recall@k 계산 시 후보가 부족해질 수 있다.
    results = scored
    for rank, item in enumerate(results, 1):
        item["rank"] = rank

    bm25_results = sorted(
        [{"doc_id": doc_id, "source_type": st, "rank": rank} for doc_id, (rank, st) in bm25_hits.items()],
        key=lambda x: x["rank"],
    )
    dense_results = sorted(
        [{"doc_id": doc_id, "source_type": st, "rank": rank} for doc_id, (rank, st) in dense_hits.items()],
        key=lambda x: x["rank"],
    )

    log.info("    RRF 최종: %d건", len(results))
    return {"bm25": bm25_results, "dense": dense_results, "rrf": results}


def _type_ranked_ids(clause_result: dict, method: str, source_type: str) -> list[str]:
    """clause_result[method]에서 지정한 타입만, 순위 순서 그대로 doc_id 리스트로 뽑는다."""
    return [r["doc_id"] for r in clause_result[method] if r["source_type"] == source_type]


def _recall_at_k(
    per_clause: list[dict],
    method: str,
    k: int,
    gt_laws: set[str],
    gt_cases: set[str],
) -> dict:
    """법령/판례를 각각 독립적으로 상위 k개씩 잘라서 recall/precision/F1을 계산한다.

    - recall@k    = (상위 k개 안에 든 정답 수) / (전체 정답 수)
    - precision@k = (상위 k개 중 정답인 것 수) / (내놓은 상위 k개 결과 수)
      * gt에 특약과 무관한 조문이 섞여 정답 수가 부풀려진 상황에서, recall만
        보면 억울하게 낮아진다. precision은 분모가 '검색 결과 수'라 이 문제에
        덜 휘둘린다. 단 precision만 보면 '적게 내놓고 맞히면 만점' 함정이 있어
        recall과 함께 봐야 한다.
    - f1@k = recall과 precision의 조화평균
    """
    pool_laws: set[str] = set()
    pool_cases: set[str] = set()
    for clause_result in per_clause:
        pool_laws.update(_type_ranked_ids(clause_result, method, "law")[:k])
        pool_cases.update(_type_ranked_ids(clause_result, method, "precedent")[:k])
    law_hits = _count_hits(gt_laws, pool_laws)
    prec_hits = _count_hits(gt_cases, pool_cases)
    law_total = len({_article_of(g) for g in gt_laws})   # 조 단위 중복 제거
    prec_total = len({_article_of(g) for g in gt_cases})

    # precision 분모: 실제로 내놓은 결과 수 (pool 크기)
    law_returned = len(pool_laws)
    case_returned = len(pool_cases)
    # precision hit: 내놓은 결과 중 gt에 실제로 속하는 것 수
    law_correct = sum(1 for d in pool_laws if _is_hit(gt_laws, d))
    case_correct = sum(1 for d in pool_cases if _is_hit(gt_cases, d))

    law_recall = law_hits / law_total if law_total else None
    law_precision = law_correct / law_returned if law_returned else None
    prec_recall = prec_hits / prec_total if prec_total else None
    prec_precision = case_correct / case_returned if case_returned else None

    return {
        "law_hits": law_hits,
        "law_total": law_total,
        "law_returned": law_returned,
        "law_correct": law_correct,
        "law_recall": round(law_recall, 4) if law_recall is not None else None,
        "law_precision": round(law_precision, 4) if law_precision is not None else None,
        "law_f1": round(_f1(law_recall, law_precision), 4) if (law_recall and law_precision) else None,
        "precedent_hits": prec_hits,
        "precedent_total": prec_total,
        "precedent_returned": case_returned,
        "precedent_correct": case_correct,
        "precedent_recall": round(prec_recall, 4) if prec_recall is not None else None,
        "precedent_precision": round(prec_precision, 4) if prec_precision is not None else None,
        "precedent_f1": round(_f1(prec_recall, prec_precision), 4) if (prec_recall and prec_precision) else None,
    }


def _f1(recall: float | None, precision: float | None) -> float:
    if not recall or not precision:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def _article_of(key: str) -> str:
    """clause_key에서 항/호/목을 떼고 '조' 단위까지만 남긴다."""
    return re.sub(r'_(제\d+항|제\d+호|제\d+목).*$', '', key)


def _is_hit(gt_set: set[str], doc_id: str) -> bool:
    """조 단위로 매칭한다. gt와 검색결과 둘 다 '조'까지만 비교하므로,
    항이 달라도 같은 조이면 정답으로 처리한다."""
    doc_art = _article_of(doc_id)
    return any(_article_of(gt) == doc_art for gt in gt_set)


def _count_hits(gt_set: set[str], pool: set[str]) -> int:
    pool_arts = {_article_of(d) for d in pool}
    gt_arts = {_article_of(g) for g in gt_set}
    count = 0
    for gt_art in gt_arts:
        if gt_art in pool_arts:
            count += 1
    return count


def _precision1_and_mrr(
    per_clause: list[dict],
    method: str,
    gt_laws: set[str],
    gt_cases: set[str],
) -> dict:
    """법령/판례 각각에 대해 precision@1과 MRR을 계산한다.

    - precision@1: 해당 타입 후보 중 1위로 뽑힌 문서가 gt에 속하는 케이스의 비율
      (gt가 있는 케이스만 분모에 포함)
    - MRR: gt에 처음 맞은 후보의 순위(rank)의 역수(1/rank) 평균
      (그 타입 후보 안에 gt가 아예 없으면 0으로 취급)
    """
    law_p1_hits, law_p1_total = 0, 0
    prec_p1_hits, prec_p1_total = 0, 0
    law_rr_sum, law_rr_total = 0.0, 0
    prec_rr_sum, prec_rr_total = 0.0, 0

    for clause_result in per_clause:
        law_ranked = _type_ranked_ids(clause_result, method, "law")
        prec_ranked = _type_ranked_ids(clause_result, method, "precedent")

        if gt_laws:
            law_p1_total += 1
            if law_ranked and _is_hit(gt_laws, law_ranked[0]):
                law_p1_hits += 1
            law_rr_total += 1
            for rank, doc_id in enumerate(law_ranked, 1):
                if _is_hit(gt_laws, doc_id):
                    law_rr_sum += 1 / rank
                    break

        if gt_cases:
            prec_p1_total += 1
            if prec_ranked and _is_hit(gt_cases, prec_ranked[0]):
                prec_p1_hits += 1
            prec_rr_total += 1
            for rank, doc_id in enumerate(prec_ranked, 1):
                if _is_hit(gt_cases, doc_id):
                    prec_rr_sum += 1 / rank
                    break

    return {
        "law_precision_at_1": round(law_p1_hits / law_p1_total, 4) if law_p1_total else None,
        "law_precision_at_1_n": law_p1_total,
        "law_mrr": round(law_rr_sum / law_rr_total, 4) if law_rr_total else None,
        "law_mrr_n": law_rr_total,
        "precedent_precision_at_1": round(prec_p1_hits / prec_p1_total, 4) if prec_p1_total else None,
        "precedent_precision_at_1_n": prec_p1_total,
        "precedent_mrr": round(prec_rr_sum / prec_rr_total, 4) if prec_rr_total else None,
        "precedent_mrr_n": prec_rr_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="법령/판례 검색 recall / precision@1 / MRR 평가")
    parser.add_argument("--eval-set", type=Path, required=True, help="평가셋 JSON 경로")
    parser.add_argument("--results", type=Path, required=True, help="결과 저장 경로")
    args = parser.parse_args()

    log.info("▶ eval_set 로드: %s", args.eval_set)
    cases = json.loads(args.eval_set.read_text(encoding="utf-8"))
    log.info("  케이스 %d개", len(cases))

    corpus = load_corpus()

    case_results = []
    for i, case in enumerate(cases):
        case_id = case["id"]
        clauses = [c["normalized"] for c in case["clauses"]]
        gt_laws = set(case["gt_laws"])
        # gt_laws_explicit이 없는(예전 스키마) eval_set은 gt_laws 전체를 explicit으로 취급
        gt_laws_explicit = set(case.get("gt_laws_explicit", case["gt_laws"]))
        gt_cases = set(case["gt_cases"])
        log.info(
            "[%d/%d] %s: 특약 %d개, gt_laws=%d, gt_cases=%d",
            i + 1, len(cases), case_id, len(clauses), len(gt_laws), len(gt_cases),
        )

        per_clause: list[dict] = [{"bm25": [], "dense": [], "rrf": []} for _ in clauses]

        def _retrieve(args_tuple: tuple[int, str]) -> tuple[int, dict]:
            j, clause = args_tuple
            log.info("  특약 [%d/%d]: %s...", j + 1, len(clauses), clause[:60])
            return j, retrieve_clause(clause, corpus)

        with ThreadPoolExecutor(max_workers=CLAUSE_WORKERS) as pool:
            futures = {pool.submit(_retrieve, (j, c)): j for j, c in enumerate(clauses)}
            for future in as_completed(futures):
                try:
                    j, results = future.result()
                    per_clause[j] = results
                except Exception as exc:
                    log.error("  특약 검색 실패: %s", exc)

        recall_at_k: dict[int, dict] = {}
        recall_at_k_explicit: dict[int, dict] = {}
        for k in RECALL_K_VALUES:
            recall_at_k[k] = {
                method: _recall_at_k(per_clause, method, k, gt_laws, gt_cases)
                for method in ("bm25", "dense", "rrf")
            }
            recall_at_k_explicit[k] = {
                method: _recall_at_k(per_clause, method, k, gt_laws_explicit, gt_cases)
                for method in ("bm25", "dense", "rrf")
            }

        precision_mrr = {
            method: _precision1_and_mrr(per_clause, method, gt_laws, gt_cases)
            for method in ("bm25", "dense", "rrf")
        }

        rrf20 = recall_at_k[20]["rrf"]
        rrf_pm = precision_mrr["rrf"]
        log.info(
            "  Recall@20 [RRF]  법령: %.3f (%d/%d)  판례: %.3f (%d/%d)",
            rrf20["law_recall"] or 0, rrf20["law_hits"], rrf20["law_total"],
            rrf20["precedent_recall"] or 0, rrf20["precedent_hits"], rrf20["precedent_total"],
        )
        log.info(
            "  [RRF] 법령 P@1=%s MRR=%s  판례 P@1=%s MRR=%s",
            rrf_pm["law_precision_at_1"], rrf_pm["law_mrr"],
            rrf_pm["precedent_precision_at_1"], rrf_pm["precedent_mrr"],
        )
        clause_records = [
            {
                "clause": clauses[j],
                "bm25": per_clause[j]["bm25"],
                "dense": per_clause[j]["dense"],
                "rrf": per_clause[j]["rrf"],
            }
            for j in range(len(clauses))
        ]
        case_results.append({
            "id": case_id,
            "gt_laws": list(gt_laws),
            "gt_laws_explicit": list(gt_laws_explicit),
            "gt_cases": list(gt_cases),
            "recall_at_k": recall_at_k,
            "recall_at_k_explicit": recall_at_k_explicit,
            "precision_mrr": precision_mrr,
            "clauses": clause_records,
        })
        args.results.write_text(
            json.dumps(case_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info("  결과 저장: %s (%d/%d)", args.results, i + 1, len(cases))

    def _agg_law(key: str, k: int, method: str) -> tuple:
        hits = sum(c[key][k][method]["law_hits"] for c in case_results)
        total = sum(c[key][k][method]["law_total"] for c in case_results)
        returned = sum(c[key][k][method]["law_returned"] for c in case_results)
        correct = sum(c[key][k][method]["law_correct"] for c in case_results)
        recall = hits / total if total else 0.0
        precision = correct / returned if returned else 0.0
        f1 = _f1(recall, precision)
        return recall, precision, f1, hits, total, correct, returned

    print("\n" + "=" * 70)
    print(f"전체 집계 - 직접인용+보강 결합 ({len(case_results)}케이스)  [법령 기준]")
    print("=" * 70)
    for k in RECALL_K_VALUES:
        for method in ("bm25", "dense", "rrf"):
            r, p, f1, hits, total, correct, returned = _agg_law("recall_at_k", k, method)
            print(
                f"K={k:>2} [{method:>4}]  recall={r:.4f} ({hits}/{total})  "
                f"precision={p:.4f} ({correct}/{returned})  F1={f1:.4f}"
            )

    print("\n" + "=" * 70)
    print(f"전체 집계 - 직접인용만 (신뢰도 높음) ({len(case_results)}케이스)  [법령 기준]")
    print("=" * 70)
    for k in RECALL_K_VALUES:
        for method in ("bm25", "dense", "rrf"):
            r, p, f1, hits, total, correct, returned = _agg_law("recall_at_k_explicit", k, method)
            print(
                f"K={k:>2} [{method:>4}]  recall={r:.4f} ({hits}/{total})  "
                f"precision={p:.4f} ({correct}/{returned})  F1={f1:.4f}"
            )

    print("\n" + "=" * 70)
    print("전체 Precision@1 / MRR 집계")
    print("=" * 70)
    for method in ("bm25", "dense", "rrf"):
        law_p1_hits = sum(c["precision_mrr"][method]["law_precision_at_1_n"] * (c["precision_mrr"][method]["law_precision_at_1"] or 0) for c in case_results)
        law_p1_total = sum(c["precision_mrr"][method]["law_precision_at_1_n"] for c in case_results)
        law_rr_sum = sum((c["precision_mrr"][method]["law_mrr"] or 0) * c["precision_mrr"][method]["law_mrr_n"] for c in case_results)
        law_rr_total = sum(c["precision_mrr"][method]["law_mrr_n"] for c in case_results)
        prec_p1_hits = sum(c["precision_mrr"][method]["precedent_precision_at_1_n"] * (c["precision_mrr"][method]["precedent_precision_at_1"] or 0) for c in case_results)
        prec_p1_total = sum(c["precision_mrr"][method]["precedent_precision_at_1_n"] for c in case_results)
        prec_rr_sum = sum((c["precision_mrr"][method]["precedent_mrr"] or 0) * c["precision_mrr"][method]["precedent_mrr_n"] for c in case_results)
        prec_rr_total = sum(c["precision_mrr"][method]["precedent_mrr_n"] for c in case_results)

        law_p1 = round(law_p1_hits / law_p1_total, 4) if law_p1_total else None
        law_mrr = round(law_rr_sum / law_rr_total, 4) if law_rr_total else None
        prec_p1 = round(prec_p1_hits / prec_p1_total, 4) if prec_p1_total else None
        prec_mrr = round(prec_rr_sum / prec_rr_total, 4) if prec_rr_total else None
        print(
            f"[{method:>4}]  법령 P@1={law_p1} (n={law_p1_total})  법령 MRR={law_mrr} (n={law_rr_total})  "
            f"판례 P@1={prec_p1} (n={prec_p1_total})  판례 MRR={prec_mrr} (n={prec_rr_total})"
        )


if __name__ == "__main__":
    main()