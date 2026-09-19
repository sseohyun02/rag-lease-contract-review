"""Dense 코사인 유사도 임계값(min_similarity) 스윕 평가

평가셋의 각 케이스에 대해 Dense 검색을 수행하고,
임계값을 0.0~1.0 사이로 변화시키며 Recall@K와 평균 반환 건수를 측정한다.

사용 예:
    python evaluation/dense_retrieval.py --threshold-step 0.05 --top-k 10 20
    python evaluation/dense_retrieval.py --threshold-start 0.1 --threshold-end 0.5 --threshold-step 0.05
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.legal_retrieval_eval_multi import (
    RESULTS_DIR,
    DatasetValidationError,
    PipelineImportError,
    count_law_hits,
    count_precedent_hits,
    load_dataset,
    load_qe_cache,
    log_progress,
    make_cached_expand_fn,
    ratio,
    average,
    select_cases,
)
from pipeline.retrieval import dense_retrieval
from pipeline.retrieval.dense_retrieval import EMBED_COL, LAW_TABLE, PREC_TABLE
from pipeline.retrieval.evaluation_retrieval import expand_case_queries_with_llm

THRESHOLD_WORKERS = 3

# 청크 캐시 (케이스마다 CSV 재로드 방지)
_law_chunks  = None
_prec_chunks = None
_chunks_lock = threading.Lock()

def _get_chunks():
    global _law_chunks, _prec_chunks
    if _law_chunks is None:
        with _chunks_lock:
            if _law_chunks is None:
                log_progress("chunks_load_start")
                _law_chunks  = dense_retrieval.load_chunks(LAW_TABLE)
                _prec_chunks = dense_retrieval.load_chunks(PREC_TABLE)
                log_progress("chunks_load_done")
    return _law_chunks, _prec_chunks


# ── 케이스 단위 평가 ──────────────────────────────────────────────────────────

def evaluate_case(
    case: dict[str, Any],
    thresholds: list[float],
    top_k_values: list[int],
    dense_top_k: int = 100,
    expand_fn=None,
) -> dict[str, Any]:
    """Dense 검색 1회 수행 후 모든 임계값 × top_k 조합을 평가."""
    if expand_fn is None:
        expand_fn = expand_case_queries_with_llm
    expanded_queries = expand_fn(case)
    law_chunks, prec_chunks = _get_chunks()

    # min_similarity=0.0 으로 모든 후보 확보 → 이후 임계값별 필터링
    raw_results: list[dict[str, Any]] = []

    for qi, query in enumerate(expanded_queries):
        dense_query = query["retrieval_payload"]["dense_query"]
        query_vec   = dense_retrieval.embed_query(dense_query)
        for rows, source_type in [
            (dense_retrieval.search_similar(query_vec, law_chunks,  top_k=dense_top_k, min_similarity=0.0), "law"),
            (dense_retrieval.search_similar(query_vec, prec_chunks, top_k=dense_top_k, min_similarity=0.0), "precedent"),
        ]:
            for _, row in rows.iterrows():
                rd = row.to_dict()
                raw_results.append({
                    # 매칭에 필요한 필드 (count_law_hits / count_precedent_hits 참조)
                    "source_type":  source_type,
                    "result_id":    str(rd.get("clause_key") or rd.get("case_id", "")),
                    "clause_key":   rd.get("clause_key"),
                    "law_name":     rd.get("law_name"),
                    "article_key":  rd.get("article_key"),
                    "case_name":    rd.get("case_name"),
                    "clause_index": qi,
                    "similarity":   float(rd.get("similarity", 0.0)),
                })

    # 동일 문서가 여러 쿼리에서 중복 등장하면 최고 유사도만 보존
    seen: dict[str, int] = {}
    for i, r in enumerate(raw_results):
        key = r["source_type"] + "|" + r["result_id"]
        if key not in seen or raw_results[seen[key]]["similarity"] < r["similarity"]:
            seen[key] = i
    raw_results = [raw_results[i] for i in seen.values()]

    law_refs  = case["law_references"]
    prec_refs = case["precedent_references"]
    law_total  = len(law_refs)
    prec_total = len(prec_refs)

    threshold_metrics: dict[str, Any] = {}
    for thr in thresholds:
        filtered = sorted(
            [r for r in raw_results if r["similarity"] >= thr],
            key=lambda x: x["similarity"],
            reverse=True,
        )
        by_k: dict[str, Any] = {}
        for k in top_k_values:
            top = filtered[:k]
            law_hits  = count_law_hits(law_refs, top)
            prec_hits = count_precedent_hits(prec_refs, top)
            by_k[str(k)] = {
                "law_hits":          law_hits,
                "law_total":         law_total,
                "law_recall":        ratio(law_hits, law_total),
                "prec_hits":         prec_hits,
                "prec_total":        prec_total,
                "prec_recall":       ratio(prec_hits, prec_total),
                "integrated_hits":   law_hits + prec_hits,
                "integrated_total":  law_total + prec_total,
                "integrated_recall": ratio(law_hits + prec_hits, law_total + prec_total),
            }
        threshold_metrics[f"{thr:.4f}"] = {
            "result_count": len(filtered),
            "by_k": by_k,
        }

    return {
        "case_id":          case["case_id"],
        "status":           "completed",
        "expanded_queries": expanded_queries,
        "threshold_metrics": threshold_metrics,
    }


# ── 집계 ─────────────────────────────────────────────────────────────────────

def aggregate(
    case_results: list[dict[str, Any]],
    thresholds: list[float],
    top_k_values: list[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for thr in thresholds:
        tk = f"{thr:.4f}"
        by_k: dict[str, Any] = {}
        for k in top_k_values:
            kk = str(k)
            lh = sum(cr["threshold_metrics"].get(tk, {}).get("by_k", {}).get(kk, {}).get("law_hits", 0) for cr in case_results)
            lt = sum(cr["threshold_metrics"].get(tk, {}).get("by_k", {}).get(kk, {}).get("law_total", 0) for cr in case_results)
            ph = sum(cr["threshold_metrics"].get(tk, {}).get("by_k", {}).get(kk, {}).get("prec_hits", 0) for cr in case_results)
            pt = sum(cr["threshold_metrics"].get(tk, {}).get("by_k", {}).get(kk, {}).get("prec_total", 0) for cr in case_results)
            avg_cnt = average([
                cr["threshold_metrics"].get(tk, {}).get("result_count", 0)
                for cr in case_results
            ])
            macro_int = average([
                cr["threshold_metrics"].get(tk, {}).get("by_k", {}).get(kk, {}).get("integrated_recall", 0.0)
                for cr in case_results
            ])
            by_k[kk] = {
                "micro_law":        ratio(lh, lt),
                "micro_prec":       ratio(ph, pt),
                "micro_integrated": ratio(lh + ph, lt + pt),
                "macro_integrated": macro_int,
                "avg_result_count": round(avg_cnt, 1),
            }
        summary[tk] = {"threshold": thr, "by_k": by_k}
    return summary


def find_best(
    summary: dict[str, Any],
    primary_k: int = 10,
) -> tuple[float, float]:
    best_thr, best_score = 0.0, -1.0
    for data in summary.values():
        score = data["by_k"].get(str(primary_k), {}).get("micro_integrated", 0.0)
        if score > best_score:
            best_score = score
            best_thr = data["threshold"]
    return best_thr, best_score


# ── 출력 ─────────────────────────────────────────────────────────────────────

def print_table(
    summary: dict[str, Any],
    thresholds: list[float],
    top_k_values: list[int],
) -> None:
    k_headers = "  ".join(f"@{k}(micro_int) avg_cnt" for k in top_k_values)
    header = f"{'threshold':>10}  {k_headers}"
    print(header)
    print("-" * len(header))
    for thr in thresholds:
        tk = f"{thr:.4f}"
        parts = []
        for k in top_k_values:
            kk = str(k)
            mi  = summary.get(tk, {}).get("by_k", {}).get(kk, {}).get("micro_integrated", 0.0)
            cnt = summary.get(tk, {}).get("by_k", {}).get(kk, {}).get("avg_result_count", 0.0)
            parts.append(f"{mi:.4f}      {cnt:>6.1f}")
        print(f"{thr:>10.4f}  " + "  ".join(parts))


# ── 실행 ─────────────────────────────────────────────────────────────────────

def run_sweep(
    input_path: Path,
    thresholds: list[float],
    top_k_values: list[int],
    primary_k: int,
    dense_top_k: int,
    case_id: str | None,
    output_path: Path | None,
    qe_cache_path: Path | None = None,
) -> Path:
    log_progress(f"threshold_sweep_start thresholds={len(thresholds)} top_k={top_k_values}")

    dataset = load_dataset(input_path)
    cases   = select_cases(dataset, case_id)
    log_progress(f"dataset_loaded cases={len(cases)}")

    _get_chunks()  # 병렬 실행 전 사전 로드

    if qe_cache_path is not None:
        cache = load_qe_cache(qe_cache_path)
        expand_fn = make_cached_expand_fn(cache)
        log_progress(f"qe_cache_loaded path={qe_cache_path} entries={len(cache)}")
    else:
        expand_fn = expand_case_queries_with_llm

    case_results: list[dict[str, Any]] = [None] * len(cases)  # type: ignore[list-item]
    completed = 0

    def _run_one(idx_case: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        idx, case = idx_case
        log_progress(f"case_start {idx + 1}/{len(cases)} case_id={case['case_id']}")
        try:
            result = evaluate_case(
                case=case,
                thresholds=thresholds,
                top_k_values=top_k_values,
                dense_top_k=dense_top_k,
                expand_fn=expand_fn,
            )
        except Exception as exc:  # noqa: BLE001
            log_progress(f"case_failed case_id={case['case_id']} error={exc}")
            result = {
                "case_id": case["case_id"],
                "status": "failed",
                "error": str(exc),
                "threshold_metrics": {},
            }
        return idx, result

    with ThreadPoolExecutor(max_workers=THRESHOLD_WORKERS) as pool:
        futures = {pool.submit(_run_one, (i, case)): i for i, case in enumerate(cases)}
        for fut in as_completed(futures):
            idx, result = fut.result()
            case_results[idx] = result  # type: ignore[index]
            completed += 1
            log_progress(
                f"case_done {completed}/{len(cases)} "
                f"case_id={result['case_id']} status={result['status']}"
            )

    summary = aggregate(case_results, thresholds, top_k_values)
    best_thr, best_score = find_best(summary, primary_k=primary_k)

    report = {
        "schema_version": "dense_sweep_eval.v1",
        "run": {
            "input_path": str(input_path),
            "embed_col": EMBED_COL,
            "thresholds": thresholds,
            "top_k_values": top_k_values,
            "primary_k": primary_k,
            "dense_top_k": dense_top_k,
            "case_count": len(cases),
        },
        "best_threshold": best_thr,
        "best_micro_integrated": best_score,
        "summary": summary,
        "cases": case_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = RESULTS_DIR / f"threshold_sweep_{ts}.json"
    else:
        save_path = RESULTS_DIR / output_path.name

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    log_progress(f"report_saved path={save_path}")
    return save_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense 코사인 유사도 임계값 스윕 평가 (embed_vertex)"
    )
    parser.add_argument(
        "--input", default=str(PROJECT_ROOT / "evaluation" / "eval_set.json"),
    )
    parser.add_argument("--threshold-start", type=float, default=0.0)
    parser.add_argument("--threshold-end",   type=float, default=0.9)
    parser.add_argument(
        "--threshold-step", type=float, default=0.05,
        help="임계값 간격 (default: 0.05 → 19단계)",
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[5, 10, 20],
        help="평가할 K 값 목록 (default: 5 10 20)",
    )
    parser.add_argument(
        "--primary-k", type=int, default=10,
        help="최적 임계값 선택 기준 K (default: 10)",
    )
    parser.add_argument(
        "--dense-top-k", type=int, default=100,
        help="Dense 1차 검색 후보 수 (default: 100). 임계값 필터링 전 최대 후보.",
    )
    parser.add_argument("--case-id", help="특정 case_id만 평가")
    parser.add_argument("--output")
    parser.add_argument(
        "--qe-cache",
        metavar="PATH",
        default=None,
        help=(
            "이전 실행 결과 JSON 경로. "
            "지정 시 QE LLM 호출을 스킵하고 캐시된 expanded_queries를 재사용합니다. "
            "캐시에 없는 case_id는 LLM을 호출합니다."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    n = round((args.threshold_end - args.threshold_start) / args.threshold_step)
    thresholds = [
        round(args.threshold_start + i * args.threshold_step, 8)
        for i in range(n + 1)
        if args.threshold_start - 1e-9 <= args.threshold_start + i * args.threshold_step <= args.threshold_end + 1e-9
    ]

    try:
        report_path = run_sweep(
            input_path=Path(args.input),
            thresholds=thresholds,
            top_k_values=args.top_k,
            primary_k=args.primary_k,
            dense_top_k=args.dense_top_k,
            case_id=args.case_id,
            output_path=Path(args.output) if args.output else None,
            qe_cache_path=Path(args.qe_cache) if args.qe_cache else None,
        )
    except (PipelineImportError, DatasetValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    print(f"\n=== Threshold Sweep — embed_col={EMBED_COL}, dense_top_k={args.dense_top_k} ===")
    print(f"케이스 수: {report['run']['case_count']}\n")
    print_table(report["summary"], thresholds, args.top_k)
    print(
        f"\n최적 threshold={report['best_threshold']:.4f}  "
        f"micro_integrated@{args.primary_k}={report['best_micro_integrated']:.6f}"
    )
    print(f"report_path={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
