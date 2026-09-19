"""Helpers for legal retrieval evaluation datasets and metric matching."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

# Windows에서 ProactorEventLoop의 socketpair 실패(WinError 10014) 방지
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPELINE_IMPORT_ERROR: ImportError | None = None
try:
    from evaluation import rrf_baseline
    from pipeline.retrieval.bm25_retrieval import (
        build_query_tokens,
        load_case_law_from_db,
        load_law_child_from_db,
        tokenize,
    )
    from pipeline.retrieval import dense_retrieval
    from pipeline.retrieval.evaluation_retrieval import (
        collect_case_documents,
        expand_case_queries_with_llm,
        inspect_retrieved_documents,
    )
    from rank_bm25 import BM25Okapi
except ImportError as error:
    PIPELINE_IMPORT_ERROR = error


DEFAULT_RECALL_K_VALUES = [1, 3, 5, 10, 20]
REPORT_SCHEMA_VERSION = "legal_retrieval_eval.v2"
REQUIRED_CASE_FIELDS = ("case_id", "clauses", "law_references", "precedent_references")
DEFAULT_INPUT_PATH = PROJECT_ROOT / "evaluation" / "eval_set.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_SEMANTIC_EMBED_COL = "embed_vertex"
LAW_SEMANTIC_KEEP_COLS = [
    "clause_key",
    "law_name",
    "article_no",
    "paragraph_no",
    "child_text",
]
PRECEDENT_SEMANTIC_KEEP_COLS = [
    "case_id",
    "case_number",
    "judgment_summary",
    "referenced_law",
]
# 모든 임베딩 모델에 공통으로 존재하는 판례만 사용 (공정 비교)
PRECEDENT_COMMON_FILTER = "embed_vertex IS NOT NULL"

# ── embedding 프로파일 ──────────────────────────────────────────────────────
SEMANTIC_EMBED_CONFIGS = {
    "embed_vertex": {
        "query_embed_col": "embed_vertex",
        "law_embed_col": "embed_vertex",
        "precedent_embed_col": "embed_vertex",
    }
}
DEFAULT_SEMANTIC_EMBED_COLS = tuple(SEMANTIC_EMBED_CONFIGS)
DEFAULT_SEMANTIC_EMBED_COLS_ARG = ",".join(DEFAULT_SEMANTIC_EMBED_COLS)
SEMANTIC_EMBED_COL = DEFAULT_SEMANTIC_EMBED_COL
LAW_SEMANTIC_EMBED_COL = SEMANTIC_EMBED_CONFIGS[DEFAULT_SEMANTIC_EMBED_COL]["law_embed_col"]
PRECEDENT_SEMANTIC_EMBED_COL = SEMANTIC_EMBED_CONFIGS[DEFAULT_SEMANTIC_EMBED_COL]["precedent_embed_col"]

# reranker 종류
RERANKERS = ["alpha_hybrid"]

# Alpha hybrid — 법령/판례 각각 BM25 가중치 (1-α = Dense 가중치)
# 분리 reranking + sweep 최적값 (alpha_sweep_2d_20260530_235115)
ALPHA_LAW  = 0.20  # 법령: BM25 20% + Dense 80%
ALPHA_PREC = 0.70  # 판례: BM25 70% + Dense 30%

# vertex 모델로 고정
EMBED_WORKERS = 1
# 케이스 단위 병렬 처리 (API 호출 제한 고려해 보수적으로 설정)
CASE_WORKERS = 3
# QE API 동시 호출 제한 (WinError 10014 방지: Windows 소켓 버퍼 고갈 예방)
_QE_SEMAPHORE = threading.Semaphore(3)


# ── 설정 헬퍼 ─────────────────────────────────────────────────────────────

def semantic_embed_config(semantic_embed_col: str) -> dict[str, str]:
    try:
        return SEMANTIC_EMBED_CONFIGS[semantic_embed_col]
    except KeyError as error:
        choices = ", ".join(sorted(SEMANTIC_EMBED_CONFIGS))
        raise ValueError(
            f"unsupported semantic embedding column: {semantic_embed_col}. "
            f"Expected one of: {choices}"
        ) from error


def normalize_semantic_embed_cols(
    semantic_embed_cols: str | list[str] | tuple[str, ...] | None = None,
    *,
    semantic_embed_col: str | None = None,
    embed_col: str | None = None,
) -> tuple[str, ...]:
    if embed_col is not None:
        semantic_embed_cols = embed_col
    elif semantic_embed_col is not None:
        semantic_embed_cols = semantic_embed_col

    if semantic_embed_cols is None:
        candidates = list(DEFAULT_SEMANTIC_EMBED_COLS)
    elif isinstance(semantic_embed_cols, str):
        candidates = [item.strip() for item in semantic_embed_cols.split(",") if item.strip()]
    else:
        candidates = list(semantic_embed_cols)

    if not candidates:
        raise ValueError("at least one semantic embedding column must be selected")

    normalized: list[str] = []
    for candidate in candidates:
        semantic_embed_config(candidate)  # 유효성 검증
        if candidate not in normalized:
            normalized.append(candidate)
    return tuple(normalized)


def semantic_embed_configs(
    semantic_embed_cols: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    return {col: semantic_embed_config(col) for col in semantic_embed_cols}


def log_progress(message: str) -> None:
    print(f"[legal-eval] {message}", file=sys.stderr, flush=True)


# ── 예외 ──────────────────────────────────────────────────────────────────

class DatasetValidationError(ValueError):
    """평가 데이터셋이 CLI 계약을 만족하지 못할 때."""


class PipelineImportError(RuntimeError):
    """실제 retrieval/reranking 파이프라인을 import할 수 없을 때."""


# ── 케이스 실행 컨텍스트 ───────────────────────────────────────────────────

@dataclass
class CaseExecutionContext:
    """케이스 단위 실행 상태. 나머지 평가 런과 격리."""

    case: dict[str, Any]
    case_index: int
    input_path: Path
    semantic_embed_cols: tuple[str, ...] = field(default_factory=lambda: DEFAULT_SEMANTIC_EMBED_COLS)
    expanded_queries: list[dict[str, Any]] = field(default_factory=list)
    keyword_results: list[dict[str, Any]] = field(default_factory=list)

    # embed_col → results 매핑
    semantic_results_by_model: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # f"{embed_col}__{rerank_type}" → results 매핑
    reranked_results_by_config: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    inspected_documents: list[dict[str, Any]] = field(default_factory=list)
    stage_trace: list[dict[str, Any]] = field(default_factory=list)
    status: str = "pending"
    error: dict[str, str] | None = None

    @property
    def case_id(self) -> str:
        return self.case["case_id"]

    @property
    def semantic_results(self) -> list[dict[str, Any]]:
        """embed_col별 결과를 단일 리스트로 flatten (stage-level 지표 계산용)."""
        results: list[dict[str, Any]] = []
        for model_results in self.semantic_results_by_model.values():
            results.extend(model_results)
        return results

    @property
    def reranked_results(self) -> list[dict[str, Any]]:
        """실험 config별 결과를 병합, (clause_index, result_id) 기준 최고 rank 유지.
        
        여러 embed_col 실험에서 동일 문서가 중복될 수 있으므로 de-duplicate한다.
        """
        return flatten_reranked_results(self.reranked_results_by_config)


def flatten_reranked_results(
    reranked_results_by_config: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """config별 reranked_results를 합쳐 (clause_index, result_id) 당 best rank 유지."""
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for results in reranked_results_by_config.values():
        for result in results:
            key = (int(result["clause_index"]), str(result["result_id"]))
            existing = best.get(key)
            if existing is None or result.get("rank", 10**9) < existing.get("rank", 10**9):
                best[key] = result
    return sorted(best.values(), key=lambda r: r.get("rank", 10**9))


# ── 파이프라인 임포트 체크 ─────────────────────────────────────────────────

def assert_real_pipeline_imports_available() -> None:
    if PIPELINE_IMPORT_ERROR is None:
        return
    raise PipelineImportError(
        "real retrieval/reranking pipeline import failed: "
        f"{type(PIPELINE_IMPORT_ERROR).__name__}: {PIPELINE_IMPORT_ERROR}\n"
        "Required actions: restore the project retrieval/reranking modules and "
        "install their dependencies, then rerun evaluation/legal_retrieval_eval.py."
    )


# ── 데이터 유틸 ───────────────────────────────────────────────────────────

def parse_text_array(value: Any) -> list[str]:
    """Cloud SQL text-array 포맷 {a,b} → Python list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if not isinstance(value, str):
        return [str(value)]

    text = value.strip()
    if text in {"", "{}"}:
        return []
    if not (text.startswith("{") and text.endswith("}")):
        return [text]

    inner = text[1:-1]
    if not inner:
        return []
    return [item for item in next(csv.reader([inner])) if item]


def append_unique(items: list[str], additions: list[str]) -> None:
    for item in additions:
        if item not in items:
            items.append(item)


def stage_clauses(
    clause_type: str,
    extracted_clauses: list[str],
    stage5: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if clause_type == "explicit_quote":
        return [
            {"raw": clause, "normalized": clause, "clause_type": clause_type}
            for clause in extracted_clauses
            if clause
        ]
    return [
        {
            "raw": item["raw"],
            "normalized": item["normalized"],
            "clause_type": clause_type,
        }
        for item in stage5
        if item.get("raw") and item.get("normalized")
    ]


# ── 데이터셋 로딩 / 검증 ──────────────────────────────────────────────────

def normalize_eval_record_for_pipeline(
    record: dict[str, Any],
    case_index: int,
) -> dict[str, Any]:
    return {
        "case_id": record.get("id") or f"eval_{case_index}",
        "clauses": [clause["normalized"] for clause in record.get("clauses", [])],
        "law_references": [
            {"law_child": law_key} for law_key in record.get("gt_laws", [])
        ],
        "precedent_references": [
            {"case_number": case_number} for case_number in record.get("gt_cases", [])
        ],
        "source_type": record.get("source_type"),
        "source_id": record.get("source_id"),
        "source_text": record.get("source_text", {}),
        "meta": record.get("meta", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run legal QA retrieval evaluation for a JSON array dataset."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Absolute path to a JSON array evaluation dataset.",
    )
    parser.add_argument("--case-id", help="Optional case_id to evaluate.")
    parser.add_argument(
        "--output",
        help="Optional JSON report filename. Reports are always written under evaluation/results/.",
    )
    parser.add_argument(
        "--semantic-embed-cols",
        default=DEFAULT_SEMANTIC_EMBED_COLS_ARG,
        help=(
            "Comma-separated semantic embedding profiles to evaluate. "
            "Default: embed_vertex"
        ),
    )
    parser.add_argument(
        "--case-workers",
        type=int,
        default=CASE_WORKERS,
        help=(
            f"케이스 단위 병렬 처리 스레드 수 (default: {CASE_WORKERS}). "
            "API 호출 제한이 있으면 낮추세요. 1이면 순차 실행."
        ),
    )
    parser.add_argument(
        "--alpha-law",
        type=float,
        default=ALPHA_LAW,
        help=(
            f"법령 hybrid 가중치 — BM25 α, Dense (1-α) (default: {ALPHA_LAW}). "
            "값이 낮을수록 Dense 가중치가 커집니다."
        ),
    )
    parser.add_argument(
        "--alpha-prec",
        type=float,
        default=ALPHA_PREC,
        help=(
            f"판례 hybrid 가중치 — BM25 α, Dense (1-α) (default: {ALPHA_PREC}). "
            "값이 높을수록 BM25 가중치가 커집니다."
        ),
    )
    parser.add_argument(
        "--qe-cache",
        metavar="PATH",
        default=None,
        help=(
            "이전 실행 결과 JSON 경로 (evaluation/results/ 아래 파일). "
            "지정 시 Query Expansion LLM 호출을 스킵하고 캐시된 expanded_queries를 재사용합니다. "
            "캐시에 없는 case_id는 LLM을 호출합니다."
        ),
    )
    return parser.parse_args()


def load_dataset(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.is_absolute():
        raise DatasetValidationError("--input must be an absolute path")
    if not input_path.exists():
        raise DatasetValidationError(f"input file does not exist: {input_path}")

    try:
        with input_path.open("r", encoding="utf-8") as file:
            dataset = json.load(file)
    except json.JSONDecodeError as error:
        raise DatasetValidationError(f"malformed JSON in {input_path}: {error}") from error

    if not isinstance(dataset, list):
        raise DatasetValidationError("input JSON must be an array of case objects")

    normalized_dataset = [
        normalize_eval_record_for_pipeline(case, index)
        if is_eval_set_record(case)
        else case
        for index, case in enumerate(dataset)
    ]
    for index, case in enumerate(normalized_dataset):
        validate_case(case, index)
    return normalized_dataset


def is_eval_set_record(case: Any) -> bool:
    return isinstance(case, dict) and {
        "id", "source_type", "source_id", "source_text",
        "clauses", "gt_laws", "gt_cases", "meta",
    }.issubset(case)


def validate_case(case: Any, index: int) -> None:
    if not isinstance(case, dict):
        raise DatasetValidationError(f"case at index {index} must be an object")

    missing = [f for f in REQUIRED_CASE_FIELDS if f not in case]
    if missing:
        raise DatasetValidationError(
            f"case at index {index} is missing required fields: {', '.join(missing)}"
        )
    if not isinstance(case["case_id"], str) or not case["case_id"].strip():
        raise DatasetValidationError(f"case at index {index} must include a case_id string")

    for fld in ("clauses", "law_references", "precedent_references"):
        if not isinstance(case[fld], list):
            raise DatasetValidationError(f"case {case['case_id']} field {fld} must be an array")

    for clause_index, clause in enumerate(case["clauses"]):
        validate_non_empty_string(clause, f"case {case['case_id']} clauses[{clause_index}]")

    validate_reference_objects(
        case_id=case["case_id"],
        field="law_references",
        references=case["law_references"],
        required_key="law_child",
    )
    validate_reference_objects(
        case_id=case["case_id"],
        field="precedent_references",
        references=case["precedent_references"],
        required_key="case_number",
    )


def validate_non_empty_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{label} must be a non-empty string")


def validate_reference_objects(
    *,
    case_id: str,
    field: str,
    references: list[Any],
    required_key: str,
) -> None:
    for ref_index, reference in enumerate(references):
        item_label = f"case {case_id} {field}[{ref_index}]"
        if not isinstance(reference, dict):
            raise DatasetValidationError(f"{item_label} must be an object")
        validate_non_empty_string(reference.get(required_key), f"{item_label}.{required_key}")


def select_cases(
    dataset: list[dict[str, Any]],
    case_id: str | None,
) -> list[dict[str, Any]]:
    if case_id is None:
        return dataset
    selected = [case for case in dataset if case["case_id"] == case_id]
    if not selected:
        raise DatasetValidationError(f"case_id not found: {case_id}")
    return selected


# ── 컨텍스트 빌드 ─────────────────────────────────────────────────────────

def build_case_contexts(
    *,
    input_path: Path,
    cases: list[dict[str, Any]],
    semantic_embed_cols: str | list[str] | tuple[str, ...] | None = None,
) -> list[CaseExecutionContext]:
    normalized_embed_cols = normalize_semantic_embed_cols(semantic_embed_cols)
    return [
        CaseExecutionContext(
            case=case,
            case_index=index,
            input_path=input_path,
            semantic_embed_cols=normalized_embed_cols,
        )
        for index, case in enumerate(cases)
    ]


# ── QE 캐시 ──────────────────────────────────────────────────────────────

def load_qe_cache(path: Path) -> dict[str, list[dict[str, Any]]]:
    """이전 실행 결과 JSON에서 case_id → expanded_queries 매핑을 로드한다."""
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    cache: dict[str, list[dict[str, Any]]] = {}
    for case_report in report.get("cases", []):
        case_id = case_report.get("case_id")
        expanded = case_report.get("expanded_queries")
        if case_id and isinstance(expanded, list):
            cache[case_id] = expanded
    return cache


def make_cached_expand_fn(cache: dict[str, list[dict[str, Any]]]) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
    """캐시에서 expanded_queries를 반환하는 expand_fn을 생성한다.
    캐시에 없는 case_id는 LLM을 호출한다."""
    def expand_fn(case: dict[str, Any]) -> list[dict[str, Any]]:
        case_id = case.get("case_id", "")
        if case_id in cache:
            log_progress(f"qe_cache_hit case_id={case_id}")
            return cache[case_id]
        log_progress(f"qe_cache_miss case_id={case_id} falling_back_to_llm")
        return expand_case_queries_with_llm(case)
    return expand_fn


# ── 파이프라인 실행 ───────────────────────────────────────────────────────

def run_case_pipeline(
    context: CaseExecutionContext,
    expand_fn=None,
    alpha_law: float = ALPHA_LAW,
    alpha_prec: float = ALPHA_PREC,
) -> None:
    if expand_fn is None:
        expand_fn = expand_case_queries_with_llm

    current_stage = "query_expansion"
    try:
        # ── 1. Query Expansion ──────────────────────────────────────────
        # 세마포어로 동시 QE API 호출 수 제한 (WinError 10014 방지)
        log_progress(f"stage_start case_id={context.case_id} stage={current_stage}")
        with _QE_SEMAPHORE:
            expanded = expand_fn(context.case)
        context.expanded_queries.extend(expanded)
        record_stage(context, current_stage, len(context.expanded_queries))
        log_progress(
            f"stage_done case_id={context.case_id} stage={current_stage} "
            f"output_count={len(context.expanded_queries)}"
        )

        # ── 2. Hybrid Retrieval ─────────────────────────────────────────
        current_stage = "hybrid_retrieval"
        log_progress(f"stage_start case_id={context.case_id} stage={current_stage}")

        # BM25와 Dense를 병렬 실행한다.
        def _retrieve_semantic_results_by_embed_col() -> dict[str, list[dict[str, Any]]]:
            def _retrieve_for_col(embed_col: str) -> tuple[str, list[dict[str, Any]]]:
                return embed_col, run_semantic_retrieval_for_embed_col(
                    expanded_queries=context.expanded_queries,
                    top_k=100,
                    semantic_embed_col=embed_col,
                )

            semantic_results_by_col: dict[str, list[dict[str, Any]]] = {}
            n_workers = min(max(1, EMBED_WORKERS), len(context.semantic_embed_cols))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for col, results in pool.map(_retrieve_for_col, context.semantic_embed_cols):
                    semantic_results_by_col[col] = results
            return semantic_results_by_col

        with ThreadPoolExecutor(max_workers=2) as pool:
            bm25_future = pool.submit(
                run_bm25_retrieval,
                case=context.case,
                expanded_queries=context.expanded_queries,
                top_k=60,
            )
            semantic_future = pool.submit(_retrieve_semantic_results_by_embed_col)
            keyword_results = bm25_future.result()
            semantic_results_by_col = semantic_future.result()

        context.keyword_results.extend(keyword_results)
        context.semantic_results_by_model.update(semantic_results_by_col)

        semantic_result_count = sum(
            len(r) for r in context.semantic_results_by_model.values()
        )
        record_stage(
            context,
            current_stage,
            len(context.keyword_results) + semantic_result_count,
        )
        log_progress(
            f"stage_done case_id={context.case_id} stage={current_stage} "
            f"bm25_results={len(context.keyword_results)} "
            f"semantic_results={semantic_result_count}"  # FIX: int, not len(int)
        )

        # ── 3. Reranking (embed_col × rerank_type 조합) ─────────────────
        current_stage = "reranking"
        log_progress(f"stage_start case_id={context.case_id} stage={current_stage}")

        for embed_col, sem_results in context.semantic_results_by_model.items():
            for rerank_type in RERANKERS:
                reranked = run_reranking(
                    rerank_type=rerank_type,
                    keyword_results=context.keyword_results,
                    semantic_results=sem_results,
                    alpha_law=alpha_law,
                    alpha_prec=alpha_prec,
                )
                experiment_key = f"{embed_col}__{rerank_type}"
                context.reranked_results_by_config[experiment_key] = reranked

        reranked_result_count = sum(
            len(r) for r in context.reranked_results_by_config.values()
        )
        record_stage(context, current_stage, reranked_result_count)
        log_progress(
            f"stage_done case_id={context.case_id} stage={current_stage} "
            f"output_count={reranked_result_count}"
        )

        # ── 4. Document Inspection ──────────────────────────────────────
        current_stage = "document_inspection"
        log_progress(f"stage_start case_id={context.case_id} stage={current_stage}")
        context.inspected_documents.extend(
            inspect_retrieved_documents(context.reranked_results)  # property 사용
        )
        record_stage(context, current_stage, len(context.inspected_documents))
        log_progress(
            f"stage_done case_id={context.case_id} stage={current_stage} "
            f"output_count={len(context.inspected_documents)}"
        )
        context.status = "completed"

    except Exception as error:  # noqa: BLE001
        context.status = "failed"
        context.error = {
            "type": type(error).__name__,
            "message": str(error),
            "stage": current_stage,
        }
        record_stage(
            context,
            current_stage,
            current_stage_output_count(context, current_stage),
            status="failed",
            error=context.error,
        )
        log_progress(
            f"stage_failed case_id={context.case_id} stage={current_stage} "
            f"error_type={type(error).__name__} message={str(error)}"
        )


# ── BM25 ──────────────────────────────────────────────────────────────────

def run_bm25_retrieval(
    *,
    case: dict[str, Any],
    expanded_queries: list[dict[str, Any]],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    documents = collect_case_documents(case)
    if documents:
        return run_bm25_over_documents(
            documents=documents,
            expanded_queries=expanded_queries,
            top_k=top_k,
        )
    if not is_real_eval_case(case):
        return []

    law_documents, law_bm25, precedent_documents, precedent_bm25 = load_bm25_corpus()
    results: list[dict[str, Any]] = []
    for query in expanded_queries:
        query_tokens = build_query_tokens(query["retrieval_payload"]["bm25_keywords"])
        results.extend(
            bm25_results_for_documents(
                query=query,
                query_tokens=query_tokens,
                documents=law_documents,
                bm25=law_bm25,
                top_k=top_k,
            )
        )
        results.extend(
            bm25_results_for_documents(
                query=query,
                query_tokens=query_tokens,
                documents=precedent_documents,
                bm25=precedent_bm25,
                top_k=top_k,
            )
        )
    return results


def run_bm25_over_documents(
    *,
    documents: list[dict[str, Any]],
    expanded_queries: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    corpus_tokens = [tokenize(document["search_text"]) for document in documents]
    bm25 = BM25Okapi(corpus_tokens)
    results: list[dict[str, Any]] = []
    for query in expanded_queries:
        query_tokens = build_query_tokens(query["retrieval_payload"]["bm25_keywords"])
        results.extend(
            bm25_results_for_documents(
                query=query,
                query_tokens=query_tokens,
                documents=documents,
                bm25=bm25,
                top_k=top_k,
            )
        )
    return results


def bm25_results_for_documents(
    *,
    query: dict[str, Any],
    query_tokens: list[str],
    documents: list[dict[str, Any]],
    bm25: "BM25Okapi",
    top_k: int,
) -> list[dict[str, Any]]:
    scores = bm25.get_scores(query_tokens)
    top_indexes = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True,
    )[:top_k]
    results: list[dict[str, Any]] = []
    for rank, document_index in enumerate(top_indexes, 1):
        document = documents[document_index]
        score = round(float(scores[document_index]), 6)
        results.append(
            {
                "clause_index": query["clause_index"],
                "clause": query["clause"],
                "result_id": document["result_id"],
                "source_type": document["source_type"],
                "score": score,
                "keyword_score": score,
                "retrieval_method": "bm25",
                "rank": rank,
                "document_body": document["document_body"],
                "document_text": document["document_text"],
                "metadata": document["metadata"],
                "query_tokens": query_tokens,
            }
        )
    return results


@lru_cache(maxsize=1)
def load_bm25_corpus() -> tuple[
    list[dict[str, Any]], "BM25Okapi",
    list[dict[str, Any]], "BM25Okapi"
]:
    log_progress("bm25_corpus_load_start")
    law_documents = bm25_law_documents(load_law_child_from_db())
    log_progress(f"bm25_corpus_law_loaded rows={len(law_documents)}")
    precedent_documents = bm25_precedent_documents(load_case_law_from_db())
    log_progress(f"bm25_corpus_precedent_loaded rows={len(precedent_documents)}")
    law_bm25 = BM25Okapi([tokenize(d["search_text"]) for d in law_documents])
    precedent_bm25 = BM25Okapi([tokenize(d["search_text"]) for d in precedent_documents])
    log_progress("bm25_corpus_index_ready")
    return law_documents, law_bm25, precedent_documents, precedent_bm25


def bm25_law_documents(frame: pd.DataFrame) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        clause_key = clean_bm25_value(row_dict.get("clause_key"))
        child_text = clean_bm25_value(row_dict.get("child_text"))
        target = clean_bm25_value(row_dict.get("bm25_target")) or child_text
        metadata = {k: clean_bm25_value(v) for k, v in row_dict.items()}
        documents.append(
            {
                "result_id": f"law:{clause_key}",
                "source_type": "law",
                "document_body": child_text,
                "document_text": target,
                "metadata": metadata,
                "search_text": target,
            }
        )
    return documents


def bm25_precedent_documents(frame: pd.DataFrame) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        case_id = clean_bm25_value(row_dict.get("case_id"))
        case_number = clean_bm25_value(row_dict.get("case_number"))
        issue = clean_bm25_value(row_dict.get("issue"))
        summary = clean_bm25_value(row_dict.get("judgment_summary"))
        target = clean_bm25_value(row_dict.get("bm25_target")) or f"{issue} {summary}".strip()
        metadata = {k: clean_bm25_value(v) for k, v in row_dict.items()}
        metadata.setdefault("case_law", {
            "case_number": case_number,
            "case_name": metadata.get("case_name", ""),
        })
        documents.append(
            {
                "result_id": f"precedent:{case_id or case_number}",
                "source_type": "precedent",
                "document_body": summary or target,
                "document_text": target,
                "metadata": metadata,
                "search_text": target,
            }
        )
    return documents


def clean_bm25_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


# ── Dense Retrieval ────────────────────────────────────────────────────────

@lru_cache(maxsize=len(SEMANTIC_EMBED_CONFIGS))
def load_semantic_chunks(
    semantic_embed_col: str = DEFAULT_SEMANTIC_EMBED_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = semantic_embed_config(semantic_embed_col)
    law_embed_col = config["law_embed_col"]
    precedent_embed_col = config["precedent_embed_col"]
    log_progress(
        f"semantic_chunks_load_start semantic_embed_col={semantic_embed_col} "
        f"law_col={law_embed_col} precedent_col={precedent_embed_col}"
    )
    law_chunks = dense_retrieval.load_chunks(
        dense_retrieval.LAW_TABLE, law_embed_col, LAW_SEMANTIC_KEEP_COLS
    )
    log_progress(f"semantic_chunks_law_loaded rows={len(law_chunks)}")
    precedent_chunks = dense_retrieval.load_chunks(
        dense_retrieval.PREC_TABLE, precedent_embed_col, PRECEDENT_SEMANTIC_KEEP_COLS,
        extra_filter=PRECEDENT_COMMON_FILTER,
    )
    log_progress(f"semantic_chunks_precedent_loaded rows={len(precedent_chunks)}")
    return law_chunks, precedent_chunks



def run_semantic_retrieval_for_embed_col(
    *,
    expanded_queries: list[dict[str, Any]],
    top_k: int,
    semantic_embed_col: str,
) -> list[dict[str, Any]]:
    config = semantic_embed_config(semantic_embed_col)
    query_embed_col = config["query_embed_col"]
    law_embed_col = config["law_embed_col"]
    precedent_embed_col = config["precedent_embed_col"]
    law_chunks, precedent_chunks = load_semantic_chunks(semantic_embed_col)

    results: list[dict[str, Any]] = []
    for query in expanded_queries:
        dense_query = query["retrieval_payload"]["dense_query"]
        query_vec = dense_retrieval.embed_query(dense_query, query_embed_col)
        results.extend(
            semantic_results_from_rows(
                query=query,
                rows=dense_retrieval.search_similar(query_vec, law_chunks, law_embed_col, top_k),
                source_type="law",
                semantic_embed_col=semantic_embed_col,
                query_embed_col=query_embed_col,
                document_embed_col=law_embed_col,
            )
        )
        results.extend(
            semantic_results_from_rows(
                query=query,
                rows=dense_retrieval.search_similar(
                    query_vec, precedent_chunks, precedent_embed_col, top_k
                ),
                source_type="precedent",
                semantic_embed_col=semantic_embed_col,
                query_embed_col=query_embed_col,
                document_embed_col=precedent_embed_col,
            )
        )
    return results


def is_real_eval_case(case: dict[str, Any]) -> bool:
    return {"source_type", "source_id", "source_text", "meta"}.issubset(case)



def semantic_results_from_rows(
    *,
    query: dict[str, Any],
    rows: Any,
    source_type: str | None,
    semantic_embed_col: str | None = None,
    query_embed_col: str | None = None,
    document_embed_col: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(rows.iterrows(), 1):
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        similarity = round(float(row_dict.get("similarity", 0.0)), 6)
        result_source_type = source_type or str(row_dict.get("source_type", "document"))
        document_text = semantic_document_text(row_dict)
        metadata = {
            k: v
            for k, v in row_dict.items()
            if k not in {"_vec", "similarity", "document_body", "document_text", "search_text"}
        }
        if result_source_type == "precedent" and row_dict.get("case_number"):
            metadata.setdefault("case_law", {})["case_number"] = row_dict["case_number"]
        results.append(
            {
                "clause_index": query["clause_index"],
                "clause": query["clause"],
                "result_id": semantic_result_id(row_dict, result_source_type),
                "source_type": result_source_type,
                "score": similarity,
                "semantic_score": similarity,
                "similarity": similarity,
                "retrieval_method": "semantic",
                "semantic_embed_col": semantic_embed_col,
                "query_semantic_embed_col": query_embed_col,
                "document_semantic_embed_col": document_embed_col,
                "rank": rank,
                "document_body": row_dict.get("document_body", document_text),
                "document_text": row_dict.get("document_text", document_text),
                "metadata": metadata,
            }
        )
    return results


def semantic_result_id(row: dict[str, Any], source_type: str) -> str:
    if source_type == "law" and row.get("clause_key"):
        return f"law:{row['clause_key']}"
    if source_type == "precedent":
        case_number = row.get("case_number")
        if not case_number and isinstance(row.get("case_law"), dict):
            case_number = row["case_law"].get("case_number")
        if case_number:
            return f"precedent:{case_number}"
        if row.get("case_id"):
            return f"precedent:{row['case_id']}"
    return str(row.get("result_id") or f"{source_type}:{abs(hash(str(row)))}")


def semantic_document_text(row: dict[str, Any]) -> str:
    for fld in ("child_text", "judgment_summary", "document_text", "document_body", "text"):
        value = row.get(fld)
        if isinstance(value, str) and value.strip():
            return value
    return " ".join(str(v) for v in row.values() if v is not None)


# ── Reranking ─────────────────────────────────────────────────────────────

def run_reranking(
    *,
    rerank_type: str,
    keyword_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    top_k: int = 20,
    alpha_law: float = ALPHA_LAW,
    alpha_prec: float = ALPHA_PREC,
) -> list[dict[str, Any]]:
    if rerank_type == "alpha_hybrid":
        return run_alpha_hybrid_reranking(
            keyword_results=keyword_results,
            semantic_results=semantic_results,
            top_k=top_k,
            alpha_law=alpha_law,
            alpha_prec=alpha_prec,
        )
    if rerank_type == "rrf":
        return run_project_reranking(
            keyword_results=keyword_results,
            semantic_results=semantic_results,
            top_k=top_k,
        )
    raise ValueError(f"unsupported rerank_type: {rerank_type}")


def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """result_id → score 딕셔너리를 [0, 1] min-max 정규화. (공개 공통 함수)"""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def run_alpha_hybrid_reranking(
    *,
    keyword_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    top_k: int = 20,
    alpha_law: float = ALPHA_LAW,
    alpha_prec: float = ALPHA_PREC,
) -> list[dict[str, Any]]:
    """법령/판례별 alpha를 다르게 적용한 hybrid 재랭킹.
    법령/판례를 분리하여 각각 top_k개씩 독립 랭킹 후 합산.
    → 법령과 판례가 슬롯을 경쟁하지 않아 두 도메인 recall을 모두 보장.
    """
    clause_meta: dict[int, dict[str, Any]] = {}
    bm25_scores: dict[int, dict[str, float]] = {}
    dense_scores: dict[int, dict[str, float]] = {}
    result_lookup: dict[tuple[int, str], dict[str, Any]] = {}

    for r in keyword_results:
        ci = int(r["clause_index"])
        rid = str(r["result_id"])
        result_lookup.setdefault((ci, rid), dict(r))
        clause_meta.setdefault(ci, {"clause": r["clause"]})
        new_score = float(r.get("keyword_score", r.get("score", 0.0)))
        ci_map = bm25_scores.setdefault(ci, {})
        # 멀티쿼리: 동일 doc이 여러 쿼리에서 나올 때 best score(최댓값) 유지
        if rid not in ci_map or new_score > ci_map[rid]:
            ci_map[rid] = new_score

    for r in semantic_results:
        ci = int(r["clause_index"])
        rid = str(r["result_id"])
        result_lookup.setdefault((ci, rid), dict(r))
        clause_meta.setdefault(ci, {"clause": r["clause"]})
        new_score = float(r.get("score", r.get("similarity", 0.0)))
        ci_map = dense_scores.setdefault(ci, {})
        # 멀티쿼리: 동일 doc이 여러 쿼리에서 나올 때 best score(최댓값) 유지
        if rid not in ci_map or new_score > ci_map[rid]:
            ci_map[rid] = new_score

    reranked_results: list[dict[str, Any]] = []
    for ci, meta in clause_meta.items():
        norm_bm25 = minmax_normalize(bm25_scores.get(ci, {}))
        norm_dense = minmax_normalize(dense_scores.get(ci, {}))

        all_ids = set(norm_bm25) | set(norm_dense)
        law_scored: list[tuple[str, float, float, float]] = []
        prec_scored: list[tuple[str, float, float, float]] = []
        for rid in all_ids:
            original = result_lookup.get((ci, rid), {})
            stype = original.get("source_type", "")
            alpha = alpha_prec if stype == "precedent" else alpha_law
            nb = norm_bm25.get(rid, 0.0)
            nd = norm_dense.get(rid, 0.0)
            hybrid = alpha * nb + (1 - alpha) * nd
            if stype == "precedent":
                prec_scored.append((rid, hybrid, nb, nd))
            else:
                law_scored.append((rid, hybrid, nb, nd))

        # 법령/판례 각각 독립 랭킹 → 슬롯 경쟁 없이 각 도메인 top_k 보장
        law_scored.sort(key=lambda x: x[1], reverse=True)
        prec_scored.sort(key=lambda x: x[1], reverse=True)

        for domain, scored in (("law", law_scored), ("precedent", prec_scored)):
            for rank, (rid, hybrid_score, nb, nd) in enumerate(scored[:top_k], 1):
                original = result_lookup.get((ci, rid), {})
                stype = original.get("source_type", domain)
                reranked_results.append(
                    {
                        "clause_index": ci,
                        "clause": meta["clause"],
                        "result_id": rid,
                        "source_type": stype,
                        "document_body": original.get("document_body", ""),
                        "document_text": original.get("document_text", ""),
                        "metadata": original.get("metadata", {}),
                        "keyword_rank": None,
                        "semantic_rank": None,
                        "keyword_score": float(bm25_scores.get(ci, {}).get(rid, 0.0)),
                        "semantic_score": float(dense_scores.get(ci, {}).get(rid, 0.0)),
                        "norm_bm25": nb,
                        "norm_dense": nd,
                        "rerank_score": hybrid_score,
                        "score": hybrid_score,
                        "retrieval_method": "rerank",
                        "rank": rank,
                    }
                )
    return reranked_results


def run_project_reranking(
    *,
    keyword_results: list[dict[str, Any]],
    semantic_results: list[dict[str, Any]],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    result_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    bm25_map: dict[int, dict[str, Any]] = {}
    # clause → {doc_id → best_record} : 멀티쿼리에서 동일 doc의 best(최고점) rank 유지
    _dense_best: dict[str, dict[str, dict[str, Any]]] = {}

    for result in keyword_results:
        clause_index = int(result["clause_index"])
        result_id = str(result["result_id"])
        result_lookup.setdefault((clause_index, result_id), dict(result))
        bm25_item = bm25_map.setdefault(
            clause_index, {"special_terms": result["clause"], "rank_map": {}}
        )
        cur_rank = int(result["rank"])
        # 멀티쿼리: 동일 doc이 여러 쿼리에서 나올 때 best rank(최솟값) 유지
        if result_id not in bm25_item["rank_map"] or cur_rank < bm25_item["rank_map"][result_id]:
            bm25_item["rank_map"][result_id] = cur_rank

    for result in semantic_results:
        clause_index = int(result["clause_index"])
        result_id = str(result["result_id"])
        lookup_item = result_lookup.setdefault((clause_index, result_id), dict(result))
        lookup_item["semantic_score"] = result.get("semantic_score", result.get("score", 0.0))
        bm25_map.setdefault(
            clause_index, {"special_terms": result["clause"], "rank_map": {}}
        )
        clause_text = result["clause"]
        new_score = float(result.get("score", result.get("similarity", 0.0)))
        new_rank = int(result["rank"])
        clause_best = _dense_best.setdefault(clause_text, {})
        # 멀티쿼리: 동일 doc이 여러 쿼리에서 나올 때 best score(최댓값) 유지
        if result_id not in clause_best or new_score > clause_best[result_id]["score"]:
            clause_best[result_id] = {
                "doc_id": result_id,
                "score": new_score,
                "rank": new_rank,
                "doc_text": result.get("document_body") or result.get("document_text", ""),
                "summary": result.get("metadata", {}).get("judgment_summary", ""),
            }

    # best record dict → 점수 내림차순 재정렬 후 rank 재부여
    dense_map: dict[str, list[dict[str, Any]]] = {}
    for clause_text, best_by_id in _dense_best.items():
        sorted_records = sorted(best_by_id.values(), key=lambda r: r["score"], reverse=True)
        for new_rank, rec in enumerate(sorted_records, 1):
            rec["rank"] = new_rank
        dense_map[clause_text] = sorted_records

    reranked_groups = rrf_baseline.run_rrf(bm25_map, dense_map, rrf_baseline.K, top_k)
    reranked_results: list[dict[str, Any]] = []
    for group in reranked_groups:
        clause_index = int(group["index"])
        for match in group.get("top_matches", []):
            result_id = str(match["doc_id"])
            original = result_lookup.get((clause_index, result_id), {})
            reranked_results.append(
                {
                    "clause_index": clause_index,
                    "clause": group["special_terms"],
                    "result_id": result_id,
                    "source_type": original.get("source_type", "document"),
                    "document_body": original.get("document_body", match.get("doc_text", "")),
                    "document_text": original.get("document_text", match.get("doc_text", "")),
                    "metadata": original.get("metadata", {}),
                    "keyword_rank": match.get("bm25_rank"),
                    "semantic_rank": match.get("dense_rank"),
                    "keyword_score": float(original.get("keyword_score", 0.0)),
                    "semantic_score": float(original.get("semantic_score", 0.0)),
                    "rerank_score": float(match["rrf_score"]),
                    "score": float(match["rrf_score"]),
                    "retrieval_method": "rerank",
                    "rank": int(match["rank"]),
                }
            )
    return reranked_results


# ── stage output count (property 기반) ────────────────────────────────────

def current_stage_output_count(context: CaseExecutionContext, stage: str) -> int:
    if stage == "query_expansion":
        return len(context.expanded_queries)
    if stage == "hybrid_retrieval":
        return len(context.keyword_results) + len(context.semantic_results)
    if stage == "reranking":
        return sum(len(r) for r in context.reranked_results_by_config.values())
    if stage == "document_inspection":
        return len(context.inspected_documents)
    return 0


def record_stage(
    context: CaseExecutionContext,
    stage: str,
    output_count: int,
    *,
    status: str = "completed",
    error: dict[str, str] | None = None,
) -> None:
    context.stage_trace.append(
        {
            "stage": stage,
            "case_id": context.case_id,
            "status": status,
            "output_count": output_count,
            "error": error,
        }
    )


# ── 지표 계산 ──────────────────────────────────────────────────────────────

def calculate_recall_by_k(
    *,
    law_references: list[dict[str, Any]],
    precedent_references: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    law_total = len(law_references)
    precedent_total = len(precedent_references)
    metrics: dict[str, dict[str, float | int]] = {}

    for k in DEFAULT_RECALL_K_VALUES:
        top_results = top_ranked_results(reranked_results, k)
        law_hits = count_law_hits(law_references, top_results)
        precedent_hits = count_precedent_hits(precedent_references, top_results, law_references)
        metrics[str(k)] = {
            "law": ratio(law_hits, law_total),
            "precedent": ratio(precedent_hits, precedent_total),
            "law_hits": law_hits,
            "law_total": law_total,
            "precedent_hits": precedent_hits,
            "precedent_total": precedent_total,
        }
    return metrics



def calculate_law_recall_at_k(
    *,
    law_references: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    law_total = len(law_references)
    metrics: dict[str, dict[str, float | int]] = {}
    for k in DEFAULT_RECALL_K_VALUES:
        law_hits = count_law_hits(law_references, top_ranked_results(reranked_results, k))
        metrics[str(k)] = {
            "recall": ratio(law_hits, law_total),
            "hits": law_hits,
            "total": law_total,
        }
    return metrics


def calculate_stage_recall_at_k(
    *,
    law_references: list[dict[str, Any]],
    precedent_references: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    metrics = calculate_recall_by_k(
        law_references=law_references,
        precedent_references=precedent_references,
        reranked_results=results,
    )
    for values in metrics.values():
        hits = int(values["law_hits"]) + int(values["precedent_hits"])
        total = int(values["law_total"]) + int(values["precedent_total"])
        values["hits"] = hits
        values["total"] = total
        values["integrated"] = ratio(hits, total)
    return metrics


def count_source_candidates(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"law": 0, "precedent": 0, "total": 0}
    for result in results:
        source_type = result.get("source_type")
        if source_type not in {"law", "precedent"}:
            continue
        counts[source_type] += 1
        counts["total"] += 1
    return counts


def calculate_precedent_hit_flags_by_k(
    *,
    law_references: list[dict[str, Any]],
    precedent_references: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, str | bool]]]:
    return {
        str(k): precedent_hit_flags(
            precedent_references=precedent_references,
            results=top_ranked_results(reranked_results, k),
            law_references=law_references,
        )
        for k in DEFAULT_RECALL_K_VALUES
    }


def calculate_query_hit_flags_by_k(
    *,
    case_id: str,
    clauses: list[str],
    expanded_queries: list[dict[str, Any]],
    law_references: list[dict[str, Any]],
    precedent_references: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    query_records = query_records_for_hit_flags(clauses, expanded_queries)
    return {
        str(k): [
            query_hit_flags(
                case_id=case_id,
                query=query,
                query_count=len(query_records),
                law_references=law_references,
                precedent_references=precedent_references,
                reranked_results=reranked_results,
                k=k,
            )
            for query in query_records
        ]
        for k in DEFAULT_RECALL_K_VALUES
    }


def query_records_for_hit_flags(
    clauses: list[str],
    expanded_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if expanded_queries:
        return [
            {
                "clause_index": query.get("clause_index"),
                "query": query.get("query", query.get("clause", "")),
            }
            for query in expanded_queries
        ]
    return [
        {"clause_index": idx, "query": clause}
        for idx, clause in enumerate(clauses)
    ]


def query_hit_flags(
    *,
    case_id: str,
    query: dict[str, Any],
    query_count: int,
    law_references: list[dict[str, Any]],
    precedent_references: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
    k: int,
) -> dict[str, Any]:
    query_results = top_ranked_query_results(
        reranked_results=reranked_results,
        clause_index=query.get("clause_index"),
        query_count=query_count,
        k=k,
    )
    law_ref_hits = law_reference_hit_flags(law_references, query_results)
    precedent_ref_hits = precedent_hit_flags(
        precedent_references=precedent_references,
        results=query_results,
        law_references=law_references,
    )
    law_hits = sum(1 for item in law_ref_hits if item["hit"])
    precedent_hits = sum(1 for item in precedent_ref_hits if item["hit"])
    return {
        "case_id": case_id,
        "clause_index": query.get("clause_index"),
        "query": query.get("query", ""),
        "law_reference_hits": law_ref_hits,
        "precedent_reference_hits": precedent_ref_hits,
        "law_hits": law_hits,
        "law_total": len(law_ref_hits),
        "precedent_hits": precedent_hits,
        "precedent_total": len(precedent_ref_hits),
        "hits": law_hits + precedent_hits,
        "total": len(law_ref_hits) + len(precedent_ref_hits),
    }


def top_ranked_query_results(
    *,
    reranked_results: list[dict[str, Any]],
    clause_index: Any,
    query_count: int,
    k: int,
) -> list[dict[str, Any]]:
    top_results = top_ranked_results(reranked_results, k)
    if query_count == 1 and all("clause_index" not in r for r in top_results):
        return top_results
    return [r for r in top_results if r.get("clause_index") == clause_index]


def law_reference_hit_flags(
    law_references: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, str | bool]]:
    result_keys = law_result_match_pool(results)
    return [
        {
            "law_child": ref["law_child"],
            "hit": law_reference_matches_result(ref, result_keys),
        }
        for ref in law_references
    ]


def precedent_hit_flags(
    *,
    precedent_references: list[dict[str, Any]],
    results: list[dict[str, Any]],
    law_references: list[dict[str, Any]] | None = None,
) -> list[dict[str, str | bool]]:
    result_case_numbers = [
        normalize_match_value(v)
        for result in results
        for v in precedent_match_values(result)
        if normalize_match_value(v)
    ]
    law_overlap = (
        _precedent_law_overlap(law_references, results) if law_references else False
    )
    return [
        {
            "case_number": ref["case_number"],
            "hit": reference_matches_result(ref["case_number"], result_case_numbers) or law_overlap,
            "hit_by_case_number": reference_matches_result(ref["case_number"], result_case_numbers),
            "hit_by_law_overlap": law_overlap,
        }
        for ref in precedent_references
    ]


def calculate_integrated_recall(
    *,
    law_references: list[dict[str, Any]],
    precedent_references: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
) -> dict[str, float | int]:
    law_hits = count_law_hits(law_references, reranked_results)
    precedent_hits = count_precedent_hits(precedent_references, reranked_results, law_references)
    law_total = len(law_references)
    precedent_total = len(precedent_references)
    total = law_total + precedent_total
    hits = law_hits + precedent_hits
    return {
        "value": ratio(hits, total),
        "hits": hits,
        "total": total,
        "law_hits": law_hits,
        "law_total": law_total,
        "precedent_hits": precedent_hits,
        "precedent_total": precedent_total,
        "evaluated_result_count": len(reranked_results),
    }


# ── 매크로/마이크로 Recall ──────────────────────────────────────────────────

def empty_macro_recall() -> dict[str, dict[str, float]]:
    return {str(k): {"law": 0.0, "precedent": 0.0, "integrated": 0.0} for k in DEFAULT_RECALL_K_VALUES}


def empty_micro_recall(case_reports: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    law_total = sum(len(r["law_references"]) for r in case_reports)
    precedent_total = sum(len(r["precedent_references"]) for r in case_reports)
    return {
        str(k): {
            "law": 0.0, "precedent": 0.0, "integrated": 0.0,
            "law_hits": 0, "law_total": law_total,
            "precedent_hits": 0, "precedent_total": precedent_total,
            "hits": 0, "total": law_total + precedent_total,
        }
        for k in DEFAULT_RECALL_K_VALUES
    }


def calculate_macro_recall(case_reports: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not case_reports:
        return empty_macro_recall()
    macro: dict[str, dict[str, float]] = {}
    for k in DEFAULT_RECALL_K_VALUES:
        k_key = str(k)
        law_values = [c["recall_by_k"][k_key]["law"] for c in case_reports]
        precedent_values = [c["recall_by_k"][k_key]["precedent"] for c in case_reports]
        integrated_values = [
            ratio(
                c["recall_by_k"][k_key]["law_hits"] + c["recall_by_k"][k_key]["precedent_hits"],
                c["recall_by_k"][k_key]["law_total"] + c["recall_by_k"][k_key]["precedent_total"],
            )
            for c in case_reports
        ]
        macro[k_key] = {
            "law": average(law_values),
            "precedent": average(precedent_values),
            "integrated": average(integrated_values),
        }
    return macro


def calculate_micro_recall(case_reports: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    if not case_reports:
        return empty_micro_recall(case_reports)
    micro: dict[str, dict[str, float | int]] = {}
    for k in DEFAULT_RECALL_K_VALUES:
        k_key = str(k)
        law_hits = sum(c["recall_by_k"][k_key]["law_hits"] for c in case_reports)
        law_total = sum(c["recall_by_k"][k_key]["law_total"] for c in case_reports)
        precedent_hits = sum(c["recall_by_k"][k_key]["precedent_hits"] for c in case_reports)
        precedent_total = sum(c["recall_by_k"][k_key]["precedent_total"] for c in case_reports)
        hits = law_hits + precedent_hits
        total = law_total + precedent_total
        micro[k_key] = {
            "law": ratio(law_hits, law_total),
            "precedent": ratio(precedent_hits, precedent_total),
            "integrated": ratio(hits, total),
            "law_hits": law_hits, "law_total": law_total,
            "precedent_hits": precedent_hits, "precedent_total": precedent_total,
            "hits": hits, "total": total,
        }
    return micro



def calculate_experiment_micro_recall(
    case_reports: list[dict[str, Any]],
    experiment_key: str,
) -> dict[str, dict[str, float | int]]:
    """실험 config(embed_col__rerank_type) 단위 micro Recall@K.

    각 케이스의 experiment_metrics[experiment_key].recall_by_k 에서
    law_hits / law_total / precedent_hits / precedent_total 을 전체 합산해
    micro 평균을 계산한다.
    """
    metrics: dict[str, dict[str, float | int]] = {}
    for k in DEFAULT_RECALL_K_VALUES:
        k_key = str(k)

        def _field(case: dict[str, Any], field: str) -> int:
            return (
                case.get("experiment_metrics", {})
                    .get(experiment_key, {})
                    .get("recall_by_k", {})
                    .get(k_key, {})
                    .get(field, 0)
            )

        law_hits      = sum(_field(c, "law_hits")        for c in case_reports)
        law_total     = sum(_field(c, "law_total")       for c in case_reports)
        prec_hits     = sum(_field(c, "precedent_hits")  for c in case_reports)
        prec_total    = sum(_field(c, "precedent_total") for c in case_reports)
        hits  = law_hits + prec_hits
        total = law_total + prec_total
        metrics[k_key] = {
            "law":              ratio(law_hits, law_total),
            "law_hits":         law_hits,
            "law_total":        law_total,
            "precedent":        ratio(prec_hits, prec_total),
            "precedent_hits":   prec_hits,
            "precedent_total":  prec_total,
            "integrated":       ratio(hits, total),
            "hits":             hits,
            "total":            total,
        }
    return metrics


def calculate_experiment_macro_recall(
    case_reports: list[dict[str, Any]],
    experiment_key: str,
) -> dict[str, dict[str, float]]:
    """실험 config 단위 macro Recall@K (케이스별 recall 평균)."""
    metrics: dict[str, dict[str, float]] = {}
    for k in DEFAULT_RECALL_K_VALUES:
        k_key = str(k)

        def _recall(case: dict[str, Any], src: str) -> float:
            return (
                case.get("experiment_metrics", {})
                    .get(experiment_key, {})
                    .get("recall_by_k", {})
                    .get(k_key, {})
                    .get(src, 0.0)
            )

        def _integrated(case: dict[str, Any]) -> float:
            rb = (
                case.get("experiment_metrics", {})
                    .get(experiment_key, {})
                    .get("recall_by_k", {})
                    .get(k_key, {})
            )
            return ratio(
                rb.get("law_hits", 0) + rb.get("precedent_hits", 0),
                rb.get("law_total", 0) + rb.get("precedent_total", 0),
            )

        metrics[k_key] = {
            "law":        average([_recall(c, "law")        for c in case_reports]),
            "precedent":  average([_recall(c, "precedent")  for c in case_reports]),
            "integrated": average([_integrated(c)           for c in case_reports]),
        }
    return metrics


# ── 런/스테이지 집계 ───────────────────────────────────────────────────────

def calculate_run_law_recall_at_k(
    micro_recall: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    return {
        str(k): {
            "recall": micro_recall.get(str(k), {}).get("law", 0.0),
            "hits": micro_recall.get(str(k), {}).get("law_hits", 0),
            "total": micro_recall.get(str(k), {}).get("law_total", 0),
        }
        for k in DEFAULT_RECALL_K_VALUES
    }


def calculate_run_stage_recall_at_k(
    case_reports: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        stage: calculate_stage_micro_recall(case_reports, stage)
        for stage in ("bm25", "semantic", "rerank")
    }


def calculate_run_semantic_stage_recall_at_k(
    case_reports: list[dict[str, Any]],
    semantic_embed_cols: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, float | int]]]:
    return {
        col: calculate_semantic_stage_micro_recall(case_reports, col)
        for col in normalize_semantic_embed_cols(semantic_embed_cols)
    }


def calculate_stage_micro_recall(
    case_reports: list[dict[str, Any]],
    stage: str,
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for k in DEFAULT_RECALL_K_VALUES:
        k_key = str(k)
        def _sum(field: str) -> int:
            return sum(
                c.get("stage_recall_at_k", {}).get(stage, {}).get(k_key, {}).get(field, 0)
                for c in case_reports
            )
        law_hits, law_total = _sum("law_hits"), _sum("law_total")
        precedent_hits, precedent_total = _sum("precedent_hits"), _sum("precedent_total")
        hits = law_hits + precedent_hits
        total = law_total + precedent_total
        metrics[k_key] = {
            "law": ratio(law_hits, law_total),
            "law_hits": law_hits, "law_total": law_total,
            "precedent": ratio(precedent_hits, precedent_total),
            "precedent_hits": precedent_hits, "precedent_total": precedent_total,
            "integrated": ratio(hits, total),
            "hits": hits, "total": total,
        }
    return metrics


def calculate_semantic_stage_micro_recall(
    case_reports: list[dict[str, Any]],
    semantic_embed_col: str,
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for k in DEFAULT_RECALL_K_VALUES:
        k_key = str(k)
        def _sum(field: str) -> int:
            return sum(
                c.get("semantic_stage_recall_at_k", {})
                 .get(semantic_embed_col, {})
                 .get(k_key, {})
                 .get(field, 0)
                for c in case_reports
            )
        law_hits, law_total = _sum("law_hits"), _sum("law_total")
        precedent_hits, precedent_total = _sum("precedent_hits"), _sum("precedent_total")
        hits = law_hits + precedent_hits
        total = law_total + precedent_total
        metrics[k_key] = {
            "law": ratio(law_hits, law_total),
            "law_hits": law_hits, "law_total": law_total,
            "precedent": ratio(precedent_hits, precedent_total),
            "precedent_hits": precedent_hits, "precedent_total": precedent_total,
            "integrated": ratio(hits, total),
            "hits": hits, "total": total,
        }
    return metrics


def calculate_run_candidate_counts(case_reports: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals = {stage: {"law": 0, "precedent": 0, "total": 0} for stage in ("bm25", "semantic", "rerank")}
    for case in case_reports:
        for stage, counts in case.get("candidate_counts", {}).items():
            if stage not in totals:
                continue
            totals[stage]["law"] += counts.get("law", 0)
            totals[stage]["precedent"] += counts.get("precedent", 0)
            totals[stage]["total"] += counts.get("total", 0)
    return totals


def calculate_run_semantic_candidate_counts_by_embed_col(
    case_reports: list[dict[str, Any]],
    semantic_embed_cols: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, int]]:
    totals = {
        col: {"law": 0, "precedent": 0, "total": 0}
        for col in normalize_semantic_embed_cols(semantic_embed_cols)
    }
    for case in case_reports:
        for col, counts in case.get("semantic_candidate_counts_by_embed_col", {}).items():
            if col not in totals:
                continue
            totals[col]["law"] += counts.get("law", 0)
            totals[col]["precedent"] += counts.get("precedent", 0)
            totals[col]["total"] += counts.get("total", 0)
    return totals


def calculate_query_recall_by_k(
    case_reports: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for k in DEFAULT_RECALL_K_VALUES:
        k_key = str(k)
        query_flags = [
            query
            for cr in case_reports
            for query in cr.get("query_hit_flags_by_k", {}).get(k_key, [])
        ]
        law_hits = sum(q["law_hits"] for q in query_flags)
        law_total = sum(q["law_total"] for q in query_flags)
        precedent_hits = sum(q["precedent_hits"] for q in query_flags)
        precedent_total = sum(q["precedent_total"] for q in query_flags)
        hits = law_hits + precedent_hits
        total = law_total + precedent_total
        metrics[k_key] = {
            "law": ratio(law_hits, law_total),
            "precedent": ratio(precedent_hits, precedent_total),
            "integrated": ratio(hits, total),
            "law_hits": law_hits, "law_total": law_total,
            "precedent_hits": precedent_hits, "precedent_total": precedent_total,
            "hits": hits, "total": total,
            "query_count": len(query_flags),
        }
    return metrics


def calculate_query_macro_recall_by_k(
    case_reports: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for k in DEFAULT_RECALL_K_VALUES:
        k_key = str(k)
        query_flags = [
            query
            for cr in case_reports
            for query in cr.get("query_hit_flags_by_k", {}).get(k_key, [])
        ]
        metrics[k_key] = {
            "law": average([ratio(q["law_hits"], q["law_total"]) for q in query_flags]),
            "precedent": average([ratio(q["precedent_hits"], q["precedent_total"]) for q in query_flags]),
            "integrated": average([ratio(q["hits"], q["total"]) for q in query_flags]),
            "query_count": len(query_flags),
        }
    return metrics


# ── 매칭 유틸 ──────────────────────────────────────────────────────────────

def top_ranked_results(reranked_results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return [
        r for r in sorted(reranked_results, key=lambda r: r.get("rank", 10**9))
        if isinstance(r.get("rank"), int) and r["rank"] <= k
    ]


def count_law_hits(law_references: list[dict[str, Any]], results: list[dict[str, Any]]) -> int:
    result_keys = law_result_match_pool(results)
    return sum(
        1 for ref in law_references
        if law_reference_matches_result(ref, result_keys)
    )


def law_result_match_pool(results: list[dict[str, Any]]) -> list[str]:
    return [
        normalize_match_value(v)
        for result in results
        for v in (*law_match_values(result), *precedent_reference_law_values(result))
        if normalize_match_value(v)
    ]


def law_reference_matches_result(reference: dict[str, Any], result_values: list[str]) -> bool:
    return any(
        reference_matches_result(candidate, result_values)
        for candidate in law_reference_values(reference)
    )


def law_reference_values(reference: dict[str, Any]) -> list[str]:
    values: list[str] = []
    values.extend(reference_rule_candidates(reference.get("reference_rules")))
    law_child = reference.get("law_child")
    if law_child:
        values.append(str(law_child))
    return dedupe_strings(values)


def precedent_reference_law_values(result: dict[str, Any]) -> list[str]:
    if result.get("source_type") != "precedent":
        return []
    values: list[str] = []
    values.extend(reference_rule_candidates(result.get("reference_rules")))
    values.extend(reference_rule_candidates(result.get("referenced_law_keys")))
    values.extend(reference_rule_candidates(result.get("referenced_law")))
    values.extend(reference_rule_candidates(result.get("referenced_laws")))
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        values.extend(reference_rule_candidates(metadata.get("reference_rules")))
        values.extend(reference_rule_candidates(metadata.get("referenced_law_keys")))
        values.extend(reference_rule_candidates(metadata.get("referenced_law")))
        values.extend(reference_rule_candidates(metadata.get("referenced_laws")))
        case_law = metadata.get("case_law")
        if isinstance(case_law, dict):
            values.extend(reference_rule_candidates(case_law.get("reference_rules")))
            values.extend(reference_rule_candidates(case_law.get("referenced_law_keys")))
            values.extend(reference_rule_candidates(case_law.get("referenced_law")))
            values.extend(reference_rule_candidates(case_law.get("referenced_laws")))
    return dedupe_strings(values)


RULE_KEY_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+_제\d+조(?:의\d+)?(?:_제\d+항)?")
RULE_TEXT_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+(?:\s*제\d+조(?:의\d+)?(?:\s*제\d+항)?)")


def reference_rule_candidates(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(reference_rule_candidates(item))
        return dedupe_strings(values)
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(reference_rule_candidates(item))
        return dedupe_strings(values)
    text = str(value).strip()
    if not text:
        return []
    values: list[str] = []
    values.extend(RULE_KEY_PATTERN.findall(text))
    values.extend(match.group(0) for match in RULE_TEXT_PATTERN.finditer(text))
    if not values:
        values.extend(part.strip() for part in re.split(r"[|,;/\n]", text) if part.strip())
    return dedupe_strings(values)


def dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value).strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def _precedent_law_overlap(
    law_references: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> bool:
    """검색된 판례 중 하나라도 정답 법령과 공통 조문을 가지면 True."""
    gt_law_candidates = {
        normalize_match_value(c)
        for ref in law_references
        for c in law_reference_values(ref)
        if normalize_match_value(c)
    }
    if not gt_law_candidates:
        return False
    retrieved_citations = {
        normalize_match_value(v)
        for result in results
        for v in precedent_reference_law_values(result)
        if normalize_match_value(v)
    }
    return bool(gt_law_candidates & retrieved_citations)


def count_precedent_hits(
    precedent_references: list[dict[str, Any]],
    results: list[dict[str, Any]],
    law_references: list[dict[str, Any]] | None = None,
) -> int:
    result_case_numbers = [
        normalize_match_value(v)
        for result in results
        for v in precedent_match_values(result)
        if normalize_match_value(v)
    ]
    # 참조조문 기준: 검색된 판례가 정답 법령을 인용하면 hit로 인정
    law_overlap = (
        _precedent_law_overlap(law_references, results) if law_references else False
    )
    return sum(
        1 for ref in precedent_references
        if reference_matches_result(ref["case_number"], result_case_numbers) or law_overlap
    )


def law_match_values(result: dict[str, Any]) -> list[Any]:
    if result.get("source_type") != "law":
        return []
    values = [
        result.get("clause_key"),
        result.get("law_name"),
        result.get("article_key"),
        result.get("article_no"),
        result.get("paragraph_no"),
        result.get("result_id"),
    ]
    values.extend(combined_law_values(result))
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        values.extend([
            metadata.get("clause_key"),
            metadata.get("law_name"),
            metadata.get("article_key"),
            metadata.get("article_no"),
            metadata.get("paragraph_no"),
        ])
        values.extend(combined_law_values(metadata))
    return values


def precedent_match_values(result: dict[str, Any]) -> list[Any]:
    if result.get("source_type") != "precedent":
        return []
    values = [result.get("case_number"), result.get("case_name"), result.get("result_id")]
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return values
    values.extend([metadata.get("case_number"), metadata.get("case_name")])
    case_law = metadata.get("case_law")
    if isinstance(case_law, dict):
        values.extend([case_law.get("case_number"), case_law.get("case_name")])
    return values


def combined_law_values(item: dict[str, Any]) -> list[str]:
    law_name = item.get("law_name")
    clause_key = item.get("clause_key")
    article_key = item.get("article_key")
    article_no = item.get("article_no")
    paragraph_no = item.get("paragraph_no")

    if not law_name and isinstance(clause_key, str) and "_" in clause_key:
        law_name = clause_key.split("_", 1)[0]
    if not law_name:
        return []

    forms: list[str] = []

    compact_key = article_key
    if not compact_key and isinstance(clause_key, str) and "_" in clause_key:
        compact_key = clause_key.split("_", 1)[1]
    if compact_key:
        compact_key = str(compact_key).strip()
        if compact_key:
            forms.extend(
                [
                    f"{law_name}_{compact_key}",
                    f"{law_name} {compact_key}",
                    f"{law_name}{compact_key}",
                ]
            )

    korean_key = article_key_from_numbers(article_no, paragraph_no)
    if korean_key:
        forms.extend(
            [
                f"{law_name}_{korean_key}",
                f"{law_name} {korean_key}",
                f"{law_name}{korean_key}",
            ]
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for form in forms:
        normalized = form.strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def article_key_from_numbers(article_no: Any, paragraph_no: Any) -> str:
    article_text = normalize_article_no(article_no)
    if not article_text:
        return ""
    paragraph_text = normalize_paragraph_no(paragraph_no)
    return f"{article_text}_{paragraph_text}" if paragraph_text else article_text


def normalize_article_no(article_no: Any) -> str:
    if article_no is None:
        return ""
    text_value = str(article_no).strip()
    if not text_value:
        return ""
    if "조" in text_value:
        return text_value
    return f"제{text_value}조"


def normalize_paragraph_no(paragraph_no: Any) -> str:
    if paragraph_no is None:
        return ""
    text_value = str(paragraph_no).strip()
    if not text_value:
        return ""
    if text_value in {"0", "0.0", "nan", "None"}:
        return ""
    if "항" in text_value:
        return text_value
    return f"제{text_value}항"


def reference_matches_result(reference: Any, result_values: list[str]) -> bool:
    normalized_reference = normalize_match_value(reference)
    if not normalized_reference:
        return False
    return any(normalized_reference in rv for rv in result_values)


def normalize_match_value(value: Any) -> str:
    if value is None:
        return ""
    return "".join(str(value).strip().casefold().replace("_", " ").split())


def ratio(hits: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(hits / total, 6)


def average(values: list[float | int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)


def count_case_statuses(case_reports: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"completed": 0, "failed": 0}
    for cr in case_reports:
        status = cr["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


# ── 케이스 리포트 빌드 ─────────────────────────────────────────────────────

def build_case_report(
    context: CaseExecutionContext,
    expand_fn=None,
    alpha_law: float = ALPHA_LAW,
    alpha_prec: float = ALPHA_PREC,
) -> dict[str, Any]:
    run_case_pipeline(context, expand_fn=expand_fn, alpha_law=alpha_law, alpha_prec=alpha_prec)
    return case_report_payload(context)


def group_semantic_results_by_embed_col(
    results: list[dict[str, Any]],
    semantic_embed_cols: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        col: [] for col in normalize_semantic_embed_cols(semantic_embed_cols)
    }
    for result in results:
        col = result.get("semantic_embed_col") or DEFAULT_SEMANTIC_EMBED_COL
        grouped.setdefault(str(col), []).append(result)
    return grouped


def case_report_payload(context: CaseExecutionContext) -> dict[str, Any]:
    case = context.case
    clauses = list(case["clauses"])

    # semantic_results_by_model은 이미 embed_col → results 구조
    semantic_results_by_embed_col = context.semantic_results_by_model

    stage_results = {
        "bm25": context.keyword_results,
        "semantic": context.semantic_results,   # property (flatten)
        "rerank": context.reranked_results,     # property (best-rank de-dup)
    }
    stage_recall_at_k = {
        stage: calculate_stage_recall_at_k(
            law_references=case["law_references"],
            precedent_references=case["precedent_references"],
            results=results,
        )
        for stage, results in stage_results.items()
    }
    semantic_stage_recall_at_k = {
        col: calculate_stage_recall_at_k(
            law_references=case["law_references"],
            precedent_references=case["precedent_references"],
            results=results,
        )
        for col, results in semantic_results_by_embed_col.items()
    }
    candidate_counts = {
        stage: count_source_candidates(results) for stage, results in stage_results.items()
    }
    semantic_candidate_counts_by_embed_col = {
        col: count_source_candidates(results)
        for col, results in semantic_results_by_embed_col.items()
    }

    # FIX: recall_by_k 변수 선언 (merged reranked_results 기준)
    recall_by_k = calculate_recall_by_k(
        law_references=case["law_references"],
        precedent_references=case["precedent_references"],
        reranked_results=context.reranked_results,
    )

    # experiment_metrics — embed_col × rerank_type 조합별 지표
    experiment_metrics: dict[str, dict[str, Any]] = {}
    for experiment_key, exp_results in context.reranked_results_by_config.items():
        experiment_metrics[experiment_key] = {
            "recall_by_k": calculate_recall_by_k(
                law_references=case["law_references"],
                precedent_references=case["precedent_references"],
                reranked_results=exp_results,
            ),
        }

    law_recall_at_k = calculate_law_recall_at_k(
        law_references=case["law_references"],
        reranked_results=context.reranked_results,
    )
    precedent_hit_flags_by_k = calculate_precedent_hit_flags_by_k(
        law_references=case["law_references"],
        precedent_references=case["precedent_references"],
        reranked_results=context.reranked_results,
    )
    query_hit_flags_by_k = calculate_query_hit_flags_by_k(
        case_id=context.case_id,
        clauses=clauses,
        expanded_queries=context.expanded_queries,
        law_references=case["law_references"],
        precedent_references=case["precedent_references"],
        reranked_results=context.reranked_results,
    )
    query_expansion = build_query_expansion_payload(
        clauses=clauses,
        expanded_queries=context.expanded_queries,
    )
    integrated_recall = calculate_integrated_recall(
        law_references=case["law_references"],
        precedent_references=case["precedent_references"],
        reranked_results=context.reranked_results,
    )

    return {
        "case_id": context.case_id,
        "status": context.status,
        "error": context.error,
        "execution_context": {
            "case_id": context.case_id,
            "case_index": context.case_index,
            "input_path": str(context.input_path),
            "semantic_embed_cols": list(context.semantic_embed_cols),
        },
        "clauses": clauses,
        "law_references": case["law_references"],
        "precedent_references": case["precedent_references"],
        "query_expansion": query_expansion,
        "expanded_queries": context.expanded_queries,
        # "keyword_results": context.keyword_results,
        # "semantic_results": context.semantic_results,
        # "semantic_results_by_model": {
        #     col: results
        #     for col, results in context.semantic_results_by_model.items()
        # },
        "reranked_results": context.reranked_results,          # property (merged)
        # "reranked_results_by_config": {
        #     key: results
        #     for key, results in context.reranked_results_by_config.items()
        # },
        # "inspected_documents": context.inspected_documents,
        "stage_trace": context.stage_trace,
        "candidate_counts": candidate_counts,
        "semantic_candidate_counts_by_embed_col": semantic_candidate_counts_by_embed_col,
        "stage_recall_at_k": stage_recall_at_k,
        "semantic_stage_recall_at_k": semantic_stage_recall_at_k,
        "recall_by_k": recall_by_k,
        "experiment_metrics": experiment_metrics,
        "law_recall_at_k": law_recall_at_k,
        "precedent_hit_flags_by_k": precedent_hit_flags_by_k,
        "query_hit_flags_by_k": query_hit_flags_by_k,
        "integrated_recall": integrated_recall,
    }


def build_query_expansion_payload(
    *,
    clauses: list[str],
    expanded_queries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "query_count": len(expanded_queries),
        "clauses": clauses,
        "expanded_queries": expanded_queries,
        "per_clause_outputs": [query_expansion_step_output(q) for q in expanded_queries],
        "expansion_queries": [
            q["expansion_query"]
            for q in expanded_queries
            if isinstance(q.get("expansion_query"), str)
        ],
        "keywords": unique_query_keywords(expanded_queries),
    }


def query_expansion_step_output(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "clause_index": query.get("clause_index"),
        "clause": query.get("clause"),
        "query": query.get("query"),
        "expansion_query": query.get("expansion_query"),
        "keywords": query.get("keywords", []),
        "expansion": query.get("expansion", {}),
        "retrieval_payload": query.get("retrieval_payload", {}),
    }


def unique_query_keywords(expanded_queries: list[dict[str, Any]]) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for query in expanded_queries:
        query_keywords = query.get("keywords")
        if not isinstance(query_keywords, list):
            query_keywords = query.get("expansion", {}).get("keywords", [])
        for keyword in query_keywords:
            if not isinstance(keyword, str):
                continue
            normalized = keyword.strip()
            if not normalized or normalized in seen:
                continue
            keywords.append(normalized)
            seen.add(normalized)
    return keywords


# ── 런 리포트 빌드 ──────────────────────────────────────────────────────────

def build_report(
    *,
    input_path: Path,
    cases: list[dict[str, Any]],
    case_id: str | None,
    expand_fn=None,
    semantic_embed_cols: str | list[str] | tuple[str, ...] | None = None,
    semantic_embed_col: str | None = None,
    case_workers: int = CASE_WORKERS,
    alpha_law: float = ALPHA_LAW,
    alpha_prec: float = ALPHA_PREC,
) -> dict[str, Any]:
    selected_embed_cols = normalize_semantic_embed_cols(
        semantic_embed_cols, semantic_embed_col=semantic_embed_col
    )
    contexts = build_case_contexts(
        input_path=input_path,
        cases=cases,
        semantic_embed_cols=selected_embed_cols,
    )
    log_progress(f"eval_start input_path={input_path} cases={len(contexts)} case_workers={case_workers}")

    # corpus 및 임베딩 모델을 병렬 실행 전에 사전 로드 (초기화 경합 방지)
    log_progress("corpus_preload_start")
    load_bm25_corpus()
    for col in selected_embed_cols:
        load_semantic_chunks(col)
    # 임베딩 모델 단일 스레드에서 초기화 (SentenceTransformer 동시 로드 방지)
    for col in selected_embed_cols:
        dense_retrieval.embed_query("초기화", col)
    log_progress("corpus_preload_done")

    n_workers = min(case_workers, len(contexts))

    def _run_case(context: CaseExecutionContext) -> dict[str, Any]:
        log_progress(
            f"case_start index={context.case_index + 1}/{len(contexts)} "
            f"case_id={context.case_id}"
        )
        case_report = build_case_report(
            context, expand_fn=expand_fn,
            alpha_law=alpha_law, alpha_prec=alpha_prec,
        )
        log_progress(
            f"case_done index={context.case_index + 1}/{len(contexts)} "
            f"case_id={context.case_id} status={case_report['status']}"
        )
        return case_report

    if n_workers <= 1:
        case_reports = [_run_case(ctx) for ctx in contexts]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            case_reports = list(pool.map(_run_case, contexts))

    return build_run_report(
        input_path=input_path,
        requested_case_id=case_id,
        case_reports=case_reports,
        semantic_embed_cols=selected_embed_cols,
        alpha_law=alpha_law,
        alpha_prec=alpha_prec,
    )


def build_run_report(
    *,
    input_path: Path,
    requested_case_id: str | None,
    case_reports: list[dict[str, Any]],
    semantic_embed_cols: str | list[str] | tuple[str, ...] | None = None,
    semantic_embed_col: str | None = None,
    alpha_law: float = ALPHA_LAW,
    alpha_prec: float = ALPHA_PREC,
) -> dict[str, Any]:
    selected_embed_cols = normalize_semantic_embed_cols(
        semantic_embed_cols, semantic_embed_col=semantic_embed_col
    )
    selected_embed_configs = semantic_embed_configs(selected_embed_cols)
    status_counts = count_case_statuses(case_reports)
    aggregation = build_case_aggregation(case_reports, status_counts)
    macro_recall = calculate_macro_recall(case_reports)
    micro_recall = calculate_micro_recall(case_reports)
    law_recall_at_k = calculate_run_law_recall_at_k(micro_recall)
    query_recall_by_k = calculate_query_recall_by_k(case_reports)
    query_macro_recall_by_k = calculate_query_macro_recall_by_k(case_reports)
    stage_recall_at_k = calculate_run_stage_recall_at_k(case_reports)
    semantic_stage_recall_at_k = calculate_run_semantic_stage_recall_at_k(
        case_reports, selected_embed_cols
    )
    candidate_counts = calculate_run_candidate_counts(case_reports)
    semantic_candidate_counts_by_embed_col = (
        calculate_run_semantic_candidate_counts_by_embed_col(case_reports, selected_embed_cols)
    )

    # 실험 키 수집 (embed_col__rerank_type 조합)
    all_experiment_keys: list[str] = []
    for cr in case_reports:
        for key in cr.get("experiment_metrics", {}):
            if key not in all_experiment_keys:
                all_experiment_keys.append(key)

    # 실험별 Recall@K 집계 (micro + macro)
    experiment_micro_recall = {
        key: calculate_experiment_micro_recall(case_reports, key)
        for key in all_experiment_keys
    }
    experiment_macro_recall = {
        key: calculate_experiment_macro_recall(case_reports, key)
        for key in all_experiment_keys
    }

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run": {
            "input_path": str(input_path),
            "requested_case_id": requested_case_id,
            "recall_k_values": DEFAULT_RECALL_K_VALUES,
            "semantic_embed_cols": list(selected_embed_cols),
            "semantic_embed_configs": selected_embed_configs,
            "alpha_law": alpha_law,
            "alpha_prec": alpha_prec,
        },
        "case_aggregation": aggregation,
        "metrics": {
            "macro_recall": macro_recall,
            "micro_recall": micro_recall,
            "recall_at_k": micro_recall,
            "experiment_micro_recall": experiment_micro_recall,
            "experiment_macro_recall": experiment_macro_recall,
            "law_recall_at_k": law_recall_at_k,
            "query_recall_by_k": query_recall_by_k,
            "query_macro_recall_by_k": query_macro_recall_by_k,
            "stage_recall_at_k": stage_recall_at_k,
            "semantic_stage_recall_at_k": semantic_stage_recall_at_k,
            "candidate_counts": candidate_counts,
            "semantic_candidate_counts_by_embed_col": semantic_candidate_counts_by_embed_col,
        },
        "input_path": str(input_path),
        "case_id": requested_case_id,
        "case_count": len(case_reports),
        "status_counts": status_counts,
        "recall_k_values": DEFAULT_RECALL_K_VALUES,
        "semantic_embed_cols": list(selected_embed_cols),
        "semantic_embed_configs": selected_embed_configs,
        "cases": case_reports,
    }
    return report


def build_case_aggregation(
    case_reports: list[dict[str, Any]],
    status_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "case_count": len(case_reports),
        "completed_case_count": status_counts.get("completed", 0),
        "failed_case_count": status_counts.get("failed", 0),
        "status_counts": status_counts,
        "total_law_references": sum(len(cr["law_references"]) for cr in case_reports),
        "total_precedent_references": sum(len(cr["precedent_references"]) for cr in case_reports),
        "case_ids": [cr["case_id"] for cr in case_reports],
    }


# ── 리포트 저장 ────────────────────────────────────────────────────────────

def save_report(report: dict[str, Any], output_path: Path | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_path = RESULTS_DIR / f"legal_retrieval_eval_{timestamp}.json"
    else:
        report_path = RESULTS_DIR / output_path.expanduser().name

    report["report_path"] = str(report_path)
    report.setdefault("run", {})["report_path"] = str(report_path)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return report_path


# ── 콘솔 출력 포맷 ─────────────────────────────────────────────────────────

def format_core_summary_console_line(report: dict[str, Any]) -> str:
    status_counts = report.get("status_counts", {})
    return (
        f"summary input_path={report.get('input_path', '')} "
        f"cases={report.get('case_count', 0)} "
        f"completed={status_counts.get('completed', 0)} "
        f"failed={status_counts.get('failed', 0)}"
    )


def format_evaluation_unit_console_line(report: dict[str, Any]) -> str:
    aggregation = report.get("case_aggregation", {})
    return (
        "evaluation_unit=record "
        "record_unit=qa_or_case_law "
        "gt_scope=record_level "
        "note=clauses_share_record_gt "
        f"records={aggregation.get('case_count', report.get('case_count', 0))} "
        f"law_gt={aggregation.get('total_law_references', 0)} "
        f"precedent_gt={aggregation.get('total_precedent_references', 0)}"
    )


def format_semantic_embedding_console_line(report: dict[str, Any]) -> str:
    embed_cols = report.get("semantic_embed_cols")
    if not isinstance(embed_cols, list):
        embed_cols = [report.get("semantic_embed_col", DEFAULT_SEMANTIC_EMBED_COL)]
    configs = report.get("semantic_embed_configs", {})
    column_parts = []
    for col in embed_cols:
        config = configs.get(col) if isinstance(configs, dict) else None
        if not isinstance(config, dict):
            config = semantic_embed_config(col)
        column_parts.append(
            f"{col}:query={config['query_embed_col']},"
            f"law={config['law_embed_col']},"
            f"precedent={config['precedent_embed_col']}"
        )
    return (
        f"semantic_embeddings={','.join(embed_cols)} "
        f"columns={';'.join(column_parts)}"
    )


def format_experiment_recall_console_lines(report: dict[str, Any]) -> list[str]:
    """실험별(embed_col × rerank_type) Recall@K 콘솔 출력."""
    lines: list[str] = []
    metrics = report.get("metrics", {})

    experiment_micro_recall = metrics.get("experiment_micro_recall", {})
    for exp_key, recall_by_k in experiment_micro_recall.items():
        for k in (3, 5, 10):
            vals = recall_by_k.get(str(k), {})
            lines.append(
                f"experiment_micro_recall@{k} experiment={exp_key} "
                f"law={vals.get('law', 0.0):.6f} "
                f"law_hits={vals.get('law_hits', 0)} "
                f"law_total={vals.get('law_total', 0)} "
                f"precedent={vals.get('precedent', 0.0):.6f} "
                f"precedent_hits={vals.get('precedent_hits', 0)} "
                f"precedent_total={vals.get('precedent_total', 0)} "
                f"integrated={vals.get('integrated', 0.0):.6f}"
            )

    experiment_macro_recall = metrics.get("experiment_macro_recall", {})
    for exp_key, recall_by_k in experiment_macro_recall.items():
        for k in (3, 5, 10):
            vals = recall_by_k.get(str(k), {})
            lines.append(
                f"experiment_macro_recall@{k} experiment={exp_key} "
                f"law={vals.get('law', 0.0):.6f} "
                f"precedent={vals.get('precedent', 0.0):.6f} "
                f"integrated={vals.get('integrated', 0.0):.6f}"
            )

    return lines


def format_law_recall_console_lines(report: dict[str, Any]) -> list[str]:
    micro_recall = report.get("metrics", {}).get("micro_recall", {})
    lines: list[str] = []
    for k in report.get("recall_k_values", DEFAULT_RECALL_K_VALUES):
        k_key = str(k)
        recall_at_k = micro_recall.get(k_key, {})
        lines.append(
            f"law_recall@{k}={recall_at_k.get('law', 0.0):.6f} "
            f"hits={recall_at_k.get('law_hits', 0)} "
            f"total={recall_at_k.get('law_total', 0)}"
        )
    return lines


def format_recall_at_k_console_lines(report: dict[str, Any]) -> list[str]:
    recall_at_k = report.get("metrics", {}).get("recall_at_k")
    if not isinstance(recall_at_k, dict):
        recall_at_k = report.get("metrics", {}).get("micro_recall", {})
    lines: list[str] = []
    for k in report.get("recall_k_values", DEFAULT_RECALL_K_VALUES):
        k_key = str(k)
        metrics = recall_at_k.get(k_key, {})
        lines.append(
            f"recall@{k} "
            f"law={metrics.get('law', 0.0):.6f} "
            f"law_hits={metrics.get('law_hits', 0)} "
            f"law_total={metrics.get('law_total', 0)} "
            f"precedent={metrics.get('precedent', 0.0):.6f} "
            f"precedent_hits={metrics.get('precedent_hits', 0)} "
            f"precedent_total={metrics.get('precedent_total', 0)} "
            f"integrated={metrics.get('integrated', 0.0):.6f} "
            f"hits={metrics.get('hits', 0)} "
            f"total={metrics.get('total', 0)}"
        )
    return lines


def format_dataset_recall_summary_console_lines(report: dict[str, Any]) -> list[str]:
    metrics = report.get("metrics", {})
    macro_recall = metrics.get("macro_recall", {})
    micro_recall = metrics.get("micro_recall", {})
    lines: list[str] = []
    for k in report.get("recall_k_values", DEFAULT_RECALL_K_VALUES):
        k_key = str(k)
        macro = macro_recall.get(k_key, {})
        micro = micro_recall.get(k_key, {})
        lines.append(
            f"dataset_macro_recall@{k} "
            f"law={macro.get('law', 0.0):.6f} "
            f"precedent={macro.get('precedent', 0.0):.6f} "
            f"integrated={macro.get('integrated', 0.0):.6f}"
        )
        lines.append(
            f"dataset_micro_recall@{k} "
            f"law={micro.get('law', 0.0):.6f} "
            f"law_hits={micro.get('law_hits', 0)} "
            f"law_total={micro.get('law_total', 0)} "
            f"precedent={micro.get('precedent', 0.0):.6f} "
            f"precedent_hits={micro.get('precedent_hits', 0)} "
            f"precedent_total={micro.get('precedent_total', 0)} "
            f"integrated={micro.get('integrated', 0.0):.6f} "
            f"hits={micro.get('hits', 0)} "
            f"total={micro.get('total', 0)}"
        )
    return lines


def format_stage_recall_console_lines(report: dict[str, Any]) -> list[str]:
    metrics = report.get("metrics", {})
    stage_recall = metrics.get("stage_recall_at_k", {})
    candidate_counts = metrics.get("candidate_counts", {})
    semantic_stage_recall = metrics.get("semantic_stage_recall_at_k", {})
    semantic_candidate_counts = metrics.get("semantic_candidate_counts_by_embed_col", {})
    lines: list[str] = []

    for stage in ("bm25", "semantic", "rerank"):
        if stage == "semantic" and isinstance(semantic_stage_recall, dict) and semantic_stage_recall:
            embed_cols = report.get("semantic_embed_cols")
            if not isinstance(embed_cols, list) or not embed_cols:
                embed_cols = list(semantic_stage_recall)
            for col in embed_cols:
                counts = semantic_candidate_counts.get(col, {})
                for k in (3, 5, 10):
                    values = semantic_stage_recall.get(col, {}).get(str(k), {})
                    lines.append(
                        f"stage_recall stage=semantic semantic_embed_col={col} recall@{k} "
                        f"law={values.get('law', 0.0):.6f} "
                        f"precedent={values.get('precedent', 0.0):.6f} "
                        f"integrated={values.get('integrated', 0.0):.6f} "
                        f"candidates={counts.get('total', 0)} "
                        f"law_candidates={counts.get('law', 0)} "
                        f"precedent_candidates={counts.get('precedent', 0)}"
                    )
                    lines.append(
                        f"semantic_retrieval_recall@{k} semantic_embed_col={col} "
                        f"law={values.get('law', 0.0):.6f} "
                        f"precedent={values.get('precedent', 0.0):.6f} "
                        f"integrated={values.get('integrated', 0.0):.6f} "
                        f"hits={values.get('hits', 0)} "
                        f"total={values.get('total', 0)}"
                    )
            continue

        counts = candidate_counts.get(stage, {})
        for k in (3, 5, 10):
            values = stage_recall.get(stage, {}).get(str(k), {})
            lines.append(
                f"stage_recall stage={stage} recall@{k} "
                f"law={values.get('law', 0.0):.6f} "
                f"precedent={values.get('precedent', 0.0):.6f} "
                f"integrated={values.get('integrated', 0.0):.6f} "
                f"candidates={counts.get('total', 0)} "
                f"law_candidates={counts.get('law', 0)} "
                f"precedent_candidates={counts.get('precedent', 0)}"
            )
            if stage == "rerank":
                lines.append(
                    f"rerank_recall@{k} "
                    f"law={values.get('law', 0.0):.6f} "
                    f"precedent={values.get('precedent', 0.0):.6f} "
                    f"integrated={values.get('integrated', 0.0):.6f} "
                    f"hits={values.get('hits', 0)} "
                    f"total={values.get('total', 0)}"
                )
    return lines


# ── 진입점 ────────────────────────────────────────────────────────────────

def run(
    input_path: Path,
    case_id: str | None = None,
    output_path: Path | None = None,
    semantic_embed_cols: str | list[str] | tuple[str, ...] | None = None,
    semantic_embed_col: str | None = None,
    case_workers: int = CASE_WORKERS,
    alpha_law: float = ALPHA_LAW,
    alpha_prec: float = ALPHA_PREC,
    qe_cache_path: Path | None = None,
) -> Path:
    assert_real_pipeline_imports_available()
    if qe_cache_path is not None:
        cache = load_qe_cache(qe_cache_path)
        expand_fn = make_cached_expand_fn(cache)
        log_progress(f"qe_cache_loaded path={qe_cache_path} entries={len(cache)}")
    else:
        expand_fn = expand_case_queries_with_llm
    dataset = load_dataset(input_path)
    log_progress(f"dataset_loaded input_path={input_path} cases={len(dataset)}")
    cases = select_cases(dataset, case_id)
    log_progress(f"dataset_selected cases={len(cases)} case_id={case_id or 'all'}")
    log_progress(f"alpha_law={alpha_law} alpha_prec={alpha_prec}")
    report = build_report(
        input_path=input_path,
        cases=cases,
        case_id=case_id,
        expand_fn=expand_fn,
        semantic_embed_cols=semantic_embed_cols,
        semantic_embed_col=semantic_embed_col,
        case_workers=case_workers,
        alpha_law=alpha_law,
        alpha_prec=alpha_prec,
    )
    report_path = save_report(report, output_path=output_path)
    log_progress(f"report_saved path={report_path}")
    return report_path


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    qe_cache_path = Path(args.qe_cache) if args.qe_cache else None

    try:
        report_path = run(
            input_path=input_path,
            case_id=args.case_id,
            output_path=output_path,
            semantic_embed_cols=args.semantic_embed_cols,
            case_workers=args.case_workers,
            alpha_law=args.alpha_law,
            alpha_prec=args.alpha_prec,
            qe_cache_path=qe_cache_path,
        )
    except PipelineImportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except DatasetValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"report_saved path={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())