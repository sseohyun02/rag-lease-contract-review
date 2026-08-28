"""실험 전용 QE(쿼리 확장) 파일 캐시.

목적
    그리드서치처럼 같은 특약에 대해 QE를 반복 실행할 때, 매번 LLM을 호출하면
    토큰이 계속 소모된다. 특약 텍스트를 키로 QE 결과를 파일(JSON)에 저장해두고,
    다음부터는 파일에서 읽어 LLM 호출 없이 재사용한다.

원칙 (중요)
    - 서비스 파이프라인 코드(query_expansion.py 등)는 절대 수정하지 않는다.
    - 이 모듈은 '실험 스크립트'에서만 import 해서 expand_clause 대신 쓴다.
    - 캐시는 clause 텍스트 -> QE 결과(dict) 매핑. clause 가 같으면 동일 결과 재사용.
    - QE 로직/프롬프트가 바뀌면 캐시를 무효화해야 하므로, 캐시 파일에 버전 태그를 둔다.
      (프롬프트를 바꾼 뒤엔 --rebuild 로 다시 만들면 된다.)

사용 (실험 스크립트 안에서)
    from data.processors.qe_cache import CachedExpander
    expander = CachedExpander(cache_path="evaluation/_qe_cache/qe_cache.json")
    payload = build_retrieval_payload(expander.expand(clause), clause_text=clause)
    ...
    expander.flush()   # 끝나고 저장 (또는 auto_flush=True)

CLI (미리 전체 특약을 캐시에 채워두고 싶을 때)
    python data/processors/qe_cache.py --eval evaluation/eval_set_tune.json evaluation/eval_set_test.json
    python data/processors/qe_cache.py --eval ... --rebuild   # 기존 캐시 무시하고 새로
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.retrieval.query_expansion.query_expansion import expand_clause  # noqa: E402
from pipeline.retrieval.query_expansion.query_expansion_schema import (  # noqa: E402
    ClauseQueryExpansion,
)

# QE 프롬프트/스키마가 바뀌면 이 값을 올려 캐시를 무효화한다.
CACHE_VERSION = "v1"


def _key(clause: str) -> str:
    h = hashlib.sha256(clause.strip().encode("utf-8")).hexdigest()[:16]
    return f"{CACHE_VERSION}:{h}"


def _to_dict(result: Any) -> dict:
    """expand_clause 반환(ClauseQueryExpansion 또는 dict/str)을 순수 dict로."""
    if isinstance(result, ClauseQueryExpansion):
        return result.model_dump()
    if isinstance(result, dict):
        return result
    # 문자열이면 파싱 시도
    return ClauseQueryExpansion.model_validate_json(str(result)).model_dump()


class CachedExpander:
    """expand_clause 를 파일 캐시로 감싼 래퍼 (실험 전용)."""

    def __init__(self, cache_path: str | Path,
                 auto_flush: bool = True, rebuild: bool = False):
        self.path = Path(cache_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.auto_flush = auto_flush
        self._dirty = False
        self._hits = 0
        self._misses = 0
        if self.path.exists() and not rebuild:
            self._cache: dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._cache = {}

    def expand(self, clause: str) -> ClauseQueryExpansion:
        """캐시에 있으면 재사용, 없으면 LLM 호출 후 저장. 반환 타입은 원본과 동일."""
        k = _key(clause)
        if k in self._cache:
            self._hits += 1
            return ClauseQueryExpansion.model_validate(self._cache[k])
        # 미스 → 실제 QE 호출 (여기서만 토큰 소모)
        self._misses += 1
        result = expand_clause(clause)
        self._cache[k] = _to_dict(result)
        self._dirty = True
        if self.auto_flush:
            self.flush()
        return ClauseQueryExpansion.model_validate(self._cache[k])

    def flush(self):
        if self._dirty:
            self.path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._dirty = False

    @property
    def stats(self) -> str:
        return f"QE 캐시 hit={self._hits} miss={self._misses} (총 {len(self._cache)}개 저장)"


def _prefill(eval_paths: list[str], cache_path: str, rebuild: bool):
    """eval_set 들의 모든 특약을 미리 캐시에 채운다."""
    exp = CachedExpander(cache_path, auto_flush=False, rebuild=rebuild)
    clauses: list[str] = []
    for p in eval_paths:
        for c in json.loads(Path(p).read_text(encoding="utf-8")):
            cl = c["clauses"][0]["normalized"] if c.get("clauses") else c.get("clause", "")
            if cl:
                clauses.append(cl)
    print(f"특약 {len(clauses)}개 캐시 채우는 중...")
    for i, cl in enumerate(clauses, 1):
        exp.expand(cl)
        if i % 10 == 0:
            print(f"  {i}/{len(clauses)}  {exp.stats}", flush=True)
    exp.flush()
    print(f"완료. {exp.stats}\n저장: {cache_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--cache-path", default="evaluation/_qe_cache/qe_cache.json")
    ap.add_argument("--rebuild", action="store_true", help="기존 캐시 무시하고 새로 생성")
    a = ap.parse_args()
    _prefill(a.eval, a.cache_path, a.rebuild)


if __name__ == "__main__":
    main()
