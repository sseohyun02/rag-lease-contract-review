"""QE v2 — law_term_by_keyword.jsonl 기반 법령용어 사전 · stopword 자동추출 · 키워드 보강.

역할
----
1) 법령용어 사전 로드: law_term_by_keyword.jsonl(법령명 → 용어목록)을 읽는다.
2) stopword 자동추출: 여러 법령에 공통 등장하는 초고빈도 용어(계약·주택 등)를
   자동으로 골라 검색 변별력이 없는 어휘로 제외한다(수동 지정 아님).
3) 프롬프트 힌트(i): 특약과 겹치는 법령용어를 소수 추려 프롬프트에 주입할 후보로 제공.
4) 키워드 사후보강(ii): LLM keywords에 특약↔법령용어 매칭분을 얹어 BM25 표면어 겹침을 올린다.

주의: (i)(ii) 모두 '표면어 겹침'을 늘릴 뿐, 법리 추론갭(대출무산→정지조건)은
못 메운다. 그건 프롬프트의 변환표 + LLM 추론에 기댄다.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

# law_term_by_keyword.jsonl 위치(이 파일과 같은 폴더)
_TERM_JSONL = Path(__file__).with_name("law_term_by_keyword.jsonl")

# stopword 자동추출 기준: 이 개수 이상의 법령에 공통 등장하면 변별력 없음으로 제외
_STOPWORD_MIN_LAWS = 5
# 사후보강으로 얹을 수 있는 최대 키워드 수(스키마 keywords max=7 고려)
_MAX_TOTAL_KEYWORDS = 7
# 매칭에 쓸 최소 용어 길이(1글자 용어는 노이즈라 제외)
_MIN_TERM_LEN = 2


@lru_cache(maxsize=1)
def load_law_terms() -> dict[str, list[str]]:
    """법령명 → 용어목록 dict. 파일이 없으면 빈 dict."""
    if not _TERM_JSONL.exists():
        return {}
    out: dict[str, list[str]] = {}
    for line in _TERM_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        name = rec.get("법령명", "")
        terms = rec.get("용어목록", []) or []
        if name:
            out[name] = [t for t in terms if isinstance(t, str)]
    return out


@lru_cache(maxsize=1)
def auto_stopwords() -> frozenset[str]:
    """여러 법령에 공통 등장하는 초고빈도 용어를 stopword로 자동추출.

    _STOPWORD_MIN_LAWS개 이상의 법령에 나타나는 용어는 어느 특약에도 붙어
    검색 변별력이 없으므로 제외한다(계약·주택·통지 등).
    """
    laws = load_law_terms()
    doc_freq: Counter = Counter()
    for terms in laws.values():
        for t in set(terms):
            doc_freq[t] += 1
    return frozenset(t for t, n in doc_freq.items() if n >= _STOPWORD_MIN_LAWS)


@lru_cache(maxsize=1)
def _all_terms_filtered() -> frozenset[str]:
    """stopword·단문자 제외한 전체 법령용어 풀(매칭 대상)."""
    stop = auto_stopwords()
    pool: set[str] = set()
    for terms in load_law_terms().values():
        for t in terms:
            if len(t) >= _MIN_TERM_LEN and t not in stop:
                pool.add(t)
    return frozenset(pool)


def _clause_hits(clause_text: str) -> list[str]:
    """특약 텍스트에 '부분문자열로 등장'하는 법령용어를 길이 내림차순으로 반환.

    형태소 분석 없이도 법령용어는 대개 복합명사라 부분문자열 매칭으로 잡힌다.
    긴 용어(더 구체적)를 우선한다. stopword·단문자는 이미 풀에서 제외됨.
    """
    pool = _all_terms_filtered()
    hits = [t for t in pool if t in clause_text]
    # 긴 용어 우선(구체성↑), 같은 길이는 사전순
    hits.sort(key=lambda t: (-len(t), t))
    return hits


def prompt_hint_terms(clause_text: str, limit: int = 12) -> list[str]:
    """(i) 프롬프트 힌트용: 특약과 겹치는 법령용어 상위 N개.

    전체 용어를 다 넣으면 프롬프트가 폭발하므로, 특약에 실제 등장하는
    용어만 소수 추려 '이 어휘를 우선 사용하라'는 힌트로 쓴다.
    """
    return _clause_hits(clause_text)[:limit]


def augment_keywords(clause_text: str, llm_keywords: list[str], limit: int = 3) -> list[str]:
    """(ii) 사후보강: LLM keywords에 특약↔법령용어 매칭분을 얹는다.

    - LLM이 이미 낸 키워드와 중복되지 않는 법령용어만 추가.
    - 전체 키워드 수가 스키마 상한(_MAX_TOTAL_KEYWORDS)을 넘지 않게 캡.
    - 최대 limit개까지만 보강(LLM 키워드를 압도하지 않도록).
    반환: 보강된 최종 키워드 리스트.
    """
    base = list(llm_keywords)
    existing = set(base)
    room = min(limit, _MAX_TOTAL_KEYWORDS - len(base))
    if room <= 0:
        return base
    added = 0
    for t in _clause_hits(clause_text):
        if added >= room:
            break
        # 이미 있는 키워드에 부분포함되면 스킵(중복 정보)
        if t in existing or any(t in k or k in t for k in existing):
            continue
        base.append(t)
        existing.add(t)
        added += 1
    return base
