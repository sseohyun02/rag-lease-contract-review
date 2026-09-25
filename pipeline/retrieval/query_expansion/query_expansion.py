"""임대차 특약 query expansion 실행 로직.

expand_clause: 명명(legal_issues) 강제 + 사전 메뉴 grounding + 모델 지정 + 디스크 캐시.
- legal_issues(제도 표준명칭)를 dense query 앞 + BM25 keywords 앞에 주입 → 특약↔조문
  어휘 0겹침(neither)을 도달 가능으로 바꾸는 지렛대.
- 사전(legal_issue_dict.json): 특약 cue와 겹치는 제도명 메뉴를 프롬프트에 제공.
  · bm25(하드신호): 사전 등재 제도명만 keywords로 → 환각 오염 최소화.
  · dense(소프트신호): LLM 자유 추론도 허용 → recall 상방.
- 모델: QE_MODEL 환경변수로 교체(기본 gemini-2.5-pro). 전역 settings는 안 건드림.
- 캐시: qe_cache.json. 키에 모델명 포함 → 모델 바꾸면 캐시도 분리.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pipeline.retrieval.query_expansion.query_expansion_prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_user_prompt_law,
)
from pipeline.retrieval.query_expansion.query_expansion_schema import ClauseQueryExpansion
from shared.llm.gemini_client import LLMError, GeminiClient, gemini_client


QE_VERSION = "v3"

# QE 전용 모델(환경변수로 교체). 전역 settings.gemini_model은 건드리지 않는다.
# 3.x Flash ID 확인되면 set QE_MODEL=... 로 교체.
QE_MODEL = os.environ.get("QE_MODEL", "gemini-2.5-pro")

# 디스크 캐시 경로(환경변수로 교체 가능)
_DEFAULT_CACHE = Path("evaluation/qe_cache/qe_cache.json")
_CACHE_PATH = Path(os.environ.get("QE_CACHE_PATH", str(_DEFAULT_CACHE)))
_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None

# 사전 파일 위치(환경변수로 교체 가능). 없으면 사전 없이 동작.
_DICT_PATH = Path(
    os.environ.get(
        "QE_DICT_PATH",
        str(Path(__file__).with_name("legal_issue_dict.json")),
    )
)

_MAX_ISSUES_IN_KEYWORDS = 3  # keywords 상한(7) 고려, 제도명은 최대 이만큼만 하드주입


class QueryExpansionError(RuntimeError):
    """Query expansion 생성 또는 검증 실패."""


# ─────────────────────────────────────────────────────────────
# 생성/파싱 헬퍼
# ─────────────────────────────────────────────────────────────
def _is_schema_state_limit_error(error: LLMError) -> bool:
    """Vertex structured output 스키마 상태 수 제한 오류인지 판별한다."""
    message = str(error).lower()
    return (
        "invalid_argument" in message
        and (
            "too many states for serving" in message
            or "specified schema produces a constraint" in message
        )
    )


def _generate_with_schema_fallback(llm: GeminiClient, prompt: str) -> Any:
    """structured output을 사용하고, 스키마 한도 오류 시 text 모드로 폴백. QE_MODEL 사용."""
    try:
        return llm.generate(
            contents=prompt,
            model=QE_MODEL,
            system_instruction=SYSTEM_PROMPT,
            response_schema=ClauseQueryExpansion,
        )
    except LLMError as error:
        if not _is_schema_state_limit_error(error):
            raise
        return llm.generate(
            contents=prompt,
            model=QE_MODEL,
            system_instruction=SYSTEM_PROMPT,
            response_schema=None,
        )


def _parse_expansion_result(result: Any) -> ClauseQueryExpansion:
    """Gemini wrapper 반환값을 ClauseQueryExpansion으로 정규화한다."""
    if isinstance(result, ClauseQueryExpansion):
        return result
    if isinstance(result, dict):
        return ClauseQueryExpansion.model_validate(result)
    if isinstance(result, str):
        return ClauseQueryExpansion.model_validate_json(_extract_json_object(result))
    raise QueryExpansionError(
        f"지원하지 않는 Gemini 응답 타입입니다: {type(result).__name__}"
    )


def _strip_json_code_fence(text: str) -> str:
    """```json ... ``` 형태 응답을 순수 JSON 텍스트로 정리한다."""
    stripped = text.strip()
    fence_match = re.fullmatch(
        r"```(?:json|JSON)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL,
    )
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


def _extract_json_object(text: str) -> str:
    """응답에 설명문이 섞여 있어도 가장 바깥 JSON object를 추출한다."""
    stripped = _strip_json_code_fence(text)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return stripped
    return stripped[start : end + 1].strip()


def build_repair_prompt(
    *,
    clause_text: str,
    invalid_output: Any,
    error: Exception,
    extra_instructions: str | None = None,
) -> str:
    """스키마 검증 실패 시 재시도용 repair prompt를 만든다."""
    base = f"""
이전 응답은 ClauseQueryExpansion Pydantic schema 검증에 실패했다.

입력 특약:
{clause_text}

이전 응답:
{invalid_output}

검증 오류:
{repr(error)}

수정 지시:
- 순수 JSON 객체 하나만 다시 출력하라.
- markdown 코드블록이나 설명문을 붙이지 말라.
- schema에 없는 필드는 제거하라.
- 누락된 필수 필드는 채워라.
- expansion_query는 섹션 라벨 없이 산문 2~3문장으로 작성하라(300자 이내).
- legal_issues는 이 특약에 적용되는 법제도의 표준 명칭으로 1개 이상 채워라.
- keywords는 문자열 배열로 작성하라(최소 3개, 최대 7개).
- 출력 필드는 expansion_query, legal_issues, keywords만 허용한다.
""".strip()
    if extra_instructions:
        base += f"\n\n{extra_instructions.strip()}"
    return base


# ─────────────────────────────────────────────────────────────
# 사전(legal_issue_dict) 로드
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_issue_dict() -> list[dict]:
    """legal_issue_dict.json의 issues 리스트. 없으면 빈 리스트."""
    if not _DICT_PATH.exists():
        return []
    try:
        data = json.loads(_DICT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data.get("issues", []) or []


@lru_cache(maxsize=1)
def _all_dict_terms() -> frozenset[str]:
    """사전에 등재된 모든 제도명(하드주입 허용 화이트리스트)."""
    terms: set[str] = set()
    for item in load_issue_dict():
        for t in item.get("legal_terms", []):
            terms.add(t)
    return frozenset(terms)


def dict_menu_for_clause(clause_text: str, limit_groups: int = 6) -> list[str]:
    """특약 cue와 겹치는 사전 그룹의 제도명을 프롬프트 메뉴로 추린다.

    cue가 특약에 등장하는 그룹만 골라 그 제도명을 모은다.
    아무 cue도 안 맞으면 빈 리스트(그땐 LLM 자유 생성).
    """
    picked: list[str] = []
    seen: set[str] = set()
    hit_groups = 0
    for item in load_issue_dict():
        cues = item.get("clause_cues", [])
        if any(c and c in clause_text for c in cues):
            hit_groups += 1
            for t in item.get("legal_terms", []):
                if t not in seen:
                    picked.append(t)
                    seen.add(t)
            if hit_groups >= limit_groups:
                break
    return picked


# ─────────────────────────────────────────────────────────────
# 디스크 캐시
# ─────────────────────────────────────────────────────────────
def _cache_key(clause_text: str) -> str:
    """버전 + 모델명까지 키에 포함 → 모델·버전별 캐시 분리."""
    raw = f"{QE_VERSION}||{QE_MODEL}||{clause_text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_cache() -> dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            if _CACHE_PATH.exists():
                try:
                    _CACHE = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    _CACHE = {}
            else:
                _CACHE = {}
    return _CACHE


def _save_cache() -> None:
    with _CACHE_LOCK:
        if _CACHE is None:
            return
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_CACHE, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(_CACHE_PATH)


# ─────────────────────────────────────────────────────────────
# legal_issues 주입 → 하위 파이프라인용 payload 변환
# ─────────────────────────────────────────────────────────────
def _to_payload_expansion(exp: ClauseQueryExpansion) -> ClauseQueryExpansion:
    """legal_issues를 dense query 앞 + bm25 keywords 앞에 주입해 변환.

    - dense: expansion_query 앞에 제도명 전체를 붙여 소프트신호 주입(자유 추론 포함).
    - bm25 : 제도명 중 '사전 등재분만' keywords 앞에 합침(하드신호 오염 최소화).
             사전이 비어있으면 제도명 전부 허용(폴백).
    """
    issues = list(exp.legal_issues or [])

    issue_prefix = " ".join(issues)
    dense_query = f"{issue_prefix}. {exp.expansion_query}" if issue_prefix else exp.expansion_query
    dense_query = dense_query[:300]

    whitelist = _all_dict_terms()
    hard_issues = [t for t in issues if t in whitelist] if whitelist else issues

    merged: list[str] = []
    seen: set[str] = set()
    for t in hard_issues[:_MAX_ISSUES_IN_KEYWORDS] + list(exp.keywords):
        if t and t not in seen:
            merged.append(t)
            seen.add(t)
    merged = merged[:7]

    return ClauseQueryExpansion(
        expansion_query=dense_query,
        legal_issues=issues,
        keywords=merged,
    )


# ─────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────
def expand_clause(
    clause_text: str,
    *,
    client: GeminiClient | None = None,
    max_retries: int = 1,
    use_cache: bool = True,
) -> ClauseQueryExpansion:
    """특약 문장을 ClauseQueryExpansion으로 변환한다(명명 강제 + 사전 메뉴 + 캐시).

    주의: 과거 overfit_mode로 정답(gt_laws)을 프롬프트에 주입하던 기능은
    데이터 누수였음이 확인되어 완전히 제거되었다. 여기서도 정답셋을 참조하지 않는다.
    """
    key = _cache_key(clause_text)

    if use_cache:
        cached = _load_cache().get(key)
        if cached is not None:
            return ClauseQueryExpansion.model_validate(cached)

    llm = client or gemini_client
    menu = dict_menu_for_clause(clause_text)
    user_prompt = build_user_prompt_law(clause_text, issue_menu=menu)

    last_error: Exception | None = None
    last_output: Any = None
    parsed: ClauseQueryExpansion | None = None

    for attempt in range(max_retries + 1):
        if attempt == 0:
            prompt = user_prompt
        else:
            assert last_error is not None
            prompt = build_repair_prompt(
                clause_text=clause_text,
                invalid_output=last_output,
                error=last_error,
            )
        result: Any = None
        try:
            result = _generate_with_schema_fallback(llm, prompt)
            parsed = _parse_expansion_result(result)
            break
        except (ValidationError, QueryExpansionError, LLMError) as error:
            last_error = error
            last_output = result

    if parsed is None:
        assert last_error is not None
        raise QueryExpansionError(
            f"query expansion 생성/검증 실패 (retries={max_retries}): {last_error}"
        ) from last_error

    payload = _to_payload_expansion(parsed)

    if use_cache:
        _load_cache()[key] = payload.model_dump()
        _save_cache()

    return payload
