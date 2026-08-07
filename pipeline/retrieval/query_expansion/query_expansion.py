"""임대차 특약 query expansion 실행 로직."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import ValidationError

from pipeline.retrieval.query_expansion.query_expansion_prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from pipeline.retrieval.query_expansion.query_expansion_schema import ClauseQueryExpansion
from shared.llm.gemini_client import LLMError, GeminiClient, gemini_client


class QueryExpansionError(RuntimeError):
    """Query expansion 생성 또는 검증 실패."""


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
    """가능하면 structured output을 사용하고, 스키마 한도 오류 시 text 모드로 폴백."""
    try:
        return llm.generate(
            contents=prompt,
            system_instruction=SYSTEM_PROMPT,
            response_schema=ClauseQueryExpansion,
        )
    except LLMError as error:
        if not _is_schema_state_limit_error(error):
            raise

        return llm.generate(
            contents=prompt,
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
    import re

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
- expansion_query에 입력 특약의 구체 사실(금액·날짜·조건·행위 주체)을 반드시 포함하라.
- expansion_query 안에 JSON 키·중첩 JSON 형태를 넣지 말라.
- keywords는 문자열 배열로 작성하라(최소 3개, 최대 6개).
- keywords에 주택임대차보호법, 강행규정 같은 공통 배경 용어는 쓰지 않는다.
- 출력 필드는 expansion_query, keywords만 허용한다.
""".strip()
    if extra_instructions:
        base += f"\n\n{extra_instructions.strip()}"
    return base


@lru_cache(maxsize=1024)
def expand_clause(
    clause_text: str,
    *,
    client: GeminiClient | None = None,
    max_retries: int = 1,
    user_prompt: str | None = None,
) -> ClauseQueryExpansion:
    """특약 문장을 ClauseQueryExpansion으로 변환한다.

    주의: 이전에는 overfit_mode/overfit_target 인자로 evaluation/eval_set.json의
    정답(gt_laws/gt_cases)을 프롬프트에 힌트로 주입하는 기능이 있었으나,
    평가 결과를 부당하게 부풀리는 데이터 누수였음이 확인되어 완전히 제거했다.
    """
    llm = client or gemini_client

    if user_prompt is None:
        user_prompt = build_user_prompt(clause_text)
    last_error: Exception | None = None
    last_output: Any = None

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
            return _parse_expansion_result(result)
        except (ValidationError, QueryExpansionError, LLMError) as error:
            last_error = error
            last_output = result

    assert last_error is not None
    raise QueryExpansionError(
        f"query expansion 생성/검증 실패 (retries={max_retries}): {last_error}"
    ) from last_error