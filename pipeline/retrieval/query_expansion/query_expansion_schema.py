"""QE 스키마.

ClauseQueryExpansion: expansion_query + legal_issues + keywords.
- legal_issues: 이 특약에 적용되는 '법적 제도·법리의 표준 명칭' 리스트.
  예) "대출 안 나오면 무효" → ["정지조건", "조건부 법률행위"]
  이 명칭이 곧 조문 표제어라, BM25/Dense에 주입하면 특약↔조문 어휘 0겹침을
  도달 가능으로 바꾸는 지렛대가 된다. 명명을 필수로 요구한다.
"""
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _clean_str_list(values: List[str]) -> List[str]:
    """문자열 리스트를 trim/중복제거하여 정리한다."""
    cleaned: List[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        cleaned.append(normalized)
        seen.add(normalized)
    return cleaned


class ClauseQueryExpansion(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    expansion_query: str = Field(
        ...,
        min_length=40,
        max_length=300,
        description=(
            "Dense/Semantic 검색용 산문 텍스트(300자 이내). "
            "특약의 구체 사실(금액·날짜·조건·행위 주체)을 보존하면서 "
            "적용되는 법적 제도·법리를 그 표준 명칭과 함께 2~3문장으로 서술한다. "
            "유·무효 최종결론은 유보하되 제도 명칭은 반드시 포함한다."
        ),
    )
    legal_issues: List[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "이 특약에 적용되는 법적 제도·법리의 '표준 명칭' 1~5개. "
            "조문 표제어 수준의 명사(예: 정지조건, 손해배상액의 예정, 신의성실, "
            "필요비상환청구권, 동시이행항변권). 일상 서술이 아니라 제도명으로 작성한다."
        ),
    )
    keywords: List[str] = Field(
        ...,
        min_length=3,
        max_length=7,
        description=(
            "BM25 검색용 구체 명사구 3~7개. 법령 원문 표제어·법률용어 중심. "
            "조문 번호(제N조)나 clause_key 형태는 쓰지 않는다."
        ),
    )

    @field_validator("keywords")
    @classmethod
    def validate_keywords(cls, values: List[str]) -> List[str]:
        cleaned = _clean_str_list(values)
        if len(cleaned) < 3:
            raise ValueError("keywords는 중복 제거 후 최소 3개 이상이어야 합니다.")
        return cleaned

    @field_validator("legal_issues")
    @classmethod
    def clean_legal_issues(cls, values: List[str]) -> List[str]:
        return _clean_str_list(values)
