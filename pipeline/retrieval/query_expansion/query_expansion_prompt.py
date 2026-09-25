from textwrap import dedent

from pipeline.retrieval.query_expansion.law_term_augment import prompt_hint_terms


SPECIFICITY_ANCHOR_GUIDE = dedent(
    """
    [구체 사실 보존]
    - expansion_query에 금액/기간/조건/행위주체를 그대로 유지한다.
    - 일반 법률 범주로 치환하지 않는다.
    """
).strip()


NO_FINAL_JUDGMENT_GUIDE = dedent(
    """
    [최종 판단 금지]
    - 무효/적법 같은 단정 결론 대신 '~가 쟁점이다'로 표현한다.
    """
).strip()


SYSTEM_PROMPT = dedent(
    f"""
    너는 임대차 특약을 retrieval 친화 질의로 변환하는 모델이다.
    출력은 ClauseQueryExpansion schema를 따르는 JSON 객체 하나만 작성한다.
    코드블록/설명문/주석은 출력하지 않는다.
    expansion_query는 2~3문장(300자 이내) 산문으로 작성한다.

    {SPECIFICITY_ANCHOR_GUIDE}
    {NO_FINAL_JUDGMENT_GUIDE}
    """
).strip()


LAW_STATUTE_LANGUAGE_GUIDE = dedent(
    """
    [법령 조문 언어 변환]
    - expansion_query와 keywords에 법령 원문에서 쓰는 용어를 사용한다.
    - 일상 언어 → 법령 조문 언어 변환 기준:
      · 임대료 → 차임
      · 3개월 연체 → 차임연체액이 2기의 차임액에 달하는
      · 계약 해지 통보 → 해지 통고, 해지 최고
      · 집 수리 → 필요비·유익비 상환, 수선의무
      · 이사 나가기 → 명도, 인도
      · 재계약 거부 → 갱신거절, 갱신요구권
      · 보증금 반환 거부 → 동시이행항변권, 보증금반환채무
    """
).strip()


LAW_PROMPT_APPEND = dedent(
    """
    [law / dense(expansion_query)]
    - 적용 법령 쟁점을 중심으로 작성한다.
    - 관련 법령명+조문(가능하면 조+항)을 1~3개 명시한다.

    [law / BM25(keywords)]
    - 법령 BM25 검색열은 `clause_key + parent_text + child_text`다.
    - keywords는 3~6개로 작성한다.
    - 최소 2개는 '법령명 제N조(제M항)' 형태로 작성한다.
    - 반드시 1개 이상은 clause_key 스타일(예: 주택임대차보호법_제6조_제1항)로 작성한다.
    - parent_text 표제어 또는 child_text 조문표현을 1개 이상 포함한다.

    [law few-shot]
    입력 특약: "임차인은 계약만료 2개월 전까지 갱신 여부를 통지하고, 임대인은 정당한 사유 없이 갱신을 거절할 수 없다."
    {
      "expansion_query": "임차인이 계약만료 2개월 전 갱신 의사를 통지하고 임대인의 갱신거절을 제한하는 특약으로, 갱신요구권과 갱신거절 사유의 적용 범위가 쟁점이다. 통지 시기와 갱신거절의 효력 판단에서 주택임대차보호법 제6조 제1항, 제6조의2, 제6조의3 해석이 문제된다.",
      "keywords": ["주택임대차보호법 제6조 제1항", "주택임대차보호법 제6조의2", "주택임대차보호법_제6조_제1항", "갱신거절", "갱신요구권"]
    }
    """
).strip()


PRECEDENT_PROMPT_APPEND = dedent(
    """
    [precedent / dense(expansion_query)]
    - 당사자 갈등, 청구 내용, 책임 포인트 중심으로 작성한다.
    - 법령 단서는 1~2개만 포함한다.

    [precedent / BM25(keywords)]
    - 판례 BM25 검색열은 `issue + judgment_summary`다.
    - keywords는 3~5개로 작성한다.
    - 조문 나열보다 사실관계+쟁점 복합 명사구를 우선한다.
    - 법령명+조문 키워드는 최소 1개만 포함한다.
    - clause_key 스타일 키워드는 사용하지 않는다.
    - 아래는 판례 검색에 쓸모없는 단독 일반어이므로 keywords에 포함하지 않는다:
      강행규정, 임차인 불리 약정, 특약 효력, 임차인 보호, 채무불이행,
      손해배상 청구, 보증금 반환, 임대차 종료, 주택임대차보호법 (단독)
    - 이 특약 상황에 특화된 복합 명사구를 만든다:
      나쁜 예: "강행규정 위반", "특약 효력", "보증금 반환"
      좋은 예: "전입신고 대항력 존속", "실거주 목적 갱신거절 정당성", "중개보수 초과 수령 반환"

    [precedent few-shot]
    입력 특약: "임차인이 주택 인도와 전입신고를 마친 이후에도 임대인은 제3자에게 대항할 수 없다고 주장한다."
    {
      "expansion_query": "임차인이 주택의 인도와 전입신고를 마친 뒤 임대인이 대항력을 부정하는 분쟁으로, 대항요건 충족 시점과 대항력 존속 여부가 쟁점이다. 임차인의 주민등록 유지 여부와 제3자 대항 가능성이 판례상 판단요소가 된다.",
      "keywords": ["인도와 주민등록 대항요건", "전입신고 대항력 존속", "제3자 대항 가능 여부", "주택임대차보호법 제3조 제1항"]
    }
    """
).strip()


def build_user_prompt(clause_text: str, extra_instructions: str | None = None) -> str:
    prompt = dedent(
        f"""
        다음 특약 조항을 ClauseQueryExpansion schema 형식으로 변환하라.

        입력 특약:
        {clause_text}

        작성 지시:
        - 출력은 순수 JSON 객체 하나만 생성한다.
        - expansion_query는 2~3문장(300자 이내)으로 작성한다.
        - keywords는 3~7개의 구체 명사구로 작성한다.
        """
    ).strip()
    if extra_instructions:
        prompt = f"{prompt}\n\n{extra_instructions.strip()}"
    return prompt


def _hint_block(clause_text: str) -> str:
    terms = prompt_hint_terms(clause_text)
    if not terms:
        return ""
    joined = ", ".join(terms)
    return dedent(
        f"""
        [참고 법령용어 - 이 특약에서 발견된 실제 법령 표제어/용어]
        - 관련되면 legal_issues/keywords/expansion_query에 우선 사용한다. 관련 없으면 무시한다.
        - 용어: {joined}
        """
    ).strip()


def _menu_block(issue_menu):
    """사전에서 특약과 매칭된 제도명 메뉴. LLM이 여기서 우선 고르게 한다."""
    if not issue_menu:
        return ""
    joined = ", ".join(issue_menu)
    return dedent(
        f"""
        [적용 가능 제도명 후보 - 이 특약 유형에서 흔한 법제도]
        - 아래 후보 중 이 특약에 맞는 것을 legal_issues에 우선 사용한다.
        - 후보가 특약과 안 맞으면 무시하고, 더 정확한 제도명을 직접 써도 된다.
        - 후보: {joined}
        """
    ).strip()


# 명명(命名) 강제: 서술이 아니라 법제도 이름을 legal_issues에 찍게 한다.
LAW_NAMING_APPEND = dedent(
    """
    [핵심 지시 - 법제도 명명(命名)]
    - 이 특약에 적용되는 '법적 제도/법리'를 그 표준 명칭으로 legal_issues에 반드시 명시한다.
    - 상황을 풀어 서술하는 데 그치지 말고, 그 상황이 '무슨 제도'인지 이름을 찍는다.
      예) "대출이 안 되면 계약 무효" -> legal_issues=["정지조건","조건부 법률행위"]
          "중도 해지 시 비용 부담"   -> legal_issues=["신의성실의 원칙","손해배상액의 예정"]
          "수리비를 임차인이 부담"     -> legal_issues=["필요비상환청구권","유익비상환청구권"]
    - 유/무효 등 최종 법률결론은 유보한다. 단, 적용 '제도의 명칭'은 결론이 아니므로 반드시 쓴다.

    [law / dense(expansion_query)]
    - 적용 제도/법리를 그 표준 명칭과 함께 산문 2~3문장으로 서술한다.
    - 일상 표현은 변환표에 따라 법령 원문 용어로 바꾼다.

    [law / BM25(keywords)]
    - legal_issues의 제도명을 포함해, 법령 원문 표제어 중심 구체 명사구 3~7개.
    - 조문 번호(제N조)/clause_key 형태는 쓰지 않는다. 표제어/법률용어로만 작성한다.
    """
).strip()


def build_user_prompt_law(
    clause_text: str,
    issue_menu=None,
    extra_instructions: str | None = None,
) -> str:
    hint = _hint_block(clause_text)
    menu = _menu_block(issue_menu)
    tail = "".join(f"\n\n{b}" for b in (menu, hint) if b)
    prompt = dedent(
        f"""
        다음 임대차 계약서 특약 조항을 법령 검색에 최적화된 ClauseQueryExpansion으로 변환하라.

        입력 특약:
        {clause_text}

        작성 지시:
        - 출력은 순수 JSON 객체 하나만 생성한다(expansion_query, legal_issues, keywords).
        - legal_issues는 이 특약에 적용되는 법제도의 '표준 명칭'으로 1개 이상 반드시 채운다.
        - expansion_query에는 입력 특약의 구체 사실(금액/날짜/조건/주체)을 포함한다.

        {LAW_STATUTE_LANGUAGE_GUIDE}

        {LAW_NAMING_APPEND}{tail}
        """
    ).strip()
    if extra_instructions:
        prompt = f"{prompt}\n\n{extra_instructions.strip()}"
    return prompt


def build_user_prompt_precedent(
    clause_text: str,
    extra_instructions: str | None = None,
) -> str:
    prompt = dedent(
        f"""
        다음 임대차 계약서 특약 조항을 판례 검색에 최적화된 ClauseQueryExpansion으로 변환하라.

        입력 특약:
        {clause_text}

        작성 지시:
        - 출력은 순수 JSON 객체 하나만 생성한다.
        - expansion_query는 섹션 라벨 없이 산문 2~3문장으로 작성한다(300자 이내).
        - 입력 특약의 구체 사실(금액·날짜·조건·행위 주체)을 반드시 포함한다.
        - 최종 법률 판단을 하지 말고 판례 쟁점만 서술한다.

        {PRECEDENT_PROMPT_APPEND}
        """
    ).strip()
    if extra_instructions:
        prompt = f"{prompt}\n\n{extra_instructions.strip()}"
    return prompt