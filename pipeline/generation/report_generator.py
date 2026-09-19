import json
import os
import re
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel, field_validator
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")
vertexai.init(project=PROJECT_ID, location=LOCATION)

MODEL_NAME_FLASH = "gemini-2.5-flash"
MODEL_NAME_FLASH_LITE = "gemini-2.5-flash-lite"
COMMON_TERMS_COUNT = 6

# ── Pydantic 스키마 ──────────────────────────────────────────────

class LawType(str, Enum):
    law  = "법령"
    case = "판례"

class SelectedLaw(BaseModel):
    ref: str | None = None
    summary: str | None = None
    is_violation: bool = False
    is_caution: bool = False

    @field_validator("is_violation", "is_caution", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v) if v is not None else False

class ClauseSummaryOutput(BaseModel):
    clause_one_line_summary: str = ""
    clause_interpretation: str = ""
    selected_laws: list[SelectedLaw] = []

    @field_validator("clause_one_line_summary", "clause_interpretation", mode="before")
    @classmethod
    def coerce_str(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            return "\n".join(f"• {item.lstrip('•- ').strip()}" for item in v if item)
        return str(v)

    @field_validator("selected_laws", mode="before")
    @classmethod
    def coerce_list(cls, v):
        if v is None:
            return []
        return v

class RelatedClause(BaseModel):
    clause_id: str | None = None
    clause_text: str | None = None
    relation: str | None = None

class ChecklistItem(BaseModel):
    item: str | None = None
    description: str | None = None
    basis: list[str] = []

    @field_validator("basis", mode="before")
    @classmethod
    def coerce_basis(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

class FinalReportOutput(BaseModel):
    contract_checklist: list[ChecklistItem] = []
    related_clauses_map: dict[str, list[RelatedClause]] = {} # clause_id -> list of relations

class LLMRelatedClauseResult(BaseModel):
    clause_id: str
    related_clauses: list[RelatedClause] = []

    @field_validator("related_clauses", mode="after")
    @classmethod
    def split_combined_clause_ids(cls, v):
        result = []
        for item in v:
            c_id = item.clause_id or ""
            ids = [x.strip() for x in c_id.split(",") if x.strip()]
            if len(ids) > 1:
                for cid in ids:
                    result.append(RelatedClause(
                        clause_id=cid,
                        clause_text=None,
                        relation=item.relation,
                    ))
            else:
                result.append(item)
        return result

class FinalReportLLMOutput(BaseModel):
    contract_checklist: list[ChecklistItem] = []
    clause_relations: list[LLMRelatedClauseResult] = []

# ── 프롬프트 ──────────────────────────────────────────

SYSTEM_PROMPT = """[역할 및 지시]
당신은 주택임대차 계약 검토를 도와줄 법률 전문가입니다.
아래 정보를 바탕으로 임차인이 계약 전 스스로 확인해야 할 사항을 도출하세요.

- 특약 문구의 해석 가능성, 법령과의 관계, 분쟁으로 이어질 수 있는 사실관계를 객관적으로 서술하세요.
- 명백히 법령에 위반되는 경우에는 해당 법령 조문을 근거로 위반 사실을 서술합니다.
- "위험", "유리", "불리" 등의 평가적 표현은 절대 사용하지 마세요."""

def format_laws(matches: list) -> str:
    lines = []
    for m in matches:
        text = m.get("content") or m.get("doc_text") or m.get("summary") or ""
        if not text:
            continue
        doc_id = m.get("doc_id", "")
        lines.append(f"[{doc_id}]\n{text}")
    return "\n\n".join(lines) if lines else "해당 없음"

def format_precs(matches: list) -> str:
    lines = []
    for m in matches:
        text = m.get("content") or m.get("summary") or ""
        if not text or text == "nan":
            continue
        doc_id = m.get("doc_id", "")
        lines.append(f"[{doc_id}]\n{text}")
    return "\n\n".join(lines) if lines else "해당 없음"

def format_property_info(property_info: dict) -> str:
    return json.dumps(property_info, ensure_ascii=False, indent=2)

def build_clause_summary_prompt(
    target_clause: str,
    laws: list,
    precs: list,
) -> str:
    all_doc_ids = [m.get("doc_id", "") for m in laws + precs if m.get("doc_id")]
    output_format = json.dumps(
        {
            "clause_one_line_summary": "이 특약은 ... 한다는 내용입니다. (20자 내외 한 문장)",
            "clause_interpretation": "- 이 특약의 핵심 의미는 ...\n- 임차인 입장에서 주의할 점은 ...\n- 관련 법령에 따르면 ...",
            "selected_laws": [
                {
                    "ref": "<위 목록의 doc_id 그대로>",
                    "summary": "<is_violation 또는 is_caution이 true면 첫 줄 한 문장 요약 + 빈 줄 + 마크다운 상세 설명 / 둘 다 false면 빈 문자열>",
                    "is_violation": False,
                    "is_caution": False
                }
            ]
        },
        ensure_ascii=False,
        indent=2,
    )

    return f"""
[분석 대상 특약]
{target_clause}

[관련 법령]
{format_laws(laws)}

[관련 판례]
{format_precs(precs)}

[출력 지시]
위 분석 대상 특약을 검토하여 아래 JSON 형식으로만 응답하세요.

- clause_one_line_summary: 이 특약 전체를 **한 문장**으로 요약하세요.
  · 법률 지식이 없는 일반인이 바로 이해할 수 있도록 쉽고 간결하게 작성하세요.
  · 20자 내외, 마침표로 끝내세요. 마크다운 기호는 사용하지 마세요.
  · 예시: "임차인이 먼저 퇴실하면 남은 기간 임대료를 부담해야 합니다."

- clause_interpretation: 이 특약이 의미하는 바를 법률 지식이 없는 일반인도 이해할 수 있도록 쉬운 말로 바꿔 설명하세요.
  · 반드시 제공된 [관련 법령]·[관련 판례]를 근거로 설명하세요. 외부 지식을 사용하지 마세요.
  · 어떤 상황에서 어떤 권리가 보호되거나 제한될 수 있는지, 임차인이 실제로 알아야 할 핵심만 담으세요.
  · **말투는 "~했음.", "~임.", "~가능함." 형식의 개조식**으로 작성하세요. "~합니다", "~세요", "~해요" 형식은 사용하지 마세요.
  · **법률 전문 용어는 반드시 괄호 안에 풀어쓰세요.** 예) 묵시적 갱신(계약 만료 후 아무 말 없이 자동으로 계약이 연장되는 것), 중개보수(부동산 중개업소에 내는 수수료), 대항력(집이 팔려도 계속 살 수 있는 권리), 보증금반환청구권(계약 끝날 때 맡긴 돈을 돌려달라고 요구할 수 있는 권리)
  · 특약 원문을 그대로 반복하지 마세요.
  · **마크다운 형식**으로 작성하세요. `#`, `##` 등 제목 기호는 절대 사용하지 마세요.
  · `- ` 불릿 포인트로 2~4개 항목을 작성하세요.
  · **불릿 하나에 반드시 한 문장만** 작성하세요. 두 문장 이상이면 별도 불릿으로 분리하세요.
    형식 예시:
    - 이 특약의 핵심 의미는 ...입니다.
    - 묵시적 갱신(계약 만료 후 아무 말 없이 자동으로 연장되는 것)이 되면 ...할 수 있습니다.
    - 관련 법령에 따르면 ...

- selected_laws: [관련 법령]과 [관련 판례] 중 이 특약과 **직접적으로 관련 있는 상위 3개 이내**만 골라 작성하세요.
  *   **주의**: 제공된 리스트에 없는 항목을 외부 지식으로 생성하지 마세요. 반드시 식별자({', '.join(all_doc_ids)})만 사용하세요.
  *   **주의**: 연관성이 낮거나 단순히 용어가 겹치는 항목은 제외하세요. 정말 관련 있는 항목이 없으면 빈 배열([])을 반환하세요.

  is_violation / is_caution 판단 기준 (반드시 제공된 법령·판례 내용만 근거로 판단하세요):
    · **is_violation**: 특약 내용이 제공된 법령의 **강행규정(임차인에게 불리한 약정은 효력이 없다는 규정 등)**에 명백히 위배되거나, 제공된 판례상 임차인의 권리를 부당하게 제한한다고 판단된 사례와 유사하면 `true`. 단순 절차 안내이거나 상호 합의 가능한 범위이면 `false`.
    · **is_caution**: is_violation이 `false`이더라도, 제공된 법령·판례를 근거로 볼 때 임차인 또는 임대인 중 한쪽에게 실질적으로 불리하게 작용할 가능성이 있으면 `true`. 양쪽 모두에게 중립적이면 `false`.
    · is_violation이 `true`이면 is_caution은 `false`로 설정하세요 (위배가 더 상위 개념).

  summary 작성 요령:
    · **모든 summary에서 불릿 하나에 반드시 한 문장만** 작성하세요. 두 문장 이상이면 별도 불릿으로 분리하세요.
    · **말투는 "~했음.", "~임.", "~가능함." 형식의 개조식**으로 작성하세요. "~합니다", "~세요", "~해요" 형식은 사용하지 마세요.
    · **법률 전문 용어는 반드시 괄호 안에 풀어쓰세요.** 예) 묵시적 갱신(계약 만료 후 자동 연장), 중개보수(부동산 수수료), 강행규정(당사자가 바꿀 수 없는 법 조항)
    · **법령(관련 법령 섹션 항목)**:
        · is_violation 또는 is_caution이 `true`인 경우: 반드시 아래 **두 부분**으로 구성하세요. `#`, `##` 기호는 사용하지 마세요.
          1. **첫 번째 줄**: 왜 문제가 되는지(위배) 일반인이 바로 이해할 수 있는 한 문장 요약
          2. **빈 줄 하나** 삽입
          3. **이후 내용**: 구체적인 이유, 근거 조항, 임차인/임대인이 주장할 수 있는 권리를 `- ` 불릿으로 서술 (불릿 하나 = 한 문장)
        · is_violation과 is_caution 모두 `false`인 경우: 빈 문자열("")을 반환하세요.
    · **판례(관련 판례 섹션 항목)**:
        · is_violation이 `true`인 경우: 법령과 동일하게 아래 **두 부분**으로 구성하세요.
          1. **첫 번째 줄**: 이 판례가 이 특약과 어떻게 충돌하는지 일반인이 바로 이해할 수 있는 한 문장 요약
          2. **빈 줄 하나** 삽입
          3. **이후 내용**: `- ` 불릿으로 아래 항목 서술 (불릿 하나 = 한 문장)
             - **사건 개요**: 어떤 분쟁이었는지 한 문장으로 요약함.
             - **법원 판단**: 법원이 어떻게 결론 내렸는지 한 문장으로 씀.
             - **이 특약과의 관련성**: 왜 이 특약에서 주의해야 하는지 한 문장으로 씀.
        · is_violation이 `false`인 경우: 빈 줄 없이 바로 `- ` 불릿으로 작성하세요. (불릿 하나 = 한 문장)
          - **사건 개요**: 어떤 분쟁이었는지 한 문장으로 요약함.
          - **법원 판단**: 법원이 어떻게 결론 내렸는지 한 문장으로 씀.
          - **이 특약과의 관련성**: 왜 이 특약에서 주의해야 하는지 한 문장으로 씀.

- JSON 외 텍스트, 마크다운 코드블록은 포함하지 마세요.

{output_format}
"""

def build_final_report_prompt(
    property_info: dict,
    common_terms: list,
    target_terms_with_summaries: list[dict],
) -> str:
    common_text = "\n".join([f"공통특약 {i+1}: {t}" for i, t in enumerate(common_terms)])
    
    target_text_parts = []
    for item in target_terms_with_summaries:
        idx = item['index'] - COMMON_TERMS_COUNT + 1
        clause_id = f"특약{idx}"
        target_text_parts.append(f"[{clause_id}]\n원문: {item['clause']}")
        if item.get('summaries'):
            target_text_parts.append("법적 해석 요약:")
            for s in item['summaries']:
                target_text_parts.append(f"- {s['ref']}: {s['summary']}")
        target_text_parts.append("")
    
    target_text = "\n".join(target_text_parts)

    output_format = json.dumps(
        {
            "contract_checklist": [
                {"item": None, "description": None, "basis": None}
            ],
            "clause_relations": [
                {
                    "clause_id": "<특약1 등>",
                    "related_clauses": [
                        {"clause_id": "<특약2 또는 공통특약 1>", "clause_text": None, "relation": "<관계 설명>"}
                    ]
                }
            ]
        },
        ensure_ascii=False,
        indent=2,
    )

    return f"""
[계약서 정보]
{format_property_info(property_info)}

[공통 특약]
{common_text}

[타겟 특약 전체 및 법적 해석]
{target_text}

[출력 지시]
위 계약서 전체와 개별 특약의 법적 해석을 바탕으로 다음 두 가지를 도출하여 JSON 형식으로만 응답하세요.

1. contract_checklist: 임차인이 계약 전 확인해야 할 통합 체크리스트
- 입력된 모든 특약을 종합 검토한 후, 확인 항목을 통합하여 작성하세요.
- 계약서 전체 맥락에서 중요한 확인 사항을 도출하세요.
- **주의**: 모든 인덱스는 1부터 시작합니다. **`특약0`은 존재하지 않으며 절대 사용하지 마세요.**
- description: 법률 지식이 없는 일반인이 이해할 수 있도록 **쉬운 말**로 작성하세요.
  · `- ` 불릿 포인트로 2~3개 항목으로 간결하게 정리하세요.
  · `#`, `##` 등 제목 기호는 절대 사용하지 마세요.
  · **불릿 하나에 반드시 한 문장만** 작성하세요. 두 문장 이상이면 별도 불릿으로 분리하세요.
  · **말투는 "~했음.", "~임.", "~가능함." 형식의 개조식**으로 작성하세요. "~합니다", "~세요", "~해요" 형식은 사용하지 마세요.
  · **법률 전문 용어는 반드시 괄호 안에 풀어쓰세요.** 예) 묵시적 갱신(계약 만료 후 자동 연장), 중개보수(부동산 수수료)
  · 예시:
    - 계약 전 임대인과 수선 범위를 명확히 합의하고 계약서에 기재해 두세요.
    - 퇴거 시 분쟁 방지를 위해 입주 시 상태를 사진으로 기록해 두세요.
- basis: 이 체크리스트 항목의 근거가 되는 특약 ID(예: "특약1", "공통특약 1")와 법령/판례 ref를 배열로 나열하세요. 반드시 본문에 등장한 식별자만 정확히 사용해야 합니다.

2. clause_relations: 특약 간의 연관성 분석 (특약 vs 특약)
- [타겟 특약 전체]와 [공통 특약] 중 서로 동시에 적용될 때 확인이 필요하거나 충돌할 가능성이 있는 **특약들끼리의 관계(Agreement-to-Agreement)**를 분석하세요.
- **주의**: `특약0`은 절대 사용하지 마세요. 모든 식별자는 1번부터 시작합니다.
- **주의**: 법령이나 판례와 특약의 관계는 이미 개별 요약에 포함되어 있으므로, 여기서는 절대 다루지 마세요. `clause_id`에 법령 식별자(예: 민법 제00조)를 넣는 것은 엄격히 금지됩니다.
- `clause_id`들은 반드시 제공된 "특약1", "특약2", "공통특약 1" 형태의 식별자만 사용하세요.
- **주의**: 각 특약의 원문 내용을 다른 특약 번호와 절대 혼동하지 마세요. 반드시 제공된 ID와 그에 해당하는 원문 내용을 정확히 매칭하여 분석하세요.
- relation은 두 조항이 어떻게 연관되어 있으며 왜 주의해야 하는지 일반인 눈높이에서 **마크다운 형식**(`- ` 불릿)으로 설명하세요. `#`, `##` 기호는 사용하지 마세요. **불릿 하나에 반드시 한 문장만** 작성하고, **말투는 "~했음.", "~임." 개조식**으로, 법률 전문 용어는 괄호 안에 풀어쓰세요.
- 연관성이 있는 특약에 대해서만 배열에 추가하세요.
- clause_text는 null로 두세요.

[출력 형식 예시]
아래는 특약 4개가 하나의 계약서에 함께 존재하는 경우의 출력 예시입니다.
실제 입력의 특약 수와 내용에 맞게 작성하세요.
판단 표현(유리/불리/위험)은 사용하지 마세요.

---
[입력 예시]
특약A: "임차인 퇴거 시 모든 원상복구 비용은 임차인이 전액 부담한다."
특약B: "임차 기간 중 발생하는 모든 수리 및 수선은 임차인이 부담한다."
특약C: "계약 갱신 시 임대료 인상률은 당사자 간 협의로 정하며 별도 제한을 두지 않는다."
특약D: "임차인은 계약 만료 시 계약갱신을 요구하지 않기로 한다."

[출력 예시 JSON 내부 구조]
"contract_checklist": [
  {{
    "item": "원상복구 범위 및 귀책 기준 확인",
    "description": "- 계약 전 임대인과 원상복구 범위를 구체적으로 합의하고 계약서에 기재해 두세요.\n- 입주 시 내부 상태를 사진·영상으로 기록해두면 퇴거 시 분쟁을 예방할 수 있어요.",
    "basis": ["특약A"]
  }}
],
"clause_relations": [
  {{
    "clause_id": "특약A",
    "related_clauses": [
      {{
        "clause_id": "특약B",
        "clause_text": null,
        "relation": "수선의무 부담 특약이 함께 존재하는 경우, 임차인이 임차 기간 중 자비로 수선한 부분이 퇴거 시 원상복구 대상에 해당하는지 여부가 불명확해집니다. 임차인이 수선 비용을 이미 부담한 부분에 대해 추가로 원상복구 비용까지 청구될 수 있는지를 두 조항을 함께 확인합니다."
      }}
    ]
  }}
]
---

JSON 외 텍스트, 마크다운 코드블록은 포함하지 마세요.

{output_format}
"""


def call_llm(
    prompt: str,
    schema: type[BaseModel] | None = None,
    max_retries: int = 2,
    model_override: str | None = None,
) -> dict | None:
    import time
    from json import JSONDecodeError
    from pydantic import ValidationError

    model_to_use = model_override if model_override else MODEL_NAME_FLASH
    model = GenerativeModel(model_to_use, system_instruction=SYSTEM_PROMPT)
    config = GenerationConfig(temperature=0.0, response_mime_type="application/json")

    for attempt in range(1, max_retries + 2):
        try:
            response = model.generate_content(prompt, generation_config=config)
            text = response.text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            parsed = json.loads(text)
            if schema:
                validated = schema.model_validate(parsed)
                return validated.model_dump()
            return parsed
        except JSONDecodeError as e:
            print(f"  [call_llm] JSON 파싱 실패 (시도 {attempt}/{max_retries + 1}): {e}")
        except ValidationError as e:
            print(f"  [call_llm] Pydantic 검증 실패 (시도 {attempt}/{max_retries + 1}): {e}")
        except Exception as e:
            print(f"  [call_llm] LLM 호출 오류 (시도 {attempt}/{max_retries + 1}): {e}")

        if attempt <= max_retries:
            time.sleep(2 ** attempt)

    return None

from functools import lru_cache

# ... (imports)

# ... (rest of the file)

@lru_cache(maxsize=1024)
def generate_clause_summary(target_clause: str, laws_json: str, precs_json: str) -> dict:
    """1단계: 개별 특약의 법령/판례 요약만 생성 (독립적, 비동기 호출용)"""
    laws = json.loads(laws_json)
    precs = json.loads(precs_json)

    prompt = build_clause_summary_prompt(target_clause, laws, precs)
    result = call_llm(prompt, schema=ClauseSummaryOutput, model_override=MODEL_NAME_FLASH)

    rag_map: dict[str, dict] = {}
    for m in laws:
        did = m.get("doc_id", "")
        if did: rag_map[did] = {"type": LawType.law.value, "ref": did, "content": m.get("content") or m.get("doc_text") or ""}
    for m in precs:
        did = m.get("doc_id", "")
        if did: rag_map[did] = {"type": LawType.case.value, "ref": did, "content": m.get("content") or m.get("summary") or ""}

    related_laws = []
    clause_one_line_summary = ""
    clause_interpretation = ""
    if result:
        clause_one_line_summary = result.get("clause_one_line_summary") or ""
        clause_interpretation = result.get("clause_interpretation") or ""
        covered = set()
        for sel in result.get("selected_laws", []):
            ref = (sel.get("ref") or "").strip()
            if ref in rag_map and ref not in covered:
                related_laws.append({
                    **rag_map[ref],
                    "summary": sel.get("summary") or "",
                    "is_violation": sel.get("is_violation", False),
                    "is_caution": sel.get("is_caution", False),
                })
                covered.add(ref)

    return {
        "clause_one_line_summary": clause_one_line_summary,
        "clause_interpretation": clause_interpretation,
        "related_laws": related_laws,
    }


def generate_final_report(property_info: dict, common_terms: list, clauses_with_hits_and_summaries: list[dict]) -> FinalReportOutput:
    """2단계: 전체가 모인 후 통합 체크리스트와 연관성 분석 수행"""
    prompt = build_final_report_prompt(property_info, common_terms, clauses_with_hits_and_summaries)
    result = call_llm(prompt, schema=FinalReportLLMOutput, model_override=MODEL_NAME_FLASH)
    
    if not result:
        return FinalReportOutput()

    clause_text_map: dict[str, str] = {}
    for item in clauses_with_hits_and_summaries:
        label = f"특약{item['index'] - COMMON_TERMS_COUNT + 1}"
        clause_text_map[label] = item["clause"]
    for i, term in enumerate(common_terms):
        clause_text_map[f"공통특약 {i + 1}"] = term

    relations_map = {}
    
    # 중복 제거 로직 포함 (A->B, B->A)
    seen_pairs = set()

    for cr in result.get("clause_relations", []):
        cid = cr.get("clause_id")
        # cid가 실제 특약 목록에 존재하는 유효한 ID인지 검증
        if not cid or cid not in clause_text_map:
            continue
        
        valid_rels = []
        for rel in cr.get("related_clauses", []):
            other_id = rel.get("clause_id")
            # 상대방 ID도 유효한 특약 ID인지 검증
            if not other_id or other_id not in clause_text_map:
                continue
            
            pair = frozenset([cid, other_id])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                rel["clause_text"] = clause_text_map.get(other_id)
                valid_rels.append(rel)
        
        if valid_rels:
            relations_map[cid] = valid_rels

    # 체크리스트 후처리: 유효하지 않은 basis 필터링 및 텍스트 보정
    final_checklist = []
    for item in result.get("contract_checklist", []):
        # 1. basis 필터링
        if item.get("basis"):
            # 유효한 ID(clause_text_map의 키) 또는 법령/판례 형식인 것만 유지
            valid_basis = [
                b for b in item["basis"] 
                if b in clause_text_map or (re.search(r'[법령조항칙]', b) or len(b.split(",")) > 1)
            ]
            item["basis"] = [b for b in valid_basis if b != "특약0"] # 특약0은 무조건 제거

        # 2. 설명 문구 보정 (특약0 언급 제거)
        if item.get("description"):
            item["description"] = item["description"].replace("특약0", "해당 특약")

        final_checklist.append(item)

    return FinalReportOutput(
        contract_checklist=final_checklist,
        related_clauses_map=relations_map
    )


# ── 특약 재작성 ──────────────────────────────────────────

class ClauseRewriteOutput(BaseModel):
    rewritten_clause: str = ""
    reason: str = ""

    @field_validator("rewritten_clause", "reason", mode="before")
    @classmethod
    def coerce_str(cls, v):
        if v is None:
            return ""
        if isinstance(v, list):
            return "\n".join(str(x) for x in v if x)
        return str(v)


def generate_clause_rewrite(
    original_clause: str,
    violation_laws: list[dict],  # is_violation=True 항목: {ref, content, summary}
    all_related_laws: list[dict],  # 전체 related_laws: {ref, content, summary, type}
) -> dict:
    """위반 가능성이 있는 특약을 법령에 맞게 재작성한다."""

    violation_text = "\n\n".join(
        f"[{item['ref']}]\n위반 요약: {item.get('summary', '')}\n원문: {item.get('content', '')}"
        for item in violation_laws
    ) or "해당 없음"

    reference_text = "\n\n".join(
        f"[{item['ref']}] ({item.get('type', '')})\n{item.get('content', '')}"
        for item in all_related_laws
        if not item.get("is_violation")
    ) or "해당 없음"

    output_format = json.dumps(
        {
            "rewritten_clause": "재작성된 특약 전문 (원문과 동일한 형식, 법령에 맞게 수정)",
            "reason": "- **재작성 이유**:\n  - 원본 특약은 ... 때문에 문제가 됨.\n- **수정한 부분**:\n  - '원문 문구' → '수정 문구'로 변경함.\n- **법적 근거**:\n  - 주택임대차보호법 제○조에 따라 ... 임."
        },
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
[원본 특약]
{original_clause}

[위반 가능성이 있는 법령]
{violation_text}

[참고 법령·판례]
{reference_text}

[출력 지시]
위 원본 특약이 위반 가능성이 있는 법령에 위배됩니다.
법령을 준수하면서도 특약의 본래 의도를 최대한 살려 재작성하세요.

- rewritten_clause: 재작성된 특약 전문을 작성하세요.
  · 원본 특약과 동일한 문체(구어체/문어체)를 유지하세요.
  · 법령 위반 소지가 있는 문구만 수정하고, 나머지는 원문을 최대한 유지하세요.
  · 마크다운 기호는 사용하지 마세요. 특약 원문 형식 그대로 작성하세요.

- reason: 재작성 이유를 아래 **중첩 불릿 구조**로 작성하세요.
  · `#`, `##` 제목 기호는 사용하지 마세요.
  · 최상위 불릿 3개: **재작성 이유**, **수정한 부분**, **법적 근거** (볼드 처리)
  · 각 최상위 불릿 아래에 들여쓰기(`  - `)로 세부 내용을 작성하세요.
  · **들여쓰기 불릿 하나에 반드시 한 문장만** 작성하세요. 두 문장 이상이면 별도 불릿으로 분리하세요.
  · **말투는 "~했음.", "~임.", "~가능함." 형식의 개조식**으로 작성하세요.
  · **독자는 법률 지식이 전혀 없는 일반인**임을 항상 염두에 두고, 중학생도 이해할 수 있는 쉬운 말로 작성하세요.
  · **법률 전문 용어는 반드시 괄호 안에 일상 언어로 풀어쓰세요.** 예) 강행규정(법으로 정해져 있어서 당사자가 마음대로 바꿀 수 없는 조항), 계약갱신청구권(세입자가 "계약을 한 번 더 연장해 달라"고 요구할 수 있는 권리)
  · 어려운 한자어나 법조문 표현은 쉬운 우리말로 바꾸세요. 예) "위배" → "어긋남", "배제" → "없앰", "준용" → "똑같이 적용"
  · 형식 예시:
    - **재작성 이유**:
      - 원본 특약은 법으로 정해진 규정(당사자가 마음대로 바꿀 수 없는 조항)에 어긋남.
      - 세입자가 "계약을 한 번 더 연장해 달라"고 요구할 수 있는 권리를 빼앗는 내용임.
    - **수정한 부분**:
      - '임차인은 계약 만료 시 갱신을 요구하지 않는다' → '임차인은 법이 허용하는 범위 안에서 갱신을 요청할 수 있다'로 바꿈.
    - **법적 근거**:
      - 주택임대차보호법 제6조의3에 따르면, 세입자의 계약 연장 요구 권리는 특약으로 없앨 수 없음.

- JSON 외 텍스트, 마크다운 코드블록은 포함하지 마세요.

{output_format}
"""

    result = call_llm(prompt, schema=ClauseRewriteOutput, model_override=MODEL_NAME_FLASH)
    if not result:
        return {"rewritten_clause": "", "reason": ""}
    return result
