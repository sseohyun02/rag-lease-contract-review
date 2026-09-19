import time
import json
import logging
import urllib.parse
import requests
import urllib3
from dataclasses import dataclass, asdict
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Any

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =========================================================
# 1. 사용자 설정
# =========================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

OC         = os.getenv("LAW_API_KEY")
BASE_URL   = "https://www.law.go.kr/DRF"
SEARCH_URL = f"{BASE_URL}/lawSearch.do"
DETAIL_URL = f"{BASE_URL}/lawService.do"

DISPLAY     = 100
SLEEP_SEC   = 0.2
TIMEOUT_SEC = 15

MIN_LIST_HIT_COUNT   = 2
MIN_DETAIL_HIT_COUNT = 2

MAX_RESULTS_PER_LAW  = None    # 법령별 최대 수집 건수 (None = 무제한)

_BASE = Path(__file__).resolve().parent.parent.parent  # data/collectors → data → 루트
OUTPUT_JSON  = str(_BASE / "output" / "housing_precedents_final.json")
OUTPUT_JSONL = str(_BASE / "output" / "housing_precedents_final.jsonl")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SESSION        = requests.Session()
_SESSION.verify = False


# =========================================================
# 2. JO 법령별 정의
# =========================================================

FULL_COLLECT_JO_MAP: Dict[str, str] = {
    "주택임대차보호법": "주택임대차보호법",
}

FILTER_COLLECT_JO_MAP: Dict[str, str] = {
    "민법":               "민법",
    "부동산등기법":        "부동산등기법",
    "공인중개사법":        "공인중개사법",
    "화물자동차운수사업법": "화물자동차운수사업법",
    "민사소송법":          "민사소송법",
    "민사조정법":          "민사조정법",
    "민사집행법":          "민사집행법",
    "소액사건심판법":       "소액사건심판법",
    "공동주택관리법":       "공동주택관리법",
    "주민등록법":          "주민등록법",
}


# =========================================================
# 3. 주택임대차 콘텐츠 키워드
# =========================================================

HOUSING_LEASE_CONTENT_KEYWORDS: List[str] = [
    "주택임대차", "임대차", "전세", "월세", "임차인", "임대인", "주거용",
    "임차보증금", "임대차보증금", "보증금반환", "보증금증액", "보증금감액",
    "대항력", "확정일자", "우선변제", "최우선변제", "소액임차인", "선순위",
    "점유", "전입신고", "주민등록", "인도",
    "계약갱신", "갱신거절", "묵시갱신", "해지통지", "차임연체",
    "임차권등기", "임차권양도", "전대",
    "근저당", "저당", "경락", "낙찰", "압류", "담보", "배당", "경매",
    "거소", "경료", "원상회복", "수선의무", "필요비", "유익비",
]


# =========================================================
# 4. 데이터 모델
# =========================================================

@dataclass
class Precedent:
    # API 원본 필드
    판례일련번호: str
    사건명:       str
    사건번호:     str
    선고일자:     str
    선고:         str
    법원명:       str
    법원종류코드: str
    판결유형:     str
    판시사항:     str
    판결요지:     str
    참조조문:     str
    참조판례:     str
    판례내용:     str

    # 사용자 생성 필드
    최초포착법령:  str
    최초JO값:     str
    검색법령목록:  List[str]
    JO값목록:     List[str]
    매칭키워드목록: List[str]
    목록히트수최대: int
    상세히트수:    int


# =========================================================
# 5. 공통 유틸
# =========================================================

def _normalize_to_list(value: Any) -> List[dict]:
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# =========================================================
# 6. HTTP 유틸
# =========================================================

def _get_json(url: str) -> dict:
    for attempt in range(3):
        try:
            resp = _SESSION.get(url, timeout=TIMEOUT_SEC)
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            log.warning("Timeout (시도 %d/3) | %s", attempt + 1, url)
            time.sleep(1.0 * (attempt + 1))

        except requests.exceptions.ConnectionError as e:
            log.error("ConnectionError | %s | %s", e, url)
            time.sleep(1.0 * (attempt + 1))

        except requests.exceptions.HTTPError as e:
            log.error("HTTPError %s | %s", e, url)
            return {}

        except Exception as e:
            log.error("Unknown Error %s | %s", e, url)
            return {}

    log.error("3회 재시도 실패 | %s", url)
    return {}


# =========================================================
# 7. API 호출
# =========================================================

def search_page_by_jo(jo_value: str, page: int) -> dict:
    params = {
        "OC":      OC,
        "target":  "prec",
        "type":    "JSON",
        "JO":      jo_value,
        "display": DISPLAY,
        "page":    page,
    }
    url = SEARCH_URL + "?" + urllib.parse.urlencode(params, encoding="utf-8")
    return _get_json(url)


def get_precedent_detail(prec_id: str) -> dict:
    params = {
        "OC":     OC,
        "target": "prec",
        "type":   "JSON",
        "ID":     prec_id,
    }
    url = DETAIL_URL + "?" + urllib.parse.urlencode(params, encoding="utf-8")
    return _get_json(url)


# =========================================================
# 8. 키워드 히트 필터
# =========================================================

def _combined_text_from_item(item: dict) -> str:
    fields = [
        item.get("사건명",   ""),
        item.get("판시사항", ""),
        item.get("판결요지", ""),
        item.get("참조조문", ""),
        item.get("참조판례", ""),
    ]
    return " ".join(f for f in fields if f)


def _combined_text_from_detail(detail: dict) -> str:
    svc = detail.get("PrecService", detail)
    fields = [
        svc.get("사건명",   ""),
        svc.get("판시사항", ""),
        svc.get("판결요지", ""),
        svc.get("참조조문", ""),
        svc.get("참조판례", ""),
        svc.get("판례내용", ""),
    ]
    return " ".join(f for f in fields if f)


def get_matched_keywords(text: str) -> List[str]:
    if not text:
        return []
    return [kw for kw in HOUSING_LEASE_CONTENT_KEYWORDS if kw in text]


def count_keyword_hits(text: str) -> int:
    return len(get_matched_keywords(text))


def passes_list_filter(item: dict) -> Tuple[bool, int, List[str]]:
    text    = _combined_text_from_item(item)
    matched = get_matched_keywords(text)
    hits    = len(matched)
    return hits >= MIN_LIST_HIT_COUNT, hits, matched


def passes_detail_filter(detail: dict) -> Tuple[bool, int, List[str]]:
    text    = _combined_text_from_detail(detail)
    matched = get_matched_keywords(text)
    hits    = len(matched)
    return hits >= MIN_DETAIL_HIT_COUNT, hits, matched


# =========================================================
# 9. Precedent 생성 · 병합
# =========================================================

def build_precedent(
    item:           dict,
    law_name:       str,
    jo_value:       str,
    list_hits:      int,
    matched_kws:    List[str],
    detail_hits:    int = 0,
) -> Precedent:
    return Precedent(
        판례일련번호  = item.get("판례일련번호", ""),
        사건명        = item.get("사건명",       ""),
        사건번호      = item.get("사건번호",     ""),
        선고일자      = item.get("선고일자",     ""),
        선고          = item.get("선고",         ""),
        법원명        = item.get("법원명",       ""),
        법원종류코드  = item.get("법원종류코드", ""),
        판결유형      = item.get("판결유형",     ""),
        판시사항      = item.get("판시사항",     ""),
        판결요지      = item.get("판결요지",     ""),
        참조조문      = item.get("참조조문",     ""),
        참조판례      = item.get("참조판례",     ""),
        판례내용      = item.get("판례내용",     ""),
        최초포착법령   = law_name,
        최초JO값      = jo_value,
        검색법령목록   = [law_name],
        JO값목록      = [jo_value],
        매칭키워드목록 = sorted(set(matched_kws)),
        목록히트수최대 = list_hits,
        상세히트수     = detail_hits,
    )


def merge_precedent(existing: Precedent, new_p: Precedent) -> None:
    if new_p.최초포착법령 not in existing.검색법령목록:
        existing.검색법령목록.append(new_p.최초포착법령)

    if new_p.최초JO값 not in existing.JO값목록:
        existing.JO값목록.append(new_p.최초JO값)

    existing.매칭키워드목록 = sorted(
        set(existing.매칭키워드목록) | set(new_p.매칭키워드목록)
    )
    existing.목록히트수최대 = max(existing.목록히트수최대, new_p.목록히트수최대)
    existing.상세히트수     = max(existing.상세히트수,     new_p.상세히트수)

    for f in ["판례내용", "판시사항", "판결요지", "참조조문", "참조판례"]:
        if not getattr(existing, f) and getattr(new_p, f):
            setattr(existing, f, getattr(new_p, f))


def overlay_detail(item: dict, detail: dict) -> dict:
    """목록 item에 상세 조회 결과를 덮어씌워 반환."""
    svc = detail.get("PrecService", detail)
    if not isinstance(svc, dict):
        return item
    merged = dict(item)
    for f in [
        "사건명", "사건번호", "선고일자", "선고", "법원명",
        "법원종류코드", "판결유형", "판시사항", "판결요지",
        "참조조문", "참조판례", "판례내용",
    ]:
        if svc.get(f):
            merged[f] = svc[f]
    return merged


# =========================================================
# 10. 수집 함수 A — 전체 수집 (주택임대차보호법)
#     [수정] 상세 API 호출 추가 — 목록만 저장하면 본문이 비어있는 문제 수정
# =========================================================

def collect_full(law_name: str, jo_value: str) -> List[Precedent]:
    results:  List[Precedent] = []
    seen_ids: set             = set()
    page      = 1

    log.info("[전체수집 시작] 법령=%s | JO=%s", law_name, jo_value)

    while True:
        data      = search_page_by_jo(jo_value=jo_value, page=page)
        raw       = data.get("PrecSearch", {}).get("prec")
        items     = _normalize_to_list(raw)
        total_cnt = _safe_int(data.get("PrecSearch", {}).get("totalCnt", 0))

        if not items:
            log.info("  [%s] page=%d — 결과 없음, 탐색 종료", law_name, page)
            break

        for item in items:
            prec_id = item.get("판례일련번호", "")
            if not prec_id or prec_id in seen_ids:
                continue

            # ── [수정] 상세 API 호출해서 본문 채우기 ──
            detail = get_precedent_detail(prec_id)
            if detail:
                item = overlay_detail(item, detail)
                _, detail_hits, matched_kws = passes_detail_filter(detail)
            else:
                text        = _combined_text_from_item(item)
                matched_kws = get_matched_keywords(text)
                detail_hits = 0

            time.sleep(SLEEP_SEC)
            # ─────────────────────────────────────────

            seen_ids.add(prec_id)
            results.append(
                build_precedent(
                    item=item,
                    law_name=law_name,
                    jo_value=jo_value,
                    list_hits=len(matched_kws),
                    matched_kws=matched_kws,
                    detail_hits=detail_hits,
                )
            )

        log.info(
            "  [%s] page=%d | 이번=%d건 | 누계=%d건 / 전체=%d건",
            law_name, page, len(items), len(results), total_cnt,
        )

        if len(items) < DISPLAY or page * DISPLAY >= total_cnt:
            break

        page += 1
        time.sleep(SLEEP_SEC)

    log.info("[전체수집 완료] 법령=%s | 수집=%d건", law_name, len(results))
    return results


# =========================================================
# 11. 수집 함수 B — 히트 필터 수집 (나머지 10개 법령)
# =========================================================

def collect_filtered(law_name: str, jo_value: str) -> List[Precedent]:
    results:      List[Precedent] = []
    seen_ids:     set             = set()
    page          = 1
    total_scanned = 0

    log.info("[필터수집 시작] 법령=%s | JO=%s", law_name, jo_value)

    while True:
        if MAX_RESULTS_PER_LAW is not None and len(results) >= MAX_RESULTS_PER_LAW:
            log.info("  [%s] 최대 수집 건수 도달 (%d건)", law_name, MAX_RESULTS_PER_LAW)
            break

        data      = search_page_by_jo(jo_value=jo_value, page=page)
        raw       = data.get("PrecSearch", {}).get("prec")
        items     = _normalize_to_list(raw)
        total_cnt = _safe_int(data.get("PrecSearch", {}).get("totalCnt", 0))

        if not items:
            log.info("  [%s] page=%d — 결과 없음, 탐색 종료", law_name, page)
            break

        for item in items:
            total_scanned += 1

            prec_id = item.get("판례일련번호", "")
            if not prec_id or prec_id in seen_ids:
                continue

            # 1차 필터: 목록 단계 키워드 히트
            passed, list_hits, matched_kws = passes_list_filter(item)
            if not passed:
                continue

            # 상세 API 호출 (항상 실행 — 본문 보장)
            detail      = get_precedent_detail(prec_id)
            detail_hits = 0
            if detail:
                passed_d, detail_hits, matched_kws_d = passes_detail_filter(detail)
                if not passed_d:
                    time.sleep(SLEEP_SEC)
                    continue
                matched_kws = matched_kws_d
                item        = overlay_detail(item, detail)
            time.sleep(SLEEP_SEC)

            seen_ids.add(prec_id)
            results.append(
                build_precedent(
                    item=item,
                    law_name=law_name,
                    jo_value=jo_value,
                    list_hits=list_hits,
                    matched_kws=matched_kws,
                    detail_hits=detail_hits,
                )
            )

            if MAX_RESULTS_PER_LAW is not None and len(results) >= MAX_RESULTS_PER_LAW:
                break

        log.info(
            "  [%s] page=%d | 이번 검토=%d건 | 통과 누계=%d건 / 전체=%d건",
            law_name, page, len(items), len(results), total_cnt,
        )

        if len(items) < DISPLAY or page * DISPLAY >= total_cnt:
            break

        page += 1
        time.sleep(SLEEP_SEC)

    log.info(
        "[필터수집 완료] 법령=%s | 검토=%d건 | 통과=%d건",
        law_name, total_scanned, len(results),
    )
    return results


# =========================================================
# 12. 저장
# =========================================================

def save_outputs(
    precedents: List[Precedent],
    json_path:  str = OUTPUT_JSON,
    jsonl_path: str = OUTPUT_JSONL,
) -> None:
    rows = [asdict(p) for p in precedents]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log.info("저장 완료 → %s / %s", json_path, jsonl_path)


# =========================================================
# 13. 메인 파이프라인
# =========================================================

def main() -> None:
    all_map:     Dict[str, Precedent] = {}
    law_summary: Counter              = Counter()

    # ── 1단계: 주택임대차보호법 전체 수집 ──
    for law_name, jo_value in FULL_COLLECT_JO_MAP.items():
        log.info("=" * 70)
        log.info("[전체수집 법령] %s (JO=%s)", law_name, jo_value)

        batch = collect_full(law_name=law_name, jo_value=jo_value)
        law_summary[law_name] = len(batch)

        for p in batch:
            if not p.판례일련번호:
                continue
            if p.판례일련번호 in all_map:
                merge_precedent(all_map[p.판례일련번호], p)
            else:
                all_map[p.판례일련번호] = p

        log.info(
            "[전체수집 완료] %s | batch=%d | 전체고유=%d",
            law_name, len(batch), len(all_map),
        )
        time.sleep(SLEEP_SEC)

    # ── 2단계: 나머지 법령 히트 필터 수집 ──
    for law_name, jo_value in FILTER_COLLECT_JO_MAP.items():
        log.info("=" * 70)
        log.info("[필터수집 법령] %s (JO=%s)", law_name, jo_value)

        batch  = collect_filtered(law_name=law_name, jo_value=jo_value)
        before = len(all_map)
        law_summary[law_name] = len(batch)

        for p in batch:
            if not p.판례일련번호:
                continue
            if p.판례일련번호 in all_map:
                merge_precedent(all_map[p.판례일련번호], p)
            else:
                all_map[p.판례일련번호] = p

        added = len(all_map) - before
        dups  = len(batch) - added
        log.info(
            "[필터수집 완료] %s | batch=%d | 신규=%d | 중복병합=%d | 전체고유=%d",
            law_name, len(batch), added, dups, len(all_map),
        )
        time.sleep(SLEEP_SEC)

    # ── 최종 저장 ──
    all_precedents = list(all_map.values())
    save_outputs(all_precedents)

    # ── 통계 출력 ──
    log.info("=" * 70)
    log.info("최종 고유 판례 수: %d건", len(all_precedents))

    log.info("[법령별 수집 건수]")
    for law, cnt in law_summary.items():
        log.info("  %-20s %5d건", law, cnt)

    log.info("[최초포착법령 기준 분포]")
    origin = Counter(p.최초포착법령 for p in all_precedents)
    for law, cnt in origin.most_common():
        log.info("  %-20s %5d건", law, cnt)

    log.info("[매칭 키워드 빈도 상위 20개]")
    kw_counter: Counter = Counter()
    for p in all_precedents:
        for kw in p.매칭키워드목록:
            kw_counter[kw] += 1
    for kw, cnt in kw_counter.most_common(20):
        log.info("  %-15s %5d건", kw, cnt)

    # ── 본문 수집 현황 ──
    empty_cnt = sum(
        1 for p in all_precedents
        if not p.판례내용 and not p.판시사항 and not p.판결요지
    )
    log.info("본문 있음: %d건 / 본문 없음: %d건", len(all_precedents) - empty_cnt, empty_cnt)


if __name__ == "__main__":
    main()