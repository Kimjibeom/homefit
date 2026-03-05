"""
tools.py - HomeFit 프로젝트의 Tool 정의 모듈

매물 검색 더미 데이터, 취득세 계산, 대출 한도 계산 함수를
LangChain @tool 데코레이터로 등록합니다.
Property Matcher Agent와 Finance Expert Agent가 Function Calling으로 호출합니다.

2026년 최신 규제 반영:
  - 스트레스 DSR 3단계 (스트레스 금리 1.5% 100% 반영)
  - LTV: 수도권/규제지역 다주택자 0%, 주담대 최대 6억 한도, 생애 최초 70%
  - 취득세 감면: 생애 최초 200만원, 출산/양육 가구 500만원
"""

import json
from langchain_core.tools import tool


DUMMY_PROPERTIES = [
    {
        "id": "P001",
        "name": "강남 래미안 블레스티지",
        "area": "서울 강남구",
        "type": "아파트",
        "price": 145000,
        "size_m2": 84.9,
        "rooms": 3,
        "floor": "15/25층",
        "year_built": 2019,
        "description": "강남역 도보 10분, 학군 우수, 역세권 프리미엄",
    },
    {
        "id": "P002",
        "name": "마포 래미안 푸르지오",
        "area": "서울 마포구",
        "type": "아파트",
        "price": 98000,
        "size_m2": 74.5,
        "rooms": 3,
        "floor": "10/20층",
        "year_built": 2017,
        "description": "마포역 도보 5분, 한강 조망, 교통 편리",
    },
    {
        "id": "P003",
        "name": "성동 서울숲 트리마제",
        "area": "서울 성동구",
        "type": "아파트",
        "price": 78000,
        "size_m2": 59.9,
        "rooms": 2,
        "floor": "8/18층",
        "year_built": 2020,
        "description": "서울숲 인접, 왕십리역 도보 7분, 신축 단지",
    },
    {
        "id": "P004",
        "name": "노원 래미안 에스티움",
        "area": "서울 노원구",
        "type": "아파트",
        "price": 58000,
        "size_m2": 84.7,
        "rooms": 3,
        "floor": "12/22층",
        "year_built": 2021,
        "description": "노원역 도보 8분, 넓은 평수, 가성비 우수",
    },
    {
        "id": "P005",
        "name": "인천 송도 더샵 센트럴파크",
        "area": "인천 연수구",
        "type": "아파트",
        "price": 45000,
        "size_m2": 84.5,
        "rooms": 3,
        "floor": "20/30층",
        "year_built": 2022,
        "description": "센트럴파크 인접, GTX-B 예정, 신도시 인프라",
    },
    {
        "id": "P006",
        "name": "광교 자연앤힐스테이트",
        "area": "경기 수원시",
        "type": "아파트",
        "price": 62000,
        "size_m2": 99.7,
        "rooms": 4,
        "floor": "7/15층",
        "year_built": 2016,
        "description": "광교호수공원 인접, 넓은 평수, 쾌적한 주거환경",
    },
    {
        "id": "P007",
        "name": "은평 뉴타운 e편한세상",
        "area": "서울 은평구",
        "type": "아파트",
        "price": 52000,
        "size_m2": 59.6,
        "rooms": 2,
        "floor": "5/15층",
        "year_built": 2015,
        "description": "은평뉴타운 내 위치, 3호선 역세권",
    },
    {
        "id": "P008",
        "name": "하남 미사 호반써밋",
        "area": "경기 하남시",
        "type": "아파트",
        "price": 68000,
        "size_m2": 84.9,
        "rooms": 3,
        "floor": "15/25층",
        "year_built": 2018,
        "description": "미사역 도보 3분, 한강변 위치, 교통 우수",
    },
    {
        "id": "P009",
        "name": "부천 중동 롯데캐슬",
        "area": "경기 부천시",
        "type": "아파트",
        "price": 38000,
        "size_m2": 74.2,
        "rooms": 3,
        "floor": "9/20층",
        "year_built": 2014,
        "description": "7호선 역세권, 리모델링 완료, 실속형 매물",
    },
    {
        "id": "P010",
        "name": "동탄2 호반베르디움",
        "area": "경기 화성시",
        "type": "아파트",
        "price": 48000,
        "size_m2": 84.7,
        "rooms": 3,
        "floor": "18/28층",
        "year_built": 2023,
        "description": "동탄역 도보 10분, SRT 이용 가능, 신축 프리미엄",
    },
    {
        "id": "P011",
        "name": "도봉 래미안 아트리체",
        "area": "서울 도봉구",
        "type": "아파트",
        "price": 42000,
        "size_m2": 74.8,
        "rooms": 3,
        "floor": "11/20층",
        "year_built": 2020,
        "description": "도봉산역 도보 5분, 가성비 우수, 초등학교 인접",
    },
    {
        "id": "P012",
        "name": "관악 드림타운 푸르지오",
        "area": "서울 관악구",
        "type": "아파트",
        "price": 47000,
        "size_m2": 59.9,
        "rooms": 2,
        "floor": "7/15층",
        "year_built": 2018,
        "description": "서울대입구역 도보 10분, 관악산 조망, 교통 우수",
    },
    {
        "id": "P013",
        "name": "중랑 포레스트 자이",
        "area": "서울 중랑구",
        "type": "아파트",
        "price": 45000,
        "size_m2": 84.5,
        "rooms": 3,
        "floor": "14/22층",
        "year_built": 2021,
        "description": "망우역 도보 7분, 넓은 평수, 공원 인접 쾌적한 환경",
    },
]

REGULATED_AREAS = {"강남", "서초", "송파", "용산"}
METRO_AREAS = {"서울", "경기", "인천"}


def _is_metro(area: str) -> bool:
    return any(m in area for m in METRO_AREAS)


def _is_regulated(area: str) -> bool:
    return any(r in area for r in REGULATED_AREAS)


# ──────────────────────────────────────────────────────────────
# Tool 1: 매물 검색 — 예산 범위 내 매물을 가격 내림차순 리스트로 반환
# ──────────────────────────────────────────────────────────────
@tool
def search_properties(
    max_price: int,
    preferred_area: str = "",
    property_type: str = "아파트",
) -> str:
    """예산 범위와 선호 조건에 맞는 매물을 검색합니다.

    Args:
        max_price: 최대 매물 가격 (만원 단위, 예: 70000 = 7억)
        preferred_area: 선호 지역 키워드 (예: "서울", "강남", "경기")
        property_type: 매물 유형 (기본: "아파트")

    Returns:
        조건에 맞는 매물 목록 JSON (가격 내림차순, 전체 반환)
    """
    results = [p for p in DUMMY_PROPERTIES if p["price"] <= max_price]

    if property_type:
        type_filtered = [p for p in results if p["type"] == property_type]
        if type_filtered:
            results = type_filtered

    area_results = []
    other_results = []
    if preferred_area:
        for p in results:
            if preferred_area in p["area"]:
                area_results.append(p)
            else:
                other_results.append(p)
    else:
        area_results = results

    area_results.sort(key=lambda x: x["price"], reverse=True)
    other_results.sort(key=lambda x: x["price"], reverse=True)

    combined = area_results + other_results

    if not combined:
        available = [p for p in DUMMY_PROPERTIES if p["price"] <= max_price * 1.3]
        if available:
            available.sort(key=lambda x: x["price"])
            hint_items = [
                f"{p['name']}({p['area']}, {p['price']:,}만원)"
                for p in available[:3]
            ]
            hint = ", ".join(hint_items)
            return json.dumps(
                {
                    "error": f"max_price {max_price:,}만원 이하 '{preferred_area}' 매물이 없습니다. "
                    f"max_price를 높이면 다음 매물이 검색 가능합니다: {hint}"
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"error": "조건에 맞는 매물이 없습니다. 예산을 높이거나 다른 지역을 검토해 주세요."},
            ensure_ascii=False,
        )

    return json.dumps(combined, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────
# Tool 2: 취득세 계산
# 2026년 개정: 생애 최초 200만원 한도 감면(2028년까지),
#              출산/양육 가구 최대 500만원 감면
# ──────────────────────────────────────────────────────────────
@tool
def calculate_acquisition_tax(
    property_price: int,
    is_first_home: bool = True,
    is_parenting_household: bool = False,
) -> str:
    """매물 가격과 구매자 조건에 따른 취득세를 계산합니다.

    2026년 기준 한국 주택 취득세:
      - 6억 이하: 1%
      - 6억 초과~9억 이하: 1%~3% 점진 적용
      - 9억 초과: 3%
    감면:
      - 생애 최초 주택 구매: 최대 200만원 감면 (2028년까지 연장)
      - 출산/양육 가구: 최대 500만원 감면

    Args:
        property_price: 매물 가격 (만원 단위)
        is_first_home: 생애 첫 주택 구매 여부
        is_parenting_household: 출산/양육 가구 여부

    Returns:
        취득세 계산 결과 JSON
    """
    if property_price <= 60000:
        tax_rate = 0.01
    elif property_price <= 90000:
        tax_rate = 0.01 + (property_price - 60000) / 30000 * 0.02
    else:
        tax_rate = 0.03

    tax_amount = int(property_price * tax_rate)

    discount = 0
    discount_desc_parts = []
    if is_first_home and property_price <= 120000:
        first_home_discount = min(tax_amount, 200)
        discount += first_home_discount
        discount_desc_parts.append(f"생애 최초 감면 {first_home_discount:,}만원")

    if is_parenting_household and property_price <= 120000:
        parenting_discount = min(tax_amount - discount, 500)
        parenting_discount = max(parenting_discount, 0)
        discount += parenting_discount
        if parenting_discount > 0:
            discount_desc_parts.append(f"출산/양육 가구 감면 {parenting_discount:,}만원")

    tax_amount -= discount
    tax_amount = max(tax_amount, 0)

    discount_desc = ", ".join(discount_desc_parts) if discount_desc_parts else "없음"

    result = {
        "property_price_만원": property_price,
        "tax_rate_percent": round(tax_rate * 100, 2),
        "tax_before_discount_만원": tax_amount + discount,
        "discount_만원": discount,
        "discount_detail": discount_desc,
        "final_tax_만원": tax_amount,
        "description": (
            f"매물가 {property_price:,}만원 기준 취득세율 {tax_rate*100:.1f}%, "
            f"취득세 {tax_amount:,}만원"
            + (f" (감면 적용: {discount_desc})" if discount > 0 else "")
        ),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────
# Tool 3: 대출 한도 계산 (LTV + 스트레스 DSR 3단계)
#
# 2026년 규제:
#   - 스트레스 DSR 3단계: 기본금리 + 스트레스 금리 1.5%를 100% 반영
#   - LTV: 수도권/규제지역 다주택자 0%, 주담대 최대 한도 6억
#          생애 최초 구입자 수도권 LTV 70%
#   - 계산 기준: '필요 대출금'(=총 필요 자금 - 보유 자금) 기준
# ──────────────────────────────────────────────────────────────
@tool
def calculate_loan_limit(
    property_price: int,
    annual_income: int,
    user_budget: int = 0,
    acquisition_tax: int = 0,
    existing_debt_payment: int = 0,
    is_first_home: bool = True,
    area: str = "서울",
) -> str:
    """보유 자금을 우선 사용하고, 부족분만 대출받는 방식으로 대출 한도를 계산합니다.

    계산 절차:
      1) 총 필요 자금 = 매물 가격 + 취득세
      2) 필요 대출금 = 총 필요 자금 - 보유 자금
      3) 대출 불필요(필요 대출금 ≤ 0) → 대출 없이 구매 가능
      4) LTV/DSR 최대 대출 한도 산출
      5) 필요 대출금 ≤ 최대 한도 → 구매 가능
      6) 월 상환액은 '필요 대출금' 기준으로 계산

    Args:
        property_price: 매물 가격 (만원 단위)
        annual_income: 연간 소득 (만원 단위)
        user_budget: 사용자 보유 자금 (만원 단위)
        acquisition_tax: 취득세 (만원 단위)
        existing_debt_payment: 기존 연간 대출 상환액 (만원 단위, 기본 0)
        is_first_home: 생애 첫 주택 구매 여부
        area: 매물 지역 (예: "서울 강남구")

    Returns:
        대출 한도 계산 결과 JSON
    """
    total_required = property_price + acquisition_tax
    needed_loan = total_required - user_budget

    if needed_loan <= 0:
        return json.dumps({
            "total_required_만원": total_required,
            "user_budget_만원": user_budget,
            "needed_loan_만원": 0,
            "loan_needed": False,
            "max_loan_limit_만원": 0,
            "monthly_repayment_만원": 0,
            "can_purchase": True,
            "remaining_funds_만원": abs(needed_loan),
            "description": (
                f"총 필요 자금 {total_required:,}만원 (매물가 {property_price:,} + 취득세 {acquisition_tax:,}) ≤ "
                f"보유 자금 {user_budget:,}만원. 대출 없이 구매 가능합니다. "
                f"잔여 자금: {abs(needed_loan):,}만원"
            ),
        }, ensure_ascii=False, indent=2)

    # === LTV 산출 ===
    is_metro = _is_metro(area)
    is_regulated = _is_regulated(area)

    if is_first_home:
        ltv_ratio = 0.70
        ltv_label = "생애최초 70%"
    elif is_regulated:
        ltv_ratio = 0.0
        ltv_label = "규제지역 다주택 0%"
    elif is_metro:
        ltv_ratio = 0.50
        ltv_label = "수도권 50%"
    else:
        ltv_ratio = 0.70
        ltv_label = "비규제 70%"

    ltv_limit = int(property_price * ltv_ratio)

    if is_metro and ltv_limit > 60000:
        ltv_limit = 60000
        ltv_label += " (수도권 주담대 최대 6억 한도 적용)"

    # === 스트레스 DSR 3단계 ===
    base_rate = 0.04
    stress_rate = 0.015
    applied_rate = base_rate + stress_rate  # 5.5%
    monthly_rate = applied_rate / 12
    n_months = 360

    dsr_ratio = 0.40
    max_annual_repayment = int(annual_income * dsr_ratio) - existing_debt_payment

    if max_annual_repayment <= 0:
        dsr_limit = 0
    else:
        max_monthly_payment = max_annual_repayment / 12
        annuity_factor = (
            ((1 + monthly_rate) ** n_months - 1)
            / (monthly_rate * (1 + monthly_rate) ** n_months)
        )
        dsr_limit = int(max_monthly_payment * annuity_factor)

    max_loan_limit = min(ltv_limit, dsr_limit)
    limiting_factor = "LTV" if ltv_limit <= dsr_limit else "DSR"

    can_purchase = needed_loan <= max_loan_limit

    # 월 상환액: '필요 대출금' 기준 (최대 한도가 아닌 실제 빌려야 하는 금액)
    actual_loan = needed_loan if can_purchase else max_loan_limit
    if actual_loan > 0:
        repay_rate = base_rate / 12  # 실제 상환은 기본금리 기준
        actual_monthly = int(
            actual_loan
            * repay_rate
            * (1 + repay_rate) ** n_months
            / ((1 + repay_rate) ** n_months - 1)
        )
    else:
        actual_monthly = 0

    shortfall = max(needed_loan - max_loan_limit, 0)

    result = {
        "total_required_만원": total_required,
        "user_budget_만원": user_budget,
        "needed_loan_만원": needed_loan,
        "loan_needed": True,
        "ltv_ratio": ltv_label,
        "ltv_limit_만원": ltv_limit,
        "dsr_ratio": f"{dsr_ratio*100:.0f}% (스트레스 DSR 3단계, 금리 {applied_rate*100:.1f}%)",
        "dsr_limit_만원": dsr_limit,
        "max_loan_limit_만원": max_loan_limit,
        "limiting_factor": limiting_factor,
        "can_purchase": can_purchase,
        "actual_loan_만원": actual_loan,
        "monthly_repayment_만원": actual_monthly,
        "loan_term": "30년",
        "interest_rate": f"연 {base_rate*100:.1f}% (스트레스 DSR 산정금리: {applied_rate*100:.1f}%)",
        "shortfall_만원": shortfall,
        "description": (
            f"총 필요 자금: {total_required:,}만원 | 보유 자금: {user_budget:,}만원 | "
            f"필요 대출금: {needed_loan:,}만원\n"
            f"LTV({ltv_label}) 한도: {ltv_limit:,}만원, "
            f"DSR(40%, 스트레스 금리 {applied_rate*100:.1f}%) 한도: {dsr_limit:,}만원 → "
            f"최대 대출 한도: {max_loan_limit:,}만원 ({limiting_factor} 제한)\n"
            + (
                f"필요 대출금 {needed_loan:,}만원 ≤ 최대 한도 {max_loan_limit:,}만원 → ✅ 구매 가능, "
                f"월 상환액: {actual_monthly:,}만원"
                if can_purchase
                else f"필요 대출금 {needed_loan:,}만원 > 최대 한도 {max_loan_limit:,}만원 → ❌ 구매 불가, "
                f"부족분: {shortfall:,}만원"
            )
        ),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)
