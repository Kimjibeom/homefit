"""
tools.py - HomeFit 프로젝트의 Tool 정의 모듈

매물 검색 더미 데이터, 취득세 계산, 대출 한도 계산 함수를
LangChain @tool 데코레이터로 등록합니다.
Property Matcher Agent와 Finance Expert Agent가 Function Calling으로 호출합니다.
"""

import json
from langchain_core.tools import tool


# ──────────────────────────────────────────────────────────────
# 더미 매물 데이터 (가격: 만원 단위)
# 다양한 가격대와 지역을 커버하여 피드백 루프 테스트가 가능하도록 구성
# ──────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────
# Tool 1: 매물 검색
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
        조건에 맞는 매물 목록 JSON (최대 5건, 가격 내림차순)
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

    if not area_results and not other_results:
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

    if area_results:
        return json.dumps(area_results[:5], ensure_ascii=False, indent=2)

    return json.dumps(other_results[:5], ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────
# Tool 2: 취득세 계산
# ──────────────────────────────────────────────────────────────
@tool
def calculate_acquisition_tax(
    property_price: int,
    is_first_home: bool = True,
) -> str:
    """매물 가격과 생애 첫 주택 여부에 따른 취득세를 계산합니다.

    한국 주택 취득세 기준:
      - 6억 이하: 1%
      - 6억 초과~9억 이하: 1%~3% 점진 적용
      - 9억 초과: 3%
    생애 첫 주택 감면: 최대 200만원

    Args:
        property_price: 매물 가격 (만원 단위)
        is_first_home: 생애 첫 주택 구매 여부

    Returns:
        취득세 계산 결과 JSON
    """
    if property_price <= 60000:
        tax_rate = 0.01
    elif property_price <= 90000:
        # 6억~9억 구간: 1%에서 3%로 점진 증가
        tax_rate = 0.01 + (property_price - 60000) / 30000 * 0.02
    else:
        tax_rate = 0.03

    tax_amount = int(property_price * tax_rate)

    discount = 0
    if is_first_home and property_price <= 120000:
        discount = min(tax_amount, 200)
        tax_amount -= discount

    result = {
        "property_price_만원": property_price,
        "tax_rate_percent": round(tax_rate * 100, 2),
        "tax_before_discount_만원": tax_amount + discount,
        "first_home_discount_만원": discount,
        "final_tax_만원": tax_amount,
        "description": (
            f"매물가 {property_price:,}만원 기준 취득세율 {tax_rate*100:.1f}%, "
            f"취득세 {tax_amount:,}만원"
            + (f" (생애 첫 주택 감면 {discount:,}만원 적용)" if discount > 0 else "")
        ),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────
# Tool 3: 대출 한도 계산 (LTV + DSR)
# ──────────────────────────────────────────────────────────────
@tool
def calculate_loan_limit(
    property_price: int,
    annual_income: int,
    existing_debt_payment: int = 0,
) -> str:
    """LTV(70%) 및 DSR(40%) 규제를 적용하여 대출 가능 한도를 계산합니다.

    대출 조건 가정: 30년 만기, 연 4% 고정금리, 원리금 균등상환

    Args:
        property_price: 매물 가격 (만원 단위)
        annual_income: 연간 소득 (만원 단위)
        existing_debt_payment: 기존 연간 대출 상환액 (만원 단위, 기본 0)

    Returns:
        대출 한도 계산 결과 JSON (LTV/DSR 각각의 한도 및 최종 한도 포함)
    """
    # === LTV 기반 한도 (비규제지역 70%) ===
    ltv_ratio = 0.70
    ltv_limit = int(property_price * ltv_ratio)

    # === DSR 기반 한도 (40%, 30년 고정 4%) ===
    annual_rate = 0.04
    monthly_rate = annual_rate / 12
    n_months = 360  # 30년

    max_annual_repayment = int(annual_income * 0.40) - existing_debt_payment

    if max_annual_repayment <= 0:
        dsr_limit = 0
        monthly_repayment = 0
    else:
        max_monthly_payment = max_annual_repayment / 12
        # 대출 원금 역산: P = PMT × [(1+r)^n − 1] / [r × (1+r)^n]
        annuity_factor = (
            ((1 + monthly_rate) ** n_months - 1)
            / (monthly_rate * (1 + monthly_rate) ** n_months)
        )
        dsr_limit = int(max_monthly_payment * annuity_factor)
        monthly_repayment = max_monthly_payment

    final_limit = min(ltv_limit, dsr_limit)
    limiting_factor = "LTV" if ltv_limit <= dsr_limit else "DSR"
    required_equity = property_price - final_limit

    # 최종 대출 한도 기준 실제 월 상환액 재계산
    if final_limit > 0:
        actual_monthly = int(
            final_limit
            * monthly_rate
            * (1 + monthly_rate) ** n_months
            / ((1 + monthly_rate) ** n_months - 1)
        )
    else:
        actual_monthly = 0

    result = {
        "property_price_만원": property_price,
        "ltv_ratio": f"{ltv_ratio * 100:.0f}%",
        "ltv_limit_만원": ltv_limit,
        "dsr_ratio": "40%",
        "dsr_limit_만원": dsr_limit,
        "annual_income_만원": annual_income,
        "existing_debt_payment_만원": existing_debt_payment,
        "final_loan_limit_만원": final_limit,
        "limiting_factor": limiting_factor,
        "required_equity_만원": required_equity,
        "monthly_repayment_만원": actual_monthly,
        "loan_term": "30년",
        "interest_rate": "연 4.0%",
        "description": (
            f"LTV({ltv_ratio*100:.0f}%) 한도: {ltv_limit:,}만원, "
            f"DSR(40%) 한도: {dsr_limit:,}만원 → "
            f"최종 대출 한도: {final_limit:,}만원 ({limiting_factor} 제한), "
            f"필요 자기자본: {required_equity:,}만원, "
            f"월 상환액: {actual_monthly:,}만원"
        ),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)
