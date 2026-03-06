"""
agents.py - HomeFit 프로젝트의 에이전트 모듈

세 가지 핵심 에이전트의 LLM 호출 로직을 정의합니다:
  1. Profile Agent      - 사용자 질문 → 구조화된 프로필 추출 (Pydantic)
  2. Property Matcher    - 프로필 + 피드백 → Tool Calling으로 매물 검색/추천
  3. Finance Expert      - RAG 정책 검색 + Tool Calling 계산 → 구매 가능성 종합 검증

공통 요소:
  - GraphState (TypedDict): 에이전트 간 공유 상태 스키마
  - get_llm(): Azure OpenAI / Gemini 클라이언트 팩토리

2026년 개편 사항:
  - 보유 자금 최우선 사용 재무 로직
  - 스트레스 DSR 3단계 반영
  - 매물 후보 리스트(property_candidates) + 상위 1위 상세 분석 + Top 5 출력
"""

import os
import re
import json
import time
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from tools import (
    search_properties,
    calculate_acquisition_tax,
    calculate_loan_limit,
    calculate_required_funds,
    calculate_monthly_repayment,
)
from rag import retrieve_policy_context

load_dotenv()


def _content_to_str(content) -> str:
    """LLM 응답의 content를 문자열로 변환합니다."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            elif hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _parse_korean_amount(text: str) -> int | None:
    """한국어 금액 표현을 만원 단위 정수로 변환합니다.

    예: "3억" → 30000, "5억" → 50000, "2억5천" → 25000,
        "6천만원" → 6000, "5000만원" → 5000
    """
    m = re.search(r'(\d+)\s*억\s*(?:(\d+)\s*천)?', text)
    if m:
        result = int(m.group(1)) * 10000
        if m.group(2):
            result += int(m.group(2)) * 1000
        return result
    m = re.search(r'(\d+)\s*천\s*만?\s*원?', text)
    if m:
        return int(m.group(1)) * 1000
    m = re.search(r'(\d{4,})\s*만?\s*원?', text)
    if m:
        return int(m.group(1))
    return None


def _parse_profile_from_query(query: str) -> dict:
    """LLM 호출 실패 시 정규식으로 사용자 쿼리에서 프로필을 추출합니다."""
    profile = {
        "budget": 30000,
        "annual_income": 5000,
        "household_size": 1,
        "preferred_area": "서울",
        "property_type": "아파트",
        "is_first_home": True,
        "existing_debt_payment": 0,
    }

    budget_m = re.search(
        r'(?:자기\s*자본|보유\s*자금|자금|예산)[^,.\n]{0,5}?(\d+\s*억[^,.\n]*?(?:\d+\s*천)?|\d+\s*천\s*만|\d{4,}\s*만)',
        query,
    )
    if budget_m:
        val = _parse_korean_amount(budget_m.group(0))
        if val:
            profile["budget"] = val

    income_m = re.search(
        r'(?:연봉|연\s*소득|소득)[^,.\n]{0,5}?(\d+\s*억[^,.\n]*?|\d+\s*천\s*만?\s*원?|\d{4,}\s*만?\s*원?)',
        query,
    )
    if income_m:
        val = _parse_korean_amount(income_m.group(0))
        if val:
            profile["annual_income"] = val

    hh_m = re.search(r'(\d+)\s*인\s*(?:가족|가구)', query)
    if hh_m:
        profile["household_size"] = int(hh_m.group(1))

    areas = [
        "강남", "서초", "송파", "마포", "성동", "강서", "노원", "도봉",
        "관악", "은평", "중랑", "영등포", "용산", "서울",
        "경기", "인천", "수원", "성남", "하남", "광교",
    ]
    for area in areas:
        if area in query:
            profile["preferred_area"] = area
            break

    for ptype in ["오피스텔", "빌라", "아파트"]:
        if ptype in query:
            profile["property_type"] = ptype
            break

    return profile


# ══════════════════════════════════════════════════════════════
# LangGraph 상태 스키마
# ══════════════════════════════════════════════════════════════
class GraphState(TypedDict):
    user_query: str
    user_profile: dict
    target_property: dict              # 가장 비싼 매물 1위 (상세 분석 대상)
    property_candidates: list          # 예산 범위 내 매물 후보 리스트 (가격 내림차순)
    financial_report: str
    feedback: str
    is_valid: bool
    search_count: int
    llm_provider: str


# ══════════════════════════════════════════════════════════════
# Pydantic 모델: Structured Output용 사용자 프로필
# ══════════════════════════════════════════════════════════════
class UserProfile(BaseModel):
    """사용자의 재무 상태 및 주택 구매 요구사항"""

    budget: int = Field(
        description="보유 자기자본/현금 (만원 단위). 예: 3억 → 30000"
    )
    annual_income: int = Field(
        description="연간 총 소득 (만원 단위). 예: 연봉 6천만원 → 6000"
    )
    household_size: int = Field(
        default=1,
        description="가구 구성원 수",
    )
    preferred_area: str = Field(
        default="서울",
        description="선호 지역 (예: 서울, 강남, 마포, 경기 등)",
    )
    property_type: str = Field(
        default="아파트",
        description="선호 매물 유형 (아파트, 빌라, 오피스텔 등)",
    )
    is_first_home: bool = Field(
        default=True,
        description="생애 첫 주택 구매 여부",
    )
    existing_debt_payment: int = Field(
        default=0,
        description="기존 연간 대출 상환액 (만원 단위). 없으면 0",
    )


# ══════════════════════════════════════════════════════════════
# LLM 클라이언트 팩토리
# ══════════════════════════════════════════════════════════════
def get_llm(temperature: float = 0.0, provider: str = "azure"):
    """LLM 클라이언트를 생성합니다."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL_NAME", "gemini-2.0-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AOAI_ENDPOINT"),
        api_key=os.getenv("AOAI_API_KEY"),
        azure_deployment=os.getenv("AOAI_DEPLOY_GPT4O_MINI"),
        api_version="2024-08-01-preview",
        temperature=temperature,
    )


# ══════════════════════════════════════════════════════════════
# Agent 1: Profile Agent — 사용자 프로파일링
# ══════════════════════════════════════════════════════════════
PROFILE_SYSTEM_PROMPT = """당신은 부동산 구매 컨설팅 전문가입니다.
사용자의 자연어 질문에서 아래 정보를 정확히 추출하세요.

- budget: 보유 자기자본/현금 (만원 단위, 정수)
- annual_income: 연간 소득 (만원 단위, 정수)
- household_size: 가구 구성원 수 (정수)
- preferred_area: 선호 지역
- property_type: 선호 매물 유형 (기본: 아파트)
- is_first_home: 생애 첫 주택 여부 (기본: True)
- existing_debt_payment: 기존 연간 대출 상환액 (만원, 기본: 0)

[금액 변환 기준 — 반드시 만원 단위 정수로 변환]
  "3억" = 3억원 = 30000만원 → budget: 30000
  "5천만원" = 5000만원 → 5000
  "연봉 6천만원" = 6000만원 → annual_income: 6000
  "연봉 6천" = 6000만원 → annual_income: 6000
  "자기자본 2억5천" = 25000만원 → budget: 25000
  "1억" = 10000만원 → 10000

[가구원 수 변환 기준]
  "3인 가족" → household_size: 3
  "부부와 아이 1명" → household_size: 3
  "혼자" → household_size: 1
  "부부" → household_size: 2

명시되지 않은 항목은 합리적으로 추정하세요."""


def run_profile_agent(state: GraphState) -> dict:
    """사용자 질문에서 재무 상태 및 요구사항을 Pydantic 모델로 추출합니다.

    LLM 호출 실패 시 1회 재시도 후, 정규식 기반 fallback으로 프로필을 추출합니다.
    """
    provider = state.get("llm_provider", "azure")
    llm = get_llm(provider=provider)
    structured_llm = llm.with_structured_output(UserProfile)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PROFILE_SYSTEM_PROMPT),
        ("human", "{user_query}"),
    ])

    chain = prompt | structured_llm
    user_query = state["user_query"]

    last_error = None
    for attempt in range(2):
        try:
            profile: UserProfile = chain.invoke({"user_query": user_query})
            return {"user_profile": profile.model_dump()}
        except Exception as e:
            last_error = e
            if attempt == 0:
                time.sleep(2)

    fallback = _parse_profile_from_query(user_query)
    fallback["_parse_error"] = str(last_error)
    return {"user_profile": fallback}


# ══════════════════════════════════════════════════════════════
# Agent 2: Property Matcher Agent — 매물 후보 리스트 검색
# ══════════════════════════════════════════════════════════════
PROPERTY_SYSTEM_PROMPT = """당신은 부동산 매물 추천 전문가입니다.
사용자 프로필을 분석하여 예산 범위 내 매물을 검색하세요.

행동 규칙:
1. 반드시 search_properties 도구를 호출하여 매물을 검색하세요.
2. max_price는 아래에 제시된 '추천 max_price'를 사용하세요.
3. 검색 결과 전체를 반환하세요. 이 중에서 시스템이 자동으로 가격이 높은 순서대로 정렬하고 최적의 매물을 선택합니다.
4. 검색 결과가 있으면, 그 중 가격이 가장 높고 가구원 수에 맞는 매물을 아래 태그에 JSON으로 출력하세요:
   [SELECTED_PROPERTY]
   {"id": "...", "name": "...", ...}
   [/SELECTED_PROPERTY]"""


def _estimate_max_affordable_price(profile: dict) -> int:
    """프로필 기반으로 대출 포함 최대 구매 가능 매물가를 추정합니다."""
    budget = profile.get("budget", 0)
    annual_income = profile.get("annual_income", 0)

    ltv_max = int(budget / 0.3) if budget > 0 else 0

    # 스트레스 DSR 3단계 반영: 기본 4% + 스트레스 1.5% = 5.5%
    monthly_rate = 0.055 / 12
    n_months = 360
    annuity_factor = (
        ((1 + monthly_rate) ** n_months - 1)
        / (monthly_rate * (1 + monthly_rate) ** n_months)
    )
    existing_debt = profile.get("existing_debt_payment", 0)
    max_annual_repayment = max(annual_income * 0.4 - existing_debt, 0)
    max_loan = int((max_annual_repayment / 12) * annuity_factor)

    max_loan = min(max_loan, 60000)

    dsr_max = budget + max_loan

    return min(ltv_max, dsr_max)


def run_property_matcher_agent(state: GraphState) -> dict:
    """사용자 프로필과 피드백을 반영하여 Tool Calling으로 매물 후보 리스트를 검색합니다.

    Returns:
        {"target_property": dict, "property_candidates": list, "search_count": int}
    """
    provider = state.get("llm_provider", "azure")
    llm = get_llm(provider=provider)
    tools = [search_properties]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    profile = state.get("user_profile", {})
    feedback = state.get("feedback", "")
    search_count = state.get("search_count", 0)

    prev_property = state.get("target_property", {})
    prev_candidates = state.get("property_candidates", [])

    # ── 재검색: 기존 후보 중 이전 매물보다 저렴한 것으로 분석 대상 교체 ──
    if feedback and prev_property.get("price") and prev_candidates:
        prev_price = prev_property["price"]
        cheaper = [c for c in prev_candidates if c.get("price", 0) < prev_price]
        if cheaper:
            selected = _pick_best_property(cheaper, profile)
            return {
                "target_property": selected,
                "property_candidates": prev_candidates,
                "search_count": search_count + 1,
            }

    suggested_max = _estimate_max_affordable_price(profile)

    feedback_section = f"\n\n[이전 검색 피드백]\n{feedback}" if feedback else ""

    human_content = (
        f"사용자 프로필:\n"
        f"- 보유 자금: {profile.get('budget', 0):,}만원\n"
        f"- 연소득: {profile.get('annual_income', 0):,}만원\n"
        f"- 선호 지역: {profile.get('preferred_area', '서울')}\n"
        f"- 선호 유형: {profile.get('property_type', '아파트')}\n"
        f"- 가구원 수: {profile.get('household_size', 1)}명\n"
        f"- 현재 검색 회차: {search_count + 1}회\n\n"
        f"★ 추천 max_price: {suggested_max:,}만원 (보유자금 + 예상 대출한도 기준으로 계산됨)\n"
        f"  → search_properties 호출 시 이 값을 max_price로 사용하세요."
        f"{feedback_section}\n\n"
        f"위 조건에 맞는 매물을 검색하고 결과를 모두 반환해 주세요."
    )

    messages = [
        SystemMessage(content=PROPERTY_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    try:
        for _ in range(3):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                result = tool_map[tc["name"]].invoke(tc["args"])
                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

        candidates = _extract_all_properties_from_messages(messages)
        selected = _pick_best_property(candidates, profile)

        if not candidates:
            candidates, selected = _fallback_search_list(profile, suggested_max)

        return {
            "target_property": selected,
            "property_candidates": candidates,
            "search_count": search_count + 1,
        }

    except Exception:
        candidates, selected = _fallback_search_list(profile, suggested_max)
        return {
            "target_property": selected,
            "property_candidates": candidates,
            "search_count": search_count + 1,
        }


def _extract_all_properties_from_messages(messages: list) -> list:
    """도구 결과에서 모든 매물 리스트를 추출하여 가격 내림차순 정렬합니다."""
    all_props = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                parsed = json.loads(msg.content)
                if isinstance(parsed, list):
                    all_props.extend(parsed)
                elif isinstance(parsed, dict) and "error" not in parsed:
                    all_props.append(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

    seen_ids = set()
    unique = []
    for p in all_props:
        pid = p.get("id", "")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            unique.append(p)
        elif not pid:
            unique.append(p)

    unique.sort(key=lambda x: x.get("price", 0), reverse=True)
    return unique


def _pick_best_property(candidates: list, profile: dict) -> dict:
    """후보 리스트에서 선호 지역 + 가구원 수에 가장 적합한 매물을 선택합니다."""
    if not candidates:
        return {"error": "적합한 매물을 찾지 못했습니다."}

    preferred_area = profile.get("preferred_area", "")
    household = profile.get("household_size", 1)

    pool = candidates
    if preferred_area:
        area_pool = [p for p in candidates if preferred_area in p.get("area", "")]
        if area_pool:
            pool = area_pool

    suitable = [p for p in pool if p.get("rooms", 0) >= household]
    if suitable:
        return suitable[0]
    return pool[0]


def _fallback_search_list(profile: dict, max_price: int) -> tuple[list, dict]:
    """LLM이 도구를 호출하지 않았을 때 직접 검색하여 리스트와 최적 매물을 반환합니다."""
    try:
        raw = search_properties.invoke({
            "max_price": max_price,
            "preferred_area": profile.get("preferred_area", "서울"),
            "property_type": profile.get("property_type", "아파트"),
        })
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            parsed.sort(key=lambda x: x.get("price", 0), reverse=True)
            best = _pick_best_property(parsed, profile)
            return parsed, best
    except Exception:
        pass
    return [], {"error": "조건에 맞는 매물을 찾지 못했습니다."}


# ══════════════════════════════════════════════════════════════
# Agent 3: Finance Expert Agent — 재무 및 규제 검증
# ══════════════════════════════════════════════════════════════
FINANCE_SYSTEM_TEMPLATE = """당신은 부동산 재무 분석 전문가입니다.
아래 부동산 정책 정보를 참고하여 매물 구매 가능성을 정밀 분석하세요.

[참고 정책 정보 — RAG 검색 결과]
{policy_context}

⚠️⚠️⚠️ 절대 규칙: 모든 금액 계산(덧셈·뺄셈·곱셈·나눗셈·이자 계산)은 반드시 도구를 호출하여 수행하세요.
리포트에 기재하는 모든 숫자는 도구가 반환한 값을 그대로 복사해야 합니다.
절대로 머릿속으로 직접 사칙연산하지 마세요. 환각(Hallucination) 방지를 위한 필수 규칙입니다.

[분석 절차 — 반드시 아래 순서를 따르세요]
1단계. 취득세 계산: calculate_acquisition_tax 도구로 취득세를 산출합니다.
2단계. 총 필요 자금 & 필요 대출금: calculate_required_funds 도구를 호출하여 정확히 계산합니다.
   → 인자: property_price(매물가), acquisition_tax(1단계 결과), user_budget(보유 자금)
   → ⚠️ 절대 직접 더하거나 빼지 마세요. 도구 결과의 total_required_만원, needed_loan_만원을 그대로 사용하세요.
3단계. 대출 필요 여부 분기:
   - 2단계 결과의 can_buy_without_loan=true → 대출 불필요 (월 상환액 0원, 규제 검토 생략)
   - can_buy_without_loan=false → 4단계로 진행
4단계. 최대 대출 한도 산출: calculate_loan_limit 도구를 호출하여
   LTV 및 스트레스 DSR 3단계 규제를 적용한 은행 최대 대출 한도를 도출합니다.
   ★ 도구 호출 시 반드시 user_budget, acquisition_tax, area, is_first_home 인자를 전달하세요.
5단계. 최종 검증: 2단계의 needed_loan_만원 ≤ 4단계의 max_loan_limit_만원 → 구매 가능(PASS), 초과 → 구매 불가(FAIL)
6단계. 월 상환액 계산: calculate_monthly_repayment 도구를 호출합니다.
   → loan_amount에는 최대 대출 한도가 아닌 2단계의 '필요 대출금(needed_loan_만원)'을 전달하세요.
   → ⚠️ 절대 직접 이자 계산하지 마세요. 도구 결과의 monthly_repayment_만원을 그대로 사용하세요.

[최종 리포트 — 마크다운으로 아래 항목을 반드시 포함]
1. 매물 정보 요약
2. 취득세 계산 결과
3. 총 필요 자금 & 필요 대출금 분석
4. 대출 한도 분석 (LTV/DSR 각각, 스트레스 DSR 3단계 반영 내역)
5. 월 상환액 예상 (필요 대출금 기준)
6. 최종 구매 가능 여부 판정
7. 보유 자금 + 대출을 활용하여 구매 가능한 {{preferred_area}} 지역 추천 아파트 Top 5
   → 보유 자금과 예상 대출 한도를 합산한 총 구매력 기준으로 구매 가능한 매물 상위 5개를 표로 출력
   → 각 매물에 대해 순위·매물명·매매가·면적·지역을 표시
   → "보유 자금만으로는 부족하지만, 대출을 활용하면 구매 가능한 매물" 임을 섹션 서두에 안내

리포트 마지막에 반드시 아래 태그로 판정 결과를 표기하세요:
  [VERDICT]PASS[/VERDICT]  또는  [VERDICT]FAIL[/VERDICT]"""


def run_finance_expert_agent(state: GraphState) -> dict:
    """RAG 기반 정책 검색 + Tool Calling 수치 계산으로 매물 구매 가능성을 종합 검증합니다.

    Returns:
        {"is_valid": bool, "feedback": str, "financial_report": str}
    """
    provider = state.get("llm_provider", "azure")
    llm = get_llm(temperature=0.1, provider=provider)
    tools = [
        calculate_acquisition_tax,
        calculate_required_funds,
        calculate_loan_limit,
        calculate_monthly_repayment,
    ]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    profile = state.get("user_profile", {})
    prop = state.get("target_property", {})
    candidates = state.get("property_candidates", [])

    if prop.get("error"):
        return {
            "is_valid": False,
            "feedback": "매물 정보가 없어 재무 분석을 수행할 수 없습니다.",
            "financial_report": "## 분석 실패\n매물 정보를 찾을 수 없어 분석을 진행하지 못했습니다.",
        }

    property_price = prop.get("price", 0)
    preferred_area = profile.get("preferred_area", "서울")

    # ── RAG: 관련 정책 문서 검색 (buyer_type 메타데이터 필터링 적용) ──
    is_first_home = profile.get("is_first_home", True)
    buyer_type = "생애최초" if is_first_home else None

    rag_query = (
        f"주택 가격 {property_price}만원 구매 시 "
        f"LTV DSR 스트레스 DSR 취득세 대출 한도 규제 보유 자금 우선 사용"
    )
    try:
        policy_context = retrieve_policy_context(
            rag_query, k=4, provider=provider, buyer_type=buyer_type,
        )
    except Exception:
        policy_context = (
            "정책 정보 검색에 실패했습니다. "
            "2026년 기준 스트레스 DSR 3단계(금리 +1.5%), LTV 규제를 적용합니다."
        )

    # ── Top 5 후보 리스트 텍스트 ──
    user_budget = profile.get("budget", 0)
    top5 = candidates[:5]
    top5_text = ""
    if top5:
        top5_lines = []
        for i, c in enumerate(top5, 1):
            c_price = c.get("price", 0)
            top5_lines.append(
                f"  {i}. {c.get('name', 'N/A')} | "
                f"매매가 {c_price:,}만원 | "
                f"{c.get('size_m2', 'N/A')}m² | "
                f"{c.get('area', 'N/A')}"
            )
        top5_text = "\n".join(top5_lines)

    # ── 프롬프트 구성 ──
    system_content = FINANCE_SYSTEM_TEMPLATE.format(
        policy_context=policy_context,
    ).replace("{preferred_area}", preferred_area)

    human_content = (
        f"## 분석 대상 매물 (가격 1위)\n"
        f"- 이름: {prop.get('name', 'N/A')}\n"
        f"- 지역: {prop.get('area', 'N/A')}\n"
        f"- 가격: {property_price:,}만원\n"
        f"- 면적: {prop.get('size_m2', 'N/A')}m²\n"
        f"- 방 수: {prop.get('rooms', 'N/A')}개\n\n"
        f"## 사용자 재무 현황\n"
        f"- 보유 자금: {profile.get('budget', 0):,}만원\n"
        f"- 연소득: {profile.get('annual_income', 0):,}만원\n"
        f"- 생애 첫 주택: {'예' if profile.get('is_first_home', True) else '아니오'}\n"
        f"- 기존 연간 대출 상환액: {profile.get('existing_debt_payment', 0):,}만원\n\n"
        f"## 보유 자금 + 대출 활용 구매 가능 매물 Top 5 (리포트 7번 항목에 사용)\n"
        f"(보유 자금: {user_budget:,}만원 기준, 대출을 활용하면 구매 가능한 매물)\n"
        f"{top5_text if top5_text else '  (후보 매물 없음)'}\n\n"
        f"위 절차에 따라 1위 매물의 구매 가능 여부를 상세 분석하고,\n"
        f"리포트 7번 항목에는 위 Top 5를 마크다운 표(순위/매물명/매매가/면적/지역)로 출력하세요.\n"
        f"섹션 서두에 '보유 자금({user_budget:,}만원)과 대출 한도를 합산한 총 구매력 기준'임을 안내하세요."
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]

    try:
        for _ in range(10):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                result = tool_map[tc["name"]].invoke(tc["args"])
                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

        final_content = _content_to_str(getattr(response, "content", ""))

        is_valid = "[VERDICT]PASS[/VERDICT]" in final_content

        if not is_valid:
            budget = profile.get("budget", 0)
            feedback = (
                f"예산 초과. 매물 '{prop.get('name', '')}' "
                f"(가격: {property_price:,}만원)은 "
                f"보유 자금 {budget:,}만원으로 구매가 어렵습니다. "
                f"예산에 맞는 더 저렴한 매물을 다시 찾아주세요."
            )
        else:
            feedback = ""

        report = final_content
        for tag in ["[VERDICT]PASS[/VERDICT]", "[VERDICT]FAIL[/VERDICT]"]:
            report = report.replace(tag, "")

        return {
            "is_valid": is_valid,
            "feedback": feedback,
            "financial_report": report.strip(),
        }

    except Exception as e:
        return {
            "is_valid": False,
            "feedback": f"재무 분석 중 오류 발생: {str(e)}",
            "financial_report": f"## 분석 오류\n재무 분석 중 오류가 발생했습니다: {str(e)}",
        }
