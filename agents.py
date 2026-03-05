"""
agents.py - HomeFit 프로젝트의 에이전트 모듈

세 가지 핵심 에이전트의 LLM 호출 로직을 정의합니다:
  1. Profile Agent      - 사용자 질문 → 구조화된 프로필 추출 (Pydantic)
  2. Property Matcher    - 프로필 + 피드백 → Tool Calling으로 매물 검색/추천
  3. Finance Expert      - RAG 정책 검색 + Tool Calling 계산 → 구매 가능성 종합 검증

공통 요소:
  - GraphState (TypedDict): 에이전트 간 공유 상태 스키마
  - get_llm(): Azure OpenAI GPT-4o-mini 클라이언트 팩토리
"""

import os
import json
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

from tools import search_properties, calculate_acquisition_tax, calculate_loan_limit
from rag import retrieve_policy_context

load_dotenv()


# ══════════════════════════════════════════════════════════════
# LangGraph 상태 스키마
# ══════════════════════════════════════════════════════════════
class GraphState(TypedDict):
    user_query: str                # 사용자의 최초 질문 원문
    user_profile: dict             # Profile Agent가 추출한 구조화 프로필
    target_property: dict          # Property Matcher가 추천한 매물 정보
    financial_report: str          # Finance Expert의 최종 마크다운 리포트
    feedback: str                  # 에이전트 간 피드백 (예산 초과 메시지 등)
    is_valid: bool                 # 최종 구매 가능 여부 플래그
    search_count: int              # 무한 루프 방지용 매물 검색 횟수
    llm_provider: str              # LLM 제공자: "azure" 또는 "gemini"


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
    """LLM 클라이언트를 생성합니다.

    Args:
        temperature: 생성 온도 (0.0~1.0)
        provider: "azure" → Azure OpenAI, "gemini" → Google Gemini
    """
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

    Returns:
        {"user_profile": dict} — UserProfile.model_dump() 결과
    """
    provider = state.get("llm_provider", "azure")
    llm = get_llm(provider=provider)
    structured_llm = llm.with_structured_output(UserProfile)

    prompt = ChatPromptTemplate.from_messages([
        ("system", PROFILE_SYSTEM_PROMPT),
        ("human", "{user_query}"),
    ])

    try:
        chain = prompt | structured_llm
        profile: UserProfile = chain.invoke({"user_query": state["user_query"]})
        return {"user_profile": profile.model_dump()}
    except Exception as e:
        # 파싱 실패 시 안전한 기본값 반환
        return {
            "user_profile": {
                "budget": 30000,
                "annual_income": 5000,
                "household_size": 1,
                "preferred_area": "서울",
                "property_type": "아파트",
                "is_first_home": True,
                "existing_debt_payment": 0,
                "_parse_error": str(e),
            }
        }


# ══════════════════════════════════════════════════════════════
# Agent 2: Property Matcher Agent — 매물 추천
# ══════════════════════════════════════════════════════════════
PROPERTY_SYSTEM_PROMPT = """당신은 부동산 매물 추천 전문가입니다.
사용자 프로필과 이전 피드백을 분석하여 최적의 매물을 검색하세요.

행동 규칙:
1. 반드시 search_properties 도구를 호출하여 매물을 검색하세요.
2. 피드백에 '예산 초과'가 포함된 경우, 이전 매물보다 확실히 저렴한 매물을 찾으세요.
3. max_price는 아래에 제시된 '추천 max_price'를 사용하세요. 재검색 시에는 이전보다 20~30% 낮춰 검색하세요.
4. 검색 결과 중 가구원 수에 맞는 방 수(가구원 수 이상)의 매물을 우선 선택하세요.
5. 선택한 매물을 아래 태그 안에 JSON으로 출력하세요:
   [SELECTED_PROPERTY]
   {"id": "...", "name": "...", ...}
   [/SELECTED_PROPERTY]"""


def _estimate_max_affordable_price(profile: dict) -> int:
    """프로필 기반으로 대출 포함 최대 구매 가능 매물가를 추정합니다."""
    budget = profile.get("budget", 0)
    annual_income = profile.get("annual_income", 0)

    # LTV 70% 기준: budget = property * 0.3 → property = budget / 0.3
    ltv_max = int(budget / 0.3) if budget > 0 else 0

    # DSR 40% 기준: 30년 4% 고정금리 원리금균등상환
    monthly_rate = 0.04 / 12
    n_months = 360
    annuity_factor = (
        ((1 + monthly_rate) ** n_months - 1)
        / (monthly_rate * (1 + monthly_rate) ** n_months)
    )
    existing_debt = profile.get("existing_debt_payment", 0)
    max_annual_repayment = max(annual_income * 0.4 - existing_debt, 0)
    max_loan = int((max_annual_repayment / 12) * annuity_factor)
    dsr_max = budget + max_loan

    return min(ltv_max, dsr_max)


def run_property_matcher_agent(state: GraphState) -> dict:
    """사용자 프로필과 피드백을 반영하여 Tool Calling으로 매물을 추천합니다.

    ReAct 패턴: LLM이 search_properties 도구를 호출 → 결과 반환 → 최적 매물 선택

    Returns:
        {"target_property": dict, "search_count": int}
    """
    provider = state.get("llm_provider", "azure")
    llm = get_llm(provider=provider)
    tools = [search_properties]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    profile = state.get("user_profile", {})
    feedback = state.get("feedback", "")
    search_count = state.get("search_count", 0)

    suggested_max = _estimate_max_affordable_price(profile)
    prev_property = state.get("target_property", {})
    if feedback and prev_property.get("price"):
        suggested_max = int(prev_property["price"] * 0.75)

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
        f"위 조건에 맞는 매물을 검색하고 최적의 1건을 추천해 주세요."
    )

    messages = [
        SystemMessage(content=PROPERTY_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    try:
        # ReAct 루프 (최대 3회 반복)
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

        selected = _extract_property_from_messages(messages)
        return {
            "target_property": selected,
            "search_count": search_count + 1,
        }

    except Exception as e:
        return {
            "target_property": {"error": f"매물 검색 중 오류: {str(e)}"},
            "search_count": search_count + 1,
        }


def _extract_property_from_messages(messages: list) -> dict:
    """에이전트 대화 메시지에서 추천 매물 정보를 추출합니다.

    추출 우선순위:
      1. [SELECTED_PROPERTY] 태그 안의 JSON
      2. 도구 결과에서 ID 매칭
      3. 도구 결과의 첫 번째 매물 (fallback)
    """
    tool_results = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                parsed = json.loads(msg.content)
                if isinstance(parsed, list):
                    tool_results.extend(parsed)
                elif isinstance(parsed, dict) and "error" not in parsed:
                    tool_results.append(parsed)
            except (json.JSONDecodeError, TypeError):
                continue

    # LLM 최종 응답에서 태그 기반 파싱
    final_msg = messages[-1]
    if hasattr(final_msg, "content") and isinstance(final_msg.content, str):
        content = final_msg.content

        if "[SELECTED_PROPERTY]" in content and "[/SELECTED_PROPERTY]" in content:
            try:
                json_str = (
                    content
                    .split("[SELECTED_PROPERTY]")[1]
                    .split("[/SELECTED_PROPERTY]")[0]
                    .strip()
                )
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                pass

        # ID 키워드 매칭 fallback
        for prop in tool_results:
            pid = prop.get("id", "")
            if pid and pid in content:
                return prop

    # 최종 fallback: 첫 번째 도구 결과
    if tool_results:
        return tool_results[0]

    return {"error": "적합한 매물을 찾지 못했습니다."}


# ══════════════════════════════════════════════════════════════
# Agent 3: Finance Expert Agent — 재무 및 규제 검증
# ══════════════════════════════════════════════════════════════
FINANCE_SYSTEM_TEMPLATE = """당신은 부동산 재무 분석 전문가입니다.
아래 부동산 정책 정보를 참고하여 매물 구매 가능성을 정밀 분석하세요.

[참고 정책 정보 — RAG 검색 결과]
{policy_context}

분석 절차:
1. calculate_acquisition_tax 도구로 취득세를 계산하세요.
2. calculate_loan_limit 도구로 대출 가능 한도를 계산하세요.
3. 필요 자기자본 = 매물가 + 취득세 - 대출 한도
4. 필요 자기자본 ≤ 보유 자금 → 구매 가능(PASS), 초과 → 구매 불가(FAIL)

최종 리포트를 마크다운으로 작성하되, 반드시 아래 항목을 포함하세요:
  - 매물 정보 요약
  - 취득세 계산 결과
  - 대출 한도 분석 (LTV/DSR 각각)
  - 필요 자기자본 vs 보유 자금 비교
  - 월 상환액 예상
  - 최종 구매 가능 여부 판정

리포트 마지막에 반드시 아래 태그로 판정 결과를 표기하세요:
  [VERDICT]PASS[/VERDICT]  또는  [VERDICT]FAIL[/VERDICT]"""


def run_finance_expert_agent(state: GraphState) -> dict:
    """RAG 기반 정책 검색 + Tool Calling 수치 계산으로 매물 구매 가능성을 종합 검증합니다.

    처리 흐름:
      1. FAISS에서 관련 정책 문서 검색 (RAG)
      2. LLM이 취득세·대출한도 계산 Tool을 호출 (ReAct 루프)
      3. LLM이 종합 분석 리포트 작성 + PASS/FAIL 판정
      4. 판정에 따라 is_valid, feedback, financial_report 업데이트

    Returns:
        {"is_valid": bool, "feedback": str, "financial_report": str}
    """
    provider = state.get("llm_provider", "azure")
    llm = get_llm(temperature=0.1, provider=provider)
    tools = [calculate_acquisition_tax, calculate_loan_limit]
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    profile = state.get("user_profile", {})
    prop = state.get("target_property", {})

    if prop.get("error"):
        return {
            "is_valid": False,
            "feedback": "매물 정보가 없어 재무 분석을 수행할 수 없습니다.",
            "financial_report": "## 분석 실패\n매물 정보를 찾을 수 없어 분석을 진행하지 못했습니다.",
        }

    property_price = prop.get("price", 0)

    # ── RAG: 관련 정책 문서 검색 ──
    rag_query = f"주택 가격 {property_price}만원 구매 시 LTV DSR 취득세 대출 한도 규제"
    try:
        policy_context = retrieve_policy_context(rag_query, k=3, provider=provider)
    except Exception:
        policy_context = (
            "정책 정보 검색에 실패했습니다. "
            "일반적인 LTV 70%, DSR 40% 규제 기준으로 분석합니다."
        )

    # ── 프롬프트 구성 ──
    system_content = FINANCE_SYSTEM_TEMPLATE.format(policy_context=policy_context)

    human_content = (
        f"## 분석 대상 매물\n"
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
        f"취득세와 대출 한도를 계산하고, 매물 구매 가능 여부를 종합 분석해 주세요."
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]

    try:
        # ── ReAct 루프: Tool 호출 반복 (최대 6회) ──
        for _ in range(6):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                result = tool_map[tc["name"]].invoke(tc["args"])
                messages.append(
                    ToolMessage(content=result, tool_call_id=tc["id"])
                )

        final_content = getattr(response, "content", "") or ""

        # ── 판정 결과 추출 ──
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

        # 태그 제거 후 리포트 정제
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
