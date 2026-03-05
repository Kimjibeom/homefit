"""
graph.py - HomeFit 프로젝트의 LangGraph 정의 및 라우팅 모듈

멀티 에이전트 워크플로를 StateGraph로 구성합니다.

노드 연결:
  START → Profile Agent → Property Matcher → Finance Expert

조건부 엣지:
  Finance Expert 이후:
    - is_valid=True  → END (구매 가능 확인)
    - is_valid=False AND search_count < 3 → Property Matcher (재탐색 루프)
    - search_count >= 3 → END (탐색 횟수 초과)
"""

from langgraph.graph import StateGraph, START, END
from agents import (
    GraphState,
    run_profile_agent,
    run_property_matcher_agent,
    run_finance_expert_agent,
)

MAX_SEARCH_RETRIES = 3


def should_retry_or_end(state: GraphState) -> str:
    """Finance Expert 실행 후 재탐색 / 종료를 결정하는 라우팅 함수."""
    if state.get("is_valid", False):
        return "end"

    if state.get("search_count", 0) < MAX_SEARCH_RETRIES:
        return "retry"

    return "end"


def build_graph():
    """HomeFit 멀티 에이전트 그래프를 구성하고 컴파일합니다."""
    graph = StateGraph(GraphState)

    graph.add_node("profile_agent", run_profile_agent)
    graph.add_node("property_matcher", run_property_matcher_agent)
    graph.add_node("finance_expert", run_finance_expert_agent)

    graph.add_edge(START, "profile_agent")
    graph.add_edge("profile_agent", "property_matcher")
    graph.add_edge("property_matcher", "finance_expert")

    graph.add_conditional_edges(
        "finance_expert",
        should_retry_or_end,
        {
            "retry": "property_matcher",
            "end": END,
        },
    )

    return graph.compile()
