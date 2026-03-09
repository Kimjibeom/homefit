"""
graph.py - HomeFit 프로젝트의 LangGraph 정의 모듈

멀티 에이전트 워크플로를 StateGraph로 구성합니다.

노드 연결 (단방향 리니어 플로우):
  START → Profile Agent → Property Matcher → Finance Expert → END
"""

from langgraph.graph import StateGraph, START, END
from agents import (
    GraphState,
    run_profile_agent,
    run_property_matcher_agent,
    run_finance_expert_agent,
)


def build_graph():
    """HomeFit 멀티 에이전트 그래프를 구성하고 컴파일합니다."""
    graph = StateGraph(GraphState)

    graph.add_node("profile_agent", run_profile_agent)
    graph.add_node("property_matcher", run_property_matcher_agent)
    graph.add_node("finance_expert", run_finance_expert_agent)

    graph.add_edge(START, "profile_agent")
    graph.add_edge("profile_agent", "property_matcher")
    graph.add_edge("property_matcher", "finance_expert")
    graph.add_edge("finance_expert", END)

    return graph.compile()
