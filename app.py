"""
app.py - HomeFit 프로젝트의 Streamlit UI

구성:
  - 좌측 Sidebar: 모델 선택, 사용자 프로필, 검색 진행 현황, 추천 매물 정보
  - 메인 화면: st.chat_message 기반 챗봇 인터페이스
  - 기능: LangGraph 스트리밍 실행으로 각 에이전트 진행 상황을 실시간 표시

실행 방법:
  cd homefit_project
  streamlit run app.py
"""

import streamlit as st
from graph import build_graph

# ──────────────────────────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HomeFit - 내 집 마련 컨설팅",
    page_icon="🏠",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# 세션 상태 초기화
# ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_state" not in st.session_state:
    st.session_state.last_state = None
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

# ──────────────────────────────────────────────────────────────
# 에이전트 노드별 진행 상태 메시지 매핑
# ──────────────────────────────────────────────────────────────
NODE_STATUS_MAP = {
    "profile_agent": {
        "running": "🔍 사용자 프로필 분석 중...",
        "done": "✅ 프로필 분석 완료",
    },
    "property_matcher": {
        "running": "🏠 매물 검색 중...",
        "done": "✅ 매물 검색 완료",
    },
    "finance_expert": {
        "running": "💰 대출 규제 확인 및 재무 분석 중...",
        "done": "✅ 재무 분석 완료",
    },
}

PROVIDER_OPTIONS = {
    "Gemini (Google)": "gemini",
    "Azure OpenAI": "azure",
}


# ──────────────────────────────────────────────────────────────
# 사이드바 렌더링
# ──────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    """좌측 사이드바에 모델 선택, 프로필, 검색 현황, 추천 매물 정보를 표시합니다."""
    with st.sidebar:
        st.header("⚙️ 모델 설정")
        selected_label = st.selectbox(
            "LLM 모델 선택",
            list(PROVIDER_OPTIONS.keys()),
            index=0,
            help="Gemini: Google API Key 필요 | Azure: VNet 내부에서만 접근 가능",
        )
        provider = PROVIDER_OPTIONS[selected_label]

        st.divider()
        st.header("📋 분석 현황")

        state = st.session_state.last_state
        if state is None:
            st.info("질문을 입력하면 분석이 시작됩니다.")
            return provider

        # ── 사용자 프로필 ──
        profile = state.get("user_profile", {})
        if profile:
            st.subheader("👤 사용자 프로필")
            col1, col2 = st.columns(2)
            col1.metric("보유 자금", f"{profile.get('budget', 0):,}만원")
            col2.metric("연간 소득", f"{profile.get('annual_income', 0):,}만원")

            col3, col4 = st.columns(2)
            col3.metric("가구원 수", f"{profile.get('household_size', '-')}명")
            col4.metric(
                "첫 주택",
                "예" if profile.get("is_first_home") else "아니오",
            )

            st.caption(f"선호 지역: {profile.get('preferred_area', '-')}")
            st.caption(f"매물 유형: {profile.get('property_type', '-')}")

        st.divider()

        # ── 검색 진행 현황 ──
        search_count = state.get("search_count", 0)
        st.subheader("🔍 검색 현황")
        st.metric("검색 횟수", f"{search_count} / 3")

        is_valid = state.get("is_valid")
        if is_valid is True:
            st.success("✅ 구매 가능 매물 확인됨")
        elif is_valid is False and search_count >= 3:
            st.error("❌ 예산 내 매물 미확인")
        elif is_valid is False:
            st.warning("🔄 재검색 진행됨")

        st.divider()

        # ── 추천 매물 (1위) ──
        prop = state.get("target_property", {})
        if prop and "error" not in prop:
            st.subheader("🏠 분석 대상 매물 (1위)")
            st.write(f"**{prop.get('name', 'N/A')}**")
            st.write(f"📍 {prop.get('area', 'N/A')}")
            st.metric("매매가", f"{prop.get('price', 0):,}만원")

            col1, col2 = st.columns(2)
            col1.write(f"면적: {prop.get('size_m2', '-')}m²")
            col2.write(f"방: {prop.get('rooms', '-')}개")

            if prop.get("floor"):
                st.write(f"층수: {prop.get('floor')}")
            st.caption(prop.get("description", ""))

        # ── 매물 후보 리스트 ──
        candidates = state.get("property_candidates", [])
        if candidates:
            st.divider()
            st.subheader(f"📋 매물 후보 ({len(candidates)}건)")
            for i, c in enumerate(candidates[:5], 1):
                price_str = f"{c.get('price', 0):,}만원"
                st.caption(f"{i}. {c.get('name', '')} — {price_str}")

    return provider


# ──────────────────────────────────────────────────────────────
# 메인 화면
# ──────────────────────────────────────────────────────────────
st.title("🏠 HomeFit — 맞춤형 내 집 마련 컨설팅")
st.caption(
    "예산, 소득, 선호 지역 등을 알려주시면 "
    "맞춤 매물 추천과 대출/세금 분석을 해드립니다."
)

llm_provider = render_sidebar()

# ── 이전 대화 이력 표시 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── 사용자 입력 처리 ──
placeholder_text = (
    "예: 자기자본 3억, 연봉 6천만원, 서울에서 3인 가족이 살 아파트를 찾고 있어요"
)

if user_input := st.chat_input(placeholder_text):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── LangGraph 실행 ──
    with st.chat_message("assistant"):
        initial_state = {
            "user_query": user_input,
            "user_profile": {},
            "target_property": {},
            "property_candidates": [],
            "financial_report": "",
            "feedback": "",
            "is_valid": False,
            "search_count": 0,
            "llm_provider": llm_provider,
        }

        status_container = st.status(
            f"HomeFit 에이전트 분석을 시작합니다... ({llm_provider.upper()})",
            expanded=True,
        )
        final_state = dict(initial_state)

        try:
            with status_container:
                for event in st.session_state.graph.stream(initial_state):
                    for node_name, node_output in event.items():
                        status_info = NODE_STATUS_MAP.get(node_name)
                        if not status_info:
                            continue

                        st.write(status_info["done"])

                        if node_name == "profile_agent":
                            p = node_output.get("user_profile", {})
                            if p:
                                st.caption(
                                    f"  → 보유자금 {p.get('budget',0):,}만원 | "
                                    f"연소득 {p.get('annual_income',0):,}만원 | "
                                    f"선호지역: {p.get('preferred_area','')}"
                                )

                        elif node_name == "property_matcher":
                            tp = node_output.get("target_property", {})
                            cands = node_output.get("property_candidates", [])
                            count = node_output.get("search_count", 0)
                            if tp and "error" not in tp:
                                st.caption(
                                    f"  → [{count}회차] 1위: "
                                    f"{tp.get('name','')} | "
                                    f"{tp.get('area','')} | "
                                    f"{tp.get('price',0):,}만원"
                                )
                            if cands:
                                st.caption(
                                    f"  → 총 {len(cands)}건 매물 후보 확보"
                                )

                        elif node_name == "finance_expert":
                            if node_output.get("is_valid"):
                                st.caption("  → ✅ 구매 가능 판정!")
                            else:
                                fb = node_output.get("feedback", "")
                                if fb:
                                    short_fb = fb[:100] + ("..." if len(fb) > 100 else "")
                                    st.caption(f"  → ⚠️ {short_fb}")

                    for node_output in event.values():
                        final_state.update(node_output)

            status_container.update(label="분석 완료!", state="complete")

        except Exception as e:
            status_container.update(label="오류 발생", state="error")
            st.error(f"에이전트 실행 중 오류가 발생했습니다: {str(e)}")

        # ── 최종 결과 표시 ──
        st.session_state.last_state = final_state

        report = final_state.get("financial_report", "")
        if report:
            st.markdown("---")
            st.markdown("### 📊 HomeFit 분석 리포트")
            st.markdown(report)

        if final_state.get("is_valid"):
            st.success(
                "🎉 축하합니다! 추천 매물은 구매 가능 범위 내에 있습니다."
            )
        else:
            search_count = final_state.get("search_count", 0)
            if search_count >= 3:
                st.warning(
                    "😔 3회 탐색 결과, 현재 조건으로 구매 가능한 매물을 찾지 못했습니다. "
                    "예산을 조정하거나 선호 지역을 확대해 보세요."
                )
            else:
                st.info("분석이 완료되었습니다. 추가 질문이 있으시면 입력해 주세요.")

        assistant_content = report if report else "분석을 완료했습니다."
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_content}
        )

    st.rerun()
