# 🏠 HomeFit — 맞춤형 내 집 마련 AI 컨설팅

**LangGraph 멀티 에이전트 기반 부동산 구매 컨설팅 시스템**

사용자의 자연어 질문에서 재무 상태와 선호 조건을 파악하고, 매물 검색 → 대출/세금 분석 → 구매 가능성 판정까지 자동으로 수행하는 AI 컨설팅 서비스입니다.

> 2026년 한국 부동산 규제(스트레스 DSR 3단계, LTV, 취득세 개정안)를 반영합니다.

---

## 목차

- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [에이전트 워크플로](#에이전트-워크플로)
- [프로젝트 구조](#프로젝트-구조)
- [모듈 상세 설명](#모듈-상세-설명)
- [적용 규제 및 정책](#적용-규제-및-정책)
- [설치 및 실행](#설치-및-실행)
- [사용 예시](#사용-예시)
- [기술 스택](#기술-스택)
- [라이선스](#라이선스)

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **자연어 프로파일링** | 사용자의 자유 형식 질문에서 예산, 소득, 가구원 수, 선호 지역 등을 자동 추출 |
| **매물 검색 및 추천** | 예산 범위 내 매물을 Tool Calling으로 검색하고 최적 매물을 선정 |
| **재무 분석 리포트** | 취득세, 대출 한도(LTV/DSR), 월 상환액을 도구 기반으로 정밀 계산 |
| **RAG 기반 정책 검증** | FAISS 벡터 DB에서 관련 부동산 정책 문서를 검색하여 근거 기반 분석 |
| **자동 재탐색 루프** | 구매 불가 판정 시 최대 3회까지 더 저렴한 매물을 자동 재검색 |
| **듀얼 LLM 지원** | Azure OpenAI(GPT-4o-mini)와 Google Gemini 중 선택 가능 |
| **실시간 진행 표시** | Streamlit 스트리밍으로 각 에이전트의 분석 진행 상황을 실시간 확인 |

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                        │
│  ┌──────────┐  ┌──────────────────────────────────────────┐    │
│  │ Sidebar  │  │           Chat Interface                  │    │
│  │ - 모델선택│  │  사용자 입력 → 에이전트 실행 → 리포트 출력│    │
│  │ - 프로필  │  └──────────────────────────────────────────┘    │
│  │ - 검색현황│                      │                           │
│  │ - 추천매물│                      ▼                           │
│  └──────────┘         LangGraph 스트리밍 실행                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               LangGraph StateGraph (graph.py)                   │
│                                                                 │
│  START → [Profile Agent] → [Property Matcher] → [Finance Expert]│
│                                     ▲                │          │
│                                     │    FAIL & <3회 │          │
│                                     └────────────────┘          │
│                                          PASS or ≥3회 → END     │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐      ┌──────────────┐     ┌──────────────┐
   │agents.py │      │  tools.py    │     │   rag.py     │
   │ 에이전트  │      │  Tool 정의   │     │  RAG 모듈    │
   │ LLM 호출  │─────▶│ - 매물 검색  │     │ - 정책 문서  │
   │ 로직      │      │ - 취득세     │     │ - FAISS DB   │
   └──────────┘      │ - 대출 한도  │     │ - Retriever  │
                     │ - 필요 자금  │     └──────────────┘
                     │ - 월 상환액  │
                     └──────────────┘
```

---

## 에이전트 워크플로

### Agent 1: Profile Agent — 사용자 프로파일링

- **역할**: 자연어 질문에서 구조화된 사용자 프로필 추출
- **방식**: Pydantic `with_structured_output`을 사용한 Structured Output
- **추출 항목**: 보유 자금, 연소득, 가구원 수, 선호 지역, 매물 유형, 생애 첫 주택 여부, 기존 대출 상환액

### Agent 2: Property Matcher — 매물 검색/추천

- **역할**: 프로필과 피드백을 반영하여 예산 범위 내 매물 후보 리스트 검색
- **방식**: Tool Calling으로 `search_properties` 도구 호출
- **로직**: 보유 자금 + 예상 대출 한도 기반 최대 구매가 추정 → 매물 검색 → 가격 내림차순 정렬 → 최적 매물 선정
- **재검색**: 피드백 수신 시 이전 매물 대비 25% 저렴한 가격대로 재탐색

### Agent 3: Finance Expert — 재무 및 규제 검증

- **역할**: RAG 기반 정책 검색 + Tool Calling 수치 계산으로 구매 가능성 종합 검증
- **분석 절차**:
  1. 취득세 계산 (`calculate_acquisition_tax`)
  2. 총 필요 자금 & 필요 대출금 산출 (`calculate_required_funds`)
  3. 대출 필요 여부 판단 (보유 자금만으로 구매 가능한지 확인)
  4. LTV/DSR 기반 최대 대출 한도 산출 (`calculate_loan_limit`)
  5. 필요 대출금 ≤ 최대 한도 여부로 구매 가능성 판정
  6. 월 상환액 계산 (`calculate_monthly_repayment`)
- **출력**: 마크다운 리포트 + `[VERDICT]PASS/FAIL[/VERDICT]` 태그

### 조건부 라우팅 (재탐색 루프)

```
Finance Expert 판정 결과:
  ├─ PASS (구매 가능)     → END
  ├─ FAIL & 검색 < 3회   → Property Matcher (더 저렴한 매물 재검색)
  └─ FAIL & 검색 ≥ 3회   → END (탐색 횟수 초과)
```

---

## 프로젝트 구조

```
homefit/
├── app.py              # Streamlit UI (채팅 인터페이스, 사이드바)
├── graph.py            # LangGraph StateGraph 정의 및 조건부 라우팅
├── agents.py           # 3개 에이전트 LLM 호출 로직 (Profile, Matcher, Finance)
├── tools.py            # LangChain @tool 도구 (매물 검색, 세금/대출 계산)
├── rag.py              # RAG 모듈 (정책 문서, FAISS 벡터스토어, Retriever)
├── requirements.txt    # Python 의존성
├── .env                # API 키 설정 (Azure OpenAI, Google Gemini)
├── .gitignore
└── LICENSE             # MIT License
```

---

## 모듈 상세 설명

### `app.py` — Streamlit UI

| 구성 요소 | 설명 |
|-----------|------|
| 메인 채팅 | `st.chat_message` 기반 대화형 인터페이스 |
| 좌측 사이드바 | 모델 선택, 사용자 프로필, 검색 횟수, 추천 매물(1위), 후보 리스트(Top 5) |
| 진행 상태 | `st.status` + `graph.stream()`으로 에이전트별 실시간 진행 표시 |
| 결과 표시 | 재무 분석 리포트(마크다운) + 구매 가능 여부 배지 |

### `agents.py` — 에이전트 모듈

| 에이전트 | 함수 | LLM 기법 | 도구 |
|---------|------|---------|------|
| Profile Agent | `run_profile_agent()` | Structured Output (Pydantic) | 없음 |
| Property Matcher | `run_property_matcher_agent()` | Tool Calling | `search_properties` |
| Finance Expert | `run_finance_expert_agent()` | Tool Calling + RAG | 4개 계산 도구 |

**공유 상태 스키마 (`GraphState`)**:
- `user_query`, `user_profile`, `target_property`, `property_candidates`
- `financial_report`, `feedback`, `is_valid`, `search_count`, `llm_provider`

### `tools.py` — Tool 정의

| 도구 | 용도 |
|------|------|
| `search_properties` | 가격·지역·유형 조건으로 매물 검색 (30건 더미 데이터) |
| `calculate_acquisition_tax` | 취득세 계산 (생애 최초/출산 가구 감면 반영) |
| `calculate_loan_limit` | LTV + 스트레스 DSR 3단계 기반 최대 대출 한도 산출 |
| `calculate_required_funds` | 총 필요 자금 & 필요 대출금 계산 (LLM 산수 환각 방지) |
| `calculate_monthly_repayment` | 원리금균등상환 방식 월 상환액 계산 |

### `rag.py` — RAG 모듈

| 구성 요소 | 설명 |
|-----------|------|
| 정책 문서 (6건) | LTV 규제, 스트레스 DSR, 취득세, 대출 금리, 추가 비용, 보유 자금 우선 사용 원칙 |
| 임베딩 | Azure OpenAI `text-embedding-3-small` 또는 Google `gemini-embedding-001` |
| 벡터스토어 | FAISS (provider별 싱글턴) |
| Retriever | 유사도 기반 Top-k 검색 (기본 k=3~4) |

---

## 적용 규제 및 정책

> 2026년 1월 기준 한국 부동산 규제를 반영합니다.

### LTV (담보인정비율)

| 조건 | LTV |
|------|-----|
| 생애 최초 구매자 | 70% (수도권/규제지역 포함) |
| 수도권 일반 | 50% |
| 규제지역(강남·서초·송파·용산) 다주택자 | 0% |
| 비규제지역 | 70% |
| 수도권 주담대 최대 한도 | 6억원 |

### 스트레스 DSR 3단계 (2025.7~ 전면 시행)

- DSR 상한: 40%
- 스트레스 금리: 기본금리 + **1.5%p** (100% 반영)
- 예시: 기본 4.0% → DSR 산정금리 **5.5%** 적용
- 실제 상환액은 기본금리(4.0%) 기준으로 산출

### 취득세

| 매매가 | 세율 |
|--------|------|
| 6억 이하 | 1% |
| 6억~9억 | 1%~3% (점진 적용) |
| 9억 초과 | 3% |
| 생애 최초 감면 | 최대 200만원 (12억 이하, 2028년까지) |
| 출산/양육 가구 감면 | 최대 500만원 (12억 이하) |

---

## 설치 및 실행

### 사전 요구사항

- Python 3.10+
- Azure OpenAI 또는 Google Gemini API 키

### 1. 의존성 설치

```bash
cd homefit
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성합니다:

```env
# Azure OpenAI
AOAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AOAI_API_KEY=your-api-key
AOAI_DEPLOY_GPT4O_MINI=gpt-4o-mini
AOAI_DEPLOY_EMBED_3_SMALL=text-embedding-3-small

# Google Gemini (선택)
GOOGLE_API_KEY=your-google-api-key
GOOGLE_MODEL_NAME=gemini-3.5-flash
GOOGLE_EMBED_MODEL=gemini-embedding-001
```

### 3. 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속합니다.

---

## 사용 예시

### 입력 예시

```
자기자본 3억, 연봉 6천만원, 서울에서 3인 가족이 살 아파트를 찾고 있어요
```

### 처리 흐름

1. **Profile Agent**: 보유 자금 30,000만원, 연소득 6,000만원, 가구원 3명, 서울, 아파트 추출
2. **Property Matcher**: 서울 소재 예산 범위 내 아파트 검색 → 가격 내림차순 정렬 → 최적 매물 선정
3. **Finance Expert**: 취득세 → 필요 대출금 → LTV/DSR 검증 → 월 상환액 → 구매 가능성 판정
4. **결과**: 마크다운 리포트 (매물 정보, 세금, 대출 분석, Top 5 추천 매물 포함)

### 리포트 주요 항목

- 매물 정보 요약
- 취득세 계산 결과
- 총 필요 자금 & 필요 대출금 분석
- 대출 한도 분석 (LTV/DSR, 스트레스 DSR 반영)
- 월 상환액 예상
- 최종 구매 가능 여부 판정
- 보유 자금 + 대출 활용 구매 가능 매물 Top 5

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **프레임워크** | LangChain, LangGraph |
| **LLM** | Azure OpenAI (GPT-4o-mini), Google Gemini (3.5 Flash) |
| **임베딩** | Azure `text-embedding-3-small`, Google `gemini-embedding-001` |
| **벡터 DB** | FAISS (CPU) |
| **UI** | Streamlit |
| **데이터 검증** | Pydantic v2 |
| **환경 관리** | python-dotenv |

---

## 라이선스

MIT License © 2026 Kimjibeom
