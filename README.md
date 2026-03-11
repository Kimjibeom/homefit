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
| **실시간 매물 검색** | 네이버 부동산 API(세션 인증 + JWT 토큰)로 단지 내 최소 평형의 실거래가를 조회하고, 입지 점수 기준 Top 5 매물을 추천 |
| **TOP 5 일괄 재무 분석** | 취득세, 대출 한도(LTV/DSR), 월 상환액을 TOP 5 매물 각각에 대해 도구 기반으로 정밀 계산 |
| **RAG 기반 정책 검증** | FAISS 벡터 DB에서 관련 부동산 정책 문서를 검색하여 근거 기반 분석 (topic 커버리지 + 키워드 리랭킹) |
| **듀얼 LLM 지원** | Azure OpenAI(GPT-4o-mini)와 Google Gemini 중 선택 가능 |
| **실시간 진행 표시** | Streamlit 스트리밍으로 각 에이전트의 분석 진행 상황을 실시간 확인 |
| **환각 방지 설계** | 모든 금액 계산(세금, 대출, 상환액)을 LLM이 아닌 전용 도구가 수행 |

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
│    START → [Profile Agent] → [Property Matcher] → [Finance Expert] → END
│                                                                 │
│    단방향 리니어 플로우: 프로파일링 → 매물 검색 → 재무 검증       │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐      ┌──────────────┐     ┌──────────────┐
   │agents.py │      │  tools.py    │     │   rag.py     │
   │ 에이전트  │      │  Tool 정의   │     │  RAG 모듈    │
   │ LLM 호출  │─────▶│ - 매물 검색  │     │ - 정책 문서  │
   │ 로직      │      │  (평형별    │     │   (12건)     │
   └──────────┘      │   실거래가) │     │ - FAISS DB   │
                     │ - 취득세     │     │ - 캐시 영속화│
                     │ - 대출 한도  │     │ - 리랭킹     │
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
- **Fallback**: LLM 호출 실패 시 1회 재시도 후 정규식 기반 프로필 추출

### Agent 2: Property Matcher — 매물 검색/추천

- **역할**: 프로필 기반으로 예산 범위 내 매물 후보 리스트를 검색하여 TOP 5 반환
- **방식**: Tool Calling으로 `search_properties` 도구 1회 호출
- **로직**:
  1. 보유 자금 + 예상 대출 한도(스트레스 DSR 반영) 기반 `max_price` 자동 산정
  2. 네이버 부동산 API로 20개 매물 각각의 최소 평형 실거래가 조회 (overview → prices/real 2단계)
  3. 예산 이하 매물 중 입지 점수 기준 상위 5개 선정
- **Fallback**: LLM이 도구를 호출하지 않을 경우 직접 검색 수행

### Agent 3: Finance Expert — TOP 5 일괄 재무 검증

- **역할**: RAG 기반 정책 검색 + Tool Calling 수치 계산으로 TOP 5 매물 각각의 구매 가능성을 종합 검증
- **분석 절차** (TOP 5 매물 각각에 대해 반복):
  1. 취득세 계산 (`calculate_acquisition_tax`)
  2. 총 필요 자금 & 필요 대출금 산출 (`calculate_required_funds`)
  3. 대출 필요 여부 판단 (보유 자금만으로 구매 가능한지 확인)
  4. LTV/DSR 기반 최대 대출 한도 산출 (`calculate_loan_limit`)
  5. 필요 대출금 ≤ 최대 한도 여부로 구매 가능성 판정
- **출력**: 마크다운 종합 리포트 (표 형태) + `[VERDICT]PASS/FAIL[/VERDICT]` 태그
- **리포트 항목**: 순위, 매물명, 입지점수, 최소평형 실거래가, 취득세, 필요 대출금, 구매 가능 여부

---

## 프로젝트 구조

```
homefit/
├── app.py              # Streamlit UI (채팅 인터페이스, 사이드바)
├── graph.py            # LangGraph StateGraph 정의 (단방향 리니어 플로우)
├── agents.py           # 3개 에이전트 LLM 호출 로직 (Profile, Matcher, Finance)
├── tools.py            # LangChain @tool 도구 (매물 검색, 세금/대출 계산)
├── rag.py              # RAG 모듈 (정책 문서 12건, FAISS 벡터스토어, 리랭킹)
├── requirements.txt    # Python 의존성
├── .env                # API 키 설정 (Azure OpenAI, Google Gemini)
├── .faiss_cache/       # FAISS 벡터스토어 캐시 (provider별 저장, 해시 기반 재색인)
├── .gitignore
├── LICENSE             # MIT License
└── README.md
```

---

## 모듈 상세 설명

### `app.py` — Streamlit UI

| 구성 요소 | 설명 |
|-----------|------|
| 메인 채팅 | `st.chat_message` 기반 대화형 인터페이스 |
| 좌측 사이드바 | 모델 선택(Azure/Gemini), 사용자 프로필, 검색 횟수, 분석 대상 매물(1위), 후보 리스트(Top 5) |
| 진행 상태 | `st.status` + `graph.stream()`으로 에이전트별 실시간 진행 표시 |
| 결과 표시 | 재무 분석 리포트(마크다운 표) + 구매 가능 여부 배지 |

### `graph.py` — LangGraph 워크플로

단방향 리니어 플로우로 구성:

```
START → profile_agent → property_matcher → finance_expert → END
```

### `agents.py` — 에이전트 모듈

| 에이전트 | 함수 | LLM 기법 | 도구 |
|---------|------|---------|------|
| Profile Agent | `run_profile_agent()` | Structured Output (Pydantic) | 없음 (Fallback: 정규식) |
| Property Matcher | `run_property_matcher_agent()` | Tool Calling | `search_properties` |
| Finance Expert | `run_finance_expert_agent()` | Tool Calling + RAG | 4개 계산 도구 (최대 20회 반복) |

**공유 상태 스키마 (`GraphState`)**:
- `user_query`, `user_profile`, `target_property`, `property_candidates`
- `financial_report`, `feedback`, `is_valid`, `search_count`, `llm_provider`

**LLM 팩토리 (`get_llm`)**:
- `azure`: Azure OpenAI (GPT-4o-mini)
- `gemini`: Google Gemini (기본 gemini-2.0-flash)

### `tools.py` — Tool 정의

| 도구 | 용도 |
|------|------|
| `search_properties` | 네이버 부동산 API로 20개 매물의 최소 평형 실거래가 조회 후, 예산 이하 매물을 입지 점수 기준 Top 5 반환 |
| `calculate_acquisition_tax` | 취득세 계산 (생애 최초 200만원 / 출산·양육 가구 500만원 감면 반영) |
| `calculate_loan_limit` | 보유 자금 우선 사용 → 부족분 대출: LTV + 스트레스 DSR 3단계 기반 최대 대출 한도 산출 |
| `calculate_required_funds` | 총 필요 자금 & 필요 대출금 계산 (LLM 산수 환각 방지) |
| `calculate_monthly_repayment` | 원리금균등상환 방식 월 상환액 계산 |

**더미 매물 데이터**: 서울 성북구·강북구·노원구 소재 아파트 20건 (각 매물에 `hscp_no` 포함)

**실시간 가격 조회** (세션 인증 기반):

| 내부 함수 | 역할 |
|-----------|------|
| `_get_naver_session()` | 네이버 부동산 메인 페이지 방문 → 쿠키 + JWT 토큰 자동 획득 (1시간 TTL 캐싱) |
| `_naver_api_get()` | API GET 요청 + HTTP 429 지수 백오프 재시도 (최대 3회) |
| `_fetch_realtime_price(hscp_no)` | 단지 내 최소 평형의 최신 매매 실거래가 조회 |

`_fetch_realtime_price` 조회 절차:
1. **overview API** (`/api/complexes/overview/{hscp_no}`) → 평형(`pyeongs`) 목록 획득
2. 전용면적(`exclusiveArea`)이 가장 작은 평형의 `areaNo` 선택
3. **prices/real API** (`/api/complexes/{hscp_no}/prices/real?areaNo={areaNo}`) → 해당 평형의 최신 매매 실거래가 반환
4. **폴백**: 평형별 실거래 이력이 없으면 overview 대표 실거래가 → `minPrice` 순으로 폴백

### `rag.py` — RAG 모듈

| 구성 요소 | 설명 |
|-----------|------|
| 정책 문서 (12건) | LTV 규제(3건), 스트레스 DSR(2건), 취득세(3건), 대출 금리(2건), 추가 비용(1건), 재무분석 가이드(1건) |
| 메타데이터 | `chunk_id`, `law_name`, `effective_date`, `buyer_type`, `category`, `topic` |
| 임베딩 | Azure OpenAI `text-embedding-3-small` 또는 Google `gemini-embedding-001` |
| 벡터스토어 | FAISS (provider별 싱글턴, `.faiss_cache/`에 영속화) |
| 재색인 | 문서 해시(`SHA-256`) 기반 — 문서 내용 변경 시 자동 재생성 |
| topic 커버리지 | LTV/DSR/취득세 문서가 모두 포함되지 않으면 k를 늘려 재검색 |
| 리랭킹 | 쿼리 키워드 + 도메인 키워드 매칭도 기반 결과 재정렬 |
| Retriever | 유사도 기반 Top-k 검색 (기본 k=3~4) + `buyer_type` 필터링 |

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
GOOGLE_MODEL_NAME=gemini-2.0-flash
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
2. **Property Matcher**: 네이버 부동산 API로 최소 평형 실거래가 조회 → 예산 내 입지 점수 기준 TOP 5 선정
3. **Finance Expert**: TOP 5 매물 각각에 대해 취득세 → 필요 대출금 → LTV/DSR 검증 → 종합 리포트 출력
4. **결과**: 마크다운 리포트 (TOP 5 비교표 + 구매 가능 여부 판정)

### 리포트 주요 항목

- 사용자 재무 현황 요약
- TOP 5 매물 종합 분석 결과표 (순위, 매물명, 입지점수, 최소평형 실거래가, 취득세, 필요 대출금, 구매 가능 여부)
- 구매 가능 매물에 대한 상세 설명
- 종합 의견 및 추천

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| **프레임워크** | LangChain, LangGraph |
| **LLM** | Azure OpenAI (GPT-4o-mini), Google Gemini (2.0 Flash) |
| **임베딩** | Azure `text-embedding-3-small`, Google `gemini-embedding-001` |
| **벡터 DB** | FAISS (CPU, 로컬 캐시 영속화) |
| **실시간 데이터** | 네이버 부동산 API (세션 인증 + JWT, 평형별 실거래가 조회) |
| **UI** | Streamlit |
| **데이터 검증** | Pydantic v2 |
| **환경 관리** | python-dotenv |

---

## 라이선스

MIT License © 2026 Kimjibeom
