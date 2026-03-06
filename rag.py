"""
rag.py - HomeFit 프로젝트의 RAG (Retrieval-Augmented Generation) 모듈

가상의 부동산 정책/규제 문서를 FAISS 벡터 DB에 임베딩하고,
Finance Expert Agent가 정책 근거를 검색할 수 있는 Retriever를 제공합니다.

구성:
  1. 정책 문서 정의 — 청킹 + 풍부한 메타데이터(chunk_id, law_name, effective_date, buyer_type 등)
  2. AzureOpenAIEmbeddings / GoogleGenerativeAIEmbeddings 클라이언트 초기화
  3. FAISS 벡터스토어 싱글턴 관리 — save_local/load_local 영속화, 문서 해시 기반 재색인
  4. Retriever 및 편의 함수 제공 — topic 커버리지 재검색 + 키워드 기반 reranking

2026년 최신 규제 반영 (적용 기준일: 2026년 1월)
"""

import os
import hashlib
import json
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

FAISS_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".faiss_cache")

REQUIRED_TOPICS = {"LTV", "DSR", "취득세"}

# ──────────────────────────────────────────────────────────────
# 2026년 기준 부동산 정책/규제 문서 (청킹 + 풍부한 메타데이터)
# ──────────────────────────────────────────────────────────────
POLICY_DOCUMENTS = [
    # === LTV 규제 — 비규제/수도권 기본 ===
    Document(
        page_content=(
            "[주택담보대출 LTV 규제 — 비규제/수도권 기본]\n"
            "담보인정비율(LTV, Loan-to-Value)은 주택 가격 대비 대출 가능 비율입니다.\n\n"
            "■ 비규제지역: LTV 70% 적용\n"
            "■ 수도권(서울·경기·인천) 일반: LTV 50% 적용\n"
            "■ 수도권/규제지역 주담대 최대 한도: 6억 원 (LTV 계산 결과가 6억 초과 시 6억으로 제한)\n"
            "적용 기준일: 2026년 1월"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "LTV",
            "chunk_id": "LTV-001",
            "law_name": "주택담보대출 LTV 규제",
            "effective_date": "2026-01",
            "buyer_type": ["1주택", "공통"],
            "category": "대출규제",
        },
    ),
    # === LTV 규제 — 규제지역/다주택자 ===
    Document(
        page_content=(
            "[주택담보대출 LTV 규제 — 규제지역/다주택자]\n"
            "규제지역(강남·서초·송파·용산 등)에서의 LTV 적용 기준입니다.\n\n"
            "■ 규제지역 1주택자: LTV 40% 적용\n"
            "■ 규제지역 다주택자: LTV 0% 적용 (주택담보대출 원칙적 불가)\n"
            "주의: 다주택자가 수도권/규제지역에서 추가 주택 구입 시 주택담보대출이 원칙적으로 불가합니다.\n"
            "적용 기준일: 2026년 1월"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "LTV",
            "chunk_id": "LTV-002",
            "law_name": "주택담보대출 LTV 규제",
            "effective_date": "2026-01",
            "buyer_type": ["다주택"],
            "category": "대출규제",
        },
    ),
    # === LTV 규제 — 생애최초 우대 ===
    Document(
        page_content=(
            "[주택담보대출 LTV 규제 — 생애최초 구매자 우대]\n"
            "생애 최초 주택 구매자에 대한 LTV 우대 혜택입니다.\n\n"
            "■ 생애 최초 주택 구매자: 수도권/규제지역이라도 LTV 70% 적용 (우대)\n"
            "■ 수도권/규제지역 주담대 최대 한도: 6억 원 (LTV 계산 결과가 6억 초과 시 6억으로 제한)\n"
            "적용 기준일: 2026년 1월"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "LTV",
            "chunk_id": "LTV-003",
            "law_name": "주택담보대출 LTV 규제",
            "effective_date": "2026-01",
            "buyer_type": ["생애최초"],
            "category": "대출규제",
        },
    ),
    # === 스트레스 DSR 3단계 — 기본 규제 ===
    Document(
        page_content=(
            "[스트레스 DSR 3단계 규제 — 기본 규정]\n"
            "DSR(Debt Service Ratio)은 연간 소득 대비 모든 대출 원리금 상환액 비율입니다.\n\n"
            "■ 기본 DSR 규제: 40% 이하 유지 필수\n"
            "■ 스트레스 DSR 3단계 (2025.7~ 전면 시행):\n"
            "  - DSR 산정 시 스트레스 금리 1.5%p를 100% 반영\n"
            "  - 예: 기본 대출금리 4.0% → DSR 산정금리 5.5% 적용\n"
            "  - 이로 인해 동일 소득 대비 대출 한도가 기존보다 약 15~20% 축소됨\n\n"
            "■ 적용 대상: 모든 금융기관의 주택담보대출 및 신용대출 포함\n"
            "■ 계산: (연간 모든 대출 원리금 상환액 ÷ 연간 소득) × 100\n"
            "적용 기준일: 2025년 7월 (3단계 전면 시행)"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "DSR",
            "chunk_id": "DSR-001",
            "law_name": "스트레스 DSR 3단계 규제",
            "effective_date": "2025-07",
            "buyer_type": ["공통"],
            "category": "대출규제",
        },
    ),
    # === 스트레스 DSR 3단계 — 우대/합산 ===
    Document(
        page_content=(
            "[스트레스 DSR 3단계 규제 — 우대 한도 및 소득 합산]\n"
            "특정 구매자에 대한 DSR 우대 및 부부 합산 소득 인정 기준입니다.\n\n"
            "■ 생애 최초 구매자: DSR 우대 한도 적용 가능 (최대 50%)\n"
            "■ 부부 합산 소득 인정: 배우자 소득 100% 합산 가능\n"
            "주의: DSR 초과 시 추가 대출이 불가합니다.\n"
            "적용 기준일: 2025년 7월 (3단계 전면 시행)"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "DSR",
            "chunk_id": "DSR-002",
            "law_name": "스트레스 DSR 3단계 규제",
            "effective_date": "2025-07",
            "buyer_type": ["생애최초"],
            "category": "대출규제",
        },
    ),
    # === 취득세 — 기본 세율 ===
    Document(
        page_content=(
            "[주택 취득세율 — 기본 세율 (2026년 개정안)]\n"
            "주택 취득 시 부과되는 취득세율은 매매가격에 따라 차등 적용됩니다.\n\n"
            "■ 6억원 이하: 취득세율 1%\n"
            "■ 6억원 초과 ~ 9억원 이하: 취득세율 1% ~ 3% (점진적 증가)\n"
            "■ 9억원 초과: 취득세율 3%\n"
            "■ 취득세 외 부가세: 지방교육세(취득세의 10%), 농어촌특별세\n"
            "참고: 위 세율은 유상 거래(매매) 기준입니다.\n"
            "적용 기준일: 2026년 1월"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "취득세",
            "chunk_id": "TAX-001",
            "law_name": "주택 취득세법",
            "effective_date": "2026-01",
            "buyer_type": ["1주택", "공통"],
            "category": "세금",
        },
    ),
    # === 취득세 — 다주택자 중과 ===
    Document(
        page_content=(
            "[주택 취득세율 — 다주택자 중과 세율]\n"
            "다주택자에 대한 취득세 중과 세율입니다.\n\n"
            "■ 2주택자: 취득세율 8%\n"
            "■ 3주택 이상: 취득세율 12%\n"
            "적용 기준일: 2026년 1월"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "취득세",
            "chunk_id": "TAX-002",
            "law_name": "주택 취득세법",
            "effective_date": "2026-01",
            "buyer_type": ["다주택"],
            "category": "세금",
        },
    ),
    # === 취득세 — 감면 혜택 ===
    Document(
        page_content=(
            "[주택 취득세율 — 감면 혜택 (2026년 개정)]\n"
            "생애 최초 및 출산/양육 가구에 대한 취득세 감면 혜택입니다.\n\n"
            "★ 2026년 개정 감면 혜택:\n"
            "  - 생애 최초 주택 구매 시: 최대 200만원 한도 감면 (2028년까지 연장 적용)\n"
            "  - 출산/양육 가구 주택 구입: 최대 500만원 감면 혜택 유지\n"
            "  - 두 감면 혜택은 12억원 이하 주택에 적용 가능\n"
            "적용 기준일: 2026년 1월"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "취득세",
            "chunk_id": "TAX-003",
            "law_name": "주택 취득세법",
            "effective_date": "2026-01",
            "buyer_type": ["생애최초"],
            "category": "세금",
        },
    ),
    # === 대출 금리 — 기본 금리 정보 ===
    Document(
        page_content=(
            "[주택담보대출 금리 안내 — 2026년 기준]\n"
            "2026년 기준 주택담보대출 평균 금리 정보입니다.\n\n"
            "■ 고정금리: 연 3.5% ~ 4.5% (5년 고정 기준)\n"
            "■ 변동금리: 연 3.0% ~ 4.0% (6개월 COFIX 연동)\n"
            "■ 혼합금리: 초기 5년 고정 후 변동 적용\n"
            "■ 대출 기간: 최소 10년 ~ 최대 40년 (원리금 균등상환)\n"
            "■ 중도상환수수료: 대출 실행 후 3년 이내 1.4%\n"
            "적용 기준일: 2026년 1월"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "대출금리",
            "chunk_id": "RATE-001",
            "law_name": "주택담보대출 금리 가이드",
            "effective_date": "2026-01",
            "buyer_type": ["공통"],
            "category": "대출금리",
        },
    ),
    # === 대출 금리 — 우대 금리 및 스트레스 DSR 산정 ===
    Document(
        page_content=(
            "[주택담보대출 금리 — 우대 금리 및 스트레스 DSR 산정]\n"
            "특정 조건 충족 시 적용되는 우대 금리와 스트레스 DSR 산정금리입니다.\n\n"
            "■ 우대 금리: 신혼부부 0.2%p, 다자녀 0.5%p 금리 인하\n"
            "★ 스트레스 DSR 산정 시 적용금리:\n"
            "  - 기본 대출금리 + 스트레스 금리 1.5%p\n"
            "  - 예: 고정금리 4.0% 기준 → DSR 산정금리 5.5%로 계산\n"
            "  - 실제 상환액은 기본 대출금리(4.0%) 기준으로 산출\n"
            "적용 기준일: 2026년 1월"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "대출금리",
            "chunk_id": "RATE-002",
            "law_name": "주택담보대출 금리 가이드",
            "effective_date": "2026-01",
            "buyer_type": ["공통"],
            "category": "대출금리",
        },
    ),
    # === 추가 비용 ===
    Document(
        page_content=(
            "[주택 구매 시 추가 비용 안내]\n"
            "주택 매매 시 매매가 외 발생하는 추가 비용 항목입니다.\n\n"
            "■ 취득세: 매매가의 1% ~ 3%\n"
            "■ 중개수수료: 매매가의 0.3% ~ 0.9%\n"
            "■ 법무사 비용: 약 50만원 ~ 100만원\n"
            "■ 이사 비용: 약 30만원 ~ 200만원\n"
            "■ 인테리어/수리비: 약 500만원 ~ 3,000만원\n"
            "■ 대출 관련 비용: 인지세(5만원), 설정비(채권최고액의 0.2%)\n"
            "총 추가 비용은 매매가의 약 3% ~ 7%로 예상해야 합니다."
        ),
        metadata={
            "source": "housing_policy",
            "topic": "추가비용",
            "chunk_id": "COST-001",
            "law_name": "주택 매매 부대비용 안내",
            "effective_date": "2026-01",
            "buyer_type": ["공통"],
            "category": "부대비용",
        },
    ),
    # === 재무분석 가이드 ===
    Document(
        page_content=(
            "[보유 자금 우선 사용 원칙 — 재무 분석 가이드]\n"
            "주택 구매 재무 분석 시 보유 자금을 최우선으로 사용하고, "
            "부족분만 대출받는 것이 올바른 분석 방식입니다.\n\n"
            "분석 절차:\n"
            "1. 총 필요 자금 산출: 매물 가격 + 취득세\n"
            "2. 필요 대출금 산출: 총 필요 자금 - 사용자 보유 자금\n"
            "3. 대출 필요 여부 판단:\n"
            "   - 필요 대출금 ≤ 0원: 대출 불필요 (월 상환액 0원, 규제 검토 생략)\n"
            "   - 필요 대출금 > 0원: LTV/DSR 규제 검토 진행\n"
            "4. 최대 대출 한도: LTV 및 스트레스 DSR 적용하여 은행 최대 한도 도출\n"
            "5. 최종 검증: 필요 대출금 ≤ 최대 대출 한도 → 구매 가능\n"
            "6. 월 상환액: 최대 한도가 아닌 실제 '필요 대출금' 기준으로 산정"
        ),
        metadata={
            "source": "housing_policy",
            "topic": "재무분석",
            "chunk_id": "GUIDE-001",
            "law_name": "재무 분석 가이드",
            "effective_date": "2026-01",
            "buyer_type": ["공통"],
            "category": "분석가이드",
        },
    ),
]


# ──────────────────────────────────────────────────────────────
# 문서 해시: 문서 내용이 변경되면 벡터스토어를 재생성
# ──────────────────────────────────────────────────────────────
def _compute_documents_hash() -> str:
    """모든 정책 문서의 page_content + metadata를 해시합니다."""
    hasher = hashlib.sha256()
    for doc in POLICY_DOCUMENTS:
        hasher.update(doc.page_content.encode("utf-8"))
        hasher.update(json.dumps(doc.metadata, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return hasher.hexdigest()


# ──────────────────────────────────────────────────────────────
# Embeddings 클라이언트
# ──────────────────────────────────────────────────────────────
def get_embeddings(provider: str = "azure"):
    """Embeddings 클라이언트를 생성합니다."""
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embed_model = os.getenv("GOOGLE_EMBED_MODEL", "gemini-embedding-001")
        return GoogleGenerativeAIEmbeddings(
            model=f"models/{embed_model}",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AOAI_ENDPOINT"),
        api_key=os.getenv("AOAI_API_KEY"),
        azure_deployment=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL"),
        api_version="2024-08-01-preview",
    )


# ──────────────────────────────────────────────────────────────
# FAISS 벡터스토어 싱글턴 — save_local/load_local 영속화
# ──────────────────────────────────────────────────────────────
_vectorstores: dict[str, FAISS] = {}


def get_vectorstore(provider: str = "azure") -> FAISS:
    """FAISS 벡터스토어를 provider별 싱글턴으로 관리합니다.
    캐시 파일이 존재하고 문서 해시가 일치하면 load_local로 로드하고,
    그렇지 않으면 새로 생성한 뒤 save_local로 저장합니다."""
    global _vectorstores
    if provider in _vectorstores:
        return _vectorstores[provider]

    embeddings = get_embeddings(provider)
    current_hash = _compute_documents_hash()
    cache_path = os.path.join(FAISS_CACHE_DIR, provider)
    hash_file = os.path.join(cache_path, "docs_hash.txt")

    cached_hash = None
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            cached_hash = f.read().strip()

    if cached_hash == current_hash and os.path.exists(os.path.join(cache_path, "index.faiss")):
        vs = FAISS.load_local(
            cache_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        vs = FAISS.from_documents(POLICY_DOCUMENTS, embeddings)
        os.makedirs(cache_path, exist_ok=True)
        vs.save_local(cache_path)
        with open(hash_file, "w") as f:
            f.write(current_hash)

    _vectorstores[provider] = vs
    return vs


def get_retriever(k: int = 3, provider: str = "azure", filter_dict: dict | None = None):
    """FAISS 기반 Retriever를 반환합니다.

    Args:
        k: 검색할 문서 수
        provider: 임베딩 프로바이더
        filter_dict: 메타데이터 필터 (예: {"buyer_type": "생애최초"})
    """
    vectorstore = get_vectorstore(provider)
    search_kwargs = {"k": k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


# ──────────────────────────────────────────────────────────────
# Reranking: 키워드 기반 관련도 점수로 재정렬
# ──────────────────────────────────────────────────────────────
def _keyword_rerank(query: str, docs: list[Document]) -> list[Document]:
    """쿼리의 핵심 키워드와 문서 내용/메타데이터의 매칭도를 기반으로 재정렬합니다."""
    keywords = set()
    for term in query.split():
        term = term.strip()
        if len(term) >= 2:
            keywords.add(term)

    domain_keywords = {"LTV", "DSR", "스트레스", "취득세", "대출", "금리", "생애", "다주택", "규제", "한도", "보유"}
    active_domain_keywords = keywords & domain_keywords
    if not active_domain_keywords:
        active_domain_keywords = domain_keywords & set(query)

    def score(doc: Document) -> float:
        text = doc.page_content + " " + doc.metadata.get("topic", "") + " " + doc.metadata.get("law_name", "")
        buyer_types = doc.metadata.get("buyer_type", [])
        if isinstance(buyer_types, list):
            text += " " + " ".join(buyer_types)

        s = 0.0
        for kw in keywords:
            if kw in text:
                s += 1.0
                if kw in active_domain_keywords:
                    s += 0.5
        return s

    return sorted(docs, key=score, reverse=True)


# ──────────────────────────────────────────────────────────────
# topic 커버리지 체크: LTV/DSR/취득세가 모두 포함되는지 확인
# ──────────────────────────────────────────────────────────────
def _check_topic_coverage(docs: list[Document], required: set[str] | None = None) -> set[str]:
    """검색 결과에서 누락된 핵심 topic을 반환합니다."""
    if required is None:
        required = REQUIRED_TOPICS
    found = {doc.metadata.get("topic", "") for doc in docs}
    return required - found


# ──────────────────────────────────────────────────────────────
# 편의 함수: 쿼리 기반 정책 문서 검색 → 텍스트 반환
# (topic 커버리지 재검색 + 키워드 reranking 적용)
# ──────────────────────────────────────────────────────────────
def retrieve_policy_context(
    query: str,
    k: int = 3,
    provider: str = "azure",
    buyer_type: str | None = None,
) -> str:
    """주어진 쿼리와 관련된 부동산 정책 문서를 검색하여
    하나의 텍스트 블록으로 반환합니다.

    개선사항:
    1. 메타데이터 필터링: buyer_type 지정 시 해당 구매자 유형 문서 우선 검색
    2. topic 커버리지 체크: LTV/DSR/취득세 문서가 모두 포함되지 않으면 k를 늘려 재검색
    3. 키워드 기반 reranking: 쿼리 키워드와의 매칭도를 기반으로 결과를 재정렬
    """
    filter_dict = None
    if buyer_type:
        filter_dict = {"buyer_type": buyer_type}

    retriever = get_retriever(k=k, provider=provider, filter_dict=filter_dict)
    docs = retriever.invoke(query)

    missing_topics = _check_topic_coverage(docs)
    if missing_topics:
        expanded_k = min(k + len(missing_topics) * 2, len(POLICY_DOCUMENTS))
        if expanded_k > k:
            expanded_retriever = get_retriever(k=expanded_k, provider=provider)
            expanded_docs = expanded_retriever.invoke(query)

            existing_ids = {doc.metadata.get("chunk_id") for doc in docs}
            for doc in expanded_docs:
                if doc.metadata.get("chunk_id") not in existing_ids:
                    if doc.metadata.get("topic") in missing_topics:
                        docs.append(doc)
                        existing_ids.add(doc.metadata.get("chunk_id"))

    docs = _keyword_rerank(query, docs)

    if not docs:
        return "관련 정책 정보를 찾을 수 없습니다."

    parts = []
    for i, doc in enumerate(docs, 1):
        meta_parts = [doc.metadata.get("topic", "")]
        if doc.metadata.get("law_name"):
            meta_parts.append(doc.metadata["law_name"])
        if doc.metadata.get("effective_date"):
            meta_parts.append(f"시행: {doc.metadata['effective_date']}")
        if doc.metadata.get("buyer_type"):
            bt = doc.metadata["buyer_type"]
            if isinstance(bt, list):
                meta_parts.append(f"대상: {'/'.join(bt)}")

        meta_str = " | ".join(filter(None, meta_parts))
        parts.append(f"--- 정책 문서 {i} ({meta_str}) ---\n{doc.page_content}")

    return "\n\n".join(parts)
