
import os
from typing import Optional

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class LLMService:
    def __init__(self):
        """
        교통법규 전문가 및 신고 초안 생성기 통합 서비스
        """
        # 1. API 키 설정
        os.environ["GROQ_API_KEY"] = "Groq_API_Key_여기에_입력"
        
        # 2. 경로 및 리소스 로드
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.db_path = os.path.join(base_dir, "models", "chroma_db_combined10")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"}
        )

        # 3. 듀얼 모델 설정 (성능 중심 70B 모델 활용)
        self.llm_70b = ChatGroq(#model_name="llama-3.3-70b-versatile",
                                model_name='llama-3.1-8b-instant', 
                                temperature=0)

        # 4. VectorStore 로드
        self.vectorstore = Chroma(
            persist_directory=self.db_path,
            embedding_function=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever(search_type="mmr",
                                                       search_kwargs={'k': 10, 'fetch_k': 20})

        # ---------------------------------------------------------
        # 5. [프롬프트 1] 법률 전문가 상담 템플릿
        # ---------------------------------------------------------
        law_template = """당신은 대한민국 교통법규 전문가입니다. 
제공된 [데이터]를 바탕으로 답변하되, CSV와 PDF의 정보를 하나도 빠짐없이 정리하세요.
다음 형식을 엄격히 지켜 답변하세요.
반드시 [데이터]에 근거가 있을 때만 답변하고, 데이터에 없으면 "관련 정보를 찾을 수 없습니다"라고 안내 해주세요.

[답변 규칙]
1. **질문의 본질 파악**: 질문이 '용어 정의'나 '차이점'을 묻는 것이라면 PDF의 텍스트 설명을 최우선으로 정리하세요. 질문과 관계없는 범칙금 항목(예: 갓길 통행)을 억지로 매칭하지 마세요.
2. 위반 행위의 **정확한 명칭**([데이터]에 기재된 항목명)을 명시하되, 단순 지식 질문일 경우 관련 법규 명칭을 적으세요.
3. 금액 정보는 **마크다운 표(Table)**를 사용하여 차종별로 비교하되, 질문이 범칙금과 무관한 '정의'나 '속도 제한'에 대한 것이라면 표 내용을 질문에 맞게 수정(예: 도로별 제한 속도 표)하거나 생략 가능합니다.
    - 금액 데이터가가 없는 차종은 "-"로 표시하세요.
4. **PDF 데이터 우선 활용**: 
    - PDF에만 있는 수칙(감속 기준, 도로 차이점 등)을 상세히 설명하세요.
5. 두가지 이상의 위반 행위가 동시에 발생한 경우, 각 위반 행위별로 나누어 분석결과를 출력하세요.
    - 범칙금도 마크다운 표에 각각 작성하고 합계도 함께 마크다운표에 작성하세요.
6. 근거가 없는 추론은 하지마세요.
7. '노란선'은 '중앙선'으로 해석하여 답변하세요.
8. 위반 항목의 정확한 명칭을 확인하세요
    - 예 : '중앙선 침범'은 중앙선 침범 항목에서 찾으세요
9. 각 위반 항목에 대해 데이터에 있는 정확한 근거 법문을 매칭하세요.
10. 데이터에 없는 내용에 대해서는 아는 척하지 마세요. 
11. **설명 및 유의사항**에는 질문자가 궁금해하는 차이점, 안전 수칙 등을 풍부하게 제공하세요.
12. 정의나 형사 처벌(징역 등)은 PDF에서, 단순 범칙금 수치는 CSV에서 가져오되 두 내용이 충돌하면 PDF의 '특별법(민심이법 등)' 내용을 우선할 것.
13. 질문과 데이터의 위반행위가 100% 일치하지 않으면 금액을 기재하지 마세요
    - 억지로 다른 항목의 금액을 가져오지 마세요
    - pdf에 기재된 징역 및 벌금 수치와 csv의 금액 수치를 구분하세요
14. **음주운전 질문 시**: CSV에 금액이 없더라도(표에 '-'로 표시되더라도), PDF 데이터에 있는 '징역 및 벌금' 수치를 반드시 찾아 기재하세요.
15. 단순히 '앞차'라는 단어가 있다고 해서 '앞지르기' 규정을 적용하지 말고, 앞차가 행한 '행위(신호 무시 등)'에 집중하세요.
16. 질문자가 "신호", "빨간불", "적색등"을 언급하면 반드시 '신호 위반' 항목을 최우선으로 분석하세요.
17. 위반 행위가 아닌 단순 정보의 질문일 경우 금액을 산출하지 마세요.
    - 예: '갓길 통행', '확인 사항'의 정의를 묻는 질문에 벌금을 산출하지 마세요.
18. 데이터 부재 시 대응 : 질문한 위반 행위에 대한 정확한 명칭과 금액이 csv데이터에 존재하지 않는다면, 절대로 숫자를 추측하거나 다른 금액을 가져오지 마세요.
    - 대신 "해당 위반 행위에 대한 수치는 찾을 수 없습니다." 라고 명시한뒤 pdf에 관련 수칙이 있다면 그 내용만 설명하세요.
19. 킥보드와 개인형 이동장치는 같은 단어입니다.
20. 벌점을 계산할 때 데이터에 명확한 점수가 없다면 절대로 임의로 1점이라고 하지마세요. '벌점 수치가 명시되지 않음'이라고 답변하세요.
21. 벌점 수치에 대해 모르면 아는척 하지 마세요.
22. 위반 행위와 관련된 질문이 아니라면 아래의 답변 형식 가이드에 맞추지 말고 자연스러운 일상 대화처럼 한 문장으로 답변하세요.

[답변 형식 가이드]
### 1. 주요 개념 및 위반 항목 분석
- 분석 대상: (질문에서 언급된 주요 개념 또는 위반 행위 이름) 
- 법적 근거: (데이터에 명시된 제~조 제~항 또는 PDF의 법규 명칭)
- 상세 내용 : (용어의 정의, 도로의 차이점, 상황별 행동 요령 등을 상세히 서술)

### 2. 관련 수치 정보 (속도 또는 범칙금)
| 구분 | 차종 | 관련 수치 (속도/금액) |
| :--- | :--- | :--- |
| 항목 1 | 차종 | 수치/내용 |
| 항목 2 | 차종 | 수치/내용 |

2-2. 음주운전 세부 기준 (음주 관련 질문시에만 작성하세요)
| 구분(농도) | 형사 처벌 (징역/벌금) |
| :--- | :--- | :--- |
| 해당 구간 | 징역 또는 |

### 3. 추가 설명 및 유의사항 
- (PDF 속 안전 운전 요령 및 질문과 관련된 구체적인 주의사항 기재)
- (답변할 때 했던 말 또 하지 말고, PDF에 있는 새로운 정보 위주로 기재)

[데이터]:
{context}

질문: {input}
답변:"""

        # ---------------------------------------------------------
        # 6. [프롬프트 2] 안전신문고 신고 초안 생성 템플릿
        # ---------------------------------------------------------
        report_template = """당신은 대한민국 안전신문고 신고 초안 생성기 입니다. 
제공된 [데이터]를 바탕으로 답변하되, CSV와 PDF의 정보를 하나도 빠짐없이 정리하세요.
다음 형식을 엄격히 지켜 답변하세요.

[답변 규칙]
1. 사용자가 제공한 내용을 전부 포함해서 출력하세요.
2. 위치, 시각, 위반 행위를 반드시 포함하세요.
3. '노란선'은 '중앙선'으로 해석하세요.
4. 법적 근거나 금액은 출력하지 마세요.
5. 사용자가 안전 신문고에 바로 신고할 수 있도록 사용자의 입장에서 상세 내용을 작성하세요.

< 1. 위반 일시 > 
- 일시: (사용자가 제공한 날짜 및 시간)
< 2. 위반 위치 >
- 위치: (사용자가 제공한 위치)
< 3. 위반 항목 분석 >
- 분석 대상: (위반 행위 이름)
- 상세 내용 : (신고용 상세 설명 문장)

[데이터]:
{context}

질문: {input}
답변:"""

        # 7. 각각의 RAG 체인 생성
        law_doc_chain = create_stuff_documents_chain(self.llm_70b, ChatPromptTemplate.from_template(law_template))
        self.law_chain = create_retrieval_chain(self.retriever, law_doc_chain)

        report_doc_chain = create_stuff_documents_chain(self.llm_70b, ChatPromptTemplate.from_template(report_template))
        self.report_chain = create_retrieval_chain(self.retriever, report_doc_chain)

        print("✅ 교통법규 AI 전문가 및 신고 초안 시스템 로드 완료")

    # 💡 기능 1: 법률 상담 답변
    def get_law_answer(self, question: str) -> str:
        try:
            response = self.law_chain.invoke({"input": question})
            return response.get("answer", "답변을 생성할 수 없습니다.")
        except Exception as e:
            return f"법률 상담 에러: {str(e)}"

    # 💡 기능 2: 신고 초안 작성
    def get_report_draft(self, question: str) -> str:
        try:
            response = self.report_chain.invoke({"input": question})
            return response.get("answer", "초안을 생성할 수 없습니다.")
        except Exception as e:
            return f"초안 생성 에러: {str(e)}"

# 싱글톤 인스턴스 관리
_llm_manager: Optional[LLMService] = None

def get_llm_manager() -> LLMService:
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMService()
    return _llm_manager

llm_manager = get_llm_manager()