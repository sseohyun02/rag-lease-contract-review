# CLARA: RAG 기반 주택임대차 계약서 검토 서비스
계약서의 특약과 관련된 법령 및 판례를 검색하고, 이를 근거로 위험요소와 확인할 사항을 제시하는 서비스입니다.

- 동국대학교 2026-1 데이터사이언스캡스톤디자인
- 4인 팀

🌐 **[서비스 바로가기 →](https://dscd-ade-fe.web.app/)**　　🎬 **[시연 영상 보기 →](https://www.youtube.com/watch?v=ZcvXUgnevZw&feature=youtu.be)**

## 프로젝트 개요

CLARA(Contract Lease Analysis and Review Assistant) 는 RAG 기술을 활용하여 주택 임대차 계약서의 특약 조항을 자동 분석하는 AI 기반 법률 보조 서비스입니다.
법률 지식이 부족한 임차인이 특약 속 위험을 쉽게 파악하고, 계약 전 분쟁을 스스로 예방할 수 있도록 돕는 것을 목표로 합니다.
범용 LLM의 근거 없는 답변 대신, **법령·판례에 근거한 신뢰 가능한 분석**을 제공하며, 기존 법률 전문가 의뢰 시 평균 24시간 이상 걸리던 검토를 **1분 내로** 완료할 수 있도록 합니다.

## 주요 기능

1. **계약서 업로드 및 이력 관리**: PDF를 파싱하고 문서 이력을 조회·삭제합니다.
2. **특약 분석**: 특약별 한 줄 요약, 해석, 관련 법령·판례와 검토 설명을 제공합니다.
3. **특약 재작성 제안**: 법령 위반 가능성이 표시된 특약에 대해 사용자 요청 시 수정 문구와 이유를 생성합니다.
4. **특약 간 관계 분석 및 체크리스트**: 여러 특약이 함께 적용될 때 확인할 사항과 계약 전 점검 항목을 제시합니다.
5. **계약서 기반 질의응답**: 파싱된 계약서, 분석 결과, 재작성된 특약과 대화 이력을 참고해 답변합니다.

## 서비스 화면

| 메인 화면 (업로드) | 특약 분석 결과 |
|:---:|:---:|
| <img width="1280" height="705" alt="화면 캡처 2026-06-10 0214381" src="https://github.com/user-attachments/assets/8ed04030-a7a5-412a-b601-2b8d448845ff" /> | <img width="1280" height="705" alt="화면 캡처 2026-06-10 0216321" src="https://github.com/user-attachments/assets/283d908e-3838-4a2e-9ae4-b805017a73e5" /> |


| 특약 수정 | 체크리스트 |
|:---:|:---:|
| <img width="1280" height="706" alt="화면 캡처 2026-06-10 021738" src="https://github.com/user-attachments/assets/cc63c692-4f2c-4b50-832f-55d023781bf7" /> | <img width="1280" height="703" alt="화면 캡처 2026-06-10 021801" src="https://github.com/user-attachments/assets/1ad1eecc-e159-4e73-8a46-aeef53f5205e" /> |

| 챗봇 |
|:---:|
| <img width="1280" height="705" alt="화면 캡처 2026-06-10 0218132" src="https://github.com/user-attachments/assets/2e807ff2-12fb-4d37-bffe-f4449fa6d775" /> |


## 시스템 아키텍처

<table>
  <thead>
    <tr>
      <th>전체 파이프라인</th>
      <th>인프라 구조</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td valign="top">
<pre>
[계약서 PDF 업로드]
       ↓
[텍스트 추출 + 개인정보 마스킹]
       ↓
[특약 추출 + 쿼리 확장]
       ↓
[BM25 키워드 검색 / Dense 의미 검색]
       ↓
[하이브리드 검색 결과 결합]
       ↓
[검색된 법령·판례 내용 조회]
       ↓
[보고서 생성 (Gemini LLM)]
</pre>
      </td>
      <td valign="top">
<pre>
[사용자 브라우저 (Next.js)]
       ↓ HTTPS
[Cloud Run (FastAPI 백엔드)]
       ↓
[Cloud SQL / Cloud Storage]

Cloud SQL: 법령·판례 DB
Cloud Storage: 계약서 파일
</pre>
      </td>
    </tr>
  </tbody>
</table>


## 기술 스택

| 분야 | 기술 스택 |
|:---:|---|
| **Frontend** | ![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=next.js&logoColor=white) ![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=React&logoColor=white) ![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white) ![TailwindCSS](https://img.shields.io/badge/TailwindCSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white) |
| **Backend** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) |
| **문서 처리** | ![Document AI](https://img.shields.io/badge/Document_AI-4285F4?style=flat-square&logo=googlecloud&logoColor=white) ![Presidio](https://img.shields.io/badge/Presidio-555555?style=flat-square) |
| **AI / RAG** | ![Vertex AI Gemini](https://img.shields.io/badge/Vertex_AI_Gemini-8E75B2?style=flat-square&logo=google&logoColor=white) ![Gemini Embedding](https://img.shields.io/badge/gemini--embedding--001-4285F4?style=flat-square) ![BM25](https://img.shields.io/badge/BM25-555555?style=flat-square) ![Kiwi](https://img.shields.io/badge/Kiwi-555555?style=flat-square) |
| **DB / Storage** | ![Cloud SQL](https://img.shields.io/badge/Cloud_SQL-4285F4?style=flat-square&logo=googlecloud&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white) ![pgvector](https://img.shields.io/badge/pgvector-4169E1?style=flat-square) ![Cloud Storage](https://img.shields.io/badge/Cloud_Storage-4285F4?style=flat-square&logo=googlecloud&logoColor=white) |
| **Infra** | ![Cloud Run](https://img.shields.io/badge/Cloud_Run-4285F4?style=flat-square&logo=googlecloud&logoColor=white) ![Firebase Hosting](https://img.shields.io/badge/Firebase_Hosting-FFCA28?style=flat-square&logo=firebase&logoColor=black) |


## 폴더 구조

```
rag-lease-contract-review/
├── README.md
│
├── frontend/                        # Next.js 프론트엔드
│   ├── app/
│   │   ├── page.tsx                 # 메인 페이지 (계약서 업로드)
│   │   └── analysis/page.tsx        # 분석 결과 페이지
│   └── components/
│       ├── UploadDropzone.tsx        # 계약서 업로드 컴포넌트
│       ├── TopNavBar.tsx             # 상단 네비게이션
│       ├── RecentDocumentPanel.tsx   # 최근 분석 히스토리 패널
│       └── analysis/
│           ├── DocumentPanel.tsx     # 특약 목록 패널
│           ├── ChatBot.tsx           # 챗봇 컴포넌트
│           ├── LawSection.tsx        # 법령 섹션
│           └── PrecedentSection.tsx  # 판례 섹션
│
├── app/                             # FastAPI 백엔드 서버
│   ├── main.py                      # API 엔트리포인트
│   ├── requirements.txt             # API 서버 의존성
│   └── schemas.py                   # 요청/응답 스키마
│
├── pipeline/                        # RAG 파이프라인 핵심 모듈
│   ├── preprocessing/               # 계약서 파싱 및 전처리
│   │   ├── pipeline.py              # 전처리 파이프라인
│   │   ├── layout.py                # 레이아웃 파싱
│   │   ├── masking.py               # 개인정보 마스킹
│   │   └── rule_parser.py           # 특약 추출 규칙
│   ├── retrieval/                   # 검색 모듈
│   │   ├── bm25_retrieval.py        # BM25 키워드 검색
│   │   ├── dense_retrieval.py       # Dense 의미 검색
│   │   ├── retrieval_service.py     # 하이브리드 검색 서비스
│   │   └── query_expansion/         # 쿼리 확장
│   │       └── query_expansion.py   # LLM 기반 쿼리 확장
│   └── generation/
│       └── report_generator.py      # 보고서 생성
│
├── shared/                          # 공통 유틸리티
│   ├── config.py                    # 환경 설정
│   ├── llm/gemini_client.py         # Gemini API 클라이언트
│   ├── db/connection.py             # Cloud SQL 연결
│   └── storage/gcs_client.py        # GCS 클라이언트
│
├── data/                            # 데이터 수집 및 처리
│   ├── collectors/
│   │   ├── law_collector.py         # 법령 데이터 수집
│   │   └── case_collector.py        # 판례 데이터 수집
│   └── processors/
│       ├── law_embedder.py          # 법령 임베딩
│       └── caselaw_embedder.py      # 판례 임베딩
│
├── evaluation/                      # 성능 평가
│   ├── retrieval_eval.py            # 검색 성능 평가 (Recall)
│   └── report_eval.py               # 보고서 생성 평가 (LLM-as-a-Judge)
│
├── infra/
│   └── cloud_sql/init_schema.sql    # DB 스키마
│
├── scripts/                         # 유틸리티 스크립트
│   ├── deploy_cloud_run.sh          # Cloud Run 배포 스크립트
│   └── load_data_cloud_sql.py       # DB 데이터 로드
│
└── requirements.txt                 # Python 의존성
```


## 설치 및 실행

### 사전 요구사항

- Python 3.10+
- Node.js 20.9+
- Google Cloud CLI 및 ADC(Application Default Credentials) 인증
- Google Cloud 리소스 및 사용 권한
  - Vertex AI
  - Document AI Layout Parser
  - Cloud SQL PostgreSQL 및 pgvector
  - Cloud Storage

Cloud SQL에 `infra/cloud_sql/init_schema.sql`을 적용하고, 검색에 사용할 법령·판례 데이터와 임베딩을 별도로 준비해야 합니다.

### 백엔드 실행

프로젝트 루트에서 실행합니다.

```bash
# 의존성 설치
python -m pip install -r requirements.txt -r app/requirements.txt

# 개인정보 마스킹에 사용하는 NLP 모델 설치
python -m spacy download en_core_web_lg

# Google Cloud 인증
gcloud auth application-default login

# 환경 변수 설정
cp .env.example .env
# .env 파일에 Google Cloud 및 DB 정보 입력

# 서버 실행
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

Windows PowerShell에서는 복사 명령으로 `Copy-Item .env.example .env`를 사용합니다.

### 프론트엔드 실행

`frontend/.env.local` 파일에 백엔드 주소를 설정합니다.

```env
NEXT_PUBLIC_API_BASE=http://localhost:8080
```

새 터미널에서 실행합니다.

```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
# http://localhost:3000
```

### 환경 변수 설정

프로젝트 루트의 `.env.example` 참고:

```env
# Google Cloud
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us-central1
GCS_BUCKET=your-gcs-bucket-name

# Cloud SQL
CLOUD_SQL_CONNECTION=your-gcp-project-id:us-central1:your-db-instance
DB_NAME=your-db-name
DB_USER=your-db-user
DB_PASSWORD=your-db-password

# Gemini
GEMINI_MODEL=gemini-2.5-flash

# Document AI
DOCAI_LOCATION=us
DOCAI_PROCESSOR_ID=your-docai-processor-id
DOCAI_LAYOUT_PROCESSOR_ID=your-docai-layout-processor-id
DOCAI_LAYOUT_PROCESSOR_VERSION=pretrained-layout-parser-v1.5-2025-08-25
```

Gemini는 Vertex AI의 ADC 인증을 사용합니다. `DOCAI_PROCESSOR_ID`는 현재 설정 검증에서 필수이며, PDF 텍스트 추출에는 `DOCAI_LAYOUT_PROCESSOR_ID`를 사용합니다.


본 프로젝트는 동국대학교 데이터사이언스캡스톤디자인 수업의 졸업 프로젝트로 개발되었습니다.




