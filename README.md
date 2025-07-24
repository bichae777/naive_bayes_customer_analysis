# Naive Bayes Customer Data Quality Analysis
1. 프로젝트 구조
naive_bayes_customer_analysis/
├── 📊 data/
│   ├── raw/                    # 원본 데이터 (던험비 데이터셋)
│   ├── processed/              # 전처리된 데이터
│   └── features/               # 피처 엔지니어링 결과
├── 📓 notebooks/
│   ├── 01_data_exploration.ipynb        # 데이터 탐색
│   ├── 02_data_preprocessing.ipynb      # 데이터 전처리
│   ├── 03_feature_engineering.ipynb     # 피처 엔지니어링
│   ├── 04_naive_bayes_analysis.ipynb    # 나이브 베이즈 모델링
│   ├── 06_rfm_clustering_analysis.ipynb # RFM 클러스터링
│   ├── 07_standard_rfm_analysis.ipynb   # 표준 RFM 분석
│   ├── 08_coupon_only_rfm_analysis.ipynb # 쿠폰 중심 분석
├── 📈 results/
│   ├── models/                 # 저장된 모델들
│   ├── figures/               # 시각화 결과
│   └── reports/               # 분석 보고서
└── 🔧 src/
   └── __init__.py            # 분석 유틸리티 함수들



2. 분석 순서
01_data_exploration.ipynb: 데이터 이해하기
02-03: 전처리 및 피처 엔지니어링
04: 나이브 베이즈 모델링
06-08: RFM 클러스터링 분석



3. 기술 스택
데이터 분석
- Python 3.8+: 메인 프로그래밍 언어
- Pandas: 데이터 조작 및 분석
- NumPy: 수치 계산
- Scikit-learn**: 머신러닝 모델링

시각화
- Matplotlib: 기본 플롯
- Seaborn: 통계적 시각화
- Plotly: 인터랙티브 차트

모델링
- Naive Bayes: 고객 행동 예측
- K-Means: RFM 클러스터링
- Linear Regression: 시계열 예측


4. 환경 설정
```bash
# 레포지토리 클론
git clone https://github.com/bichae777/naive_bayes_customer_analysis.git
cd naive_bayes_customer_analysis

# 필요한 패키지 설치
pip install -r requirements.txt

# Jupyter 노트북 실행
jupyter notebook