# MosESD-Apriori

MosESD(Multi-online Extreme Studentized Deviate) 알고리즘에 Apriori 알고리즘을 적용하여 피처 선택 기능을 추가한 이상 탐지 시스템

## 개요

본 프로젝트는 반도체 제조 공정에서 발생하는 이상 상황을 탐지하기 위해 개발함. MosESD 알고리즘을 기반으로 하되, Apriori 알고리즘을 통한 피처 선택으로 성능을 향상.

## 주요 기능

- **MosESD 알고리즘**: 온라인 반지도 학습 기반 이상 탐지
- **Apriori 피처 선택**: 세 가지 정책(naive, hard, soft)을 통한 최적 피처 조합 탐색
- **다양한 정책 지원**:
  - `unsupervised`: 기본 MosESD 알고리즘
  - `naive`: 단순 임계값 기반 피처 선택
  - `hard`: 엄격한 Apriori 규칙 적용
  - `soft`: 확률적 완화가 적용된 Apriori

## 프로젝트 구조

```
MosESD-Apriori/
├── main.py                 # 메인 실행 파일
├── config.yaml             # 피처 설정 파일
├── requirements.txt        # 의존성 패키지
├── data/                   # 데이터 디렉토리
│   └── labeled_data/       # 라벨링된 데이터
├── src/                    # 소스코드
│   ├── data/               # 데이터 로더
│   ├── models/             # 알고리즘 구현
│   ├── log/                # 로깅 모듈
│   └── utils/              # 유틸리티 함수
├── log/                    # 실행 로그
└── result/                 # 결과 저장
```

## 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 준비
`data/labeled_data/` 디렉토리에 `scaled_data_*.csv` 형식의 라벨링된 데이터를 배치합니다.

### 3. 피처 설정
`config.yaml` 파일에서 사용할 피처들을 설정할 수 있습니다:
```yaml
features:
  default:
    - "Feature1"
    - "Feature2"
    - "Feature3"
    # ... 기타 피처들
```

## 사용법

### 기본 실행
```bash
python main.py
```

### 주요 매개변수

#### 데이터 관련
- `--dataset`: 데이터셋 이름 (기본값: "labeled")
- `--log_num`: 데이터 파일의 로그 번호 (기본값: 260)
- `--step`: 분석할 스텝 번호 (기본값: 6)
- `--anomaly_col`: 이상치 컬럼명 (기본값: "label")

#### 알고리즘 관련
- `--policy`: 피처 선택 정책 ["naive", "hard", "soft", "unsupervised"] (기본값: "hard")
- `--threshold`: 피처 선택 임계값 (기본값: 0.6)
- `--probability`: soft 정책에서 사용할 확률 (기본값: 0.2)

#### MosESD 매개변수
- `--rwin_size`: TRES 윈도우 크기 (기본값: 20)
- `--dwin_size`: TCHA 윈도우 크기 (기본값: 20)
- `--init_size`: 초기 데이터셋 크기 (기본값: 100)
- `--alpha`: 유의수준 (기본값: 0.05)
- `--maxr`: 최대 이상치 개수 (기본값: 10)
- `--epochs`: 훈련 에포크 (기본값: 1)

### 실행 예시

```bash
# Hard Apriori 정책으로 실행
python main.py --log_num 260 --step 6 --policy hard --threshold 0.6

# Soft Apriori 정책으로 실행 (확률적 완화 적용)
python main.py --log_num 260 --step 6 --policy soft --probability 0.3

# 기본 MosESD (피처 선택 없음)
python main.py --log_num 260 --step 6 --policy unsupervised
```

## 알고리즘 설명

### MosESD (Multi-dimensional online Semi-supervised ESD)
- **TRES (Trend RESidual)**: 시계열 데이터의 트렌드 잔차를 계산
- **TCHA (Trend CHAnge)**: 트렌드 변화율을 계산
- **SESD**: Sequential Extreme Studentized Deviate 테스트로 이상치 탐지

### Apriori 피처 선택
1. **Stage 1**: 개별 피처의 성능 평가 및 우선순위 부여
2. **Stage N≥2**: 이전 단계의 유효한 조합들을 병합하여 새로운 조합 생성
3. **Pruning**: 임계값 미달 조합 제거

#### 정책별 차이점
- **Naive**: 임계값을 넘는 모든 조합 유지
- **Hard**: 엄격한 Apriori 규칙 적용 (N-2개 공통 피처 필요)
- **Soft**: Hard 정책 + 확률적으로 임계값 미달 조합도 일부 유지

## 출력 결과

### 콘솔 출력
```
최고 F1 점수와 최적 피처 조합이 출력됩니다.
예: 1.0 ('RF1_Cap1Pos', 'RF1_RefPwr')
```

### 로그 파일
- 위치: `log/YYYYMMDD/{dataset}_{policy}_YYYYMMDD.log`
- 내용: 하이퍼파라미터, 각 단계별 결과, 최종 성능

### 결과 파일
- 예측 결과: `result/YYYYMMDD/{dataset}_{policy}_{anomaly_col}_YYYYMMDD_HHMMSS.csv`
- 성능 요약: `log/{policy}.csv`

## 요구사항

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
- tqdm
- pyyaml

## 라이선스

이 프로젝트는 연구 목적으로 개발되었습니다.

## 기여자
- IDSL (Intelligent Data Science Lab)
  - KINGYHWAN (김용환)
  - rtyu4236 (조유정)

## 참고문헌

본 프로젝트는 MosESD 알고리즘과 Apriori 알고리즘을 결합한 연구를 기반으로 함.
