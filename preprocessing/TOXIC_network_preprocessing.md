# 독성 네트워크 구축 전처리 과정
TOXIC_data_check_SSM.ipynb 참고

## 1. 데이터셋 통합 및 라벨링

### 1.1 원본 데이터셋 로드
- **DILI**: 475개 화합물 (Drug-Induced Liver Injury 데이터)
- **OpenFoodTox (OFT)**: 123개 화합물 (비독성 화합물)
- **T3DB**: 21개 화합물 (독성 화합물)
- **Useful Substances (US)**: 16개 화합물 (유용한 물질)

### 1.2 데이터 통합
```python
# 각 DataFrame에 origin 컬럼 추가
DILI = DILI.assign(origin='DILI')
OFT = OFT.assign(origin='OFT')
T3DB = T3DB.assign(origin='T3DB')
US = US.assign(origin='US')
```

### 1.3 라벨링 규칙
다음 규칙에 따라 독성 라벨을 부여:

| 데이터셋 | 조건 | 라벨 |
|---------|------|------|
| DILI | Y = 0 (비독성) | 0 |
| DILI | Y = 1 (독성) | 1 |
| OFT | 모든 화합물 | 0 |
| T3DB | 모든 화합물 | 1 |
| US | 모든 화합물 | 0 |

```python
conditions = [cond_dili_0, cond_dili_1, cond_oft, cond_t3db, cond_us]
choices    = [0,           1,           0,        1,         0]
TOXIC_TOTAL['label'] = np.select(conditions, choices, default=np.nan)
```

### 1.4 최종 라벨 분포
- **비독성 (0)**: 317개 화합물
- **독성 (1)**: 256개 화합물

## 2. 데이터 분할 (Scaffold Split)

### 2.1 Scaffold Splitter 구현
Murcko Scaffold 기반 분할을 위한 커스텀 클래스 구현:

```python
class ScaffoldSplitter(BaseSplitter):
    def generate_scaffold(smiles, include_chirality=False):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
        return scaffold
```

### 2.2 분할 비율
- **Train**: 80% (458개 화합물)
- **Validation**: 0% (사용하지 않음)
- **Test**: 20% (115개 화합물)

### 2.3 분할 과정
1. 각 화합물의 SMILES에서 Murcko Scaffold 추출
2. 동일한 scaffold를 가진 화합물들을 그룹화
3. Scaffold 그룹 단위로 train/test 분할
4. 중복 인덱스 검증 후 분할 완료

## 3. SUBDYVE 데이터 준비

### 3.1 SEED 데이터 생성
독성 화합물만을 SEED로 선정:
```python
SEED_FOR_SUBDYVE = TOXIC_TOTAL_labeled[
    (TOXIC_TOTAL_labeled['Y'] == 1) & (TOXIC_TOTAL_labeled['origin'] == 'DILI') | 
    (TOXIC_TOTAL_labeled['origin'] == 'T3DB')
]
```

### 3.2 COMP 데이터 생성
SEED에 포함되지 않은 모든 화합물을 COMP로 선정:
```python
COMP_FOR_SUBDYVE = TOXIC_TOTAL_labeled[~TOXIC_TOTAL_labeled['ID'].isin(SEED_FOR_SUBDYVE['ID'])]
```

### 3.3 ID 매핑 생성
- **SEED_ID2I**: SEED 화합물의 ID를 인덱스로 매핑
- **COMP_ID2I**: COMP 화합물의 ID를 인덱스로 매핑

## 4. 데이터 정리 및 저장

### 4.1 컬럼 정리
- `SMILES` 컬럼 제거
- `CAN_SMILES`를 `SMILES`로 컬럼명 변경
- 인덱스 리셋

### 4.2 최종 저장 파일
- `TOXIC_TOTAL_labeled.csv`: 전체 라벨링된 데이터
- `train_data.csv`: 훈련 데이터
- `test_data.csv`: 테스트 데이터
- `SUBDYVE_SEED.csv`: SEED 데이터
- `SUBDYVE_COMP.csv`: COMP 데이터
- `SEED_ID2I.pkl`: SEED ID 매핑
- `COMP_ID2I.pkl`: COMP ID 매핑

## 5. 데이터 품질 관리

### 5.1 중복 제거
- SMILES 기반 중복 제거
- ID 기반 중복 제거
- 정규화된 SMILES 사용

### 5.2 데이터 검증
- 분할 인덱스 중복 검사
- 라벨 분포 확인
- 데이터셋 크기 검증

## 6. 환경 설정

### 6.1 필요한 라이브러리
- `numpy`, `pandas`: 데이터 처리
- `rdkit`: 화학 구조 처리
- `chainer_chemistry`: 분자 데이터셋 처리

### 6.2 주의사항
- `distutils.msvccompiler` 모듈 오류로 인해 일부 작업은 별도 환경에서 수행
- SSM conda 환경 활성화 필요

이 전처리 과정을 통해 독성 예측을 위한 구조화된 데이터셋이 구축되었으며, scaffold 기반 분할로 인해 화학적 유사성을 고려한 모델 평가가 가능합니다. 