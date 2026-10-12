# Subdyve_bean

SIDER / TOXIC 각각에 대해 **BEAN baseline(14종 fingerprint) vs SubDyve** 비교 결과를 한 번에 얻기 위한 정리본입니다.

## 실행

```bash
cd Subdyve_bean
conda activate SSM              # 파이썬 env (아래 참고)
./run_TOXIC.sh --gpu 4          # TOXIC 전체 (bean -> ssm -> subdyve -> analysis)
./run_SIDER.sh --gpu 4          # SIDER 전체

./run_TOXIC.sh --stages bean                 # 일부 단계만
./run_TOXIC.sh --stages subdyve,analysis     # SSM 결과가 이미 있을 때
./run_TOXIC.sh --clean                       # 이전 산출물 삭제 후 처음부터
./run_pipeline.sh --help
```

- 파이썬: PATH 의 `python` 을 사용하므로 conda env `SSM` 을 활성화한 뒤 실행하세요. 다른 파이썬은 `PY=/path/python ./run_TOXIC.sh`.
- `--gpu ID` 는 `CUDA_VISIBLE_DEVICES` 로 전달됩니다 (NP_SUBDYVE 단계만 GPU 사용).
- 각 단계 로그: `logs/<TARGET>/<stage>.log`
- 각 단계가 끝나면 필요한 산출물이 있는지 검사하고, 없으면 에러로 중단합니다.
  (원본 `collect.py`, `utils.py` 는 실패해도 exit 0 으로 끝나는 부분이 있고, `analyze_all_fps.py` 는 입력이 없으면 경고만 내고 "부분 결과"로 진행하기 때문)

## 파이프라인

```
data/COMP_<T>.csv, SEED_<T>.csv
   └─[bean]──> source_BEAN/0_1_makeFP -> 0_2_makeNetwork -> 0_3_network_analysis (14종 FP, Tanimoto, edge 0.85)
                  └─> result_<T>/NP_edge_0.85_<fp>_nonseed.txt  (x14)

source_SUBDYVE/SSM/data/<T>/{train,test}_data.csv
   └─[ssm]──> ssm_smiles -> collect(best AUC iteration) -> ssm_DISC -> clean_DISC
                  └─> SSM/results/<T>/CONCAT_<T>_DISC_2000_filtered.csv
   └─[subdyve]──> SUBMINE/bin/matchFP -> source/makeNetwork_combined (cosine) -> source/<T>_NP_SUBDYVE (GCN, GPU) -> source/utils
                  └─> Network/<T>/output/NP_SUBDYVE_result_nonseed.txt

   └─[analysis]──> analyze_all_fps.py + data_check_toxic_updated.py   (14종 FP + SubDyve 비교)
                  └─> plots_<T>/bedroc_metrics_summary.csv, statistical_significance_summary.csv, *.png
                      result_<T>/filtered_results_<fp|subdyve>.csv
```

## 주요 결과물 위치

| 파일 | 내용 |
|---|---|
| `plots_<T>/bedroc_metrics_summary.csv` | 모델별 AUROC / BEDROC / EF@10·20·30% (bootstrap ± std) |
| `plots_<T>/statistical_significance_summary.csv` | SubDyve vs 최고 모델 Mann-Whitney U (아래 주의점 참고) |
| `plots_<T>/<fp>_*.png`, `net_by_<fp>.png` | US 순위 분석 / 점수 분포 / 네트워크 시각화 |
| `result_<T>/filtered_results_*.csv` | US 중 Toxic_label==0 으로 필터된 화합물 (Number 매핑) |
| `source_SUBDYVE/Network/<T>/output/` | SubDyve 네트워크·NP 점수·분석 요약 |

## 디렉토리 구조와 원본 대응

| 이 디렉토리 | 원본 (`bean/` 기준) |
|---|---|
| `source_BEAN/` | `source/` 의 `0_1_makeFP.py 0_2_makeNetwork.py 0_3_network_analysis.py statistic.py` |
| `source_SUBDYVE/SSM/bin/` | `source_SUBDYVE/SSM/bin/` (LOO_* 제외) |
| `source_SUBDYVE/SUBMINE/bin/matchFP.py` | 동일 |
| `source_SUBDYVE/source/` | `source_SUBDYVE/source/` 의 `makeNetwork_combined statistic SIDER_NP_SUBDYVE TOXIC_NP_SUBDYVE utils` |
| `analyze_all_fps.py`, `data_check_toxic_updated.py` | `bean/` 동일 이름 |
| `run_pipeline.sh` | `run_bean_*.sh` + `source_SUBDYVE/TOTAL_BEAN_SUBDYVE_*.sh` + `BEAN_SUBDYVE_*.sh` + `SSM/1_BEAN_mine_subgraph_*.sh` 통합 |
| `data/` | `COMP_/SEED_<T>.csv`, `TOXIC_TOTAL_labeled.csv`, `SIDER_TOTAL_labeled_skinonly.csv` |
| `source_SUBDYVE/SSM/data/<T>/` | `train_data.csv`, `test_data.csv` (scaffold split 결과) |
| `source_SUBDYVE/SUBMINE/data/<T>/` | `SEED_DATA.csv`, `COMP_DATA.csv`, **ChemBERTa-77M-MTR 임베딩 pkl** (미리 계산된 입력) |
| `preprocessing/` | 입력 데이터를 만든 노트북/문서 (참고용, **실행 대상 아님**) |

복사하지 않은 것: 기존 결과물(`result_*`, `plots_*`, `SSM/results`, `LOO*`, `연결안된것_혼났던결과` 등 약 11GB), `FP_*.pickle`(matchFP 가 재생성), `*_ID2I.pkl`(파이프라인에서 미사용), `US포함_*` 변형 데이터, 분석용 부가 스크립트(`DiSC_pattern_analysis_complete.py`, `USnode_graphAnaly_unified.py` 등).

## 입력 데이터

- BEAN 입력(`data/COMP_<T>.csv`, `SEED_<T>.csv`)과 SubDyve 입력(`SUBMINE/data/<T>/COMP_DATA.csv`, `SEED_DATA.csv`)은 **같은 화합물 집합**입니다
  (ID·순서 동일, `CAN_SMILES` == `SMILES` 확인함). 규모: TOXIC seed 256 / comp 317, SIDER seed 1318 / comp 184.
- 생성 과정: `preprocessing/250805_{TOXIC,SIDER}_SSM_2_SUBDYVE_all.ipynb`, TOXIC 라벨링 규칙은 `preprocessing/TOXIC_network_preprocessing.md`.
  노트북은 `chainer_chemistry` 가 필요한데 `SSM` env 에는 없고, 노트북 안 경로도 예전 절대경로입니다.
- **ChemBERTa 임베딩**: pkl 을 그대로 복사해 사용합니다. TOXIC 용 생성 코드는 `preprocessing/TOXIC_data_check_SSM.ipynb` 의 ChemBERTa 셀
  (`DeepChem/ChemBERTa-77M-MTR`, last_hidden_state mean pooling)에 있습니다. **SIDER 용 생성 코드는 이 프로젝트에서 찾지 못했습니다.**
  입력 SMILES 를 바꾸면 pkl 도 다시 만들어야 하며 자동으로 재생성되지 않습니다.
