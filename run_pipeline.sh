#!/usr/bin/env bash
# =============================================================================
# Subdyve_bean 통합 파이프라인  (SIDER / TOXIC 공용)
#
#   ./run_pipeline.sh <SIDER|TOXIC> [--top N] [--stages a,b,c] [--gpu ID] [--clean]
#
#   단계(stages)  (기본: bean,ssm,subdyve,analysis 전부, 이 순서대로)
#     bean      BEAN baseline: 14종 fingerprint 네트워크 + propagation
#               -> result_<T>/NP_edge_0.85_<fp>_nonseed.txt
#     ssm       SSM(subgraph mining) + DiSC 추출
#               -> source_SUBDYVE/SSM/results/<T>/CONCAT_<T>_DISC_<TOP>_filtered.csv
#     subdyve   DiSC 매칭 -> 네트워크 -> NP_SUBDYVE(GCN, GPU) -> seed 제외
#               -> source_SUBDYVE/Network/<T>/output/NP_SUBDYVE_result_nonseed.txt
#     analysis  14종 FP + SubDyve 비교 (AUROC/BEDROC/EF), filtered_results
#               -> plots_<T>/bedroc_metrics_summary.csv, result_<T>/filtered_results_*.csv
#
#   --top N     DiSC 패턴 개수 (기본 2000)
#   --gpu ID    NP_SUBDYVE에 쓸 물리 GPU 번호 (CUDA_VISIBLE_DEVICES). 미지정 시 환경 기본값
#   --clean     선택한 단계의 이전 산출물을 먼저 삭제 (0_1_makeFP.py는 캐시 pickle이 있으면 재사용하므로,
#               입력 데이터를 바꿨다면 반드시 사용)
#   파이썬: 기본은 PATH 의 python  ->  먼저 `conda activate SSM` 후 실행. PY=/path/to/python 으로 직접 지정도 가능
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"

usage() { sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# ---------------------------- 인자 파싱 ---------------------------------------
[[ $# -ge 1 ]] || usage 1
T="$1"; shift
TOP=2000
STAGES="bean,ssm,subdyve,analysis"
CLEAN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --top)    TOP="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --gpu)    export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        --clean)  CLEAN=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "[ERROR] 알 수 없는 옵션: $1" >&2; usage 1 ;;
    esac
done

# ---------------------------- 타겟별 설정 -------------------------------------
case "$T" in
    SIDER) DISC=3 ;;   # 원본 SSM/1_BEAN_mine_subgraph_SIDER.sh 의 --DiSC 값
    TOXIC) DISC=4 ;;   # 원본 SSM/1_BEAN_mine_subgraph_TOXIC.sh 의 --DiSC 값
    *) echo "[ERROR] 타겟은 SIDER 또는 TOXIC 이어야 합니다: $T" >&2; exit 1 ;;
esac
# BEAN baseline 상수 (원본 run_bean_*.sh)
BEAN_CUTOFF=0.0; BEAN_EDGE=0.85; BEAN_RESTART=0.1; BEAN_SIM=tanimoto
# SubDyve 상수 (원본 BEAN_SUBDYVE_*.sh)
SD_SEED_CUTOFF=0.0; SD_EDGE=0.85; SD_SIM=cosine

LOG_DIR="$ROOT/logs/$T"; mkdir -p "$LOG_DIR"

die()  { echo "[ERROR] $*" >&2; exit 1; }
need() { [[ -s "$1" ]] || die "필요한 파일이 없거나 비어 있음: $1  ($2)"; }

# ---------------------------- 단계 구현 ---------------------------------------
stage_bean() {
    cd "$ROOT"
    local out="result_${T}"
    [[ $CLEAN -eq 1 ]] && rm -rf "${ROOT:?}/result_${T:?}"
    need "data/COMP_${T}.csv" "입력 데이터"; need "data/SEED_${T}.csv" "입력 데이터"

    echo "[bean 1/3] fingerprint + similarity"
    "$PY" source_BEAN/0_1_makeFP.py -t "data/COMP_${T}.csv" -c "$BEAN_CUTOFF" -a "data/SEED_${T}.csv" \
        -o "$out" --similarity "$BEAN_SIM"

    echo "[bean 2/3] network (edge list)"
    "$PY" source_BEAN/0_2_makeNetwork.py -i "$out" -c "$BEAN_CUTOFF" -e "$BEAN_EDGE" --draw

    echo "[bean 3/3] network propagation (RWR) per fingerprint"
    local seed="$out/$BEAN_CUTOFF/seed.txt" edge base n=0
    need "$seed" "0_2_makeNetwork.py 산출물"
    for edge in "$out/$BEAN_CUTOFF"/edge_${BEAN_EDGE}_*.txt; do
        [[ -e "$edge" ]] || die "edge 파일이 없음: $out/$BEAN_CUTOFF/edge_${BEAN_EDGE}_*.txt"
        base="$(basename "$edge" .txt)"
        echo "  - propagation: $base"
        "$PY" source_BEAN/0_3_network_analysis.py "$edge" "$seed" \
            -o "$out/NP_${base}.txt" -e "$BEAN_RESTART" -addBidirectionEdge True
        # seed node(p_)로 시작하는 행 제거 (grep은 결과가 0줄이면 exit 1 -> 아래에서 비어있음을 에러 처리)
        grep -v "^p_" "$out/NP_${base}.txt" > "$out/NP_${base}_nonseed.txt" || true
        need "$out/NP_${base}_nonseed.txt" "propagation 결과가 비어 있음"
        n=$((n+1))
    done
    [[ $n -eq 14 ]] || die "fingerprint 결과가 14개가 아님: $n개"
}

stage_ssm() {
    cd "$ROOT/source_SUBDYVE"
    local train="./SSM/data/$T/train_data.csv" test="./SSM/data/$T/test_data.csv"
    local out="./SSM/results/$T/" aucdict="./SSM/logs/${T}_max_AUC_dict.csv"
    [[ $CLEAN -eq 1 ]] && rm -rf "${ROOT:?}/source_SUBDYVE/SSM/results/${T:?}" "$aucdict"
    need "$train" "입력 데이터"; need "$test" "입력 데이터"
    mkdir -p "$out" ./SSM/logs

    echo "[ssm 1/4] SSM 학습/예측 (rw=7, alpha=0.1, iterations=20, nWalker=5)"
    "$PY" ./SSM/bin/ssm_smiles.py --train_data "$train" --test_data "$test" --output_dir "$out" \
        --rw 7 --alpha 0.1 --iterations 20 --nWalker 5

    echo "[ssm 2/4] AUC 최대 iteration 선택"
    "$PY" ./SSM/bin/collect.py --target "$T" --output_path "$aucdict"
    need "$aucdict" "collect.py 는 실패해도 exit 0 으로 끝나므로 파일 유무로 확인"
    local best
    best="$(awk -F, -v seed="$T" '$1 == seed {print $2}' "$aucdict")"
    [[ -n "$best" ]] || die "bestIteration 추출 실패: $aucdict"
    echo "  best iteration = $best"

    echo "[ssm 3/4] DiSC 추출 (--DiSC $DISC, bestIteration $best)"
    "$PY" ./SSM/bin/ssm_DISC.py --train_data "$train" --test_data "$test" --output_dir "$out" \
        --rw 7 --alpha 0.1 --iterations 20 --nWalker 5 --DiSC "$DISC" --bestIteration "$best"

    echo "[ssm 4/4] DiSC 중복 제거/필터 (top $TOP)"
    "$PY" ./SSM/bin/clean_DISC.py --target "$T" --top "$TOP" --log_dir ./SSM/logs --result_root "$out"
    need "${out}CONCAT_${T}_DISC_${TOP}_filtered.csv" "clean_DISC.py 산출물"
}

stage_subdyve() {
    cd "$ROOT/source_SUBDYVE"
    local in="./SUBMINE/data/$T" out="./Network/$T/output"
    local disc="./SSM/results/$T/CONCAT_${T}_DISC_${TOP}_filtered.csv"
    if [[ $CLEAN -eq 1 ]]; then
        rm -rf "${ROOT:?}/source_SUBDYVE/Network/${T:?}"
        rm -f "$in"/FP_Seed.pickle "$in"/FP_Comp.pickle
    fi
    need "$disc" "먼저 'ssm' 단계를 실행하세요 (또는 --top 값 확인)"
    mkdir -p "$out"

    echo "[subdyve 1/4] DiSC 패턴 매칭 -> FP_Seed/FP_Comp.pickle"
    # 주의: 원본 BEAN_SUBDYVE_SIDER.sh 는 --raw_data_dir 가 빠져 있어 현재 matchFP.py 와 맞지 않음 -> 여기서는 명시
    "$PY" ./SUBMINE/bin/matchFP.py --target "$T" --top "$TOP" --disc_path "$disc" \
        --output_root "$in" --raw_data_dir "$in"

    echo "[subdyve 2/4] 네트워크 생성 (similarity=$SD_SIM)"
    "$PY" source/makeNetwork_combined.py --seed "$in/SEED_DATA.csv" --comp "$in/COMP_DATA.csv" \
        --input_fp_dir "$in" --output "$out" --seed_cutoff "$SD_SEED_CUTOFF" --edge_cutoff "$SD_EDGE" \
        --similarity "$SD_SIM"

    echo "[subdyve 3/4] NP_SUBDYVE (GCN + propagation, GPU)"
    local seed="$out/seed_${SD_SEED_CUTOFF}.txt" edge
    edge="$(find "$out" -maxdepth 1 -name "edge_${SD_SEED_CUTOFF}_${SD_EDGE}_*" | sort | head -n 1)"
    [[ -n "$edge" ]] || die "edge 파일을 찾지 못함: $out/edge_${SD_SEED_CUTOFF}_${SD_EDGE}_*"
    local edge_base result
    edge_base="$(basename "$edge")"
    result="$out/${edge_base/edge/NP_SUBDYVE}"            # edge_0.0_0.85_subgraph.txt -> NP_SUBDYVE_0.0_0.85_subgraph.txt
    "$PY" "source/${T}_NP_SUBDYVE.py" "$edge" "$seed" -o "$result" -addBidirectionEdge True \
        -data_SEED "$in/SEED_DATA.csv" -data_COMP "$in/COMP_DATA.csv" \
        -fp_SEED "$in/FP_Seed.pickle" -fp_COMP "$in/FP_Comp.pickle" \
        -PT_CB_SEED "$in/DeepChem_ChemBERTa-77M-MTR_${T}_Seed.pkl" \
        -PT_CB_COMP "$in/DeepChem_ChemBERTa-77M-MTR_${T}_Comp.pkl"
    need "$result" "NP_SUBDYVE 산출물"

    echo "[subdyve 4/4] seed 제외 + 후처리"
    "$PY" ./source/utils.py --target "$T" --seed_cutoff "$SD_SEED_CUTOFF" \
        --network_output_dir "$out" --submine_data_dir "$in" --edge_cutoff "$SD_EDGE"
    # utils.py 는 내부에서 예외를 잡고 계속 진행하므로 산출물로 성공 여부를 확인
    need "$out/NP_SUBDYVE_result_nonseed.txt" "utils.py 후처리 실패 가능성 - 위 로그 확인"
}

stage_analysis() {
    cd "$ROOT"
    [[ $CLEAN -eq 1 ]] && rm -rf "${ROOT:?}/plots_${T:?}"
    # analyze_all_fps.py 는 입력이 없으면 경고만 출력하고 '부분 결과'로 진행하므로 사전 점검
    local fp
    for fp in standard extended graph maccs pubchem estate hybridization klekota-roth cdk-substructure rdkit morgan avalon fp2 fp4; do
        need "result_${T}/NP_edge_0.85_${fp}_nonseed.txt" "먼저 'bean' 단계를 실행하세요"
    done
    need "source_SUBDYVE/Network/${T}/output/NP_SUBDYVE_result_nonseed.txt" "먼저 'subdyve' 단계를 실행하세요"

    echo "[analysis 1/2] analyze_all_fps.py  (AUROC/BEDROC/EF + 통계검정 + 플롯)"
    "$PY" analyze_all_fps.py --target "$T"
    echo "[analysis 2/2] data_check_toxic_updated.py  (filtered_results_*.csv)"
    "$PY" data_check_toxic_updated.py --target "$T"
    need "plots_${T}/bedroc_metrics_summary.csv" "analyze_all_fps.py 산출물"
}

# ---------------------------- 실행 --------------------------------------------
echo "=============================================================="
echo " target=$T  top=$TOP  stages=$STAGES  clean=$CLEAN"
echo " python=$PY"
echo " CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=============================================================="
command -v "$PY" >/dev/null 2>&1 || die "파이썬을 찾을 수 없음: $PY  (conda activate SSM 후 실행하거나 PY=... 로 지정)"

IFS=',' read -r -a STAGE_LIST <<< "$STAGES"
for s in "${STAGE_LIST[@]}"; do
    case "$s" in bean|ssm|subdyve|analysis) ;; *) die "알 수 없는 단계: $s" ;; esac
done
for s in "${STAGE_LIST[@]}"; do
    echo; echo "################ [$T] stage: $s  ($(date '+%F %T')) ################"
    t0=$SECONDS
    # 각 단계 로그: logs/<T>/<stage>.log  (화면에도 동시 출력)
    ( "stage_$s" ) 2>&1 | tee "$LOG_DIR/$s.log"
    echo "[done] $s  ($((SECONDS - t0))s)"
done

echo; echo "=== 완료: $T ==="
if [[ -f "$ROOT/plots_${T}/bedroc_metrics_summary.csv" ]]; then
    echo "요약표: $ROOT/plots_${T}/bedroc_metrics_summary.csv"
fi
