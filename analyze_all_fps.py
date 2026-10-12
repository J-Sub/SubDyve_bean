#!/usr/bin/env python3
"""
각 FP별 nonseed 파일들에 utils.py의 분석을 적용하고 네트워크 시각화를 수행하는 통합 스크립트
RDKit을 사용하여 BEDROC/EF 계산 (score 오름차순 정렬)
"""

import os
import sys
import pathlib
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score
from scipy.stats import mannwhitneyu

# utils.py의 함수들을 import하기 위해 경로 추가
sys.path.append('./source_SUBDYVE/source')
from utils import process_np_results, analyze_us_ranking, create_bar_plot_with_id_coloring, ALL_SUFFIXES, COLOR_MAP

# 전역 suffix 및 색상 매핑 정의 (utils.py에서 import)
# ALL_SUFFIXES와 COLOR_MAP은 utils.py에서 가져옴

def setup_matplotlib_style():
    """matplotlib 스타일 설정"""
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['figure.titlesize'] = 22

def normalize_scores(df):
    """
    Score를 0-1로 MinMax 정규화
    """
    min_score = df['Score'].min()
    max_score = df['Score'].max()
    
    if max_score == min_score:
        df['Normalized_Score'] = 0.5  # 모든 점수가 같으면 0.5로 설정
    else:
        df['Normalized_Score'] = (df['Score'] - min_score) / (max_score - min_score)
    
    return df

def calculate_bedroc_ef_metrics_rdkit(df, top_fractions=[0.1, 0.2, 0.3], alpha=20.0):
    """
    RDKit을 사용하여 BEDROC 및 EF 메트릭 계산 (score 오름차순 정렬)
    """
    try:
        # US 노드와 Non-US 노드 분리
        us_nodes = df[df['ID'].str.endswith('_US')]
        non_us_nodes = df[~df['ID'].str.endswith('_US')]
        
        if len(us_nodes) == 0 or len(non_us_nodes) == 0:
            return {}
        
        total_nodes = len(df)
        us_count = len(us_nodes)
        
        
        # Rank 컬럼 추가 (Score 기준 내림차순 랭킹) - LOO_analyze와 동일한 방식
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df["Rank"] = range(1, len(df) + 1)
        
        # US 노드들의 랭킹 (높은 score 기준)
        us_ranks = df[df["ID"].str.endswith("_US")]["Rank"]
        
        # Score를 오름차순으로 정렬 (낮은 score가 좋음)
        df_sorted = df.sort_values("Score", ascending=True).reset_index(drop=True)
        df_sorted["Reverse_Rank"] = range(1, len(df_sorted) + 1)
        
        # US 노드들의 역순 랭킹
        # US 노드들의 역순 랭킹
        # US 노드들의 역순 랭킹
        metrics = {}
        
        # AUROC 계산 (역순) - sklearn 없이 구현
        def calculate_auroc(y_true, y_scores):
            """AUROC 계산 (sklearn 없이)"""
            # 점수 기준으로 정렬
            sorted_indices = np.argsort(-y_scores)  # 높은 점수부터
            sorted_labels = y_true[sorted_indices]
            
            # TPR, FPR 계산
            n_pos = np.sum(y_true)
            n_neg = len(y_true) - n_pos
            
            if n_pos == 0 or n_neg == 0:
                return 0.5
            
            tpr = np.cumsum(sorted_labels) / n_pos
            fpr = np.cumsum(1 - sorted_labels) / n_neg
            
            # AUC 계산 (trapezoidal rule)
            auc = np.trapz(tpr, fpr)
            return auc
        
        y_true = df_sorted['ID'].str.endswith('_US').astype(int).values
        y_scores = -df_sorted['Score'].values  # 낮은 score가 좋으므로 음수로 변환
        auroc = calculate_auroc(y_true, y_scores)
        metrics["AUROC"] = round(auroc, 2)
        
        # RDKit을 사용한 BEDROC 및 EF 계산
        # RDKit 형식: [[score, label], [score, label], ...]
        # label: 1 (US/active), 0 (Non-US/inactive)
        scores_for_rdkit = [[score, label] for score, label in zip(y_scores, y_true)]
        col = 1  # label이 위치한 컬럼 인덱스
        
        # BEDROC 계산 (RDKit 사용)
        try:
            bedroc = CalcBEDROC(scores_for_rdkit, col, alpha=alpha)
            metrics["BEDROC"] = round(bedroc * 100, 2)  # 백분율로 변환
        except Exception as e:
            print(f"Warning: BEDROC calculation failed: {e}")
            metrics['BEDROC'] = 0.0
        
        # EF@X% 계산 (RDKit 사용)
        try:
            ef_list = CalcEnrichment(scores_for_rdkit, col, top_fractions)
            for i, fraction in enumerate(top_fractions):
                metrics[f"EF@{fraction*100:.0f}%"] = round(ef_list[i], 2) if i < len(ef_list) else 0.0
        except Exception as e:
            print(f"Warning: EF calculation failed: {e}")
            for fraction in top_fractions:
                metrics[f"EF@{fraction*100:.0f}%"] = 0.0
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating BEDROC/EF metrics with RDKit: {e}")
        return {}

def bootstrap_metrics_rdkit(df, top_fractions=[0.1, 0.2, 0.3], alpha=20.0, n_bootstrap=100, random_state=42):
    """계층적 부트스트래핑을 사용한 RDKit 메트릭 계산"""
    # US와 Non-US 노드 분리
    us_nodes = df[df['ID'].str.endswith('_US')]
    non_us_nodes = df[~df['ID'].str.endswith('_US')]
    
    us_count = len(us_nodes)
    non_us_count = len(non_us_nodes)
    
    if us_count == 0 or non_us_count == 0:
        return {}, pd.DataFrame()
    
    np.random.seed(random_state)
    
    boot_results = []
    
    for i in range(n_bootstrap):
        # US 노드에서 복원 추출
        us_bootstrap = us_nodes.sample(n=us_count, replace=True, 
                                       random_state=np.random.randint(0, 100000))
        
        # Non-US 노드에서 복원 추출
        non_us_bootstrap = non_us_nodes.sample(n=non_us_count, replace=True,
                                               random_state=np.random.randint(0, 100000))
        
        # 합치기
        bootstrap_df = pd.concat([us_bootstrap, non_us_bootstrap], ignore_index=True)
        
        # RDKit을 사용한 메트릭 계산
        metrics = calculate_bedroc_ef_metrics_rdkit(bootstrap_df, top_fractions=top_fractions, alpha=alpha)
        
        if metrics:
            boot_results.append(metrics)
    
    if not boot_results:
        return {}, pd.DataFrame()
    
    boot_df = pd.DataFrame(boot_results)

    # 원본 데이터에서 메트릭 계산
    original_metrics = calculate_bedroc_ef_metrics_rdkit(df, top_fractions=top_fractions, alpha=alpha)
    
    # 평균 + 표준편차 계산
    summary = {}
    for col in boot_df.columns:
        mean_val = boot_df[col].mean()
        std_val = boot_df[col].std()
        
        if col in ["BEDROC", "AUROC"]:
            summary[col] = f"{mean_val:.2f} ± {std_val:.2f}"
        else:
            summary[col] = f"{mean_val:.2f} ± {std_val:.2f}"
    
    return original_metrics, boot_df

def analyze_np_results_for_fp(fp_type, input_file, output_dir):
    """특정 FP 타입의 NP 결과를 분석"""
    print(f"\n[INFO] Analyzing {fp_type}...")
    
    try:
        # NP 결과 로드 (헤더가 있는 경우와 없는 경우 모두 처리)
        try:
            # 먼저 헤더가 있는지 확인
            df = pd.read_csv(input_file, sep='\t')
            if 'id' in df.columns and 'npscore' in df.columns:
                # 헤더가 있는 경우
                df = df[['id', 'npscore']]
            else:
                # 헤더가 없는 경우
                df = pd.read_csv(input_file, sep='\t', header=None, names=['id', 'npscore'])
        except:
            # 헤더가 없는 경우
            df = pd.read_csv(input_file, sep='\t', header=None, names=['id', 'npscore'])
        
        # 컬럼명 매핑
        df = df.rename(columns={'npscore': 'Score', 'id': 'ID'})
        
        # Score 정규화 적용
        df = normalize_scores(df)
        
        # utils.py의 process_np_results 함수 적용
        processed_df, analysis_results = process_np_results(df, output_dir, fp_type)
        
        print(f"[✓] {fp_type} analysis completed")
        return processed_df, analysis_results
        
    except Exception as e:
        print(f"[ERROR] Failed to analyze {fp_type}: {e}")
        return None, None

def calculate_bedroc_metrics_for_fp(fp_type, input_file, top_fractions=[0.1, 0.2, 0.3], alpha=20.0, n_bootstrap=100):
    """특정 FP 타입의 BEDROC 메트릭 계산 (RDKit 사용)"""
    print(f"[INFO] Calculating BEDROC metrics for {fp_type}...")
    
    try:
        # NP 결과 로드 (헤더가 있는 경우와 없는 경우 모두 처리)
        try:
            # 먼저 헤더가 있는지 확인
            df = pd.read_csv(input_file, sep='\t')
            if 'id' in df.columns and 'npscore' in df.columns:
                # 헤더가 있는 경우
                df = df[['id', 'npscore']]
            else:
                # 헤더가 없는 경우
                df = pd.read_csv(input_file, sep='\t', header=None, names=['id', 'npscore'])
        except:
            # 헤더가 없는 경우
            df = pd.read_csv(input_file, sep='\t', header=None, names=['id', 'npscore'])
        
        # 컬럼명 매핑
        df = df.rename(columns={'npscore': 'Score', 'id': 'ID'})
        
        # Score 정규화 적용
        df = normalize_scores(df)
        
        # RDKit을 사용한 BEDROC 메트릭 계산
        original_metrics, forward_boot_df = bootstrap_metrics_rdkit(
            df, top_fractions=top_fractions, alpha=alpha, n_bootstrap=n_bootstrap
        )
        
        # 부트스트래핑 결과의 표준편차 계산
        bootstrap_summary = {}
        if not forward_boot_df.empty:
            for col in forward_boot_df.columns:
                mean_val = forward_boot_df[col].mean()
                std_val = forward_boot_df[col].std()
                bootstrap_summary[col] = f"{mean_val:.2f} ± {std_val:.2f}"
        
        # 원본 메트릭과 표준편차를 결합하여 표시 (LOO_analyze 방식)
        combined_metrics = {}
        
        # 원본 메트릭에 표준편차 정보 추가
        for key, original_value in original_metrics.items():
            if key in bootstrap_summary:
                # 부트스트래핑 결과에서 표준편차 추출
                bootstrap_str = bootstrap_summary[key]
                std_val = float(bootstrap_str.split(' ± ')[1])
                combined_metrics[key] = f"{original_value:.2f} ± {std_val:.2f}"
            else:
                combined_metrics[key] = f"{original_value:.2f}"
        
        # 부트스트래핑 요약을 별도 키로도 저장 (참조용)
        for key, value in bootstrap_summary.items():
            combined_metrics[f"{key}_bootstrap"] = value
        
        # 역방향 메트릭도 계산 (기존 호환성을 위해)
        reverse_metrics = {}
        
        print(f"[✓] BEDROC metrics calculated for {fp_type}")
        return combined_metrics, forward_boot_df, reverse_metrics
        
    except Exception as e:
        print(f"[ERROR] Failed to calculate BEDROC metrics for {fp_type}: {e}")
        return None, None, None

def create_network_visualization(fp_type, edge_file, output_dir):
    """네트워크 시각화 생성 (suffix별 색상 통일)"""
    print(f"[INFO] Creating network visualization for {fp_type}...")
    try:
        df = pd.read_csv(edge_file, sep=r"\s+", header=None, names=["src", "dst", "weight"])
        G = nx.from_pandas_edgelist(df, source="src", target="dst", edge_attr="weight", create_using=nx.Graph())
        def get_dataset(node:str) -> str:
            return node.split('_')[-1]
        node_suffixes = [get_dataset(n) for n in G.nodes()]
        node_colors = [COLOR_MAP.get(suffix, 'gray') for suffix in node_suffixes]
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(10, 8))
        nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.4)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=60, linewidths=0.2, edgecolors="k")
        # 범례(데이터셋 → 색) - 실제 존재하는 suffix만 표시
        existing_suffixes = list(set(node_suffixes))  # 실제 데이터에 존재하는 suffix들
        
        for suffix in existing_suffixes:
            if suffix in COLOR_MAP:
                plt.scatter([], [], c=[COLOR_MAP[suffix]], label=suffix, s=80)
        plt.legend(title="Dataset", frameon=False, loc="upper right")
        plt.axis("off")
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"net_by_{fp_type}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[✓] Network visualization saved to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to create network visualization for {fp_type}: {e}")
        
def parse_args():
    parser = argparse.ArgumentParser(description="analyze all fps including SubDyve")
    parser.add_argument("--target", type=str, required=True, help="Target name (e.g. SIDER, TOXIC)")
    return parser.parse_args()

def find_best_model_for_each_metric(all_bootstraps):
    """각 메트릭별로 최고 성능을 보이는 모델을 찾습니다."""
    best_models = {}
    
    # 사용 가능한 메트릭들 확인
    if not all_bootstraps:
        return best_models
    
    # 첫 번째 모델에서 메트릭 목록 추출
    first_model = list(all_bootstraps.keys())[0]
    available_metrics = all_bootstraps[first_model].columns.tolist()
    
    for metric in available_metrics:
        best_model = None
        best_mean = -np.inf
        
        for model_name, boot_df in all_bootstraps.items():
            if metric in boot_df.columns:
                mean_val = boot_df[metric].mean()
                if not np.isnan(mean_val) and mean_val > best_mean:
                    best_mean = mean_val
                    best_model = model_name
        
        if best_model:
            best_models[metric] = best_model
    
    return best_models

def perform_statistical_tests_subdyve_vs_best(all_bootstraps):
    """SubDyve를 각 메트릭의 최고 성능 모델과 비교하여 통계적 유의성을 검정합니다."""
    significance_results = {}
    
    # SubDyve 데이터 확인
    if "SubDyve" not in all_bootstraps:
        print("Warning: SubDyve not found in bootstrap results")
        return significance_results
    
    subdyve_data = all_bootstraps["SubDyve"]
    
    # 각 메트릭별 최고 성능 모델 찾기
    best_models = find_best_model_for_each_metric(all_bootstraps)
    
    # 검정할 메트릭 목록 (기존 + 역방향)
    metrics_to_test = [
        'AUROC', 'BEDROC', 'EF@10%', 'EF@20%', 'EF@30%'
    ]
    
    for metric_name in metrics_to_test:
        if metric_name in best_models and metric_name in subdyve_data.columns:
            best_model_name = best_models[metric_name]
            best_model_data = all_bootstraps[best_model_name]
            
            if metric_name in best_model_data.columns:
                try:
                    # NaN 값 제거
                    subdyve_values = subdyve_data[metric_name].dropna()
                    best_values = best_model_data[metric_name].dropna()
                    
                    if len(subdyve_values) > 0 and len(best_values) > 0:
                        # Mann-Whitney U test 수행 (SubDyve vs Best)
                        u_statistic, p_value = mannwhitneyu(
                            subdyve_values, 
                            best_values,
                            alternative='less'  # SubDyve가 Best보다 낮은지 검정
                        )
                        
                        significance_results[metric_name] = {
                            'best_model': best_model_name,
                            'best_mean': best_values.mean(),
                            'subdyve_mean': subdyve_values.mean(),
                            'p_value': p_value,
                            'significance': '',
                            'statistic': u_statistic
                        }
                        
                        # 유의성 표시 (SubDyve가 Best보다 유의미하게 낮은 경우)
                        if p_value < 0.001:
                            significance_results[metric_name]['significance'] = '***'
                        elif p_value < 0.01:
                            significance_results[metric_name]['significance'] = '**'
                        elif p_value < 0.05:
                            significance_results[metric_name]['significance'] = '*'
                        else:
                            significance_results[metric_name]['significance'] = ''
                    else:
                        significance_results[metric_name] = {
                            'best_model': best_model_name,
                            'best_mean': np.nan,
                            'subdyve_mean': np.nan,
                            'p_value': np.nan,
                            'significance': '',
                            'statistic': np.nan
                        }
                        
                except Exception as e:
                    print(f"Error in statistical test for {metric_name}: {e}")
                    significance_results[metric_name] = {
                        'best_model': best_model_name,
                        'best_mean': np.nan,
                        'subdyve_mean': np.nan,
                        'p_value': np.nan,
                        'significance': '',
                        'statistic': np.nan
                    }
    
    return significance_results

def main():
    args = parse_args()
    target = args.target
    
    """메인 함수"""
    print("[INFO] Starting comprehensive FP analysis...")
    
    # matplotlib 스타일 설정
    setup_matplotlib_style()
    
    # FP 타입 정의
    cdktypes = ['standard', 'extended', 'graph', 'maccs', 'pubchem', 'estate', 'hybridization', 'klekota-roth', 'cdk-substructure']
    rdktypes = ['rdkit', 'morgan', 'avalon']
    babeltypes = ['fp2', 'fp4']
    alltypes = cdktypes + rdktypes + babeltypes
    
    # 경로 설정
    base_dir = Path(".")
    result_dir = base_dir / f"result_{target}"
    plots_dir = base_dir / f"plots_{target}"
    
    # 출력 디렉토리 생성
    plots_dir.mkdir(exist_ok=True)
    
    # BEDROC 메트릭 저장용 리스트
    bedroc_results = []
    
    # 부트스트랩 결과 저장용 딕셔너리 (통계 검정용)
    all_bootstraps = {}
    
    # 각 FP 타입별로 분석 수행
    for fp_type in alltypes:
        print(f"\n{'='*50}")
        print(f"Processing {fp_type}")
        print(f"{'='*50}")
        
        # 파일 경로 설정
        np_file = result_dir / f"NP_edge_0.85_{fp_type}_nonseed.txt"
        edge_file = result_dir / "0.0" / f"edge_0.85_{fp_type}.txt"
        
        # NP 결과 분석
        if np_file.exists():
            processed_df, analysis_results = analyze_np_results_for_fp(
                fp_type, np_file, plots_dir
            )
            
            # BEDROC 메트릭 계산 (RDKit 사용)
            bedroc_metrics, forward_boot_df, reverse_metrics = calculate_bedroc_metrics_for_fp(fp_type, np_file)
            if bedroc_metrics:
                bedroc_metrics['Model'] = f"NP_{fp_type}"
                bedroc_results.append(bedroc_metrics)
                
                # 부트스트랩 결과 저장 (통계 검정용)
                all_bootstraps[f"NP_{fp_type}"] = forward_boot_df
        else:
            print(f"[WARNING] NP file not found: {np_file}")
        
        # 네트워크 시각화
        if edge_file.exists():
            create_network_visualization(fp_type, edge_file, plots_dir)
        else:
            print(f"[WARNING] Edge file not found: {edge_file}")
    
    # SubDyve 결과 분석
    print(f"\n{'='*50}")
    print("Processing SubDyve results")
    print(f"{'='*50}")
    
    subdyve_file = base_dir / "source_SUBDYVE" / "Network" / target / "output" / "NP_SUBDYVE_result_nonseed.txt"
    if subdyve_file.exists():
        # SubDyve 결과 분석
        processed_df, analysis_results = analyze_np_results_for_fp(
            "SUBDYVE", subdyve_file, plots_dir
        )
        
        # BEDROC 메트릭 계산 (RDKit 사용)
        bedroc_metrics, forward_boot_df, reverse_metrics = calculate_bedroc_metrics_for_fp("SUBDYVE", subdyve_file)
        if bedroc_metrics:
            bedroc_metrics['Model'] = "SubDyve"
            bedroc_results.append(bedroc_metrics)
            
            # 부트스트랩 결과 저장 (통계 검정용)
            all_bootstraps["SubDyve"] = forward_boot_df
    else:
        print(f"[WARNING] SubDyve file not found: {subdyve_file}")
        print(f"Expected path: {subdyve_file}")
        print(f"Available directories in source_SUBDYVE/Network/: {list((base_dir / 'source_SUBDYVE' / 'Network').glob('*')) if (base_dir / 'source_SUBDYVE' / 'Network').exists() else 'Directory not found'}")
    
    # 통계적 유의성 검정 수행
    print(f"\n{'='*50}")
    print("Performing statistical significance tests...")
    print(f"{'='*50}")
    
    if len(all_bootstraps) > 1:
        significance_results = perform_statistical_tests_subdyve_vs_best(all_bootstraps)
        
        # 통계적 유의성 결과를 별도 CSV로 저장
        significance_summary = []
        for metric, sig_info in significance_results.items():
            if not np.isnan(sig_info['p_value']):
                significance_summary.append({
                    'Metric': metric,
                    'Best_Model': sig_info['best_model'],
                    'Best_Mean': sig_info['best_mean'],
                    'SubDyve_Mean': sig_info['subdyve_mean'],
                    'p_value': sig_info['p_value'],
                    'Significance': sig_info['significance'],
                    'U_statistic': sig_info['statistic']
                })
        
        if significance_summary:
            significance_df = pd.DataFrame(significance_summary)
            significance_csv = plots_dir / "statistical_significance_summary.csv"
            significance_df.to_csv(significance_csv, index=False)
            print(f"\n[✓] Statistical significance results saved to: {significance_csv}")
            
            # 통계적 유의성 요약 출력
            print(f"\nStatistical Significance Summary (SubDyve vs Best Model):")
            print("*** p < 0.001, ** p < 0.01, * p < 0.05")
            print("Alternative: 'less' (SubDyve < Best)")
            for metric, sig_info in significance_results.items():
                if not np.isnan(sig_info['p_value']):
                    print(f"\n{metric}:")
                    print(f"  Best Model: {sig_info['best_model']} (Mean: {sig_info['best_mean']:.4f})")
                    print(f"  SubDyve: {sig_info['subdyve_mean']:.4f}")
                    print(f"  p-value: {sig_info['p_value']:.4f} {sig_info['significance']}")
                    print(f"  U-statistic: {sig_info['statistic']:.2f}")
    
    # BEDROC 결과를 CSV로 저장
    if bedroc_results:
        bedroc_df = pd.DataFrame(bedroc_results)
        
        # 컬럼 순서 정리
        ordered_cols = ['Model', 'AUROC', 'BEDROC', 'EF@10%', 'EF@20%', 'EF@30%']
        ordered_cols = [col for col in ordered_cols if col in bedroc_df.columns]
        bedroc_df = bedroc_df[ordered_cols]
        
        # CSV 저장
        csv_file = plots_dir / "bedroc_metrics_summary.csv"
        bedroc_df.to_csv(csv_file, index=False)
        print(f"\n[✓] BEDROC metrics saved to: {csv_file}")
        print("\nBEDROC Metrics Summary:")
        print(bedroc_df.to_string(index=False))
    
    print(f"\n{'='*50}")
    print("Analysis completed!")
    print(f"Results saved in: {plots_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()

# python analyze_all_fps.py --target TOXIC