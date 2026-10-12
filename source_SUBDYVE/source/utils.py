

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
from collections import Counter
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score

# 전역 색상 매핑 상수 (중앙 집중식 관리)
ALL_SUFFIXES = ['DILI', 'OFT', 'US', 'T3DB', 'SIDER']

COLOR_MAP = {
    'DILI': '#1f77b4',  # 파란색
    'OFT': '#ff7f0e',   # 주황색  
    'US': 'red',        # 빨간색
    'T3DB': '#2ca02c',   # 초록색
    'SIDER': '#9467bd'   # 보라색
}

def sorting_key(id_value):
    if id_value.startswith('C'):    # make harder to find
        return (2, id_value)
    elif id_value.startswith('D'):
        return (0, id_value)
    else:
        return (1, id_value)

def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def analyze_us_ranking(df, output_dir, target):
    """US 노드의 랭킹 분석 및 시각화"""
    
    # 1. 기본 통계
    total_nodes = len(df)
    us_nodes = df[df["Is_US"]]
    non_us_nodes = df[~df["Is_US"]]
    
    us_count = len(us_nodes)
    us_ranks = us_nodes["Rank"].tolist()
    us_scores = us_nodes["Score"].tolist()
    
    # 2. 랭킹 통계
    us_rank_stats = {
        "total_nodes": total_nodes,
        "us_count": us_count,
        "us_percentage": us_count / total_nodes * 100,
        "mean_rank": np.mean(us_ranks),
        "median_rank": np.median(us_ranks),
        "min_rank": np.min(us_ranks),
        "max_rank": np.max(us_ranks),
        "std_rank": np.std(us_ranks),
        "mean_score": np.mean(us_scores),
        "median_score": np.median(us_scores)
    }
    
    # 3. 분위수 분석
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    us_percentile_ranks = np.percentile(us_ranks, percentiles)
    
    # 4. 하위권 집중도 분석
    bottom_10_percent = total_nodes * 0.1
    bottom_25_percent = total_nodes * 0.25
    bottom_50_percent = total_nodes * 0.5
    
    us_in_bottom_10 = len(us_nodes[us_nodes["Rank"] > total_nodes - bottom_10_percent])
    us_in_bottom_25 = len(us_nodes[us_nodes["Rank"] > total_nodes - bottom_25_percent])
    us_in_bottom_50 = len(us_nodes[us_nodes["Rank"] > total_nodes - bottom_50_percent])
    
    concentration_stats = {
        "us_in_bottom_10%": us_in_bottom_10,
        "us_in_bottom_25%": us_in_bottom_25,
        "us_in_bottom_50%": us_in_bottom_50,
        "concentration_bottom_10%": us_in_bottom_10 / us_count * 100 if us_count > 0 else 0,
        "concentration_bottom_25%": us_in_bottom_25 / us_count * 100 if us_count > 0 else 0,
        "concentration_bottom_50%": us_in_bottom_50 / us_count * 100 if us_count > 0 else 0
    }
    
    # 5. Mann-Whitney U 검정 (US vs non-US 랭킹 차이)
    if len(non_us_nodes) > 0:
        mw_stat, mw_pvalue = stats.mannwhitneyu(us_ranks, non_us_nodes["Rank"].tolist(), alternative='greater')
        rank_difference_test = {
            "mann_whitney_stat": mw_stat,
            "mann_whitney_pvalue": mw_pvalue,
            "us_ranked_higher": mw_pvalue < 0.05
        }
    else:
        rank_difference_test = {"error": "No non-US nodes for comparison"}
    
    # 6. 시각화
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    
    # 6.1 US 랭킹 분포 히스토그램
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 히스토그램
    axes[0,0].hist(us_ranks, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0,0].axvline(np.mean(us_ranks), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(us_ranks):.1f}')
    axes[0,0].axvline(np.median(us_ranks), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(us_ranks):.1f}')
    axes[0,0].set_xlabel('Rank')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('US Nodes Rank Distribution')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 랭킹 vs 스코어 산점도
    axes[0,1].scatter(us_ranks, us_scores, alpha=0.7, color='red', s=50)
    axes[0,1].set_xlabel('Rank')
    axes[0,1].set_ylabel('Score')
    axes[0,1].set_title('US Nodes: Rank vs Score')
    axes[0,1].grid(True, alpha=0.3)
    
    # 분위수 플롯
    axes[1,0].bar(range(len(percentiles)), us_percentile_ranks, color='orange', alpha=0.7)
    axes[1,0].set_xlabel('Percentile')
    axes[1,0].set_ylabel('Rank')
    axes[1,0].set_title('US Nodes Rank Percentiles')
    axes[1,0].set_xticks(range(len(percentiles)))
    axes[1,0].set_xticklabels([f'{p}%' for p in percentiles])
    axes[1,0].grid(True, alpha=0.3)
    
    # 하위권 집중도
    bottom_categories = ['Bottom 10%', 'Bottom 25%', 'Bottom 50%']
    bottom_counts = [us_in_bottom_10, us_in_bottom_25, us_in_bottom_50]
    axes[1,1].bar(bottom_categories, bottom_counts, color=['red', 'orange', 'yellow'], alpha=0.7)
    axes[1,1].set_ylabel('Number of US Nodes')
    axes[1,1].set_title('US Nodes in Bottom Percentiles')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target}_US_ranking_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. 결과 저장
    analysis_results = {
        "rank_stats": us_rank_stats,
        "percentile_ranks": dict(zip(percentiles, us_percentile_ranks)),
        "concentration_stats": concentration_stats,
        "rank_difference_test": rank_difference_test
    }
    
    return analysis_results

def assign_average_rank(df, score_column='Score'):
    """Tie 처리를 위한 평균 랭킹 할당"""
    df = df.copy()
    df['is_hit'] = df['ID'].str.endswith('_US').astype(int)
    df['_random'] = np.random.rand(len(df))
    df = df.sort_values(by=[score_column, '_random'], ascending=[False, True]).reset_index(drop=True)
    df['rank'] = df.index + 1
    df.drop(columns=['_random'], inplace=True)
    return df

def calculate_metrics_with_tie_handling(data, top_fractions=[0.01], alpha=20.0, k_values=[5, 10, 20]):
    """Tie 처리를 포함한 메트릭 계산"""
    data['is_hit'] = data['ID'].str.endswith('_US').astype(int)
    data = assign_average_rank(data, score_column='Score')

    scores = [[score, label] for score, label in zip(data['Score'], data['is_hit'])]
    col = 1

    ef_list = CalcEnrichment(scores, col, top_fractions)
    bedroc = CalcBEDROC(scores, col, alpha=alpha)

    ranks = np.where(data['is_hit'] == 1)[0] + 1
    mrr = np.mean([1.0 / rank for rank in ranks]) if len(ranks) > 0 else 0.0

    ap_at_k, recall_at_k, ndcg_at_k = {}, {}, {}
    for k in k_values:
        top_k = data.head(k)
        y_true = top_k['is_hit'].values
        y_scores = top_k['Score'].values
        ap_at_k[k] = average_precision_score(y_true, y_scores)
        recall_at_k[k] = np.sum(y_true) / np.sum(data['is_hit']) if np.sum(data['is_hit']) > 0 else 0.0
        ndcg_at_k[k] = ndcg_score([data['is_hit']], [data['Score']], k=k)

    return {
        "Enrichment Factors (EF)": ef_list,
        "BEDROC": bedroc,
        "MRR": mrr,
        "AP@k": ap_at_k,
        "Recall@k": recall_at_k,
        "NDCG@k": ndcg_at_k
    }

def bootstrap_metrics(df, top_fractions, alpha=20.0, k_values=[5, 10, 20], n_bootstrap=100):
    """부트스트랩을 통한 메트릭 계산"""
    df['label'] = df['ID'].str.endswith('_US').astype(int)
    
    boot_results = []

    for _ in range(n_bootstrap):
        bootstrap_df = df.sample(frac=1.0, replace=True, random_state=np.random.randint(0, 100000))

        # AUROC
        if bootstrap_df['label'].nunique() > 1:
            auroc = roc_auc_score(bootstrap_df['label'], bootstrap_df['Score'])
        else:
            auroc = np.nan

        # BEDROC, EF
        metrics = calculate_metrics_with_tie_handling(bootstrap_df, top_fractions=top_fractions, alpha=alpha, k_values=k_values)

        boot_results.append({
            "AUROC": auroc,
            "BEDROC": metrics["BEDROC"],
            **{f"EF@{top_fraction}": ef for top_fraction, ef in zip(top_fractions, metrics["Enrichment Factors (EF)"])},
        })

    boot_df = pd.DataFrame(boot_results)

    # 평균 + 표준편차 계산
    summary = {}
    for col in boot_df.columns:
        mean_val = boot_df[col].mean()
        std_val = boot_df[col].std()

        if col in ["BEDROC", "AUROC"]:
            summary[col] = f"{mean_val * 100:.0f} ± {std_val * 100:.0f}"
        else:
            summary[col] = f"{mean_val:.1f} ± {std_val:.1f}"

    return summary, boot_df  # 부트스트랩 결과도 함께 반환

def calculate_reverse_metrics(df, top_fractions=[0.1, 0.2, 0.3], alpha=20.0, n_bootstrap=100):
    """역방향 메트릭 계산 (US 노드가 하위권에 위치할수록 좋은 경우)"""
    df['label'] = df['ID'].str.endswith('_US').astype(int)
    
    boot_results = []

    for _ in range(n_bootstrap):
        bootstrap_df = df.sample(frac=1.0, replace=True, random_state=np.random.randint(0, 100000))

        # Reverse AUROC (1 - AUROC)
        if bootstrap_df['label'].nunique() > 1:
            auroc = roc_auc_score(bootstrap_df['label'], bootstrap_df['Score'])
            reverse_auroc = 1 - auroc
        else:
            reverse_auroc = np.nan

        # Reverse BEDROC (1 - BEDROC)
        metrics = calculate_metrics_with_tie_handling(bootstrap_df, top_fractions=top_fractions, alpha=alpha)
        reverse_bedroc = 1 - metrics["BEDROC"]

        # Bottom Enrichment Factors (하위권에서 US 노드 농축도)
        total_nodes = len(bootstrap_df)
        us_nodes = bootstrap_df[bootstrap_df['label'] == 1]
        
        bottom_efs = []
        for fraction in top_fractions:
            bottom_count = int(total_nodes * fraction)
            bottom_nodes = bootstrap_df.tail(bottom_count)  # 하위권 노드들
            us_in_bottom = len(bottom_nodes[bottom_nodes['label'] == 1])
            
            if len(us_nodes) > 0:
                expected_us = len(us_nodes) * fraction
                ef = us_in_bottom / expected_us if expected_us > 0 else 0
            else:
                ef = 0
            bottom_efs.append(ef)

        boot_results.append({
            "RAUROC": reverse_auroc,
            "RBEDROC": reverse_bedroc,
            **{f"BEF@{fraction}": ef for fraction, ef in zip(top_fractions, bottom_efs)},
        })

    boot_df = pd.DataFrame(boot_results)

    # 평균 + 표준편차 계산
    summary = {}
    for col in boot_df.columns:
        mean_val = boot_df[col].mean()
        std_val = boot_df[col].std()

        if col in ["RAUROC", "RBEDROC"]:
            summary[col] = f"{mean_val * 100:.0f} ± {std_val * 100:.0f}"
        else:
            summary[col] = f"{mean_val:.1f} ± {std_val:.1f}"

    return summary

def perform_statistical_tests(all_bootstraps, reference_model="SubDyve"):
    """Mann-Whitney U test를 통한 통계적 유의성 검정"""
    from scipy.stats import mannwhitneyu
    
    results = {}
    
    # 기준 모델의 부트스트랩 결과 찾기
    if reference_model not in all_bootstraps:
        print(f"Warning: Reference model {reference_model} not found in bootstrap results")
        return results
    
    reference_data = all_bootstraps[reference_model]
    
    for model_name, model_bootstraps in all_bootstraps.items():
        if model_name == reference_model:
            continue
            
        model_results = {}
        
        # 각 메트릭에 대해 통계 검정 수행
        for metric in reference_data.columns:
            if metric in model_bootstraps.columns:
                # Mann-Whitney U test
                try:
                    stat, p_value = mannwhitneyu(
                        reference_data[metric].dropna(), 
                        model_bootstraps[metric].dropna(), 
                        alternative='two-sided'
                    )
                    
                    # 유의성 표시
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    else:
                        significance = ""
                    
                    model_results[metric] = {
                        'p_value': p_value,
                        'significance': significance,
                        'statistic': stat
                    }
                except:
                    model_results[metric] = {
                        'p_value': np.nan,
                        'significance': '',
                        'statistic': np.nan
                    }
        
        results[model_name] = model_results
    
    return results

def bootstrap_metrics_with_reverse(df, top_fractions=[0.1, 0.2, 0.3], alpha=20.0, n_bootstrap=100):
    """기존 메트릭과 역방향 메트릭을 모두 계산"""
    # 기존 메트릭
    forward_summary, forward_boot_df = bootstrap_metrics(df, top_fractions, alpha, n_bootstrap=n_bootstrap)
    
    # 역방향 메트릭
    reverse_summary = calculate_reverse_metrics(df, top_fractions, alpha, n_bootstrap)
    
    # 통합
    combined_metrics = {**forward_summary, **reverse_summary}
    
    return combined_metrics, forward_boot_df, reverse_summary

def create_bar_plot_with_id_coloring(df, output_dir, target):
    """ID 기반 색상 구분으로 바 플롯 생성"""
    
    # 컬럼명 매핑 (id -> ID)
    if 'id' in df.columns and 'ID' not in df.columns:
        df = df.rename(columns={'id': 'ID'})
    
    # ID에서 '_' 뒤 문자열 추출
    df["ID_suffix"] = df["ID"].str.split('_').str[-1]
    
    # 중앙 집중식 색상 매핑 사용
    color_map = COLOR_MAP
    
    # US 노드는 빨간색으로 강조
    df["Plot_Color"] = df.apply(lambda row: 'red' if row["Is_US"] else color_map.get(row["ID_suffix"], 'gray'), axis=1)
    
    plt.figure(figsize=(15, 8))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    
    # 바 플롯 생성
    bars = plt.bar(df.index, df["Score"], color=df["Plot_Color"], edgecolor="black", linewidth=0.5, alpha=0.8)
    
    # US 노드 강조
    us_indices = df[df["Is_US"]].index
    for idx in us_indices:
        bars[idx].set_alpha(1.0)
        bars[idx].set_edgecolor('darkred')
        bars[idx].set_linewidth(2)
    
    plt.xlabel("Rank")
    plt.ylabel("Score")
    plt.title(f"[{target}] NP score Distribution")
    
    # 범례 생성 (실제 데이터에 존재하는 suffix만 표시)
    legend_elements = []
    
    # 데이터에 실제로 존재하는 suffix들만 범례에 추가
    existing_suffixes = df["ID_suffix"].unique()
    
    for suffix in existing_suffixes:
        if suffix in COLOR_MAP and suffix != 'US':  # US는 별도로 처리
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=COLOR_MAP[suffix], alpha=0.8, label=suffix))
    
    # US는 항상 마지막에 추가 (빨간색으로 강조)
    if 'US' in existing_suffixes:
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='red', alpha=1.0, label='Beans useful substances'))
    
    plt.legend(handles=legend_elements, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target}_score_distribution_id_colored.png'), dpi=300, bbox_inches='tight')
    plt.close()

def process_np_results(df, output_dir, target):
    """NP 결과 처리 및 다양한 분석 수행"""
    
    # 컬럼명 매핑 (npscore -> Score, id -> ID)
    if 'npscore' in df.columns:
        df = df.rename(columns={'npscore': 'Score', 'id': 'ID'})
    
    # 1. 기본 전처리
    df["Rank"] = df["Score"].rank(ascending=False, method="min")
    df["Group"] = df["ID"].apply(lambda x: "US" if x.endswith("_US") else "non-US")
    df["Is_US"] = df["ID"].str.endswith("_US")
    
    # 2. Score가 0인 경우 처리
    non_zero_min = df[df["Score"] > 0]["Score"].min()
    replacement_value = non_zero_min * 0.1 if non_zero_min > 0 else 0.01
    df["Score"] = df["Score"].apply(lambda x: replacement_value if x == 0 else x)
    
    # 3. US 노드 랭킹 처리
    df_us = df[df["Is_US"]]
    if (df_us["Score"] == 0).all():
        max_rank = len(df)
        df.loc[df["Is_US"], "Rank"] = max_rank
    
    # 4. 정규화 및 라벨링
    df["Normalized_Score"] = (df["Score"] - df["Score"].min()) / (df["Score"].max() - df["Score"].min())
    df["Softmax_Score"] = softmax(df["Normalized_Score"].values)
    df["Sigmoid_Score"] = sigmoid(df["Normalized_Score"].values)
    
    # 5. 독성 라벨 생성 (여러 방법)
    df["Toxic_label_softmax"] = (df["Softmax_Score"] > df["Softmax_Score"].mean()).astype(int)
    df["Toxic_label_sigmoid"] = (df["Sigmoid_Score"] > 0.5).astype(int)
    df["Toxic_label_normalized"] = (df["Normalized_Score"] > 0.5).astype(int)
    
    # 6. 시각화 및 분석
    create_bar_plot_with_id_coloring(df, output_dir, target)
    analysis_results = analyze_us_ranking(df, output_dir, target)
    
    # 7. 결과 저장
    output_path = os.path.join(output_dir, f"{target}_processed_results.csv")
    df.to_csv(output_path, index=False)
    
    # 8. 분석 결과 저장
    analysis_path = os.path.join(output_dir, f"{target}_analysis_summary.txt")
    with open(analysis_path, 'w') as f:
        f.write(f"Analysis Results for {target}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Rank Statistics:\n")
        for key, value in analysis_results["rank_stats"].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nConcentration Statistics:\n")
        for key, value in analysis_results["concentration_stats"].items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nRank Difference Test:\n")
        for key, value in analysis_results["rank_difference_test"].items():
            f.write(f"  {key}: {value}\n")
    
    return df, analysis_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--seed_cutoff', type=float, required=True, default=0.5)
    parser.add_argument('--network_output_dir', type=str, required=True)
    parser.add_argument('--submine_data_dir', type=str, required=True)
    parser.add_argument('--edge_cutoff', type=float, default=0.85)

    args = parser.parse_args()
    target = args.target
    seed_cutoff = str(args.seed_cutoff)

    output_dir = os.path.abspath(args.network_output_dir)
    # data_dir = os.path.join(os.path.abspath(args.submine_data_dir), target)
    data_dir = os.path.abspath(args.submine_data_dir)

    sNPresult_path = os.path.join(output_dir, f"NP_SUBDYVE_{args.seed_cutoff}_{args.edge_cutoff}_subgraph.txt")
    sNPSeed_path = os.path.join(output_dir, f"seed_{seed_cutoff}.txt")
    output_final_path = os.path.join(output_dir, "NP_SUBDYVE_result_nonseed.txt")

    try:
        print(f"Post-processing {target}")

        # NP 결과 로드
        np_df = pd.read_csv(sNPresult_path, sep='\t', header=None, names=['id', 'npscore'])
        np_df.sort_values(by='npscore', ascending=False, inplace=True)
        np_df.reset_index(drop=True, inplace=True)

        # Seed 데이터 로드
        seed_df = pd.read_csv(sNPSeed_path, sep='\t', header=None, names=['id', 'weight'])
        seed_ids = seed_df['id'].tolist()

        # Seed 제외한 결과
        np_excl = np_df[~np_df['id'].isin(seed_ids)].reset_index(drop=True)
        np_excl = np_excl[~np_excl['id'].str.startswith('t')].reset_index(drop=True)

        # COMP 데이터 로드
        comp_ids = pd.read_csv(os.path.join(data_dir, "COMP_DATA.csv"))['ID'].tolist()

        # 누락된 ID 추가
        existing_ids = set(np_excl['id'])
        missing = [
            {'id': i, 'npscore': 0.0} for i in comp_ids if i not in existing_ids
        ]
        np_full = pd.concat([np_excl, pd.DataFrame(missing)], ignore_index=True)

        # 정렬
        np_full_sorted = np_full.sort_values(
            by=['npscore', 'id'],
            ascending=[False, True],
            key=lambda x: x if x.name == 'npscore' else x.map(sorting_key)
        ).reset_index(drop=True)

        # 최종 결과 저장
        np_full_sorted.to_csv(output_final_path, sep='\t', index=False)
        print(f"Saved: {output_final_path}")

        # 추가 분석 수행
        print("Performing additional analysis...")
        analysis_df, analysis_results = process_np_results(np_full_sorted, output_dir, target)
        
        print(f"Analysis complete. Results saved in {output_dir}")

    except Exception as e:
        print(f"Error during post-processing for {target}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
