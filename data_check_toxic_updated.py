import pandas as pd
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TOXIC data and apply DiSC substructure matching.")
    parser.add_argument("--target", type=str, required=True, help="Target name (e.g. SIDER, TOXIC)")
    return parser.parse_args()


# Softmax 함수 정의
def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))  # 안정성을 위해 점수에서 최대값을 뺌
    return exp_scores / exp_scores.sum()

# Sigmoid 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

args = parse_args()
target = args.target

# 모든 타입 리스트
alltypes = ['standard', 'extended', 'graph', 'maccs', 'pubchem', 'estate', 'hybridization', 'klekota-roth', 'cdk-substructure', 'rdkit', 'morgan', 'avalon', 'fp2', 'fp4']

# DATA_TOTAL_labeled 데이터 로드
if target == 'SIDER':
    DATA_TOTAL_labeled = pd.read_csv(f'data/{target}_TOTAL_labeled_skinonly.csv')
else:
    DATA_TOTAL_labeled = pd.read_csv(f'data/{target}_TOTAL_labeled.csv')

    

# 1. 기존 fingerprint 타입들 처리
for ftype in alltypes:
    # 데이터 로드
    df = pd.read_csv(f"result_{target}/NP_edge_0.85_{ftype}_nonseed.txt", sep="\t", header=None, names=["ID", "Score"])
    df["Group"] = df["ID"].apply(lambda x: "US" if x.endswith("_US") else "non-US")
    df["Is_US"] = df["ID"].str.endswith("_US")

    # Rank 계산
    df["Rank"] = df["Score"].rank(ascending=False, method="min")
    df_us = df[df["Is_US"]]
    if (df_us["Score"] == 0).all():
        max_rank = len(df)
        df.loc[df["Is_US"], "Rank"] = max_rank

    # 정규화 및 Toxic_label 생성
    df["Normalized_Score"] = (df["Score"] - df["Score"].min()) / (df["Score"].max() - df["Score"].min())
    df["Sigmoid_Score"] = sigmoid(df["Normalized_Score"].values)
    df["Toxic_label"] = (df["Sigmoid_Score"] > 0.5).astype(int)
    df["Toxic_label"] = (df["Normalized_Score"] > 0.5).astype(int)

    # 필터링
    filtered_data = df[(df["Is_US"]) & (df["Toxic_label"] == 0)]

    # Number 매핑
    toxic_mapping = DATA_TOTAL_labeled[["ID", "Number"]]
    filtered_table = filtered_data.merge(toxic_mapping, on="ID", how="left")

    # 필요한 열 선택
    result_table = filtered_table[['Number', 'Normalized_Score', 'Toxic_label']]

    # CSV 파일로 저장
    result_table.to_csv(f"result_{target}/filtered_results_{ftype}.csv", index=False)

    print(f"{ftype} - Filtered Table 저장 완료: filtered_results_{ftype}.csv")

# 2. SubDyve 결과 처리
print("\n=== SubDyve 결과 처리 시작 ===")

# SubDyve 데이터 로드
subdyve_df = pd.read_csv(f"source_SUBDYVE/Network/{target}/output/NP_SUBDYVE_result_nonseed.txt", sep="\t")
subdyve_df.columns = ["ID", "Score"]  # 컬럼명 통일

# US 그룹 식별 (ID가 _US로 끝나는 경우)
subdyve_df["Group"] = subdyve_df["ID"].apply(lambda x: "US" if x.endswith("_US") else "non-US")
subdyve_df["Is_US"] = subdyve_df["ID"].str.endswith("_US")

# Rank 계산
subdyve_df["Rank"] = subdyve_df["Score"].rank(ascending=False, method="min")
subdyve_df_us = subdyve_df[subdyve_df["Is_US"]]
if (subdyve_df_us["Score"] == 0).all():
    max_rank = len(subdyve_df)
    subdyve_df.loc[subdyve_df["Is_US"], "Rank"] = max_rank

# 정규화 및 Toxic_label 생성
subdyve_df["Normalized_Score"] = (subdyve_df["Score"] - subdyve_df["Score"].min()) / (subdyve_df["Score"].max() - subdyve_df["Score"].min())
subdyve_df["Sigmoid_Score"] = sigmoid(subdyve_df["Normalized_Score"].values)
subdyve_df["Toxic_label"] = (subdyve_df["Sigmoid_Score"] > 0.5).astype(int)
subdyve_df["Toxic_label"] = (subdyve_df["Normalized_Score"] > 0.5).astype(int)

# 필터링
subdyve_filtered_data = subdyve_df[(subdyve_df["Is_US"]) & (subdyve_df["Toxic_label"] == 0)]

# Number 매핑
toxic_mapping = DATA_TOTAL_labeled[["ID", "Number"]]
subdyve_filtered_table = subdyve_filtered_data.merge(toxic_mapping, on="ID", how="left")

# 필요한 열 선택
subdyve_result_table = subdyve_filtered_table[['Number', 'Normalized_Score', 'Toxic_label']]

# CSV 파일로 저장
subdyve_result_table.to_csv(f"result_{target}/filtered_results_subdyve.csv", index=False)

print("SubDyve - Filtered Table 저장 완료: filtered_results_subdyve.csv")
print("모든 결과 처리 완료!") 