import pandas as pd
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Collect best AUC iteration for each target")
    parser.add_argument("--target", type=str, required=True, help="Target name (e.g., TOXIC)")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the resulting CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    target = args.target
    output_path = args.output_path
    
    target_max_AUC_dict = {}
    
    # 지정된 target에 대해 performance.csv 읽기
    try:
        # results 폴더에서 performance.csv 읽기
        performance_path = f"./SSM/results/{target}/performance.csv"
        
        if not os.path.exists(performance_path):
            print(f"Performance file not found: {performance_path}")
            return
            
        df = pd.read_csv(performance_path)
        
        # AUC 컬럼이 있는지 확인
        if "AUC" not in df.columns:
            print(f"AUC column not found in {performance_path}")
            print(f"Available columns: {df.columns.tolist()}")
            return
            
        # AUC 기준으로 최고값을 가진 iteration 찾기
        max_auc_idx = df["AUC"].idxmax()
        max_iter = df.loc[max_auc_idx, "iteration"]
        max_auc_value = df.loc[max_auc_idx, "AUC"]
        
        target_max_AUC_dict[target] = int(max_iter)
        
        print(f"Target: {target}")
        print(f"Best AUC: {max_auc_value:.4f}")
        print(f"Best iteration: {max_iter}")
        
    except Exception as e:
        print(f"Failed to load performance for {target}: {e}")
        return

    # 결과를 CSV로 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df = pd.DataFrame(target_max_AUC_dict.items(), columns=["target", "max_AUC_iteration"])
    result_df.to_csv(output_path, index=False)
    print(f"Best AUC iteration saved to {output_path}")

if __name__ == "__main__":
    main()
