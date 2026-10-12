import os
import pickle
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm


def count_patterns(smiles, patterns):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [sum(1 for _ in mol.GetSubstructMatches(pattern)) for pattern in patterns]
    else:
        return [0] * len(patterns)


def process_dataframe(df, name, patterns, id_key, output_dir):
    tqdm.pandas()
    pattern_counts = df["SMILES"].progress_apply(lambda x: count_patterns(x, patterns))
    pattern_df = pd.DataFrame(pattern_counts.tolist(), columns=[f"P_{i+1}" for i in range(len(patterns))])
    result_df = pd.concat([df, pattern_df], axis=1)

    id2i = {v: i for i, v in enumerate(df[id_key])}
    subgraphFreq_matrix = np.zeros((len(id2i), len(patterns)), dtype=float)

    for i, row in tqdm(result_df.iterrows(), total=result_df.shape[0], desc=f"Building FP for {name}"):
        idx = id2i.get(row[id_key], None)
        if idx is not None:
            subgraphFreq_matrix[idx, :] = row.iloc[-len(patterns):].values

    out_path = output_dir / f"FP_{name}.pickle"
    pickle.dump({"subgraph": subgraphFreq_matrix}, open(out_path, "wb"))
    print(f"Saved: {out_path}")


def prepare_and_submine(args):
    raw_data_dir = Path(args.raw_data_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_dir = output_root
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DiSC patterns
    disc_df = pd.read_csv(args.disc_path)
    patterns = [Chem.MolFromSmarts(s) for s in disc_df["DiSC"].dropna()][:args.top]

    print(f"\n Processing target: {args.target}")

    # === Seed ===
    # seed_path = raw_data_dir / args.target / f"SEED_DATA.csv" #! for BEAN_SUBDYVE
    seed_path = raw_data_dir / f"SEED_DATA.csv" #! for LOO
    seed_df = pd.read_csv(seed_path)
    print(f"Loaded: SEED_DATA.csv")

    # === Comp ===
    # comp_path = raw_data_dir / args.target / f"COMP_DATA.csv" #! for BEAN_SUBDYVE
    comp_path = raw_data_dir / f"COMP_DATA.csv" #! for LOO
    comp_df = pd.read_csv(comp_path)
    print(f"Loaded: COMP_DATA.csv")

    print("Matching DiSC patterns...")
    process_dataframe(seed_df, "Seed", patterns, "ID", output_dir)
    process_dataframe(comp_df, "Comp", patterns, "ID", output_dir)

    print(f"Done: {args.target}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare TOXIC data and apply DiSC substructure matching.")
    parser.add_argument("--target", type=str, required=True, help="Target name (e.g. ADA, EGFR)")
    parser.add_argument("--top", type=int, default=100, help="Top N DiSC patterns")
    parser.add_argument("--disc_path", type=str, required=True, help="Path to DiSC pattern CSV")
    parser.add_argument("--raw_data_dir", type=str, default="./SUBMINE/data", help="Root directory for raw data")
    parser.add_argument("--output_root", type=str, default="./SUBMINE/data", help="Root directory for output data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_and_submine(args)
