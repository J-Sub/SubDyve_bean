import os
import pickle
import time
import itertools
import argparse
import pandas as pd
import math
import statistic

def compute_similarity(fp_dict, iteration, sim_type, cross_fp=None):
    if cross_fp is None:
        return {
            "subgraph": statistic.ParallelCalcTanimotoForFPs(
                (fp_dict["subgraph"], iteration, 'subgraph', sim_type)
            )
        }
    else:
        return {
            "subgraph": statistic.ParallelCalcTanimotoForTwoFPs(
                (fp_dict["subgraph"], cross_fp["subgraph"], iteration, 'subgraph', sim_type)
            )
        }

def save_similarity(out_path, name, data):
    file = os.path.join(out_path, f"Sim_{name}.pickle")
    with open(file, "wb") as f:
        pickle.dump(data, f)
    return file

def make_seed_file(dfSeed, seed_cutoff, out_dir):
    path = os.path.join(out_dir, f"seed_{seed_cutoff}.txt")
    # dLogCut = math.log10(seed_cutoff)
    nPos = 0
    with open(path, "w") as f:
        for idx, row in dfSeed.iterrows():
        #     dacvalue = math.log10(row['acvalue'])
        #     if dacvalue < seed_cutoff:
            dfSeed.loc[idx, 'ID'] = row['ID'].replace("t", "p")
            f.write(dfSeed.loc[idx, 'ID'] + "\t1\n")
            nPos += 1
    print(f"Seed file written ({nPos} positive): {path}")
    return path

def make_edge_file(dfS, dfC, sims, edge_cut, out_dir, seed_cutoff):
    idmap = lambda df: {str(i): v for i, v in enumerate(df['ID'].tolist())}
    dictS, dictC = idmap(dfS), idmap(dfC)
    sType = 'subgraph'
    nSeed = len(sims["seed"][sType])

    out_path = os.path.join(out_dir, f"edge_{seed_cutoff}_{edge_cut}_{sType}.txt")
    with open(out_path, "w") as f:
        for line in sims["seed"][sType]:
            i1, i2, sim = line.strip().split('\t')
            f.write(f"{dictS[i1]}\t{dictS[i2]}\t{sim}\n")
        for line in sims["comp"][sType]:
            i1, i2, sim = line.strip().split('\t')
            f.write(f"{dictC[i1]}\t{dictC[i2]}\t{sim}\n")
        def write_cross(sim_list, dict1, dict2, label):
            buckets = [[] for _ in range(101)]
            for line in sim_list[sType]:
                i1, i2, sim = line.strip().split('\t')
                b = min(int(float(sim) * 100), 100)
                buckets[b].append(f"{dict1[i1]}\t{dict2[i2]}\t{sim}\n")
            count = 0
            for i in reversed(range(101)):
                if i >= 50 or (count < nSeed and i >= 1): #! 85 -> 50 / 40 -> 1
                    for l in buckets[i]:
                        f.write(l)
                        count += 1
                else:
                    break
        write_cross(sims["seed_comp"], dictS, dictC, "S-C")
    print(f"Edge file written: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", required=True)
    parser.add_argument("--comp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed_cutoff", type=float, default=0.5)
    parser.add_argument("--edge_cutoff", type=float, default=0.85)
    parser.add_argument("--similarity", default='tanimoto')
    parser.add_argument("--input_fp_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load dataframes
    dfS = pd.read_csv(args.seed)
    dfC = pd.read_csv(args.comp)

    # Load fingerprints
    fps = {
        "seed": pickle.load(open(os.path.join(args.input_fp_dir, "FP_Seed.pickle"), "rb")),
        "comp": pickle.load(open(os.path.join(args.input_fp_dir, "FP_Comp.pickle"), "rb")),
    }

    # Compute similarities
    sims = {
        "seed": compute_similarity(fps["seed"], 0.5, args.similarity), #! 0.85 -> 0.5
        "comp": compute_similarity(fps["comp"], 0.5, args.similarity), #! 0.85 -> 0.5
        "seed_comp": compute_similarity(fps["seed"], 0.01, args.similarity, cross_fp=fps["comp"]), #! 0.5 -> 0.01
    }

    for k, v in sims.items():
        save_similarity(args.output, k.capitalize(), v)
    # Optionally, implement make_seed_file if needed for your workflow
    seed_path = make_seed_file(dfS, args.seed_cutoff, args.output)
    make_edge_file(dfS, dfC, sims, args.edge_cutoff, args.output, args.seed_cutoff)
    print("All processing complete.")


if __name__ == "__main__":
    main()
