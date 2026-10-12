import os
import sys
import argparse
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import statistic
import matplotlib.pyplot as plt
from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
import pickle
import math
import time

MAX_CUT = 5000

# def MakeSeedFile(sSeedFile, dfSeed, dSeedCut):
def MakeSeedFile(sSeedFile, dfSeed):
    nNumPositive = 0
    nNumNegative = 0
    new_ids = []

    with open(sSeedFile, "w") as fp:
        for idx, row in dfSeed.iterrows():
            new_id = row['id'].replace("t", "p")
            new_ids.append(new_id)
            fp.write(new_id + "\t" + "1" + "\n")
            nNumPositive += 1

    dfSeed['ID'] = new_ids  # ✅ 핵심: ID 컬럼을 업데이트해서 downstream에서 반영되게
    print("- Tested Compound(all/seed(p/n)): ", len(dfSeed), nNumPositive, nNumNegative)    
    return dfSeed


# def MakeEdgeFile(sEdgeFile, dfComp, dfSeed, dictSimComp, dictSimSeed, dictSimCompSeed, dEdgeCut):
#     # make edge
#     listComp = dfComp['ID'].tolist()
#     listSeed = dfSeed['ID'].tolist()
#     dictComp, dictSeed = {}, {}
#     for i in range(len(listComp)):  dictComp[str(i)] = str(listComp[i])
#     for i in range(len(listSeed)):  dictSeed[str(i)] = str(listSeed[i])
#     for sType in dictSimSeed.keys():
#         sFinalEdgeFile = sEdgeFile + "_" + sType + ".txt"
#         nCount = 0
#         with open(sFinalEdgeFile, "w") as fp:
#             # Seed
#             for sLine in dictSimSeed[sType]:
#                 item = sLine.split("\t")
#                 fp.write(dictSeed[item[0]] + "\t" + dictSeed[item[1]] + "\t" + item[2] + "\n")
#             # Comp
#             for sLine in dictSimComp[sType]:
#                 item = sLine.split("\t")
#                 fp.write(dictComp[item[0]] + "\t" + dictComp[item[1]] + "\t" + item[2] + "\n")
#             # Comp and Seed
#             nCountSeed = len(dictSimSeed[sType])
#             listT = []
#             for i in range(101):    listT.append([])
#             for sLine in dictSimCompSeed[sType]:
#                 item = sLine.split("\t")
#                 listT[int(float(item[2]) * 100)].append(dictSeed[item[0]] + "\t" + dictComp[item[1]] + "\t" + item[2] + "\n")

#             for i in reversed(range(101)):
#                 if i >= 85:     
#                     for sLine in listT[i]:  fp.write(sLine)
#                     nCount += len(listT[i])
#                 elif i < 40 or nCount >= nCountSeed:
#                     print(i+1)
#                     break
#                 else:
#                     for sLine in listT[i]:  fp.write(sLine)
#                     nCount += len(listT[i])


#             print(sType, "# of (Comp/Seed/combined): ", len(dictSimComp[sType]), nCountSeed, nCount)
#     return
def MakeEdgeFile(sEdgeFile, dfComp, dfSeed, dictSimComp, dictSimSeed, dictSimCompSeed, dEdgeCut):
    for sType in dictSimSeed.keys():
        sFinalEdgeFile = sEdgeFile + "_" + sType + ".txt"
        nCount = 0
        with open(sFinalEdgeFile, "w") as fp:
            # # Seed-Seed
            # for sLine in dictSimSeed[sType]:
            #     fp.write(sLine + "\n")
            # Seed–Seed
            for sLine in dictSimSeed[sType]:
                fp.write(sLine.replace("t_", "p_") + "\n")

            # Comp-Comp
            for sLine in dictSimComp[sType]:
                fp.write(sLine + "\n")
                
            # Comp-Seed
            nCountSeed = len(dictSimSeed[sType])
            listT = [[] for _ in range(101)]

            for sLine in dictSimCompSeed[sType]:
                item = sLine.strip().split("\t")  # [seed_id, comp_id, sim_score]
                sim = float(item[2])
                sim_int = int(sim * 100)
                
                comp_id = item[0]  # compound는 그대로 유지
                seed_id = item[1].replace("t_", "p_")  # ✅ seed 쪽만 변환
                score = item[2]

                listT[sim_int].append(f"{seed_id}\t{comp_id}\t{score}\n")

            # # Comp-Seed
            # nCountSeed = len(dictSimSeed[sType])
            # listT = [[] for _ in range(101)]

            # for sLine in dictSimCompSeed[sType]:
            #     item = sLine.strip().split("\t")
            #     sim = float(item[2])
            #     sim_int = int(sim * 100)
            #     listT[sim_int].append(sLine + "\n")

            for i in reversed(range(101)):
                if i >= 85:
                    for sLine in listT[i]:
                        fp.write(sLine)
                        nCount += 1
                elif i < 20 or nCount >= nCountSeed: #? 40 -> 20 -> 10
                    break
                else:
                    for sLine in listT[i]:
                        fp.write(sLine)
                        nCount += 1

            print(sType, "# of (Comp/Seed/combined): ",
                  len(dictSimComp[sType]), nCountSeed, nCount)



# def MakeNetwork(sInputPath, dCompCut, dEdgeCut, dSeedCut, bIsDraw):
def MakeNetwork(sInputPath, dCompCut, dEdgeCut, bIsDraw):
    sOutputPath = sInputPath + str(dCompCut) + "/"
    sSimCompFile = sInputPath + "Sim_Comp_" + str(dCompCut) + ".pickle"
    sSimCompSeedFile = sInputPath + "Sim_CompSeed_" + str(dCompCut) + ".pickle"
    sSimSeedFile = sInputPath + "Sim_Seed_" + str(dCompCut) + ".pickle"

    sDataCompFile = sInputPath + "Data_Comp_" + str(dCompCut) + ".csv"
    sDataSeedFile = sInputPath + "Data_Seed_" + str(dCompCut) + ".csv"

    # check the necessary files
    if not os.path.isdir(sOutputPath):  os.mkdir(sOutputPath)
    try:
        dfComp = pd.read_csv(sDataCompFile)
        dfSeed = pd.read_csv(sDataSeedFile)
        with open(sSimCompFile, "rb") as handle:        dictSimComp = pickle.load(handle)
        with open(sSimSeedFile, "rb") as handle:        dictSimSeed = pickle.load(handle)
        with open(sSimCompSeedFile, "rb") as handle:    dictSimCompSeed = pickle.load(handle)
    except Exception as e:
        print(e)
        return True
    print("Done: Load dataset")

    # sSeedFile = sOutputPath + "seed_" + str(int(dSeedCut)) + ".txt"
    # sEdgeFile = sOutputPath + "edge_" + str(int(dSeedCut)) + "_" + str(dEdgeCut)
    # dfSeed = MakeSeedFile(sSeedFile, dfSeed, dSeedCut)
    sSeedFile = sOutputPath + "seed" + ".txt"
    sEdgeFile = sOutputPath + "edge_" + str(dEdgeCut)
    dfSeed = MakeSeedFile(sSeedFile, dfSeed)
    
    start_edge_time = time.time()
    MakeEdgeFile(sEdgeFile, dfComp, dfSeed, dictSimComp, dictSimSeed, dictSimCompSeed, dEdgeCut)
    with open(sOutputPath + "time_makeNetwork.txt", "w") as f:
        elapsed_time = time.time() - start_edge_time
        readable_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))  # 시간, 분, 초로 변환
        f.write(readable_time)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",type=str,dest="compCutoff",action="store",help="Compound Cutoff", default="0")
    parser.add_argument("-e",type=str,dest="edgeCutoff",action="store",help="Edge Cutoff")
    parser.add_argument("-i",type=str,dest="inputPath",action="store",help="Input Path")
    # parser.add_argument("-s",type=str,dest="seedCutoff",action="store",help="Seed Cutoff")
    parser.add_argument("-d","--draw",action="store_true")
    args = parser.parse_args()

    #if args.anchor == None or args.simMetric == None or args.targetGeneOrCompound == None:
    #    print("Error: -a -s -t are necessary options")
    sInputPath = args.inputPath
    # MakeNetwork(sInputPath+"/", float(args.compCutoff), float(args.edgeCutoff), float(args.seedCutoff), args.draw)
    MakeNetwork(sInputPath+"/", float(args.compCutoff), float(args.edgeCutoff), args.draw)