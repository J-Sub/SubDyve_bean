from multiprocessing import Pool
import itertools
import os
import pickle
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import argparse
# from openbabel import pybel
# import openbabel
# from PyFingerprint.fingerprint import get_fingerprint, get_fingerprints
import statistic
import time

# cdktypes = []
# rdktypes = []
cdktypes = ['standard',
            'extended',
            'graph',
            'maccs',
            'pubchem',
            'estate',
            'hybridization',
            # 'lingo',
            'klekota-roth',
            # 'shortestpath', 'signature',
            'cdk-substructure']
rdktypes = ['rdkit', 'morgan',  # 'rdk-maccs', 'topological-torsion',
            'avalon']  # , 'atom-pair']
babeltypes = ['fp2', 'fp4']  # ,fp3']
# vectypes = ['mol2vec']#, 'heteroencoder']


def MakeNetwork(sCompFile, sAnchorFile, dCutoffScore, bIsDraw, sOutputPath, similarity):
    """
    Function: Make network
    """

    print(sCompFile, sAnchorFile, dCutoffScore)
    # read compounds
    dfComp = pd.read_csv(sCompFile)
    #!
    # dfComp = dfComp[dfComp['pred'] >= dCutoffScore]
    # dfComp = dfComp[dfComp['acvalue'].isna()]
    #!
    # dfComp = dfComp[(dfComp['zinc_id']!=100095998) & (dfComp['zinc_id']!=100096000)]
    # dfComp.drop_duplicates(subset=['SMILES'], keep='first', inplace=True)
    # dfComp.rename(columns={'SMILES' : 'smiles'}, inplace=True)
    # dfComp = dfComp.rename(columns={'zinc_id': 'id'})

    dfSeed = pd.read_csv(sAnchorFile)
    # dfSeed.rename(columns={'SMILES' : 'smiles'}, inplace=True)
    # dfSeed.drop_duplicates(subset='smiles', keep='first', inplace=True)
    # dfSeed = dfSeed[dfSeed['matched_pattern'].notnull()]
    dfSeed["id"] = ""
    # dfSeed = dfSeed[dfSeed['activity'] == 'Active']
    #!
    # for idx, row in dfSeed.iterrows():
    #     if pd.isnull(row['IC50 (nM)']):
    #         if not pd.isnull(row['Kd (nM)']):
    #             dfSeed.loc[idx, 'IC50 (nM)'] = row['Kd (nM)']
    #         elif not pd.isnull(row['EC50 (nM)']):
    #             dfSeed.loc[idx, 'IC50 (nM)'] = row['EC50 (nM)'] 
    #         elif not pd.isnull(row['Ki (nM)']):
    #             dfSeed.loc[idx, 'IC50 (nM)'] = row['Ki (nM)']

    # dfSeed = dfSeed[['Ligand SMILES', 'IC50 (nM)']]
    #!
    
    # dfSeed.columns = ['smiles', 'pred', "id"]
    # dfSeed = dfSeed.dropna()

    # remove duplication
    print("### Remove Duplications ###")
    print("Before:", len(dfComp), len(dfSeed))
    dfSeed, dfComp = statistic.RemoveDup(dfSeed, dfComp)
    dfSeed = dfSeed.reset_index(drop=True)
    dfComp = dfComp.reset_index(drop=True)
    print("After:", len(dfComp), len(dfSeed))

    dfSeed['index'] = dfSeed.index
    dfSeed = dfSeed.astype({'index': 'str'})
    # dfSeed = dfSeed.astype({'acvalue': 'str'})
    # dfSeed['id'] = 't' + dfSeed['index'] + '_' + dfSeed['acvalue']
    dfSeed['id'] = 't' + '_' + dfSeed['ID']
    # dfSeed['pred'] = dfSeed['pred'].str.replace(">", "")
    # dfSeed['pred'] = dfSeed['pred'].str.replace("<", "")
    # dfSeed = dfSeed.astype({'acvalue': 'float'})
    dfSeed.drop(['index'], axis=1, inplace=True)

    # save compound data
    dfComp.to_csv(sOutputPath + "Data_Comp_"+str(dCutoffScore)+".csv")
    dfSeed.to_csv(sOutputPath + "Data_Seed_"+str(dCutoffScore)+".csv")

    alltypes = cdktypes
    alltypes.extend(rdktypes)
    alltypes.extend(babeltypes)

    # make FP for seed
    dictSeedFP = {}
    sFPSeedFile = sOutputPath + "FP_Seed_"+str(dCutoffScore)+".pickle"
    if os.path.isfile(sFPSeedFile):
        with open(sFPSeedFile, "rb") as handle:
            dictSeedFP = pickle.load(handle)
    else:
        for sType in alltypes:
            print(sType)
            dictSeedFP[sType] = statistic.makeFP(dfSeed, sType)
        with open(sFPSeedFile, "wb") as handle:
            pickle.dump(dictSeedFP, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # if bIsDraw:
    #     for sType in alltypes:
    #         statistic.ReportNetwork(dfSeed, dictSeedFP[sType],
    #                                 sType, sOutputPath)

    # make FP for comp
    sFPCompFile = sOutputPath + "FP_Comp_"+str(dCutoffScore)+".pickle"
    dictCompFP = {}
    if os.path.isfile(sFPCompFile):
        with open(sFPCompFile, "rb") as handle:
            dictCompFP = pickle.load(handle)
    else:
        for sType in alltypes:
            dictCompFP[sType] = statistic.makeFP(dfComp, sType)
        with open(sFPCompFile, 'wb') as handle:
            pickle.dump(dictCompFP, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Calc Similarity
    sSimSeedFile = sOutputPath + "Sim_Seed_"+str(dCutoffScore)+".pickle"
    sSimCompSeedFile = sOutputPath+"Sim_CompSeed_"+str(dCutoffScore)+".pickle"
    sSimCompFile = sOutputPath + "Sim_Comp_"+str(dCutoffScore)+".pickle"
    dictSeedSim = {}
    dictCompSeedSim = {}
    dictCompSim = {}
    nCore = 16
    pool = Pool(nCore)

    if os.path.isfile(sSimSeedFile):
        with open(sSimSeedFile, "rb") as handle:
            dictSeedSim = pickle.load(handle)
    else:
        start_seed_sim = time.time()
        listSeedFP = []
        for sType in alltypes:
            listSeedFP.append(dictSeedFP[sType])
        listResult = pool.map(statistic.ParallelCalcTanimotoForFPs,
                              zip(listSeedFP, itertools.repeat(0.5), alltypes, itertools.repeat(similarity)))
        # for i in range(len(alltypes)):
        #     sType = alltypes[i]
        #     dictSeedSim[sType] = listResult[i]
        for i in range(len(alltypes)):
            sType = alltypes[i]
            raw_sim = listResult[i]    # 예: ['0\t3\t0.85', '2\t4\t0.73', ...]
            named_sim = []
            for rec in raw_sim:
                idx_i, idx_j, score = rec.split('\t')
                id_i = dfSeed.iloc[int(idx_i)]['id']
                id_j = dfSeed.iloc[int(idx_j)]['id']
                named_sim.append(f'{id_i}\t{id_j}\t{score}')
            dictSeedSim[sType] = named_sim

        with open(sSimSeedFile, "wb") as handle:
            pickle.dump(dictSeedSim, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(sOutputPath + "time_seed_sim.txt", "w") as f:
            elapsed_time = time.time() - start_seed_sim
            readable_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))  # 시간, 분, 초로 변환
            f.write(readable_time)
    for sType in alltypes:
        print(len(dictSeedSim[sType]), end=" ")
    print()

    if os.path.isfile(sSimCompSeedFile):
        with open(sSimCompSeedFile, "rb") as handle:
            dictCompSeedSim = pickle.load(handle)
    else:
        start_comp_seed_sim = time.time()
        listCompFP = []
        listSeedFP = []
        for sType in alltypes:
            listCompFP.append(dictCompFP[sType])
            listSeedFP.append(dictSeedFP[sType])
        listResult = pool.map(statistic.ParallelCalcTanimotoForTwoFPs,
                              zip(listCompFP, listSeedFP,
                                  itertools.repeat(0.2), alltypes, itertools.repeat(similarity))) #! 0.3 -> 0.15 -> 0.01
        # for i in range(len(alltypes)):
        #     sType = alltypes[i]
        #     dictCompSeedSim[sType] = listResult[i]
        for i in range(len(alltypes)):
            sType = alltypes[i]

            raw_sim = listResult[i]          # ['0\t3\t0.87', …]  ← index 기반
            named_sim = []
            for rec in raw_sim:
                idx_i, idx_j, score = rec.split('\t')
                id_i = dfComp.iloc[int(idx_i)]['ID']        # index → ID
                id_j = dfSeed.iloc[int(idx_j)]['id']
                named_sim.append(f'{id_i}\t{id_j}\t{score}')

            dictCompSeedSim[sType] = named_sim              # ID 기반으로 덮어씀            
            
        with open(sSimCompSeedFile, "wb") as handle:
            pickle.dump(dictCompSeedSim, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(sOutputPath + "time_comp_seed_sim.txt", "w") as f:
            elapsed_time = time.time() - start_comp_seed_sim
            readable_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))  # 시간, 분, 초로 변환
            f.write(readable_time)
    for sType in alltypes:
        print(len(dictCompSeedSim[sType]), end=" ")
    print()

    if os.path.isfile(sSimCompFile):
        with open(sSimCompFile, "rb") as handle:
            dictCompSim = pickle.load(handle)
    else:
        start_comp_sim = time.time()
        listCompFP = []
        for sType in alltypes:
            listCompFP.append(dictCompFP[sType])
        listResult = pool.map(statistic.ParallelCalcTanimotoForFPs,
                              zip(listCompFP, itertools.repeat(0.5), alltypes, itertools.repeat(similarity))) #! 0.85 -> 0.75
        # for i in range(len(alltypes)):
        #     sType = alltypes[i]
        #     dictCompSim[sType] = listResult[i]
        for i in range(len(alltypes)):
            sType = alltypes[i]
            raw_sim = listResult[i]    # 예: ['0\t3\t0.85', '2\t4\t0.73', ...]
            named_sim = []
            for rec in raw_sim:
                idx_i, idx_j, score = rec.split('\t')
                id_i = dfComp.iloc[int(idx_i)]['ID']
                id_j = dfComp.iloc[int(idx_j)]['ID']
                named_sim.append(f'{id_i}\t{id_j}\t{score}')
            dictCompSim[sType] = named_sim
        with open(sSimCompFile, "wb") as handle:
            pickle.dump(dictCompSim, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(sOutputPath + "time_comp_sim.txt", "w") as f:
            elapsed_time = time.time() - start_comp_sim
            readable_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))  # 시간, 분, 초로 변환
            f.write(readable_time)
    for sType in alltypes:
        print(len(dictCompSim[sType]), end=" ")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, dest="compoundFile", action="store",
                        help="Compounds List")
    parser.add_argument("-c", type=str, dest="cutoffScore", action="store",
                        help="Cutoff Score", default="0")
    parser.add_argument("-a", type=str, dest="anchor", action="store",
                        help="AnchorCompound")
    parser.add_argument("-o", type=str, dest="outputpath", action="store",
                        help="Output Path")
    parser.add_argument("-d", '--draw', action='store_true')
    
    parser.add_argument("-s", '--similarity', default='tanimoto')
    
    args = parser.parse_args()

    if args.compoundFile is None or args.cutoffScore is None:
        print("Error: -t -c are necessary options")

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)
    
    MakeNetwork(args.compoundFile, args.anchor, float(args.cutoffScore),
                args.draw, args.outputpath+"/", args.similarity)