"""
Implementation of tissue-specific graph walk with RWR

"""
import sys
import numpy as np
import sys
import argparse
from scipy.sparse import coo_matrix
from sklearn.preprocessing import normalize
import networkx as nx
# convergence criterion - when vector L1 norm drops below 10^(-6)
# (this is the same as the original RWR paper)
CONV_THRESHOLD = 0.000001

def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

def isNum(x):
        try:
                float(x)
                return True
        except:
                return False

class OptimizedWalker:
    def __init__(self, original_ppi, constantWeight=False, absWeight=False, addBidirectionEdge=False):
        self._build_matrices(original_ppi, constantWeight, absWeight, addBidirectionEdge)    
        self.N = self.adjacency_matrix.shape[0]
        print(f"[OptimizedWalker] Node={self.N}")

    def run_exp(self, seed2weight, restart_prob, normalize=False, node_list=[]):
        p_0 = self._set_up_p0(seed2weight)
        if normalize and np.sum(p_0) > 0:
            p_0 = p_0 / np.sum(p_0)

        p_t = p_0.copy()
        arr_p = np.empty((len(p_t), 1))
        arr_p[:, 0] = p_t

        diff_norm = 1.0
        iteration = 0
        CONV_THRESHOLD = 1e-6

        while diff_norm > CONV_THRESHOLD:
            iteration += 1
            p_t_1 = self._calculate_next_p(p_t, p_0, restart_prob)
            if normalize and np.sum(p_t_1) > 0:
                p_t_1 = p_t_1 / np.sum(p_t_1)
            diff_norm = np.linalg.norm(p_t_1 - p_t, 1)
            p_t = p_t_1
            arr_p = np.c_[arr_p, p_t]
            if iteration >= 50000:
                print("[NP] Reached 50,000 iterations, stopping early...")
                break
        print(f"[NP finished] {iteration} iterations, final diff_norm={diff_norm:.6e}")
        total_idx = {node: i for i, node in enumerate(sorted(self.dic_node2idx))}
        output = []
        if node_list:
            for node in node_list:
                i = total_idx[node]
                output.append([node, p_t[i], arr_p[i, :].tolist()])
        else:
            for node in sorted(self.dic_node2idx):
                i = total_idx[node]
                output.append([node, p_t[i], arr_p[i, :].tolist()])
        return output

    def _calculate_next_p(self, p_t, p_0, restart_prob):
        no_restart = self.adjacency_matrix.dot(p_t) * (1 - restart_prob)
        restart = p_0 * restart_prob
        return no_restart + restart

    def _set_up_p0(self, seed2weight):
        p_0 = np.zeros(self.N, dtype=np.float32)
        for seed, weight in seed2weight.items():
            if seed in self.dic_node2idx:
                idx = self.dic_node2idx[seed]
                p_0[idx] = weight
        return p_0

    def _build_matrices(self, original_ppi, constantWeight, absWeight, addBidirectionEdge):
        nodes = set()
        edges = []
        with open(original_ppi, 'r') as f:
            for line in f:
                src, tgt, *weight = line.strip().split('\t')
                weight = float(weight[0]) if weight else 1.0
                if constantWeight:
                    weight = 1.0
                if absWeight:
                    weight = abs(weight)
                nodes.update([src, tgt])
                edges.append((src, tgt, weight))
                if addBidirectionEdge:
                    edges.append((tgt, src, weight))
        self.dic_node2idx = {node: idx for idx, node in enumerate(sorted(nodes))}
        size = len(self.dic_node2idx)
        row, col, data = [], [], []
        for src, tgt, weight in edges:
            i, j = self.dic_node2idx[src], self.dic_node2idx[tgt]
            row.append(i)
            col.append(j)
            data.append(weight)
        self.adjacency_matrix = self._normalize_cols(row, col, data, size)

    def _normalize_cols(self, row, col, data, size):
        matrix = coo_matrix((data, (row, col)), shape=(size, size))
        matrix = normalize(matrix, norm='l1', axis=0)
        return matrix


def main_propagation(argv):
        # set up argument parsing
        parser = argparse.ArgumentParser(usage='''\
        python %(prog)s input_graphs seed -o myout -e 0.01
        ''')
        parser.add_argument('input_graphs', nargs='+', help='Original graph input file, in edge list format')
        parser.add_argument('seed', help='Seed file, to pull start nodes from')
        parser.add_argument('-o',required=True, help='outfile')
        #parser.add_argument('-TFlist', default=None, help='TFlist for masking non-TF genes')
        parser.add_argument('-e', '--restart_prob', type=float, default=0.1, help='Restart probability for random walk')
        parser.add_argument('-constantWeight', default='False', choices=['True', 'False'], help='Whether constant weight or not')
        parser.add_argument('-absoluteWeight', default='False', choices=['True', 'False'], help='Whether absolute weight or not')
        parser.add_argument('-addBidirectionEdge', default='False', choices=['True', 'False'], help='Whether adding bidirection edges')
        parser.add_argument('-normalize', default='False', choices=['True', 'False'], help='Whether normalizing p0 or not')
        args = parser.parse_args()
        #if args.TFlist != None:
        #        set_TF=set()
        #        IF=open(args.TFlist,'r')
        #        for line in IF:
        #                s=line.strip().split()
        #                TF=s[0]
        #                set_TF.add(TF)

        try:
                fp = open(args.seed, "r")
        except IOError:
                 sys.exit("Error opening file {}".format(args.seed))
        
        lst_columnName=['0']
        lst_seed=[]
        lst_weights=[]
        for line in fp.readlines():
                s = line.rstrip().split()
                if len(s) >= 2:
                        if not isNum(s[1]):#header
                                lst_columnName=s[1:]
                        else:
                                lst_columnName=[str(i) for i in np.arange(len(s[1:]))+1]      
                seed = s[0]
                if len(s) == 1: #if only the gene lists are given, set weights to 1
                        weights = [1.0]
                if len(s) >= 2:
                        weights = list(map(float,s[1:]))
                lst_seed.append(seed)
                lst_weights.append(weights)
        arr_weights=np.array(lst_weights)
        fp.close()
        
        # run the experiments, and write a rank list to stdout
        dic_node2weights={}
        set_nodes=set()
        lst_wk = []
        network_name=[]
        for input_graph in args.input_graphs:
                if len(input_graph.split('/')) >= 2:
                        network_name.append(input_graph.split('/')[-2])
                else:
                        network_name.append(input_graph)
                # wk = Walker(input_graph, constantWeight=str2bool(args.constantWeight), absWeight=str2bool(args.absoluteWeight), addBidirectionEdge=str2bool(args.addBidirectionEdge))#1 wk for 1 input graph
                wk = OptimizedWalker(input_graph, constantWeight=str2bool(args.constantWeight), 
                     absWeight=str2bool(args.absoluteWeight), 
                     addBidirectionEdge=str2bool(args.addBidirectionEdge))

                # set_nodes |= set(wk.OG.nodes())
                set_nodes |= set(wk.dic_node2idx.keys())
                lst_wk.append(wk)
        column_name=[]
         
        for idx, wk in enumerate(lst_wk):#if there's multiple input graphs
                for j in range(arr_weights.shape[1]): #iterate (# samples) times
                        if len(network_name) > 1:
                                column_name.append(lst_columnName[j]+"_"+network_name[idx])
                        else:
                                column_name.append(lst_columnName[j])
                        if sum(np.abs(arr_weights[:,j])) == 0.0:
                                for node in set_nodes:
                                        if node not in dic_node2weights:
                                                dic_node2weights[node]=[]
                                        dic_node2weights[node].append(0.0)
                                continue
                        seed2weight=dict()
                        for ii in range(len(lst_seed)):
                                seed2weight[lst_seed[ii]]=arr_weights[ii,j]
                        lst_node_weight = wk.run_exp(seed2weight, args.restart_prob, normalize=str2bool(args.normalize))
                        set_tmpNodes=set()
                        for node, weight, all_weight in lst_node_weight:
                                if node not in dic_node2weights:
                                        dic_node2weights[node]=[]
                                dic_node2weights[node].append(weight)
                                set_tmpNodes.add(node)
                        for node in set_nodes-set_tmpNodes:
                                if node not in dic_node2weights:
                                        dic_node2weights[node]=[]
                                dic_node2weights[node].append(0.0)
                                
        # OF=open(args.o,'w')
        # #OF.write('Gene\t'+'\t'.join(column_name)+'\n')
        # for node, weights in dic_node2weights.items():
        #         #OF.write('\t'.join(map(str,[node]+all_weight))+'\n')
        #         OF.write('\t'.join(map(str,[node]+weights))+'\n')
        #         OF.flush()
        # OF.close()
        with open(args.o, 'w') as OF:
                # 각 노드별 평균 또는 첫 번째 열(score) 기준으로 정렬
                sorted_items = sorted(dic_node2weights.items(), key=lambda x: -x[1][0])  # ← 0번째 열 기준 내림차순

                for node, weights in sorted_items:
                        OF.write('\t'.join(map(str, [node] + weights)) + '\n')


if __name__ == '__main__':
        main_propagation(sys.argv)
