import sys
import multiprocessing

import numpy as np
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from random_walk_hyper import (
    HyperGraphRandomWalk, get_src_dst2e, get_first_order,
    parallel_get_second_order, simulate_walks_para)
from main import MyDataset, fix_seed


def random_walk_hyper(
        dataset, node_list, hyperedge_list,
        p=2, q=0.25, num_walks=10, walk_length=40, window_size=10):
    G = HyperGraphRandomWalk(p, q)
    G.data = dataset
    G.build_graph(node_list, hyperedge_list)
    edges = np.array(range(len(G.edges)))
    get_src_dst2e(G, edges)
    G.alias_n2n_1st, G.node2ff_1st = get_first_order(G)
    parallel_get_second_order(G)
    walks = simulate_walks_para(G, num_walks, walk_length)
    return walks


fix_seed()
dataset = sys.argv[1]
dim = 44
data = MyDataset('dataset', dataset).data
print('Dataset: ', dataset)
print(data)
num_nodes = data.x.shape[0]
num_edges = data.edge_index[0].max().item() + 1
hedge2node = [[] for _ in range(num_edges)]
for x, y in data.edge_index.T.tolist():
    hedge2node[x].append(y)
walks = random_walk_hyper(
    dataset, np.arange(num_nodes).astype('int'), hedge2node,
    p=2, q=0.25, num_walks=10, walk_length=40, window_size=10)
wv = Word2Vec(
    [list(map(str, walk)) for walk in walks],
    vector_size=dim, window=10, min_count=0, sg=1, epochs=1,
    workers=multiprocessing.cpu_count(),
).wv
A = [wv[str(i)] for i in range(num_nodes)]
A = StandardScaler().fit_transform(A).astype('float32')
np.savetxt('nodefeatures_%dd.txt' % dim, A)
