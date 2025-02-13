import os
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from sklearn.metrics import f1_score


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyDataset(InMemoryDataset):
    def __init__(self, root, name):
        self.root = root
        self.name = name
        super().__init__(root)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name)

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name)

    @property
    def raw_file_names(self):
        return [
            'valid_hindex_0.txt', 'test_hindex_0.txt', 'hypergraph.txt',
            'hypergraph_pos.txt', 'degree_nodecentrality_0.txt',
            'eigenvec_nodecentrality_0.txt', 'kcore_nodecentrality_0.txt',
            'pagerank_nodecentrality_0.txt', 'nodefeatures_44d.txt']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        dn = self.raw_dir + '/'
        with open(dn + 'valid_hindex_0.txt') as file:
            valids = set(line.strip() for line in file.readlines())
        with open(dn + 'test_hindex_0.txt') as file:
            tests = set(line.strip() for line in file.readlines())
        valid_mask, test_mask, e0, e1 = [], [], [], []
        with open(dn + 'hypergraph.txt') as file:
            e = 0
            for line in file.readlines():
                line = line.strip()
                if not line:
                    continue
                cs = line.split('\t')
                if self.name == 'DBLP':
                    idx, cs = cs[0].strip("'"), cs[1:]
                else:
                    idx = str(e)
                if not cs:
                    continue
                vm = idx in valids
                tm = idx in tests
                for c in cs:
                    e0.append(e)
                    e1.append(int(c))
                    valid_mask.append(vm)
                    test_mask.append(tm)
                e += 1
        valid_mask = torch.tensor(valid_mask, dtype=bool)
        test_mask = torch.tensor(test_mask, dtype=bool)
        train_mask = ~(valid_mask | test_mask)
        e0 = torch.tensor(e0)
        e1 = torch.tensor(e1)
        ids, inv = e1.unique(return_inverse=True)
        E = torch.cat((e0.view(1, -1), inv.view(1, -1)))  # E2V

        Y = []
        with open(dn + 'hypergraph_pos.txt') as file:
            for line in file.readlines():
                line = line.strip()
                if not line:
                    continue
                cs = line.split('\t')
                if self.name == 'DBLP':
                    cs = cs[1:]
                if not cs:
                    continue
                for c in cs:
                    Y.append(int(c))
        Y = torch.tensor(Y)
        Y = Y - Y.min()

        num_nodes = ids.shape[0]
        idx = torch.zeros(ids.max() + 1, dtype=int)
        idx += idx.shape[0]  # an Out-of-bounds indicator
        idx[ids] = torch.arange(num_nodes)
        centrality = torch.zeros(num_nodes, 4)
        for i, key in enumerate('degree eigenvec kcore pagerank'.split()):
            csv = pd.read_csv(
                dn + '%s_nodecentrality_0.txt' % key, sep='\t', header=0)
            j = idx[torch.from_numpy(csv.node.values)]
            centrality[j, i] = torch.from_numpy(csv[key].values).float()
        fn = dn + 'withinorderpe.txt'
        if os.path.exists(fn):
            edge_attr = torch.from_numpy(np.loadtxt(fn)).float()
        else:
            edge_attr = []
            for i in range(E[0].max().item() + 1):
                x = centrality[E[1, E[0] == i]]
                edge_attr.append((1 + x.sort(dim=0).indices) / x.shape[0])
            edge_attr = torch.cat(edge_attr)
            np.savetxt(fn, edge_attr.numpy())

        X = torch.from_numpy(np.loadtxt(dn + 'nodefeatures_44d.txt')).float()

        self.save([Data(
            edge_index=E,
            edge_attr=edge_attr,
            x=X, y=Y, centrality=centrality,
            train=train_mask,
            valid=valid_mask,
            test=test_mask,
        )], self.processed_paths[0])


def load_model(data, **kwargs):
    if args.method == 'NT2':
        from nt2 import NT2
        return NT2(
            incidence_din=0 if data.edge_attr is None else data.edge_attr.shape[1],
            node_din=data.x.shape[1],
            dout=data.y.max().item() + 1,
            **kwargs)


def main(args):
    device = 'cpu' if args.gpu < 0 else ('cuda:%d' % args.gpu)
    g = MyDataset('dataset', args.dataset).data.to(device)
    if args.without_pe:
        g.edge_attr = None
        g.x = torch.cat((g.x, g.centrality), dim=1)
    n_nodes = g.x.shape[0]
    n_edges = g.edge_index[0].max().item() + 1
    n_proxy = 0
    edge_type = g.edge_index[0] * 0
    if args.with_node_proxy:
        node_proxy = torch.arange(n_nodes).view(1, -1).repeat(2, 1).to(device)
        node_proxy[0] += n_edges
        g.edge_index = torch.cat((g.edge_index, node_proxy), dim=1)
        edge_type = torch.cat((edge_type, edge_type.max() + g.edge_index.new_ones(n_nodes)))
        n_proxy += n_nodes
    if args.with_edge_proxy:
        g.x = torch.cat((g.x, g.x.new_zeros(n_edges, g.x.shape[1])))
        edge_proxy = torch.arange(n_edges).view(1, -1).repeat(2, 1).to(device)
        edge_proxy[1] += n_nodes
        g.edge_index = torch.cat((g.edge_index, edge_proxy), dim=1)
        edge_type = torch.cat((edge_type, edge_type.max() + g.edge_index.new_ones(n_edges)))
        n_proxy += n_edges
    if n_proxy:
        edge_attr = F.one_hot(edge_type).float()
        if g.edge_attr is None:
            g.edge_attr = edge_attr
        else:
            g.edge_attr = torch.cat((
                g.edge_attr, g.edge_attr.new_zeros(n_proxy, g.edge_attr.shape[1])))
            g.edge_attr = torch.cat((g.edge_attr, edge_attr), dim=1)
    net = load_model(g, **args.__dict__).to(device)
    if n_proxy:
        fwd = net.forward
        net.forward = lambda g: fwd(g)[:-n_proxy]
    # Label weights to balance loss 
    w = 1 - g.y[g.train].bincount() / g.y[g.train].shape[0]
    print('# Params:', sum(p.numel() for p in net.parameters()))
    for run in range(args.runs):
        fix_seed(run)
        net.reset_parameters()
        opt = torch.optim.Adam(
            net.parameters(),
            lr=args.lr, weight_decay=args.weight_decay)
        best_val = {'micro': 0, 'macro': 0}
        best_test = {'micro': 0, 'macro': 0}
        best_epoch = 0
        for epoch in range(args.epochs):
            # train
            net.train()
            opt.zero_grad()
            F.cross_entropy(net(g)[g.train], g.y[g.train], weight=w).backward()
            opt.step()
            # evaluate
            with torch.no_grad():
                net.eval()
                pred = net(g).argmax(dim=1)
                log = []
                for avg in ['micro', 'macro']:
                    accs = []
                    for split in 'train valid test'.split():
                        acc = 100 * f1_score(
                            g.y[g[split]].cpu(), pred[g[split]].cpu(),
                            average=avg)
                        log.append('%s %s: %.2f' % (split, avg, acc))
                        accs.append(acc)
                    if accs[1] > best_val[avg]:
                        best_val[avg] = accs[1]
                        best_test[avg] = accs[2]
                        best_epoch = epoch
                print('Epoch: %d, %s' % (epoch, ', '.join(log)))
            if epoch - best_epoch >= args.patience:
                break
        print('Run: %s, micro: %.2f, macro: %.2f' % (
            run, best_test['micro'], best_test['macro']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, default='MLP', help=(
        'HyperPassing | ...'
    ))
    parser.add_argument('dataset', type=str, default='cora', help=(
        'DBLP | AMinerAuthor | emailEu | ...'
    ))
    parser.add_argument('--gpu', type=int, default=0, help='Default: 0')
    parser.add_argument('--runs', type=int, default=1, help='Default: 1')
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Maximum epochs. Default: 100')
    parser.add_argument(
        '--patience', type=int, default=200,
        help='Patience for early stopping. Default: 200')
    parser.add_argument(
        '--lr', type=float, default=0.001, help='Learning Rate. Default: 0.001')
    parser.add_argument(
        '--weight-decay', type=float, default=0.0, help='Default: 0')
    parser.add_argument(
        '--dropout', type=float, default=0.0, help='Default: 0')
    parser.add_argument(
        '--input-dropout', type=float, default=0.0, help='Default: 0')
    parser.add_argument('--n-layers', type=int, default=3, help='Default: 3')
    parser.add_argument(
        '--hidden', type=int, default=8,
        help='Dimension of hidden representations. Default: 32')
    parser.add_argument(
        '--heads', type=int, default=4,
        help='Number of attention heads. Default: 1')
    parser.add_argument(
        '--without-pe', action='store_true',
        help='Whether to use WithinOrderPE')
    parser.add_argument(
        '--with-node-proxy', action='store_true',
        help='Whether to add self-loops for nodes')
    parser.add_argument(
        '--with-edge-proxy', action='store_true',
        help='Whether to add self-loops for edges')
    args = parser.parse_args()
    main(args)
