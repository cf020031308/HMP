

import torch
import pickle
import os
import ipdb
import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from torch_sparse import coalesce
from sklearn.feature_extraction.text import CountVectorizer

def load_LE_dataset(path, dataset):
    # load edges, features, and labels.
    print('Loading {} dataset...'.format(dataset))
    
    file_name = f'{dataset}.content'
    p2idx_features_labels = os.path.join(path, file_name)
    idx_features_labels = np.genfromtxt(p2idx_features_labels,
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = torch.LongTensor(idx_features_labels[:, -1].astype(float))


    print ('load features')

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    file_name = f'{dataset}.edges'
    p2edges_unordered = os.path.join(path, file_name)
    edges_unordered = np.genfromtxt(p2edges_unordered,
                                    dtype=np.int32)
    
    
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    print ('load edges')


    projected_features = torch.FloatTensor(np.array(features.todense()))

    
    # From adjacency matrix to edge_list
    edge_index = edges.T 
    assert edge_index[0].max() == edge_index[1].min() - 1

    # check if values in edge_index is consecutive. i.e. no missing value for node_id/he_id.
    assert len(np.unique(edge_index)) == edge_index.max() + 1
    
    num_nodes = edge_index[0].max() + 1
    num_he = edge_index[1].max() - num_nodes + 1
    
    edge_index = np.hstack((edge_index, edge_index[::-1, :]))
    
    # build torch data class
    data = Data(
            x = torch.FloatTensor(np.array(features[:num_nodes].todense())), 
            edge_index = torch.LongTensor(edge_index),
            y = labels[:num_nodes])
    data.y = data.y - data.y.min()

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = len(np.unique(edge_index))
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)
            
    
    data.num_features = data.x.shape[-1]
    data.num_classes = len(np.unique(labels[:num_nodes].numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = num_he
    
    return data

def load_citation_dataset(path, dataset):
    '''
    this will read the citation dataset from HyperGCN, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]
    '''
    print(f'Loading hypergraph dataset from hyperGCN: {dataset}')

    # first load node features:
    with open(os.path.join(path, 'features.pickle'), 'rb') as f:
        features = pickle.load(f)
        features = features.todense()

    # then load node labels:
    with open(os.path.join(path, 'labels.pickle'), 'rb') as f:
        labels = pickle.load(f)

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    with open(os.path.join(path, 'hypergraph.pickle'), 'rb') as f:
        # hypergraph in hyperGCN is in the form of a dictionary.
        # { hyperedge: [list of nodes in the he], ...}
        hypergraph = pickle.load(f)

    print(f'number of hyperedges: {len(hypergraph)}')

    edge_idx = num_nodes
    node_list = []
    edge_list = []
    for he in hypergraph.keys():
        cur_he = hypergraph[he]
        cur_size = len(cur_he)

        node_list += list(cur_he)
        edge_list += [edge_idx] * cur_size

        edge_idx += 1

    edge_index = np.array([ node_list + edge_list,
                            edge_list + node_list], dtype = np.int32)
    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)

    data.num_features = features.shape[-1]
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = len(hypergraph)

    return data

def load_yelp_dataset(path, dataset, name_dictionary_size = 1000):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - latitude, longitude
        - state, in one-hot coding. 
        - city, in one-hot coding. 
        - name, in bag-of-words

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    print(f'Loading hypergraph dataset from {dataset}')

    # first load node features:
    # load longtitude and latitude of restaurant.
    latlong = pd.read_csv(os.path.join(path, 'yelp_restaurant_latlong.csv')).values

    # city - zipcode - state integer indicator dataframe.
    loc = pd.read_csv(os.path.join(path, 'yelp_restaurant_locations.csv'))
    state_int = loc.state_int.values
    city_int = loc.city_int.values

    num_nodes = loc.shape[0]
    state_1hot = np.zeros((num_nodes, state_int.max()))
    state_1hot[np.arange(num_nodes), state_int - 1] = 1

    city_1hot = np.zeros((num_nodes, city_int.max()))
    city_1hot[np.arange(num_nodes), city_int - 1] = 1

    # convert restaurant name into bag-of-words feature.
    vectorizer = CountVectorizer(max_features = name_dictionary_size, stop_words = 'english', strip_accents = 'ascii')
    res_name = pd.read_csv(os.path.join(path, 'yelp_restaurant_name.csv')).values.flatten()
    name_bow = vectorizer.fit_transform(res_name).todense()

    features = np.hstack([latlong, state_1hot, city_1hot, name_bow])

    # then load node labels:
    df_labels = pd.read_csv(os.path.join(path, 'yelp_restaurant_business_stars.csv'))
    labels = df_labels.values.flatten()

    num_nodes, feature_dim = features.shape
    assert num_nodes == len(labels)
    print(f'number of nodes:{num_nodes}, feature dimension: {feature_dim}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)

    # The last, load hypergraph.
    # Yelp restaurant review hypergraph is store in a incidence matrix.
    H = pd.read_csv(os.path.join(path, 'yelp_restaurant_incidence_H.csv'))
    node_list = H.node.values - 1
    edge_list = H.he.values - 1 + num_nodes

    edge_index = np.vstack([node_list, edge_list])
    edge_index = np.hstack([edge_index, edge_index[::-1, :]])

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)
    # assert data.y.min().item() == 0
    data.y = data.y - data.y.min()

    # data.coalesce()
    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)

    data.num_features = features.shape[-1]
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = int(H.he.values.max())

    return data

def synthesize_heterophily(dataset, feature_noise, feature_dim):
    hetero = int(dataset.split('-')[-1])
    num_nodes = 5000
    num_classes = 2
    num_edges = 1000
    edge_size = 15

    labels = torch.randint(0, num_classes, (num_nodes, ))
    features = torch.nn.functional.one_hot(
        labels, feature_dim or num_classes).float()
    features += torch.randn(features.shape) * float(feature_noise)
    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    class0 = torch.arange(num_nodes)[labels == 0]
    class1 = torch.arange(num_nodes)[labels == 1]
    # NOTE: The original designed distribution is imbalanced wrt the two classes
    # e0 = torch.cat((
    #     class0[torch.randint(0, class0.shape[0], (num_edges, hetero))],
    #     class1[torch.randint(0, class1.shape[0], (num_edges, edge_size - hetero))]), dim=1)
    e1 = torch.arange(num_edges).view(-1, 1).repeat(1, edge_size)
    m = num_edges // 2
    e0 = torch.cat((
        torch.cat((
            class0[torch.randint(0, class0.shape[0], (m, hetero))],
            class1[torch.randint(0, class1.shape[0], (m, hetero))])),
        torch.cat((
            class1[torch.randint(0, class1.shape[0], (m, edge_size - hetero))],
            class0[torch.randint(0, class0.shape[0], (m, edge_size - hetero))]))), dim=1)

    edge_index = torch.cat((
        e0.view(1, -1), (num_nodes + e1).view(1, -1)), dim=0)
    edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

    data = Data(x=features, edge_index=edge_index, y=labels)
    data.num_features = features.shape[-1]
    data.num_classes = num_classes
    data.num_nodes = num_nodes
    data.num_hyperedges = num_edges
    return data

def synthesize_hyperchain(feature_noise, feature_dim, extra):
    width = extra['width']
    num_edges_perchain = extra['length']
    num_classes = feature_dim
    num_chains = 1000
    num_nodes_perchain = (num_edges_perchain + 1) * width
    num_nodes = num_nodes_perchain * num_chains
    num_edges = num_edges_perchain * num_chains

    labels = torch.randint(0, num_classes, (num_chains, num_nodes_perchain))
    labels[:, :width] = labels[:, -width:] = labels[:, :1]
    labels = labels.view(-1)
    features = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    features = features.view(num_chains, num_edges_perchain + 1, width, num_classes)
    features[:, 1:-1] *= float(feature_noise)
    features[:, -1] = 0
    features = features.view(num_nodes, -1)
    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    e0 = torch.arange(num_nodes_perchain - width).view(-1, 1).repeat(1, 2)
    e0[:, 1] += width
    e0 = e0.view(1, -1)
    e1 = num_nodes + torch.arange(num_edges_perchain).view(-1, 1).repeat(1, 2 * width).view(1, -1)
    edge_index = torch.cat([
        torch.cat((e0 + i * num_nodes_perchain, e1 + i * num_edges_perchain))
        for i in range(num_chains)], dim=1)
    edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)

    data = Data(x=features, edge_index=edge_index, y=labels)
    data.num_features = features.shape[-1]
    data.num_classes = num_classes
    data.num_nodes = num_nodes
    data.num_hyperedges = num_edges

    data.has_y = torch.arange(num_nodes).view(num_chains, num_edges_perchain + 1, width)[:, -1].flatten()
    return data

def load_cornell_dataset(path, dataset, feature_noise = 0.1, feature_dim = None, extra=None):
    '''
    this will read the yelp dataset from source files, and convert it edge_list to 
    [[ -V- | -E- ]
     [ -E- | -V- ]]

    each node is a restaurant, a hyperedge represent a set of restaurants one user had been to.

    node features:
        - add gaussian noise with sigma = nosie, mean = one hot coded label.

    node label:
        - average stars from 2-10, converted from original stars which is binned in x.5, min stars = 1
    '''
    if dataset.startswith('synthetic'):
        return synthesize_heterophily(dataset, feature_noise, feature_dim)
    if dataset == 'chain':
        return synthesize_hyperchain(feature_noise, feature_dim, extra)

    print(f'Loading hypergraph dataset from cornell: {dataset}')

    # first load node labels
    df_labels = pd.read_csv(os.path.join(path, f'node-labels-{dataset}.txt'), names = ['node_label'])
    num_nodes = df_labels.shape[0]
    labels = df_labels.values.flatten()

    # then create node features.
    num_classes = df_labels.values.max()
    features = np.zeros((num_nodes, num_classes))

    features[np.arange(num_nodes), labels - 1] = 1
    if feature_dim is not None:
        num_row, num_col = features.shape
        zero_col = np.zeros((num_row, feature_dim - num_col), dtype = features.dtype)
        features = np.hstack((features, zero_col))

    features = np.random.normal(features, feature_noise, features.shape)
    print(f'number of nodes:{num_nodes}, feature dimension: {features.shape[1]}')

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    labels = labels - labels.min() # shift label to 0

    # The last, load hypergraph.
    # Corenll datasets are stored in lines of hyperedges. Each line is the set of nodes for that edge.
    p2hyperedge_list = os.path.join(path, f'hyperedges-{dataset}.txt')
    node_list = []
    he_list = []
    he_id = num_nodes

    with open(p2hyperedge_list, 'r') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            cur_set = line.split(',')
            cur_set = [int(x) for x in cur_set]

            node_list += cur_set
            he_list += [he_id] * len(cur_set)
            he_id += 1
    # shift node_idx to start with 0.
    node_idx_min = np.min(node_list)
    node_list = [x - node_idx_min for x in node_list]

    edge_index = [node_list + he_list, 
                  he_list + node_list]

    edge_index = torch.LongTensor(edge_index)

    data = Data(x = features,
                edge_index = edge_index,
                y = labels)
    assert data.y.min().item() == 0

    # There might be errors if edge_index.max() != num_nodes.
    # used user function to override the default function.
    # the following will also sort the edge_index and remove duplicates. 
    total_num_node_id_he_id = edge_index.max() + 1
    data.edge_index, data.edge_attr = coalesce(data.edge_index, 
            None, 
            total_num_node_id_he_id, 
            total_num_node_id_he_id)

    data.num_features = features.shape[-1]
    data.num_classes = len(np.unique(labels.numpy()))
    data.num_nodes = num_nodes
    data.num_hyperedges = he_id - num_nodes
    
    return data
