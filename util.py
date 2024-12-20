# -*- coding: utf-8 -*-
import numpy as np
import torch
from scipy.sparse import csr_matrix
from operator import itemgetter
import scipy.sparse as sp
np.random.seed(42)

def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))

    return matrix

class Data():
    def __init__(self, data, shuffle=False, n_node=None, graph_data = None):
        if graph_data is None:
            graph_data = data[0]

        self.raw = np.asarray(data[0])
        H_T = data_masks(np.asarray(graph_data), n_node)
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)

        self.adjacency=sparse_mx_to_torch_sparse_tensor(DHBH_T)
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length) 
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg] 
            self.targets = self.targets[shuffled_arg] 
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        inp = self.raw[index]
        reversed_sess_item = torch.nn.utils.rnn.pad_sequence([torch.tensor(t[::-1], dtype=torch.long) for t in inp], batch_first=True)
        pos_id = torch.nn.utils.rnn.pad_sequence([torch.arange(len(t)) for t in inp], batch_first=True)
        return torch.tensor(self.targets[index], dtype=torch.long) - 1,  reversed_sess_item, pos_id

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)