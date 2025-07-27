import os
import logging
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from gnn.libs.utils import normalize


class DataLoader:
    def __init__(self, data_path: str, data_name: str, logger: logging.Logger) -> None:
        self.node_data_path = os.path.join(data_path, f"{data_name}.content")
        self.edge_data_path = os.path.join(data_path, f"{data_name}.cites")
        self.logger = logger

    def load(
        self,
    ) -> Tuple[sparse.coo_matrix, sparse.coo_matrix, Tensor, Dict[int, int]]:
        idx_features_labels = np.genfromtxt(
            fname=self.node_data_path,
            dtype=np.dtype(str),
        )
        features = sparse.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = OneHotEncoder(sparse_output=False).fit_transform(
            idx_features_labels[:, -1].reshape(-1, 1)
        )
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

        self.logger.info(f"Number of nodes: {labels.shape[0]}")
        self.logger.info(f"Number of output labels: {labels.shape[1]}")

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unmapped = np.genfromtxt(
            fname=self.edge_data_path,
            dtype=np.int32,
        )
        edges = np.array(
            list(map(idx_map.get, edges_unmapped.flatten())),
            dtype=np.int32,
        ).reshape(edges_unmapped.shape)
        adj = sparse.coo_matrix(
            (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
            shape=(labels.shape[0], labels.shape[0]),
            dtype=np.float32,
        )

        self.logger.info(f"Number of edges: {edges.shape[0]}")

        # build symmetric adjacency matrix.
        # when adj[i,j] != adj[j,i], set max(adj[i,j], adj[j,i]) = adj[i,j] = adj[j,i]
        # in the kipf GCN paper, the author assumes undirected, symmetric adjacency matrix.
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # row-wise normalize feature.
        # it is not mandatory to do this normalization,
        # but kipf mentioned feature normalization in the GCN paper.
        features = normalize(features)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(labels)[1])

        return features, adj, labels, idx_map
