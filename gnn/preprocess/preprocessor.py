from torch import Tensor
from scipy import sparse

from gnn.libs.utils import normalize, sparse_mx_to_torch_sparse_tensor


def preprocess_adjacency_matrix(adj: sparse.coo_matrix, model_name: str) -> Tensor:
    if model_name == "gcn":
        # add self loop in gcn
        adj = adj + sparse.eye(adj.shape[0])
    # row-wise normalization.
    # in gcn, this equals to D^{-1/2} A D^{-1/2}
    # for graphsage, this equals to mean aggregator
    adj = normalize(adj)
    return sparse_mx_to_torch_sparse_tensor(adj)
