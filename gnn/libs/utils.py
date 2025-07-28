from typing import Union, Type, Tuple

import numpy as np
import torch
from torch import Tensor
from scipy import sparse
from sklearn.metrics import f1_score

from gnn.model.gcn import GCN
from gnn.model.graphsage import GraphSage
from gnn.model.gat import GraphAttention
from gnn.model.fastgcn import FastGCN


def normalize(mx: sparse.csr_matrix) -> sparse.csr_matrix:
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sparse.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(
    sparse_mx: sparse.csr_matrix,
) -> torch.sparse.FloatTensor:
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_model_module(
    model_name: str,
) -> Type[Union[GCN, GraphSage, GraphAttention, FastGCN]]:
    if model_name == "gcn":
        return GCN
    elif model_name == "graphsage":
        return GraphSage
    elif model_name == "gat":
        return GraphAttention
    elif model_name == "fastgcn":
        return FastGCN
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def evaluate(output: Tensor, labels: Tensor) -> Tuple[float, float]:
    preds = output.max(1)[1].type_as(labels)
    f1_macro = f1_score(labels.numpy(), preds.numpy(), average="macro")
    acc = accuracy(preds, labels)
    return acc.item(), f1_macro


def accuracy(preds: Tensor, labels: Tensor) -> float:
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
