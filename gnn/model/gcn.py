import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from gnn.layer.convolution import GraphConvolution


class GCN(nn.Module):
    def __init__(
        self, num_feature: int, hidden_dim: int, num_class: int, dropout: float
    ) -> None:
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(num_feature, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, num_class)
        self.dropout = dropout

    def forward(self, feature: Tensor, adj: Tensor) -> Tensor:
        x = F.relu(self.gc1(feature, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
