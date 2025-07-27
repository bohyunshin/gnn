from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from gnn.layer.attention import GraphAttentionLayer


class GraphAttention(nn.Module):
    def __init__(
        self,
        num_feature: int,
        hidden_dim: int,
        num_class: int,
        num_heads: int = 3,
        alpha: float = 0.2,
        dropout: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super(GraphAttention, self).__init__()
        self.gat1 = GraphAttentionLayer(
            in_features=num_feature,
            out_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
        )
        self.gat2 = GraphAttentionLayer(
            in_features=hidden_dim,
            out_features=num_class,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha,
        )
        self.dropout = dropout

    def forward(self, feature: Tensor, adj: Tensor) -> Tensor:
        x = F.relu(self.gat1(feature, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj)
        return F.log_softmax(x, dim=1)
