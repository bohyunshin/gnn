from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from gnn.layer.aggregation import SageLayer


class GraphSage(nn.Module):
    def __init__(
        self,
        num_feature: int,
        hidden_dim: int,
        num_class: int,
        aggregator: str,
        dropout: float,
        **kwargs: Any,
    ) -> None:
        super(GraphSage, self).__init__()
        self.sage1 = SageLayer(num_feature, hidden_dim, aggregator)
        self.sage2 = SageLayer(hidden_dim, num_class, aggregator)
        self.dropout = dropout

    def forward(self, feature: Tensor, adj: Tensor) -> Tensor:
        x = F.relu(self.sage1(feature, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage2(x, adj)
        return F.log_softmax(x, dim=1)
