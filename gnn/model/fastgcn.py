from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from gnn.layer.convolution import FastGraphConvolution


class FastGCN(nn.Module):
    def __init__(
        self,
        num_feature: int,
        hidden_dim: int,
        num_class: int,
        dropout: float,
        sample_size: int = 5,
        sampling_method: str = "importance",
        **kwargs: Any,
    ) -> None:
        super(FastGCN, self).__init__()
        self.fastgc1 = FastGraphConvolution(
            in_features=num_feature,
            out_features=hidden_dim,
            sample_size=sample_size,
            sampling_method=sampling_method,
        )
        self.fastgc2 = FastGraphConvolution(
            in_features=hidden_dim,
            out_features=num_class,
            sample_size=sample_size,
            sampling_method=sampling_method,
        )
        self.dropout = dropout

    def forward(self, feature: Tensor, adj: Tensor) -> Tensor:
        x = F.relu(self.fastgc1(feature, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fastgc2(x, adj)
        return F.log_softmax(x, dim=1)
