import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features: int, out_features: int, bias=True) -> None:
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: Tensor, adj: Tensor):
        # convolution
        support = torch.mm(input, self.weight)
        # multiplication with adjacency matrix using sparse matrix
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class FastGraphConvolution(nn.Module):
    """
    FastGCN Layer implementation with importance sampling

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        sample_size (int): Number of neighbors to sample for each node
        sampling_method (str): Sampling method ('uniform', 'importance')
        bias (bool): Whether to use bias in linear transformation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sample_size=5,
        sampling_method="importance",
        bias=True,
    ):
        super(FastGraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sample_size = sample_size
        self.sampling_method = sampling_method

        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def compute_sampling_probabilities(self, A: torch.sparse.FloatTensor) -> Tensor:
        """
        Compute sampling probabilities for each node based on FastGCN paper

        Args:
            A (torch.sparse.FloatTensor): Sparse adjacency matrix [N, N]

        Returns:
            Tensor: Sampling probabilities [N]
        """
        N = A.size(0)

        if self.sampling_method == "uniform":
            # Uniform sampling - all nodes have equal probability
            probs = torch.ones(N, device=A.device) / N

        elif self.sampling_method == "importance":
            # Importance sampling based on degree (as in FastGCN paper)
            # The importance sampling probability is proportional to the node's "importance"
            # In FastGCN, this is often the degree or degree^2

            # Compute node degrees from adjacency matrix
            degrees = A.sum(dim=1).to_dense()  # Row sums = out-degrees

            # FastGCN uses degree-based importance sampling
            # Probability proportional to degree (or degree^2 for higher variance reduction)
            importance_scores = (
                degrees + 1e-16
            )  # Add small epsilon to avoid zero degrees

            # Normalize to get probabilities
            probs = importance_scores / importance_scores.sum()

        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        return probs

    def sample_neighbors(
        self,
        A: torch.sparse.FloatTensor,
        node_idx: int,
        sampling_probs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Sample neighbors for a specific node

        Args:
            A (torch.sparse.FloatTensor): Sparse adjacency matrix [N, N]
            node_idx (int): Index of the node
            sampling_probs (Tensor): Sampling probabilities [N]

        Returns:
            tuple: (sampled_neighbors, sample_probs, sample_weights)
        """
        # Get neighbors of the node
        A_coalesced = A.coalesce()
        indices = A_coalesced.indices()
        src_nodes = indices[0]
        dst_nodes = indices[1]

        # Find neighbors of node_idx
        neighbor_mask = src_nodes == node_idx
        if not neighbor_mask.any():
            # No neighbors - return empty
            return (
                torch.tensor([], dtype=torch.long),
                torch.tensor([]),
                torch.tensor([]),
            )

        neighbors = dst_nodes[neighbor_mask]

        # Get sampling probabilities for neighbors
        neighbor_probs = sampling_probs[neighbors]

        # Normalize probabilities
        neighbor_probs = neighbor_probs / (neighbor_probs.sum() + 1e-16)

        # Sample from neighbors
        num_samples = min(self.sample_size, len(neighbors))

        if num_samples == 0:
            return (
                torch.tensor([], dtype=torch.long),
                torch.tensor([]),
                torch.tensor([]),
            )

        # Sample with replacement
        sampled_indices = torch.multinomial(
            neighbor_probs, num_samples, replacement=True
        )
        sampled_neighbors = neighbors[sampled_indices]
        sampled_probs = neighbor_probs[sampled_indices]

        # Compute importance sampling weights (1/p_u)
        sample_weights = 1.0 / (sampled_probs + 1e-16)

        return sampled_neighbors, sampled_probs, sample_weights

    def aggregate_neighbors(self, H: Tensor, A: torch.sparse.FloatTensor) -> Tensor:
        """
        Aggregate neighbor features using importance sampling

        Args:
            H (Tensor): Node features [N, in_features]
            A (torch.sparse.FloatTensor): Sparse adjacency matrix [N, N]

        Returns:
            Tensor: Aggregated features [N, in_features]
        """
        N = H.size(0)

        # Compute sampling probabilities **globally**: N x 1
        sampling_probs = self.compute_sampling_probabilities(A)

        # Initialize aggregated features
        aggregated = torch.zeros_like(H)

        # For each node, sample and aggregate neighbors
        for i in range(N):
            sampled_neighbors, _, sample_weights = self.sample_neighbors(
                A, i, sampling_probs
            )

            if len(sampled_neighbors) > 0:
                # Get features of sampled neighbors
                sampled_features = H[sampled_neighbors]  # [sample_size, in_features]

                # Apply importance sampling weights
                weighted_features = sampled_features * sample_weights.unsqueeze(
                    1
                )  # [sample_size, in_features]

                # Average over samples: (1/|S(v)|) * Σ(h_u / p_u)
                aggregated[i] = weighted_features.mean(dim=0)

        return aggregated

    def forward(self, H: Tensor, A: torch.sparse.FloatTensor) -> Tensor:
        """
        Forward pass of FastGCN layer

        Args:
            H (Tensor): Node feature matrix [N, in_features]
            A (torch.sparse.FloatTensor): Sparse adjacency matrix [N, N]

        Returns:
            Tensor: Updated node features [N, out_features]
        """
        # Aggregate neighbor features using importance sampling
        H_agg = self.aggregate_neighbors(H, A)

        # Apply linear transformation: W * aggregated_features
        H_out = self.linear(H_agg)

        return H_out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_features}, {self.out_features}, "
            f"sample_size={self.sample_size}, method={self.sampling_method})"
        )
