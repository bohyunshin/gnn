from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) Layer implementation
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        num_heads (int): Number of attention heads
        bias (bool): Whether to use bias in linear transformation
        dropout (float): Dropout rate
        alpha (float): LeakyReLU negative slope for attention mechanism
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads=3,
        bias=True,
        dropout=0.0,
        alpha=0.2,
    ):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha

        # Linear transformation for each attention head
        self.W = nn.ModuleList(
            [nn.Linear(in_features, out_features, bias=False) for _ in range(num_heads)]
        )

        # Attention mechanism parameters for each attention head
        self.a = nn.ModuleList(
            [nn.Linear(2 * out_features, 1, bias=False) for _ in range(num_heads)]
        )

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.dropout_layer = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        for i in range(self.num_heads):
            nn.init.xavier_uniform_(self.W[i].weight)
            nn.init.xavier_uniform_(self.a[i].weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def compute_attention(
        self, H: Tensor, A: torch.sparse.FloatTensor, head_idx: int
    ) -> Tuple[torch.sparse.FloatTensor, Tensor]:
        """
        Compute attention coefficients for a single head (vectorized)
        Args:
            H (torch.Tensor): Node features [N, in_features]
            A (torch.sparse.FloatTensor): Sparse adjacency matrix [N, N]
            head_idx (int): Index of the attention head
        Returns:
            torch.sparse.FloatTensor: Attention coefficients [N, N]
        """
        # Number of nodes
        N = H.size(0)

        # Linear transformation: H' = W * H
        H_transformed = self.W[head_idx](H)  # [N, out_features]

        # Get edge indices from sparse adjacency matrix
        A_coalesced = A.coalesce()
        edge_indices = A_coalesced.indices()  # [2, num_edges]
        src_nodes = edge_indices[0]  # Source nodes
        dst_nodes = edge_indices[1]  # Destination nodes

        # Prepare attention input: [h_i || h_j] for each edge (i,j)
        h_src = H_transformed[src_nodes]  # [num_edges, out_features]
        h_dst = H_transformed[dst_nodes]  # [num_edges, out_features]
        attention_input = torch.cat(
            [h_src, h_dst], dim=1
        )  # [num_edges, 2*out_features]

        # Compute attention scores: e_ij = LeakyReLU(a^T [h_i || h_j])
        e = self.a[head_idx](attention_input).squeeze(-1)  # [num_edges]
        e = self.leakyrelu(e)

        # Vectorized softmax computation using scatter operations
        # Step 1: Find maximum for each source node (for numerical stability)
        e_max = torch.full((N,), float("-inf"), device=e.device, dtype=e.dtype)
        e_max.scatter_reduce_(0, src_nodes, e, reduce="amax")

        # Step 2: Subtract max and compute exp
        e_shifted = e - e_max[src_nodes]  # [num_edges]
        e_exp = torch.exp(e_shifted)  # [num_edges]

        # Step 3: Compute sum of exp for each source node
        exp_sum = torch.zeros(N, device=e.device, dtype=e.dtype)
        exp_sum.scatter_add_(0, src_nodes, e_exp)

        # Step 4: Compute attention coefficients (softmax)
        attention_coeffs = e_exp / (
            exp_sum[src_nodes] + 1e-16
        )  # Add small epsilon for stability

        # Apply dropout to attention coefficients
        attention_coeffs = self.dropout_layer(attention_coeffs)

        # Create sparse attention matrix
        attention_sparse = torch.sparse.FloatTensor(
            edge_indices, attention_coeffs, A.size()
        )

        return attention_sparse, H_transformed

    def forward(self, H: Tensor, A: torch.sparse.FloatTensor) -> Tensor:
        """
        Forward pass of GAT layer
        Args:
            H (torch.Tensor): Node feature matrix [N, in_features]
            A (torch.sparse.FloatTensor): Sparse adjacency matrix [N, N]
        Returns:
            Tensor: Updated node features [N, out_features * num_heads] if concat
                    or [N, out_features] if not concat
        """
        outputs = []

        # Process each attention head
        for head in range(self.num_heads):
            # Compute attention coefficients and transformed features
            attention_matrix, H_transformed = self.compute_attention(H, A, head)

            # Apply attention: h_i' = sum_j(alpha_ij * h_j')
            output = torch.sparse.mm(attention_matrix, H_transformed)
            outputs.append(output)

        # Combine multi-head outputs
        # Average all heads following gat paper
        final_output = torch.stack(outputs, dim=0).mean(dim=0)  # [N, out_features]

        # Add bias if present
        if self.bias is not None:
            final_output = final_output + self.bias

        return final_output

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_features}, {self.out_features}, "
            f"heads={self.num_heads}"
        )
