import torch
import torch.nn as nn
from torch import Tensor


class SageLayer(nn.Module):
    """
    GraphSAGE Layer implementation
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        aggregator (str): Aggregation function ('mean', 'max', 'sum', 'lstm')
        bias (bool): Whether to use bias in linear transformation
        dropout (float): Dropout rate
    """

    def __init__(
        self, in_features: int, out_features: int, aggregator: str, bias: bool = True
    ):
        super(SageLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator

        # Linear transformation for concatenated features [h_v || h_N(v)]
        # Input dimension is 2 * in_features due to concatenation
        self.linear = nn.Linear(2 * in_features, out_features, bias=bias)

        # LSTM aggregator (if used)
        if aggregator == "lstm":
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        if hasattr(self, "lstm"):
            for name, param in self.lstm.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def aggregate_neighbors(self, H: Tensor, A: torch.sparse.FloatTensor) -> Tensor:
        """
        Aggregate neighbor features based on the specified aggregator
        Args:
            H (Tensor): Node features [N, in_features]
            A (torch.sparse.FloatTensor): Row-normalized sparse adjacency matrix [N, N]
        Returns:
            Tensor: Aggregated neighbor features [N, in_features]
        """
        if self.aggregator == "mean":
            # A is already row-normalized, so direct sparse matrix multiplication gives mean
            return torch.sparse.mm(A, H)

        elif self.aggregator == "sum":
            # For sum aggregation, we need the unnormalized adjacency matrix
            # Since A is row-normalized, we need to "unnormalize" it
            # This requires knowing the original degree of each node
            raise NotImplementedError(
                "Sum aggregation requires unnormalized adjacency matrix. "
                "Please provide the original (unnormalized) sparse adjacency matrix "
                "or use mean aggregation instead."
            )

        elif self.aggregator == "max":
            # Note: we cannot do matrix operation to do max pooling, so makes it as dense matrix.
            # If node size is huge, this could takes long time.

            # Max aggregation requires extracting neighbors
            A_coalesced = A.coalesce()
            indices = A_coalesced.indices()  # [2, num_edges]

            N = A.size(0)
            aggregated = torch.zeros_like(H)

            # Group edges by source node
            src_nodes = indices[0]  # Source nodes
            dst_nodes = indices[1]  # Destination nodes

            for i in range(N):
                # Find neighbors of node i
                neighbor_mask = src_nodes == i
                if neighbor_mask.any():
                    neighbors = dst_nodes[neighbor_mask]
                    neighbor_features = H[neighbors]  # [num_neighbors, in_features]
                    aggregated[i] = torch.max(neighbor_features, dim=0)[0]
                # If no neighbors, aggregated[i] remains zero

            return aggregated

        elif self.aggregator == "lstm":
            # Note: we cannot do matrix operation to do lstm, so makes it as dense matrix.
            # If node size is huge, this could takes long time.

            A_coalesced = A.coalesce()
            indices = A_coalesced.indices()  # [2, num_edges]

            N = A.size(0)
            aggregated = torch.zeros_like(H)

            # Group edges by source node
            src_nodes = indices[0]  # Source nodes
            dst_nodes = indices[1]  # Destination nodes

            for i in range(N):
                # Find neighbors of node i
                neighbor_mask = src_nodes == i
                if neighbor_mask.any():
                    neighbors = dst_nodes[neighbor_mask]
                    neighbor_features = H[neighbors].unsqueeze(
                        0
                    )  # [1, num_neighbors, in_features]
                    _, (h_n, _) = self.lstm(neighbor_features)
                    aggregated[i] = h_n.squeeze(0).squeeze(0)  # [in_features]
                # If no neighbors, aggregated[i] remains zero

            return aggregated

        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

    def forward(self, H: Tensor, A: torch.sparse.FloatTensor) -> Tensor:
        """
        Forward pass of GraphSAGE layer
        Args:
            H (Tensor): Node feature matrix [N, in_features]
            A (torch.sparse.FloatTensor): Row-normalized sparse adjacency matrix [N, N]
        Returns:
            Tensor: Updated node features [N, out_features]
        """

        # Step 1: Aggregate neighbor features
        H_neigh = self.aggregate_neighbors(H, A)

        # Step 2: Concatenate self features with aggregated neighbor features
        H_concat = torch.cat([H, H_neigh], dim=1)  # [N, 2 * in_features]

        # Step 3: Apply linear transformation
        H_out = self.linear(H_concat)  # [N, out_features]

        return H_out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features}, aggregator={self.aggregator})"
