import argparse
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from gnn.model.gcn import GCN


class GCNInference:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize GCN model for inference

        Args:
            model_path: Path to the saved model weights
            device: Device to run inference on ('cpu' or 'cuda')
        """

        # Load the saved weights
        checkpoint = torch.load(model_path, map_location=device)

        # Extract model parameters from checkpoint
        state_dict = checkpoint["state_dict"]
        config = checkpoint["config"]
        self.features = checkpoint["features"].to(device)
        self.adj = checkpoint["adj"].to(device)
        self.idx_map = checkpoint["idx_map"]

        # Initialize model with saved configuration
        self.model = GCN(
            num_feature=config["num_feature"],
            hidden_dim=config["hidden_dim"],
            num_class=config["num_class"],
            dropout=config["dropout"],
        )

        # Load the weights
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode

    def predict(self) -> Tuple[Tensor, Tensor]:
        """
        Perform inference and return probs and predictions
        """
        self.model.eval()
        with torch.no_grad():
            # Forward pass
            log_probs = self.model(self.features, self.adj)
            # Convert log probabilities to probabilities
            probs = torch.exp(log_probs)
        y_pred = torch.argmax(probs, dim=1)
        return probs, y_pred

    def extract_embeddings(self, layer: str = "hidden") -> torch.Tensor:
        """
        Extract node embeddings from specified layer

        Args:
            layer: Which layer to extract embeddings from
                  - 'hidden': After first GCN layer (before final classification)
                  - 'final': After second GCN layer (before softmax)

        Returns:
            Node embeddings [num_nodes, embedding_dim]
        """
        self.model.eval()
        with torch.no_grad():
            # Forward pass through first layer
            x = F.relu(self.model.gc1(self.features, self.adj))

            if layer == "hidden":
                # Return embeddings after first GCN layer (most common choice)
                return x
            elif layer == "final":
                # Apply dropout and second layer
                x = self.model.gc2(x, self.adj)
                return x
            else:
                raise ValueError("layer must be 'hidden' or 'final'")

    def make_result_dict(
        self, probs: Tensor, y_pred: Tensor, embeddings: Tensor
    ) -> Dict[int, Any]:
        result = {}
        idx_map_rev = {j: i for i, j in self.idx_map.items()}
        num_nodes = probs.size(0)
        for i in range(num_nodes):
            original_id = idx_map_rev[i]
            prob = probs[i].numpy()
            pred = y_pred[i].numpy()
            embedding = embeddings[i].numpy()

            result[original_id] = {"prob": prob, "pred": pred, "embedding": embedding}

        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    return parser.parse_args()


def main(args: argparse.ArgumentParser):
    gcn_inference = GCNInference(args.result_path)
    probs, y_pred = gcn_inference.predict()
    print(probs[0])
    print(y_pred[0])

    embeddings = gcn_inference.extract_embeddings("hidden")
    print(embeddings.size())
    print(embeddings[0])

    result_dict = gcn_inference.make_result_dict(probs, y_pred, embeddings)
    print(result_dict[31336])


if __name__ == "__main__":
    args = parse_args()
    main(args)
