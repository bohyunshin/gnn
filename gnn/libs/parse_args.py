import argparse


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True, choices=["cora"])
    parser.add_argument(
        "--model_name", type=str, required=True, choices=["gcn", "graphsage", "gat"]
    )
    parser.add_argument(
        "--sage_aggregator", type=str, default="mean", choices=["mean", "max", "lstm"]
    )
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--leaky_relu_alpha", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--is_test", action="store_true")
    return parser.parse_args()
