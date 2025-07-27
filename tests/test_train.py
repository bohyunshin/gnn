import pytest

from gnn.train import main


@pytest.mark.parametrize(
    "setup_config",
    [
        ("cora", "gcn"),
        ("cora", "graphsage"),
        ("cora", "gat"),
        ("cora", "fastgcn"),
    ],
    indirect=["setup_config"],
)
def test_train(setup_config):
    main(setup_config)
