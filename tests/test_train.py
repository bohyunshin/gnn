import pytest

from gnn.train import main


@pytest.mark.parametrize(
    "setup_config",
    [("cora", "gcn")],
    indirect=["setup_config"],
)
def test_train(setup_config):
    main(setup_config)
