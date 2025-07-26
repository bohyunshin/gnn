import argparse

import pytest


@pytest.fixture(scope="function")
def setup_config(request):
    data_name, model_name = request.param
    args = argparse.ArgumentParser()
    args.data_name = data_name
    args.model_name = model_name
    args.learning_rate = 0.01
    args.weight_decay = 5e-4
    args.dropout = 0.5
    args.patience = 5
    args.epochs = 5
    args.is_test = True
    return args
