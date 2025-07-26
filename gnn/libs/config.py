import yaml
from easydict import EasyDict


def load_yaml(path: str) -> EasyDict:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return EasyDict(config)
