from typing import Tuple

import torch
from torch import Tensor


def train_val_test_split(
    num_train: int, num_val: int, n
) -> Tuple[Tensor, Tensor, Tensor]:
    idx_train = range(num_train)
    idx_val = range(num_train, num_val)
    idx_test = range(num_val, n)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return idx_train, idx_val, idx_test
