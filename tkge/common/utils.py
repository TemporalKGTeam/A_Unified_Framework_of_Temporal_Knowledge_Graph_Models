from typing import Tuple

import torch
import numpy as np

from tkge.common.config import Config


def repeat_interleave(inputs: torch.Tensor, n_tile: int, dim: int = None) -> torch.Tensor:
    init_dim = inputs.size(dim)
    repeat_idx = [1] * inputs.dim()
    repeat_idx[dim] = n_tile

    if hasattr(torch.Tensor, 'repeat_interleave'):
        return torch.Tensor.repeat_interleave(inputs, repeat_idx, dim)
    else:

        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))

        return torch.index_select(inputs, dim, order_index)


class LocalConfig():
    def __init__(self, config: Config, fusion: str, transformation: str):
        self.global_config  = config

        local_options = None
        self.local_config = Config.create_from_dict(local_options)

    def __enter__(self):
        return self.local_config

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.local_config

        return True



