import torch

from tkge.common.error import *


def all_candidates_of_ent_queries(queries: torch.Tensor, vocab_size: int):
    """
    Generate all candidate tuples of the queries with absent entities.
    args:
        queries: entity prediction queries with either head or tail absent / value: float('nan')
            size: [query_num, query_dim]
        vocab_size: the vocabulary size of the dataset
    return:
        candidates: size [query_num * vocab_size, query_dim]
    """

    assert torch.isnan(queries).sum(1).byte().all(), "Either head or tail should be absent."

    dim_size = queries.size(1)

    missing_pos = torch.isnan(queries).nonzero()
    candidates = queries.repeat((1, vocab_size)).view(-1, dim_size)

    for p in missing_pos:
        candidates[p[0] * vocab_size:(p[0] + 1) * vocab_size, p[1]] = torch.arange(vocab_size)

    return candidates


def forward_checking(func):
    def wrapper(*args, **kwargs):
        return_res = func(*args, **kwargs)
        if not (isinstance(return_res, tuple) and len(return_res) == 2):
            raise CodeError(f'User-defined forward methods should return scores and factors')
        if torch.isnan(return_res[0]).any():
            raise NaNError(f'Catch abnormal value(NaN) in returned scores')

        epsilon = torch.tensor(10e-16)
        if return_res[0].max().abs() < epsilon and return_res[0].max().abs():
            raise AbnormalValueError(f'Abnormal scores detected: all scores are close to zero')

        return return_res

    return wrapper
