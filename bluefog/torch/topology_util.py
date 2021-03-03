from typing import Any, List, Optional, Union
import collections

import numpy as np
import torch
import bluefog.torch as bf


def _check_ranks(rank_list: List[Any], self_rank: int, size: int) -> [bool, str]:
    for rank in rank_list:
        if not isinstance(rank, int):
            return False, "contain element that is not integer."
        if (rank < 0) or (rank >= size):
            return False, "contain element that is not between 0 and size-1."
    if len(set(rank_list)) != len(rank_list):
        return False, "contain duplicated elements."
    if self_rank in rank_list:
        return False, "contain self rank."
    return True, ""


def InferDestinationSourceRanks(
    *,
    dst_ranks: Optional[List[int]] = None,
    src_ranks: Optional[List[int]] = None,
    construct_adjacency_matrix: bool = False,
) -> Union[List[int], np.array]:
    """Infer the destination(source) ranks from source(destination) ranks.

    Args:
        dst_ranks: A list of destination ranks. If provided the dst_link, a corresponding
            src_ranks will be returned.
        src_ranks: A list of destination ranks. If provided the src_ranks, a corresponding
            dst_link will be returned.
        construct_adjacency_matrix: If true, adjacency matrix will be return instead.
            Element w_{ij} represents the weights sending from node i to node j.
            We use column normalized style, i.e. the sum of receiving weight is 1.

    Raises:
        ValueError: If dst_ranks or src_ranks does not contain integer from 0 to size-1.

    Returns:
        If construct_adjacency_matrix is false, returns a rank list.
        If construct_adjacency_matrix is true, returns a 2-D numpy array.
    """
    if dst_ranks is None and src_ranks is None:
        raise ValueError("Either dst_ranks or src_ranks need to be provided.")
    if dst_ranks is not None and src_ranks is not None:
        raise ValueError(
            "Only one of two argument dst_ranks or src_ranks should be provided."
        )

    rank_list = dst_ranks or src_ranks
    is_valid, error_msg = _check_ranks(rank_list, bf.rank(), bf.size())
    assert is_valid, f"The format of dst_ranks or src_ranks is wrong: {error_msg}"

    degree = len(rank_list)
    all_degree_list = bf.allgather(torch.tensor([degree], dtype=torch.int32)).numpy()
    all_rank_list = bf.allgather(torch.tensor(rank_list, dtype=torch.int32)).numpy()
    adjacency_dict = dict()
    displacement = 0
    for i, degree in enumerate(all_degree_list):
        adjacency_dict[i] = sorted(all_rank_list[displacement : displacement + degree])
        displacement += degree

    inv_adjacency_dict = collections.defaultdict(list)
    for k, adj in adjacency_dict.items():
        for v in adj:
            inv_adjacency_dict[v].append(k)
    return_list = inv_adjacency_dict.get(bf.rank())

    if not construct_adjacency_matrix:
        return return_list

    # construct_adjacency_matrix
    W = np.eye(bf.size())
    for k, adj in adjacency_dict.items():
        W[k, adj] = 1
    if dst_ranks is None:
        W = W.T

    return return_list, W / W.sum(axis=1)
