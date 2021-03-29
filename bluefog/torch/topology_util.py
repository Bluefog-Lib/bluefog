from typing import Any, List, Optional, Tuple, Union
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


def InferSourceFromDestinationRanks(
    dst_ranks: List[int], construct_adjacency_matrix: bool = False,
) -> Union[List[int], Tuple[List[int], np.array]]:
    """Infer the source ranks from destination ranks. This is collective communication call.

    Args:
        dst_ranks: A list of destination ranks.
        construct_adjacency_matrix: If true, adjacency matrix will be return instead.
            Element w_{ij} represents the weights sending from node i to node j.
            We use column normalized style, i.e. the sum of receiving weight is 1.

    Raises:
        ValueError: If dst_ranks or src_ranks does not contain integer from 0 to size-1.

    Returns:
        If construct_adjacency_matrix is false, returns the source ranks list.
        If construct_adjacency_matrix is true, returns the the source ranks list
        and a 2-D numpy array.
    """
    is_valid, error_msg = _check_ranks(dst_ranks, bf.rank(), bf.size())
    assert is_valid, f"The format of dst_ranks is wrong: {error_msg}"
    return _infer_topo(
        dst_ranks,
        transpose=False,
        construct_adjacency_matrix=construct_adjacency_matrix,
    )


def InferDestinationFromSourceRanks(
    src_ranks: List[int], construct_adjacency_matrix: bool = False,
) -> Union[List[int], np.array]:
    """Infer the destination ranks from source ranks. This is collective communication call.

    Args:
        src_ranks: A list of destination ranks.
        construct_adjacency_matrix: If true, adjacency matrix will be return instead.
            Element w_{ij} represents the weights sending from node i to node j.
            We use column normalized style, i.e. the sum of receiving weight is 1.

    Raises:
        ValueError: If dst_ranks or src_ranks does not contain integer from 0 to size-1.

    Returns:
        If construct_adjacency_matrix is false, returns the destination ranks list.
        If construct_adjacency_matrix is true, returns the the sodestinationrce ranks
        list and a 2-D numpy array.
    """
    is_valid, error_msg = _check_ranks(src_ranks, bf.rank(), bf.size())
    assert is_valid, f"The format of src_ranks is wrong: {error_msg}"
    return _infer_topo(
        src_ranks,
        transpose=True,
        construct_adjacency_matrix=construct_adjacency_matrix,
    )


def _infer_topo(
    rank_list: List[int], transpose: bool, construct_adjacency_matrix: bool
):
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
    if return_list is None:
        return_list = []

    if not construct_adjacency_matrix:
        return return_list

    # construct_adjacency_matrix
    W = np.eye(bf.size())
    for k, adj in adjacency_dict.items():
        W[k, adj] = 1
    if transpose:
        W = W.T

    return return_list, W / W.sum(axis=1)
