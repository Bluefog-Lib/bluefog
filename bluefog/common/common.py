from enum import Enum
from dataclasses import dataclass
from typing import Callable, Any, Union, List

import numpy as np


class DeviceType(Enum):
    CPU = "CPU"
    GPU = "GPU"


class Communicator(Enum):
    GLOBAL = 0
    LOCAL = 1
    NEIGHBOR = 2

    def __repr__(self):
        return self.name


class Framework(Enum):
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    MXNET = "mxnet"

    def __repr__(self):
        return self.value


class MPIOpsType(Enum):
    UNKNOWN = 0
    ALLREDUCE = 1
    ALLGATHER = 2
    BROADCASE = 3
    NEIGHBOR_ALLREDUCE = 4
    NEIGHBOR_ALLGATHER = 5
    WIN_PUT = 6
    WIN_GET = 7
    WIN_ACCUMULATE = 8

    def __repr__(self):
        return self.name


class Status(Enum):
    OK = "Ok"
    INPROGRESS = "In progress"
    ERROR = "Error"


class BlueFogError(Exception):
    def __init__(self, message):
        super(BlueFogError, self).__init__(message)
        self.message = message

    def __str__(self):
        return "{}: {}".format("BlueFog error", self.message)


# Table storing Tensors to be reduced, keyed by unique name.
# This table contains everything necessary to do the reduction.
# We use mpi4py on cpu for now so the tensor is just a numpy array.
@dataclass
class TensorTableEntry:
    tensor_name: str
    tensor_type: type
    tensor: np.ndarray
    output: np.ndarray
    root_rank: int = 0
    device: DeviceType = DeviceType.CPU
    callback: Callable = None


# Similar to TensorTableEntry, storing the necessary information
# related with one-sided communication ops (i.e., win_* ops).
@dataclass
class WindowTableEntry:
    tensor_type: type
    memory_shape: list
    window_name: str
    window_object: Any = None
    disp_unit: int = 1
    src_rank: Union[int, List[int]] = -1
    dst_rank: Union[int, List[int]] = -1
    memory: np.ndarray = None
    device: DeviceType = DeviceType.CPU
    device_id: int = -1
    callback: Callable = None
