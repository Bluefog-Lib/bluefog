import threading
import queue
from dataclasses import dataclass
from typing import DefaultDict

import numpy as np
import networkx as nx

from bluefog.common.mpi_context import MPIContext
from bluefog.common.mpi_controller import MPIController


class Singleton(type):
    _instances = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Make the singleton thread safe.
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# Something mimic the horovod global state shared in multiple threads in C++
# Because GIL we don't need to worry about atomic in python?
@dataclass
class BlueFogGlobalState(metaclass=Singleton):
    initialized: bool = False
    shut_down: bool = False
    ops_cnt: DefaultDict = None  # How to test and ensure it is thread safe?

    topology: nx.DiGraph = None
    context: MPIContext = None
    controller: MPIController = None
    background_thread: threading.Thread = None
    background_window_thread: threading.Thread = None
    tensor_queue: queue.Queue = None
    window_queue: queue.Queue = None
    queue_cv: threading.Condition = None
